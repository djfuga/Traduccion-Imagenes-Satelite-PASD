"""
download_maps.py — Descarga y preparación del dataset de pares satelital/mapa
==============================================================================

Dos fuentes de datos soportadas:

1. **Dataset Maps de Berkeley (pix2pix)**
   URL:     http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz
   Formato: pares side-by-side ya listos (satelite | mapa de carreteras)
   Tamano:  ~255 MB, ~1096 pares train + ~1098 pares val

2. **Tiles propios OSM + satelite por bounding box**
   Satelite: ESRI World Imagery  (sin API key necesaria)
   Mapa OSM: tile.openstreetmap.org
   Proceso:  descargar tiles → ensamblar → recortar → crear pares side-by-side

Sistema de coordenadas de tiles (Slippy Map de OpenStreetMap):
--------------------------------------------------------------
El mundo se divide en 2^z × 2^z cuadros a nivel de zoom z.
  z=10: tile ≈ 38km × 38km    (escala regional)
  z=13: tile ≈ 4.8km × 4.8km  (escala de ciudad)
  z=15: tile ≈ 1.2km × 1.2km  (escala de calle)

Conversion lat/lon → (tile_x, tile_y):
  x = floor( (lon + 180) / 360 * 2^z )
  y = floor( (1 - asinh(tan(lat_rad)) / pi) / 2 * 2^z )

Referencia: https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
Arquitectura Pix2Pix: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""

import gc
import io
import math
import os
import random
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, UnidentifiedImageError


# ---------------------------------------------------------------------------
# Constantes de URL
# ---------------------------------------------------------------------------

URL_DATASET_MAPS = (
    "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz"
)

# URL_SATELITE: ESRI World Imagery — no requiere API key, resolución ~1m/px
# El orden de las coordenadas en ESRI es {z}/{y}/{x}, diferente a OSM.
URL_SATELITE_TEMPLATE = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

# URL_OSM: OpenStreetMap — requiere User-Agent descriptivo por política de uso.
URL_OSM_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"

# Cabeceras HTTP para respetar la política de uso de OSM.
# Identificar el proyecto evita ser bloqueado por el servidor de tiles.
HEADERS_OSM = {
    "User-Agent": "satelite-blog-downloader/1.0 (proyecto universitario; educativo)"
}


# ===========================================================================
# DESCARGA DE ARCHIVOS
# ===========================================================================

def descargar_con_progreso(
    url: str,
    ruta_destino: Path,
    timeout: int = 60,
    reintentos: int = 3,
    mostrar_progreso: bool = True,
) -> Path:
    """
    Descarga un archivo con barra de progreso y reintentos automáticos.

    Los reintentos son necesarios porque las conexiones a servidores de tiles
    pueden fallar ocasionalmente (timeouts, rate limits, errores de red).

    Args:
        url:              URL del archivo a descargar.
        ruta_destino:     Ruta local donde guardar el archivo.
        timeout:          Tiempo máximo de espera por conexión (segundos).
        reintentos:       Número máximo de reintentos en caso de error.
        mostrar_progreso: Si True, muestra el porcentaje de descarga.

    Returns:
        Path del archivo descargado.

    Raises:
        RuntimeError: Si todos los reintentos fallan.
    """
    ruta_destino = Path(ruta_destino)
    ruta_destino.parent.mkdir(parents=True, exist_ok=True)

    for intento in range(1, reintentos + 1):
        try:
            # Construir la solicitud con cabecera User-Agent
            solicitud = urllib.request.Request(
                url,
                headers={"User-Agent": "satelite-blog-downloader/1.0"},
            )

            with urllib.request.urlopen(solicitud, timeout=timeout) as respuesta:
                tamano_total = int(respuesta.headers.get("Content-Length", 0))
                tamano_descargado = 0
                bloque = 8192  # 8 KB por bloque

                with open(ruta_destino, "wb") as archivo:
                    while True:
                        chunk = respuesta.read(bloque)
                        if not chunk:
                            break
                        archivo.write(chunk)
                        tamano_descargado += len(chunk)

                        if mostrar_progreso and tamano_total > 0:
                            porcentaje = tamano_descargado / tamano_total * 100
                            mb_actual = tamano_descargado / 1e6
                            mb_total = tamano_total / 1e6
                            print(
                                f"\r  [{porcentaje:5.1f}%] {mb_actual:.1f} / "
                                f"{mb_total:.1f} MB",
                                end="",
                                flush=True,
                            )

            if mostrar_progreso:
                print()  # Nueva linea tras la barra de progreso

            return ruta_destino

        except Exception as error:
            print(f"\n  [!] Intento {intento}/{reintentos} fallido: {error}")
            if intento < reintentos:
                espera = 2 ** intento  # Backoff exponencial: 2s, 4s, 8s
                print(f"      Esperando {espera}s antes de reintentar...")
                time.sleep(espera)

    raise RuntimeError(
        f"No se pudo descargar '{url}' tras {reintentos} intentos."
    )


# ===========================================================================
# FUENTE 1: DATASET MAPS DE BERKELEY
# ===========================================================================

def descargar_dataset_maps(
    directorio_destino: str = "data/raw",
    forzar: bool = False,
) -> Path:
    """
    Descarga y extrae el dataset Maps de Berkeley (~255 MB).

    El dataset ya viene con pares side-by-side (satelite izquierda, mapa
    derecha en una sola imagen 512x256). Esto es el formato nativo de Pix2Pix
    que nuestro DatasetParesSideBySide puede leer directamente.

    Estructura tras la extraccion:
        {directorio_destino}/maps/
            train/  — ~1096 imagenes (512x256 px c/u)
            val/    — ~1098 imagenes (512x256 px c/u)

    Args:
        directorio_destino: Directorio base donde extraer el dataset.
        forzar:             Si True, re-descarga aunque ya exista.

    Returns:
        Path al directorio 'maps/' ya extraido.
    """
    dir_destino = Path(directorio_destino)
    dir_maps = dir_destino / "maps"
    ruta_tar = dir_destino / "maps.tar.gz"

    # Comprobar si ya existe para evitar descargas innecesarias
    if dir_maps.exists() and not forzar:
        n_train = len(list((dir_maps / "train").glob("*.jpg"))) if (dir_maps / "train").exists() else 0
        n_val = len(list((dir_maps / "val").glob("*.jpg"))) if (dir_maps / "val").exists() else 0
        print(f"[Maps] Dataset ya existe: {n_train} train, {n_val} val")
        print(f"       Ruta: {dir_maps}")
        print(f"       (Usa forzar=True para re-descargar)")
        return dir_maps

    print(f"[Maps] Descargando dataset Maps de Berkeley...")
    print(f"       URL: {URL_DATASET_MAPS}")
    print(f"       Destino: {ruta_tar}")

    descargar_con_progreso(URL_DATASET_MAPS, ruta_tar)

    print(f"[Maps] Extrayendo archivo tar.gz...")
    dir_destino.mkdir(parents=True, exist_ok=True)

    with tarfile.open(ruta_tar, "r:gz") as tar:
        # Calcular total de archivos para mostrar progreso
        miembros = tar.getmembers()
        print(f"       {len(miembros)} archivos a extraer...")
        tar.extractall(path=dir_destino)

    # Eliminar el .tar.gz para liberar espacio en Colab (disco limitado)
    ruta_tar.unlink()
    print(f"[Maps] .tar.gz eliminado para liberar espacio.")

    # Contar resultados
    n_train = len(list((dir_maps / "train").glob("*.jpg")))
    n_val   = len(list((dir_maps / "val").glob("*.jpg")))
    print(f"[Maps] Extraccion completa: {n_train} train | {n_val} val")
    print(f"       Directorio: {dir_maps.resolve()}")

    return dir_maps


# ===========================================================================
# FUENTE 2: TILES PROPIOS (OSM + SATELITE)
# ===========================================================================

def lat_lon_a_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """
    Convierte coordenadas geograficas (lat, lon) al indice de tile (x, y).

    Esta es la formula estandar del sistema Slippy Map de OpenStreetMap.
    El eje Y de tiles esta invertido respecto al eje Y geografico:
    Y aumenta hacia el sur (igual que las filas de una matriz de imagen).

    Formula:
        x = floor( (lon + 180) / 360 * 2^z )
        y = floor( (1 - asinh(tan(lat_rad)) / pi) / 2 * 2^z )

    Args:
        lat:  Latitud en grados decimales (-90 a +90).
        lon:  Longitud en grados decimales (-180 a +180).
        zoom: Nivel de zoom (0 = mundo entero, 19 = maxima resolucion).

    Returns:
        Tupla (tile_x, tile_y) con los indices del tile.
    """
    n = 2 ** zoom
    tile_x = int((lon + 180.0) / 360.0 * n)

    lat_rad = math.radians(lat)
    # asinh(tan(lat)) = log(tan(lat) + 1/cos(lat)) — formula equivalente
    tile_y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)

    # Asegurar que los indices esten dentro del rango valido
    tile_x = max(0, min(n - 1, tile_x))
    tile_y = max(0, min(n - 1, tile_y))

    return tile_x, tile_y


def tile_a_bbox(tile_x: int, tile_y: int, zoom: int) -> Tuple[float, float, float, float]:
    """
    Obtiene el bounding box geografico de un tile (en grados decimales).

    Util para verificar que los tiles descargados cubren el area esperada
    y para calcular los tiles vecinos necesarios para una region.

    Args:
        tile_x: Indice X del tile.
        tile_y: Indice Y del tile.
        zoom:   Nivel de zoom.

    Returns:
        Tupla (lat_norte, lon_oeste, lat_sur, lon_este).
    """
    n = 2 ** zoom

    # Longitudes: linealmente proporcionales al indice X
    lon_oeste = tile_x / n * 360.0 - 180.0
    lon_este  = (tile_x + 1) / n * 360.0 - 180.0

    # Latitudes: transformacion de Mercator inversa
    lat_norte = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n))))
    lat_sur   = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (tile_y + 1) / n))))

    return lat_norte, lon_oeste, lat_sur, lon_este


def _descargar_tile_imagen(
    url: str,
    headers: Optional[Dict] = None,
    timeout: int = 15,
) -> Optional[Image.Image]:
    """
    Descarga un tile individual y lo devuelve como imagen PIL.

    Descarga en memoria (sin archivo temporal) para maximizar velocidad
    cuando se descargan cientos de tiles.

    Args:
        url:     URL del tile.
        headers: Cabeceras HTTP (requeridas por OSM).
        timeout: Tiempo maximo de espera.

    Returns:
        Imagen PIL (256x256 px) o None si falla la descarga.
    """
    try:
        solicitud = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(solicitud, timeout=timeout) as respuesta:
            datos = respuesta.read()
        return Image.open(io.BytesIO(datos)).convert("RGB")
    except Exception:
        return None


def descargar_region(
    bbox: Tuple[float, float, float, float],
    zoom: int,
    tipo: str = "satelite",
    directorio_salida: str = "data/raw",
    nombre_region: str = "region",
    pausa_entre_tiles: float = 0.2,
) -> Optional[Path]:
    """
    Descarga todos los tiles de una region geografica y los ensambla en una imagen.

    El proceso es:
    1. Calcular que tiles (x, y) cubren el bounding box solicitado.
    2. Descargar cada tile (256x256 px) con pausa para no sobrecargar el servidor.
    3. Ensamblar los tiles en un canvas de (N_filas*256) x (N_cols*256) px.
    4. Recortar exactamente al bounding box solicitado.
    5. Redimensionar a 256x256 px.

    IMPORTANTE sobre las politicas de uso:
    - OSM: requiere User-Agent, limite de 1 req/s por IP. Uso no comercial.
    - ESRI: limite de uso. Para produccion, considerar servicios con API key.

    Args:
        bbox:                  (lat_norte, lon_oeste, lat_sur, lon_este).
        zoom:                  Nivel de zoom (recomendado: 13-15 para ciudades).
        tipo:                  'satelite' (ESRI) o 'osm' (OpenStreetMap).
        directorio_salida:     Donde guardar la imagen ensamblada.
        nombre_region:         Nombre base del archivo de salida.
        pausa_entre_tiles:     Segundos entre tiles (respetar rate limits).

    Returns:
        Path de la imagen ensamblada, o None si no se pudo descargar.
    """
    lat_norte, lon_oeste, lat_sur, lon_este = bbox

    # Calcular el rango de tiles que cubre el bounding box
    x_min, y_min = lat_lon_a_tile(lat_norte, lon_oeste, zoom)
    x_max, y_max = lat_lon_a_tile(lat_sur,   lon_este,  zoom)

    n_cols = x_max - x_min + 1
    n_filas = y_max - y_min + 1
    total_tiles = n_cols * n_filas

    print(f"[Tiles] Region '{nombre_region}' | zoom={zoom} | {n_cols}x{n_filas}={total_tiles} tiles")

    if total_tiles > 100:
        print(f"[!] ADVERTENCIA: {total_tiles} tiles es mucho. Considera reducir el zoom o el area.")
        print(f"    Esto puede tardar varios minutos y violar los rate limits.")

    # Configurar URL template y headers segun el tipo
    if tipo == "satelite":
        url_template = URL_SATELITE_TEMPLATE
        headers = {"User-Agent": "satelite-blog-downloader/1.0"}
    elif tipo == "osm":
        url_template = URL_OSM_TEMPLATE
        headers = HEADERS_OSM
    else:
        raise ValueError(f"tipo debe ser 'satelite' u 'osm'. Recibido: '{tipo}'")

    # Crear el canvas para ensamblar los tiles
    tam_tile = 256
    canvas = Image.new("RGB", (n_cols * tam_tile, n_filas * tam_tile))

    tiles_exitosos = 0
    for j, tile_y in enumerate(range(y_min, y_max + 1)):
        for i, tile_x in enumerate(range(x_min, x_max + 1)):

            # Construir URL del tile segun el tipo de servidor
            if tipo == "satelite":
                # ESRI usa el orden {z}/{y}/{x} (inversion respecto a OSM)
                url_tile = url_template.format(z=zoom, y=tile_y, x=tile_x)
            else:
                url_tile = url_template.format(z=zoom, x=tile_x, y=tile_y)

            imagen_tile = _descargar_tile_imagen(url_tile, headers=headers)

            if imagen_tile is not None:
                # Pegar el tile en el canvas en la posicion correcta
                pos_x = i * tam_tile
                pos_y = j * tam_tile
                canvas.paste(imagen_tile, (pos_x, pos_y))
                tiles_exitosos += 1
            else:
                print(f"  [!] Tile ({tile_x},{tile_y}) no disponible — se omite.")

            # Pausa para respetar los rate limits del servidor
            if pausa_entre_tiles > 0:
                time.sleep(pausa_entre_tiles)

    if tiles_exitosos == 0:
        print(f"[Tiles] ERROR: no se pudo descargar ningun tile para '{nombre_region}'.")
        return None

    print(f"[Tiles] {tiles_exitosos}/{total_tiles} tiles descargados.")

    # Recortar al bounding box exacto (los tiles pueden extenderse mas alla)
    bbox_tile_norte, bbox_tile_oeste, _, _ = tile_a_bbox(x_min, y_min, zoom)
    _, _, bbox_tile_sur, bbox_tile_este    = tile_a_bbox(x_max, y_max, zoom)

    span_lon_total = bbox_tile_este - bbox_tile_oeste
    span_lat_total = bbox_tile_norte - bbox_tile_sur  # norte > sur

    px_total_x = n_cols * tam_tile
    px_total_y = n_filas * tam_tile

    # Convertir coordenadas geograficas a pixeles en el canvas
    px_izq = int((lon_oeste - bbox_tile_oeste) / span_lon_total * px_total_x)
    px_der = int((lon_este  - bbox_tile_oeste) / span_lon_total * px_total_x)
    px_top = int((bbox_tile_norte - lat_norte)  / span_lat_total * px_total_y)
    px_bot = int((bbox_tile_norte - lat_sur)    / span_lat_total * px_total_y)

    # Asegurar limites validos
    px_izq = max(0, px_izq)
    px_der = min(px_total_x, px_der)
    px_top = max(0, px_top)
    px_bot = min(px_total_y, px_bot)

    if px_der > px_izq and px_bot > px_top:
        canvas = canvas.crop((px_izq, px_top, px_der, px_bot))

    # Redimensionar a 256x256 para compatibilidad con el modelo
    canvas = canvas.resize((256, 256), Image.BICUBIC)

    # Guardar la imagen ensamblada
    dir_salida = Path(directorio_salida) / tipo
    dir_salida.mkdir(parents=True, exist_ok=True)
    ruta_imagen = dir_salida / f"{nombre_region}.jpg"
    canvas.save(ruta_imagen, "JPEG", quality=95)

    print(f"[Tiles] Imagen guardada: {ruta_imagen}")
    return ruta_imagen


# ===========================================================================
# CREACION DE PARES SIDE-BY-SIDE
# ===========================================================================

def crear_pares_side_by_side(
    carpeta_a: str,
    carpeta_b: str,
    directorio_salida: str,
    extension_salida: str = "jpg",
    verificar_integridad: bool = True,
) -> int:
    """
    Combina imagenes de dos carpetas en pares side-by-side (A|B en 512x256 px).

    Este es el formato nativo del dataset Maps de Berkeley y del repositorio
    junyanz/pix2pix. Cada archivo de salida contiene:
        [ imagen_A (256x256) | imagen_B (256x256) ] → archivo 512x256

    Donde tipicamente A = satelite (entrada) y B = mapa OSM (objetivo).

    Los archivos se emparejan por nombre (sin extension).
    Si hay archivos sin pareja, se omiten con una advertencia.

    Args:
        carpeta_a:           Carpeta con imagenes del dominio A (ej: satelite/).
        carpeta_b:           Carpeta con imagenes del dominio B (ej: osm/).
        directorio_salida:   Donde guardar los pares side-by-side.
        extension_salida:    Extension del archivo de salida ('jpg' o 'png').
        verificar_integridad: Si True, comprueba que las imagenes no esten corruptas.

    Returns:
        Numero de pares creados exitosamente.
    """
    carpeta_a = Path(carpeta_a)
    carpeta_b = Path(carpeta_b)
    dir_salida = Path(directorio_salida)
    dir_salida.mkdir(parents=True, exist_ok=True)

    # Indexar archivos de cada carpeta por nombre base (sin extension)
    archivos_a = {p.stem: p for p in carpeta_a.iterdir() if p.is_file()
                  and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}}
    archivos_b = {p.stem: p for p in carpeta_b.iterdir() if p.is_file()
                  and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}}

    # Encontrar nombres en comun (emparejamiento por nombre)
    nombres_comunes = sorted(set(archivos_a.keys()) & set(archivos_b.keys()))
    solo_en_a = set(archivos_a.keys()) - set(archivos_b.keys())
    solo_en_b = set(archivos_b.keys()) - set(archivos_a.keys())

    print(f"[Pares] Encontrados: {len(nombres_comunes)} pares | "
          f"Solo en A: {len(solo_en_a)} | Solo en B: {len(solo_en_b)}")

    if solo_en_a:
        print(f"[!] Sin pareja en B: {list(solo_en_a)[:5]}{'...' if len(solo_en_a) > 5 else ''}")
    if solo_en_b:
        print(f"[!] Sin pareja en A: {list(solo_en_b)[:5]}{'...' if len(solo_en_b) > 5 else ''}")

    pares_creados = 0
    pares_fallidos = 0

    for nombre in nombres_comunes:
        ruta_a = archivos_a[nombre]
        ruta_b = archivos_b[nombre]
        ruta_salida = dir_salida / f"{nombre}.{extension_salida}"

        try:
            img_a = Image.open(ruta_a).convert("RGB")
            img_b = Image.open(ruta_b).convert("RGB")

            if verificar_integridad:
                img_a.verify()
                img_a = Image.open(ruta_a).convert("RGB")  # Reabrir tras verify()
                img_b.verify()
                img_b = Image.open(ruta_b).convert("RGB")

            # Redimensionar ambas imagenes a 256x256 si es necesario
            if img_a.size != (256, 256):
                img_a = img_a.resize((256, 256), Image.BICUBIC)
            if img_b.size != (256, 256):
                img_b = img_b.resize((256, 256), Image.BICUBIC)

            # Crear el canvas side-by-side: 512 ancho, 256 alto
            par = Image.new("RGB", (512, 256))
            par.paste(img_a, (0, 0))    # A en la mitad izquierda
            par.paste(img_b, (256, 0))  # B en la mitad derecha

            par.save(ruta_salida, quality=95 if extension_salida == "jpg" else None)
            pares_creados += 1

        except Exception as e:
            print(f"  [!] Error al procesar par '{nombre}': {e}")
            pares_fallidos += 1

        # Liberar memoria cada 100 pares para evitar acumulacion en Colab
        if pares_creados % 100 == 0 and pares_creados > 0:
            gc.collect()

    print(f"[Pares] Creados: {pares_creados} | Fallidos: {pares_fallidos}")
    return pares_creados


# ===========================================================================
# DIVISION DEL DATASET EN TRAIN / VAL / TEST
# ===========================================================================

def dividir_dataset(
    directorio_origen: str,
    directorio_destino: str = "data/processed",
    ratio_train: float = 0.8,
    ratio_val: float = 0.15,
    ratio_test: float = 0.05,
    semilla: int = 42,
    extension: str = "jpg",
    copiar: bool = True,
) -> Dict[str, int]:
    """
    Divide aleatoriamente las imagenes en conjuntos train / val / test.

    La division aleatoria con semilla fija garantiza que diferentes ejecuciones
    produzcan exactamente la misma division. Esto es fundamental para:
    - Reproducibilidad del experimento.
    - Asegurar que no hay contaminacion train/test (data leakage).

    Estructura de salida:
        {directorio_destino}/
            train/  — {ratio_train * 100}% de los pares
            val/    — {ratio_val * 100}% de los pares
            test/   — {ratio_test * 100}% de los pares

    Args:
        directorio_origen:  Carpeta con todas las imagenes a dividir.
        directorio_destino: Carpeta base donde crear los subdirectorios.
        ratio_train:        Fraccion para entrenamiento (0.0 a 1.0).
        ratio_val:          Fraccion para validacion.
        ratio_test:         Fraccion para test.
        semilla:            Semilla aleatoria para reproducibilidad.
        extension:          Extension de los archivos a buscar.
        copiar:             Si True, copia los archivos. Si False, los mueve.

    Returns:
        Diccionario {'train': N, 'val': N, 'test': N} con el numero de imagenes.

    Raises:
        ValueError: Si los ratios no suman exactamente 1.0.
    """
    suma = ratio_train + ratio_val + ratio_test
    if abs(suma - 1.0) > 1e-6:
        raise ValueError(
            f"Los ratios deben sumar 1.0. Suma actual: {suma:.4f} "
            f"(train={ratio_train}, val={ratio_val}, test={ratio_test})"
        )

    dir_origen = Path(directorio_origen)
    dir_destino = Path(directorio_destino)

    # Buscar todas las imagenes en el directorio origen
    imagenes = sorted(dir_origen.glob(f"**/*.{extension}"))
    if not imagenes:
        # Intentar con otras extensiones comunes
        for ext in ["png", "jpeg", "jpg"]:
            imagenes = sorted(dir_origen.glob(f"**/*.{ext}"))
            if imagenes:
                break

    if not imagenes:
        raise FileNotFoundError(
            f"No se encontraron imagenes .{extension} en: {dir_origen}"
        )

    n_total = len(imagenes)
    print(f"[Split] {n_total} imagenes encontradas en: {dir_origen}")

    # Mezclar aleatoriamente con semilla fija
    random.seed(semilla)
    indices = list(range(n_total))
    random.shuffle(indices)

    # Calcular cortes
    n_train = int(n_total * ratio_train)
    n_val   = int(n_total * ratio_val)
    # El resto va a test (para que no haya rounding errors)
    n_test  = n_total - n_train - n_val

    grupos = {
        "train": [imagenes[i] for i in indices[:n_train]],
        "val":   [imagenes[i] for i in indices[n_train:n_train + n_val]],
        "test":  [imagenes[i] for i in indices[n_train + n_val:]],
    }

    accion = shutil.copy2 if copiar else shutil.move
    nombre_accion = "Copiando" if copiar else "Moviendo"

    conteo = {}
    for modo, archivos in grupos.items():
        dir_modo = dir_destino / modo
        dir_modo.mkdir(parents=True, exist_ok=True)

        for ruta_archivo in archivos:
            ruta_destino_archivo = dir_modo / ruta_archivo.name
            accion(ruta_archivo, ruta_destino_archivo)

        conteo[modo] = len(archivos)
        porcentaje = len(archivos) / n_total * 100
        print(f"  {nombre_accion} {len(archivos):4d} imagenes ({porcentaje:.1f}%) -> {dir_modo}")

    print(f"[Split] Division completada: {conteo}")
    return conteo


# ===========================================================================
# VERIFICACION DE INTEGRIDAD
# ===========================================================================

def verificar_integridad_dataset(
    directorio: str,
    modos: Optional[List[str]] = None,
    eliminar_corruptos: bool = False,
) -> Dict[str, Dict[str, int]]:
    """
    Verifica que todas las imagenes del dataset se puedan abrir correctamente.

    Detecta archivos corruptos, truncados o con formato incorrecto.
    Esto es especialmente importante despues de descargar tiles o copiar
    datasets entre sistemas de archivos.

    Args:
        directorio:          Directorio raiz del dataset (contiene train/, val/, test/).
        modos:               Subdirectorios a verificar. Default: ['train', 'val', 'test'].
        eliminar_corruptos:  Si True, elimina los archivos corruptos encontrados.

    Returns:
        Diccionario {'modo': {'ok': N, 'corrompidas': N, 'tamano_medio_px': W}} por modo.
    """
    dir_raiz = Path(directorio)
    modos = modos or ["train", "val", "test"]
    extensiones = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    resultados: Dict[str, Dict[str, int]] = {}

    for modo in modos:
        dir_modo = dir_raiz / modo
        if not dir_modo.exists():
            continue

        archivos = [p for p in dir_modo.iterdir()
                    if p.is_file() and p.suffix.lower() in extensiones]

        ok = 0
        corruptos = 0
        anchos = []

        for ruta in archivos:
            try:
                with Image.open(ruta) as img:
                    img.verify()  # Detecta archivos truncados o con cabeceras invalidas
                # Reabrir para obtener las dimensiones (verify() cierra la imagen)
                with Image.open(ruta) as img:
                    anchos.append(img.width)
                ok += 1
            except (UnidentifiedImageError, Exception):
                print(f"  [!] Corrompida: {ruta}")
                corruptos += 1
                if eliminar_corruptos:
                    ruta.unlink()
                    print(f"      Eliminada.")

        tamano_medio = int(sum(anchos) / len(anchos)) if anchos else 0
        resultados[modo] = {
            "ok": ok,
            "corruptas": corruptos,
            "total": ok + corruptos,
            "tamano_medio_px": tamano_medio,
        }

        estado = "OK" if corruptos == 0 else f"[!] {corruptos} CORRUPTAS"
        print(f"[Verificar] {modo:6s}: {ok} ok | {corruptos} corruptas | "
              f"ancho medio: {tamano_medio}px | {estado}")

    return resultados


# ===========================================================================
# DESCARGA REAL DEL DATASET
# Ejecutar: py -3 src/data/download_maps.py
#
# Descarga el dataset Maps de Berkeley (~255 MB) y lo organiza en:
#   data/raw/maps/        <- archivos originales extraidos del .tar.gz
#   data/processed/
#       train/            <- ~1096 imagenes side-by-side 512x256
#       val/              <- ~1098 imagenes side-by-side 512x256
#
# El dataset ya viene pre-dividido en train/ y val/ por Berkeley.
# No se redistribuye: se conservan los splits originales para que
# los resultados sean comparables con los del paper Pix2Pix.
#
# Tiempo estimado: 5-15 min dependiendo de la conexion de red.
# Espacio en disco: ~255 MB descarga + ~510 MB extraidos = ~765 MB total.
# ===========================================================================
if __name__ == "__main__":

    # Rutas definitivas del proyecto (relativas al directorio de trabajo)
    DIR_RAW       = Path("data/raw")
    DIR_PROCESSED = Path("data/processed")

    print("=" * 65)
    print("  Descarga del dataset Maps (Berkeley / Pix2Pix)")
    print("=" * 65)
    print(f"  Destino raw       : {DIR_RAW.resolve()}")
    print(f"  Destino procesado : {DIR_PROCESSED.resolve()}")
    print()

    # ------------------------------------------------------------------
    # [1] Descargar y extraer el dataset Maps de Berkeley
    # ------------------------------------------------------------------
    # El dataset viene en formato side-by-side (512x256): la mitad
    # izquierda es la imagen satelite y la derecha el mapa de carreteras.
    # Berkeley ya lo divide en train/ (~1096 imgs) y val/ (~1098 imgs).
    print("[1] Descargando dataset Maps de Berkeley...")
    print(f"    URL: {URL_DATASET_MAPS}")
    print(f"    Tamano aproximado: ~255 MB")
    print()

    dir_maps = descargar_dataset_maps(
        directorio_destino=str(DIR_RAW),
        forzar=False,   # no re-descarga si ya existe
    )

    # Verificar que la estructura extraida es la esperada
    dir_train_raw = dir_maps / "train"
    dir_val_raw   = dir_maps / "val"

    if not dir_train_raw.exists():
        raise FileNotFoundError(
            f"No se encontro {dir_train_raw} tras la extraccion.\n"
            f"Intenta borrar {dir_maps} y volver a ejecutar."
        )

    n_train_raw = len(list(dir_train_raw.glob("*.jpg")))
    n_val_raw   = len(list(dir_val_raw.glob("*.jpg"))) if dir_val_raw.exists() else 0
    print(f"\n    Extraidos: {n_train_raw} train | {n_val_raw} val")

    # ------------------------------------------------------------------
    # [2] Copiar a data/processed/ respetando los splits originales
    # ------------------------------------------------------------------
    # NO redistribuimos aleatoriamente: Berkeley ya hizo un split
    # cuidadoso. Redistribuirlo podria introducir data leakage si
    # alguien compara con resultados publicados en el paper Pix2Pix.
    print("\n[2] Copiando a data/processed/ (split original de Berkeley)...")

    for modo, dir_origen in [("train", dir_train_raw), ("val", dir_val_raw)]:
        if not dir_origen.exists():
            print(f"    [!] {dir_origen} no existe, se omite el modo '{modo}'")
            continue

        dir_destino = DIR_PROCESSED / modo
        dir_destino.mkdir(parents=True, exist_ok=True)

        imagenes = sorted(dir_origen.glob("*.jpg"))
        ya_copiadas = len(list(dir_destino.glob("*.jpg")))

        if ya_copiadas == len(imagenes):
            print(f"    {modo:6s}: ya existen {ya_copiadas} imagenes -> se omite copia")
            continue

        copiadas = 0
        for ruta in imagenes:
            destino = dir_destino / ruta.name
            if not destino.exists():  # no sobreescribir si ya esta
                shutil.copy2(ruta, destino)
                copiadas += 1

        total_en_destino = len(list(dir_destino.glob("*.jpg")))
        print(f"    {modo:6s}: {copiadas} copiadas | {total_en_destino} totales en {dir_destino}")

    # ------------------------------------------------------------------
    # [3] Verificar integridad de data/processed/
    # ------------------------------------------------------------------
    print("\n[3] Verificando integridad de data/processed/...")

    modos_existentes = [
        m for m in ["train", "val"]
        if (DIR_PROCESSED / m).exists()
    ]

    resultados = verificar_integridad_dataset(
        str(DIR_PROCESSED),
        modos=modos_existentes,
        eliminar_corruptos=False,   # solo informar, no eliminar
    )

    n_corruptas_total = sum(r["corruptas"] for r in resultados.values())
    if n_corruptas_total > 0:
        print(f"\n  [!] ATENCION: {n_corruptas_total} imagen(es) corruptas detectadas.")
        print(f"      Vuelve a ejecutar con eliminar_corruptos=True para limpiarlas,")
        print(f"      o borra data/raw/ y data/processed/ y ejecuta de nuevo.")
    else:
        print("\n  Sin imagenes corruptas.")

    # ------------------------------------------------------------------
    # [4] Mostrar estructura final y resumen
    # ------------------------------------------------------------------
    print("\n[4] Estructura final en disco:")
    print(f"    data/")
    print(f"    +-- raw/")
    print(f"    |   +-- maps/")

    for modo in ["train", "val"]:
        dir_raw_m = dir_maps / modo
        n = len(list(dir_raw_m.glob("*.jpg"))) if dir_raw_m.exists() else 0
        print(f"    |       +-- {modo}/   ({n} imagenes originales)")

    print(f"    +-- processed/")
    for modo in ["train", "val"]:
        dir_proc_m = DIR_PROCESSED / modo
        n = len(list(dir_proc_m.glob("*.jpg"))) if dir_proc_m.exists() else 0
        print(f"            +-- {modo}/   ({n} imagenes 512x256 listas para entrenar)")

    # Mostrar una muestra del formato side-by-side
    primer_train = next((DIR_PROCESSED / "train").glob("*.jpg"), None)
    if primer_train:
        with Image.open(primer_train) as img:
            w, h = img.size
        print(f"\n    Formato de cada imagen: {w}x{h} px")
        print(f"    Mitad izquierda  (0..255):   imagen satelite (entrada del modelo)")
        print(f"    Mitad derecha  (256..511):   mapa de carreteras (objetivo del modelo)")

    print()
    print("=" * 65)
    print("  Dataset listo. Para iniciar el entrenamiento ejecuta:")
    print()
    print("  py -3 train.py --datos data/processed --direction AtoB")
    print()
    print("  AtoB: satelite -> mapa de carreteras")
    print("  BtoA: mapa de carreteras -> satelite")
    print("=" * 65)
