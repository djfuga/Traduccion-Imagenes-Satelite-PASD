"""
dataset_loader.py — Cargador de datos optimizado para pares de imagenes
========================================================================

PROPOSITO
---------
Este modulo implementa la capa de entrada de datos del pipeline Pix2Pix.
Su diseno responde a dos restricciones del entorno de entrenamiento:

  1. Google Colab T4 gratuita: GPU con ~13GB VRAM, 2 CPUs, disco lento.
  2. Dataset Maps de Pix2Pix: pares (mapa, satelite) de 600x600 pixeles
     almacenados en dos formatos segun la fuente de descarga.

FORMATOS DE DATASET SOPORTADOS
--------------------------------

  Formato 1 — Side-by-Side (formato nativo Pix2Pix):
  ---------------------------------------------------
  Cada imagen es un archivo unico de 512x256 (o similar) con ambos
  dominios concatenados horizontalmente:

      archivo.jpg
      ┌──────────────┬──────────────┐
      │  Dominio A   │  Dominio B   │
      │  (mapa/OSM)  │ (satelite)   │
      └──────────────┴──────────────┘

  Este es el formato del dataset Maps oficial disponible en:
  http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz

  Formato 2 — Carpetas Separadas:
  --------------------------------
  Los dominios A y B se almacenan en carpetas distintas con el mismo
  nombre de archivo:

      datos/
        trainA/  imagen_001.jpg  imagen_002.jpg ...
        trainB/  imagen_001.jpg  imagen_002.jpg ...

  Este formato es comun cuando las imagenes se descargan de fuentes
  independientes (ej: Sentinel-2 para satelite, OSM para mapas).
  Los archivos se emparejan por nombre o por posicion en la lista.

PIPELINE DE TRANSFORMACIONES (por que cada paso)
-------------------------------------------------
Para entrenamiento (modo='train'):

  [1] Resize a 286x286
      Por que: ligeramente mas grande que el objetivo (256x256).
      Permite el recorte aleatorio del siguiente paso.

  [2] RandomCrop a 256x256
      Por que: introduce variacion espacial aleatoria (data augmentation).
      El mismo recorte se aplica a A y B para mantener la alineacion.
      Sin este paso, el modelo veria siempre exactamente la misma region.

  [3] RandomHorizontalFlip (p=0.5)
      Por que: duplica el dataset efectivo. Mapas y satelites son
      geometricamente simetricos bajo reflexion horizontal.
      El mismo flip se aplica a A y B.

  [4] ToTensor: PIL [0,255] -> float32 [0.0, 1.0]
      Por que: PyTorch trabaja con tensores float32, no con uint8.

  [5] Normalize(mean=0.5, std=0.5): [0,1] -> [-1, 1]
      Formula: tensor_norm = (tensor - 0.5) / 0.5 = tensor*2 - 1
      Por que: la activacion Tanh del generador produce valores en [-1,1].
      Normalizar la entrada al mismo rango garantiza que la funcion de
      perdida L1 opera en un espacio simetrico y la red no tiene que
      aprender a reescalar su propia salida.

Para validacion/test (modo='val' o 'test'):
  Solo pasos [1->256] (resize directo), [4] y [5]. Sin augmentacion.

OPTIMIZACIONES DE VRAM Y VELOCIDAD
------------------------------------
  pin_memory=True  : reserva memoria CPU fijada (no paginable) para la
                     copia CPU->GPU. Permite transferencia asincrona
                     DMA, solapando la copia con el computo en GPU.
                     Solo util con CUDA; en CPU no tiene efecto.

  num_workers=2    : procesos separados que pre-cargan los proximos
                     batches mientras la GPU procesa el actual.
                     Maximo 2 en Colab gratuito (2 CPUs disponibles).
                     Mas de 2 causa OOM en la memoria del sistema.

  persistent_workers=True : los procesos worker sobreviven entre
                     epocas. Evita el overhead de fork() al inicio de
                     cada epoca (~2-5 segundos por epoch en Colab).

  prefetch_factor=2 : cada worker pre-carga 2 batches anticipadamente.
                      Reduce el tiempo de espera cuando el generador
                      de datos es mas lento que la GPU.

  drop_last=True   : descarta el ultimo batch si no tiene batch_size
                     completo. InstanceNorm funciona con batch=1 pero
                     no con batch=0. Tambien evita gradientes sesgados
                     por batches de tamano variable.

  cache_en_ram=True: carga TODO el dataset en memoria RAM al inicio.
                     Elimina completamente el cuello de botella de disco.
                     Recomendado para datasets < 2GB y RAM > 8GB.
                     En Colab: los datasets Maps pequenos (~200MB) caben
                     facilmente en los 12GB de RAM del runtime.

Referencias:
  Isola et al., "Image-to-Image Translation with cGANs", CVPR 2017.
  https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
  Sketch2Map: https://github.com/PerlMonker303/S2MP
"""

import os
import time
import random
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple

import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
import numpy as np


# ===========================================================================
# CONSTANTES GLOBALES
# ===========================================================================

TAMANIO_CARGA  = 286    # Tamano de resize previo al crop (Pix2Pix estandar)
TAMANIO_FINAL  = 256    # Tamano final de las imagenes (entrada a la U-Net)
MEDIA_NORM     = (0.5, 0.5, 0.5)
STD_NORM       = (0.5, 0.5, 0.5)
EXTENSIONES    = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


# ===========================================================================
# PIPELINE DE TRANSFORMACIONES SINCRONIZADAS
# ===========================================================================

class TransformacionesSincronizadas:
    """
    Aplica las mismas transformaciones aleatorias a ambas imagenes del par.

    El problema fundamental del augmentation en datasets pareados:
    Si aplicamos RandomCrop de forma independiente a la imagen A y a la
    imagen B, cada una sera recortada en una posicion diferente. El par
    ya no estara alineado: la esquina superior izquierda de A no
    corresponderia geograficamente con la de B.

    Solucion: decidir los parametros aleatorios UNA sola vez y aplicarlos
    a ambas imagenes en el mismo orden.

    Esta clase implementa el pipeline completo descrito en el docstring del
    modulo para los tres modos de uso:
      - 'train': resize 286 -> crop 256 -> flip -> tensor -> normalize
      - 'val':   resize 256            ->        -> tensor -> normalize
      - 'test':  resize 256            ->        -> tensor -> normalize
    """

    def __init__(self, modo: Literal["train", "val", "test"] = "train"):
        """
        Args:
            modo: Controla que transformaciones se aplican.
                  'train' aplica augmentation; 'val'/'test' no.
        """
        if modo not in ("train", "val", "test"):
            raise ValueError(f"Modo '{modo}' invalido. Usa 'train', 'val' o 'test'.")
        self.modo = modo

    def __call__(
        self,
        imagen_a: Image.Image,
        imagen_b: Image.Image,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transforma el par (A, B) de forma sincronizada.

        Args:
            imagen_a: Imagen PIL del dominio A (cualquier tamano).
            imagen_b: Imagen PIL del dominio B (cualquier tamano).

        Returns:
            Tupla (tensor_a, tensor_b) float32 en [-1, 1], forma (3, 256, 256).
        """
        # Garantizar modo RGB (elimina imagenes RGBA o en escala de grises)
        imagen_a = imagen_a.convert("RGB")
        imagen_b = imagen_b.convert("RGB")

        # ---- Paso 1: Resize ----
        # BICUBIC produce artefactos minimos para imagenes de mapas/satelite
        # (preserva bordes mejor que BILINEAR, mas rapido que LANCZOS)
        tamano_resize = TAMANIO_CARGA if self.modo == "train" else TAMANIO_FINAL
        imagen_a = TF.resize(imagen_a, [tamano_resize, tamano_resize], TF.InterpolationMode.BICUBIC)
        imagen_b = TF.resize(imagen_b, [tamano_resize, tamano_resize], TF.InterpolationMode.BICUBIC)

        if self.modo == "train":
            # ---- Paso 2: RandomCrop sincronizado ----
            # get_params() calcula i,j (origen) y h,w (tamano) del crop.
            # Llamamos una sola vez (sobre imagen_a) y usamos los MISMOS
            # parametros para imagen_b. Esto garantiza la alineacion espacial.
            i, j, h, w = T.RandomCrop.get_params(
                imagen_a, output_size=(TAMANIO_FINAL, TAMANIO_FINAL)
            )
            imagen_a = TF.crop(imagen_a, i, j, h, w)
            imagen_b = TF.crop(imagen_b, i, j, h, w)

            # ---- Paso 3: RandomHorizontalFlip sincronizado ----
            # Tomamos la decision de voltear UNA vez (random.random() > 0.5)
            # y la aplicamos a ambas imagenes.
            if random.random() > 0.5:
                imagen_a = TF.hflip(imagen_a)
                imagen_b = TF.hflip(imagen_b)

        # ---- Paso 4: PIL -> tensor float32 en [0, 1] ----
        # to_tensor() convierte HxWxC uint8 a CxHxW float32 dividiendo por 255
        tensor_a = TF.to_tensor(imagen_a)   # (3, 256, 256), rango [0.0, 1.0]
        tensor_b = TF.to_tensor(imagen_b)

        # ---- Paso 5: Normalizar a [-1, 1] ----
        # Formula: t_norm = (t - media) / std = (t - 0.5) / 0.5 = 2*t - 1
        # La activacion Tanh del generador tambien produce [-1, 1],
        # por lo que la funcion de perdida L1 opera en un espacio simetrico.
        tensor_a = TF.normalize(tensor_a, mean=MEDIA_NORM, std=STD_NORM)
        tensor_b = TF.normalize(tensor_b, mean=MEDIA_NORM, std=STD_NORM)

        return tensor_a, tensor_b

    def __repr__(self) -> str:
        return (
            f"TransformacionesSincronizadas(modo='{self.modo}', "
            f"resize={TAMANIO_CARGA if self.modo=='train' else TAMANIO_FINAL}, "
            f"crop={TAMANIO_FINAL}, flip={'si' if self.modo=='train' else 'no'})"
        )


def desnormalizar(tensor: torch.Tensor) -> torch.Tensor:
    """
    Invierte la normalizacion: convierte de [-1, 1] a [0, 1].

    Formula: t_orig = t_norm * std + media = t_norm * 0.5 + 0.5 = (t_norm + 1) / 2

    Necesario para visualizar imagenes generadas o calcular metricas como
    SSIM que esperan valores en [0, 1].

    Args:
        tensor: Tensor normalizado de cualquier forma con valores en [-1, 1].

    Returns:
        Tensor en [0, 1], misma forma. Clampeado para manejar imprecision
        numerica (valores ligeramente fuera del rango teorico).
    """
    return torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)


# ===========================================================================
# DATASET 1: FORMATO SIDE-BY-SIDE (Maps, Pix2Pix oficial)
# ===========================================================================

class DatasetParesSideBySide(Dataset):
    """
    Dataset para imagenes pareadas en formato side-by-side.

    Cada archivo almacena ambos dominios concatenados horizontalmente:
        [  A (izquierda)  |  B (derecha)  ]
    El ancho del archivo es exactamente el doble del ancho de cada dominio.

    Este es el formato del dataset Maps descargable desde:
        http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz

    Estructura esperada del directorio:
        directorio_raiz/
            00001.jpg   (cada uno es un par A|B)
            00002.jpg
            ...

    El parametro 'direction' permite usar el mismo dataset para entrenar
    en ambas direcciones sin duplicar archivos:
        'AtoB': entrada=izquierda (mapa), objetivo=derecha (satelite)
        'BtoA': entrada=derecha (satelite), objetivo=izquierda (mapa)

    Opcion cache_en_ram:
        Si True, carga todas las imagenes PIL en memoria al inicio.
        Elimina el I/O de disco durante el entrenamiento.
        Recomendado cuando el dataset cabe en RAM (tipicamente < 2GB).
        En Google Colab con dataset Maps (~200MB): reduce el tiempo de
        carga por item de ~5ms (disco) a ~0.1ms (RAM), una mejora 50x.
    """

    def __init__(
        self,
        directorio_raiz: str,
        direction: Literal["AtoB", "BtoA"] = "AtoB",
        modo: Literal["train", "val", "test"] = "train",
        cache_en_ram: bool = False,
        verificar_integridad: bool = False,
    ):
        """
        Args:
            directorio_raiz:       Ruta a la carpeta con las imagenes side-by-side.
            direction:             'AtoB' o 'BtoA'. Controla que mitad es entrada/objetivo.
            modo:                  'train', 'val' o 'test'. Controla augmentation.
            cache_en_ram:          Si True, pre-carga todas las imagenes en RAM.
            verificar_integridad:  Si True, valida que todos los archivos sean imagenes
                                   validas antes de iniciar el entrenamiento.
        """
        super().__init__()

        self.direction     = direction
        self.modo          = modo
        self.cache_en_ram  = cache_en_ram
        self.transform     = TransformacionesSincronizadas(modo=modo)
        self._cache: Dict[int, Tuple[Image.Image, Image.Image]] = {}

        raiz = Path(directorio_raiz)
        if not raiz.exists():
            raise FileNotFoundError(f"Directorio no encontrado: '{directorio_raiz}'")

        self.rutas: List[Path] = sorted([
            p for p in raiz.iterdir()
            if p.suffix.lower() in EXTENSIONES
        ])

        if len(self.rutas) == 0:
            raise ValueError(
                f"No se encontraron imagenes en '{directorio_raiz}'.\n"
                f"Extensiones buscadas: {EXTENSIONES}"
            )

        if verificar_integridad:
            self._verificar_integridad()

        if cache_en_ram:
            self._cargar_cache()

        print(
            f"[DatasetSideBySide] {len(self.rutas)} pares | "
            f"modo={modo} | direction={direction} | "
            f"cache_ram={'si' if cache_en_ram else 'no'}"
        )

    # ------------------------------------------------------------------
    # Metodos internos
    # ------------------------------------------------------------------

    def _verificar_integridad(self) -> None:
        """
        Verifica que todos los archivos son imagenes validas y legibles.

        Ejecutar antes del primer entrenamiento para detectar archivos
        corruptos o incompletos (frecuente con descargas interrumpidas).

        Imprime el numero de archivos validos e invalidos.
        """
        print(f"[Integridad] Verificando {len(self.rutas)} archivos...")
        invalidos = []
        for ruta in self.rutas:
            try:
                with Image.open(ruta) as img:
                    img.verify()   # detecta archivos truncados/corruptos
            except (UnidentifiedImageError, Exception):
                invalidos.append(ruta)

        if invalidos:
            warnings.warn(
                f"[Integridad] {len(invalidos)} archivos invalidos encontrados:\n"
                + "\n".join(f"  {r}" for r in invalidos[:5])
                + (f"\n  ... y {len(invalidos)-5} mas" if len(invalidos) > 5 else "")
            )
            # Remover archivos invalidos de la lista
            invalidos_set = set(invalidos)
            self.rutas = [r for r in self.rutas if r not in invalidos_set]
            print(f"[Integridad] {len(self.rutas)} pares validos tras la verificacion.")
        else:
            print(f"[Integridad] Todos los archivos son validos.")

    def _cargar_cache(self) -> None:
        """
        Pre-carga todas las imagenes PIL en memoria RAM.

        Divide cada imagen side-by-side en sus dos mitades (A y B) y las
        almacena como objetos PIL ya divididos, evitando el split en cada
        llamada a __getitem__.

        La RAM requerida es aproximadamente:
            n_imagenes * 2 * (ancho/2 * alto * 3 bytes) / 1e6  MB
        Para 1096 imagenes de 600x600: ~1096 * 2 * 600*600*3 / 1e6 ~= 2.4 GB
        Para 256x256 ya redimensionadas: mucho menos.
        """
        print(f"[Cache] Cargando {len(self.rutas)} imagenes en RAM...")
        inicio = time.time()
        for idx, ruta in enumerate(self.rutas):
            img = Image.open(ruta).convert("RGB")
            ancho, alto = img.size
            mitad = ancho // 2
            self._cache[idx] = (
                img.crop((0, 0, mitad, alto)),      # dominio A
                img.crop((mitad, 0, ancho, alto)),  # dominio B
            )
        elapsed = time.time() - inicio
        print(f"[Cache] {len(self._cache)} pares cargados en {elapsed:.1f}s")

    def _cargar_par(self, idx: int) -> Tuple[Image.Image, Image.Image]:
        """Retorna el par (imagen_A, imagen_B) desde cache o desde disco."""
        if self.cache_en_ram and idx in self._cache:
            return self._cache[idx]

        img = Image.open(self.rutas[idx]).convert("RGB")
        ancho, alto = img.size
        mitad = ancho // 2
        return img.crop((0, 0, mitad, alto)), img.crop((mitad, 0, ancho, alto))

    # ------------------------------------------------------------------
    # Interface Dataset
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.rutas)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Carga, divide y transforma un par de imagenes.

        Args:
            idx: Indice del par (0 a len-1).

        Returns:
            Tupla (tensor_entrada, tensor_objetivo).
            Ambos tensores: float32, forma (3, 256, 256), valores en [-1, 1].
        """
        imagen_a, imagen_b = self._cargar_par(idx)

        # direction controla que dominio es entrada y cual es objetivo
        if self.direction == "AtoB":
            entrada, objetivo = imagen_a, imagen_b
        else:
            entrada, objetivo = imagen_b, imagen_a

        return self.transform(entrada, objetivo)

    def info(self) -> dict:
        """Retorna un resumen de la configuracion del dataset."""
        return {
            "formato":     "side-by-side",
            "n_pares":     len(self.rutas),
            "modo":        self.modo,
            "direction":   self.direction,
            "cache_ram":   self.cache_en_ram,
            "tamanio_out": f"{TAMANIO_FINAL}x{TAMANIO_FINAL}",
            "augmentation": self.modo == "train",
        }


# ===========================================================================
# DATASET 2: FORMATO CARPETAS SEPARADAS
# ===========================================================================

class DatasetParesCarpetas(Dataset):
    """
    Dataset para imagenes pareadas en carpetas separadas (trainA/ y trainB/).

    Estructura esperada:
        directorio_raiz/
            trainA/   imagen_001.jpg  imagen_002.jpg  ...
            trainB/   imagen_001.jpg  imagen_002.jpg  ...

    o simplemente dos rutas de carpeta:
        carpeta_a/  imagen_001.jpg  ...
        carpeta_b/  imagen_001.jpg  ...

    Emparejamiento:
        Las imagenes se emparejan por NOMBRE DE ARCHIVO. Si los nombres no
        coinciden, se emparejan por POSICION en la lista ordenada.
        Este comportamiento se controla con el parametro 'emparejar_por_nombre'.

    Cuando usar este formato:
        - Cuando las imagenes A y B se descargaron por separado.
        - Cuando se genera el dataset OSM/satelite con scripts propios.
        - Cuando A y B tienen diferentes extensiones (ej: A=.tiff, B=.jpg).
    """

    def __init__(
        self,
        carpeta_a: str,
        carpeta_b: str,
        modo: Literal["train", "val", "test"] = "train",
        emparejar_por_nombre: bool = True,
        cache_en_ram: bool = False,
    ):
        """
        Args:
            carpeta_a:            Ruta a la carpeta con imagenes del dominio A.
            carpeta_b:            Ruta a la carpeta con imagenes del dominio B.
            modo:                 'train', 'val' o 'test'.
            emparejar_por_nombre: Si True, empareja A con B por nombre de archivo.
                                  Si False, empareja por posicion (orden alfabetico).
            cache_en_ram:         Si True, pre-carga todas las imagenes en RAM.
        """
        super().__init__()

        self.modo          = modo
        self.cache_en_ram  = cache_en_ram
        self.transform     = TransformacionesSincronizadas(modo=modo)
        self._cache_a: Dict[int, Image.Image] = {}
        self._cache_b: Dict[int, Image.Image] = {}

        rutas_a_raw = self._listar_imagenes(carpeta_a)
        rutas_b_raw = self._listar_imagenes(carpeta_b)

        if emparejar_por_nombre:
            self.pares = self._emparejar_por_nombre(rutas_a_raw, rutas_b_raw)
        else:
            # Emparejar por posicion: el i-esimo archivo de A con el i-esimo de B
            n = min(len(rutas_a_raw), len(rutas_b_raw))
            self.pares = list(zip(rutas_a_raw[:n], rutas_b_raw[:n]))
            if len(rutas_a_raw) != len(rutas_b_raw):
                warnings.warn(
                    f"[Dataset] Las carpetas tienen diferente numero de imagenes "
                    f"(A={len(rutas_a_raw)}, B={len(rutas_b_raw)}). "
                    f"Se usaran los primeros {n} pares."
                )

        if len(self.pares) == 0:
            raise ValueError(
                f"No se encontraron pares de imagenes.\n"
                f"  Carpeta A: {carpeta_a} ({len(rutas_a_raw)} imagenes)\n"
                f"  Carpeta B: {carpeta_b} ({len(rutas_b_raw)} imagenes)"
            )

        if cache_en_ram:
            self._cargar_cache()

        print(
            f"[DatasetCarpetas] {len(self.pares)} pares | "
            f"modo={modo} | emparejamiento={'nombre' if emparejar_por_nombre else 'posicion'}"
        )

    @staticmethod
    def _listar_imagenes(carpeta: str) -> List[Path]:
        """Lista y ordena todas las imagenes de una carpeta."""
        raiz = Path(carpeta)
        if not raiz.exists():
            raise FileNotFoundError(f"Carpeta no encontrada: '{carpeta}'")
        return sorted([p for p in raiz.iterdir() if p.suffix.lower() in EXTENSIONES])

    @staticmethod
    def _emparejar_por_nombre(
        rutas_a: List[Path],
        rutas_b: List[Path],
    ) -> List[Tuple[Path, Path]]:
        """
        Empareja imagenes de A con imagenes de B por nombre de archivo (sin extension).

        Permite que A tenga extension .tiff y B tenga .jpg, siempre que el
        nombre base sea el mismo (ej: 'imagen_001.tiff' con 'imagen_001.jpg').
        """
        nombres_b = {p.stem: p for p in rutas_b}
        pares = []
        sin_par = []
        for ruta_a in rutas_a:
            if ruta_a.stem in nombres_b:
                pares.append((ruta_a, nombres_b[ruta_a.stem]))
            else:
                sin_par.append(ruta_a.name)

        if sin_par:
            warnings.warn(
                f"[Dataset] {len(sin_par)} imagenes de A sin par en B: "
                + ", ".join(sin_par[:3])
                + ("..." if len(sin_par) > 3 else "")
            )
        return pares

    def _cargar_cache(self) -> None:
        """Pre-carga todas las imagenes en RAM."""
        print(f"[Cache] Cargando {len(self.pares)} pares en RAM...")
        inicio = time.time()
        for idx, (ruta_a, ruta_b) in enumerate(self.pares):
            self._cache_a[idx] = Image.open(ruta_a).convert("RGB")
            self._cache_b[idx] = Image.open(ruta_b).convert("RGB")
        elapsed = time.time() - inicio
        print(f"[Cache] {len(self.pares)} pares cargados en {elapsed:.1f}s")

    def __len__(self) -> int:
        return len(self.pares)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Carga y transforma el par de imagenes en el indice dado.

        Returns:
            Tupla (tensor_a, tensor_b) float32, forma (3, 256, 256), valores [-1, 1].
        """
        if self.cache_en_ram and idx in self._cache_a:
            img_a = self._cache_a[idx]
            img_b = self._cache_b[idx]
        else:
            ruta_a, ruta_b = self.pares[idx]
            img_a = Image.open(ruta_a).convert("RGB")
            img_b = Image.open(ruta_b).convert("RGB")

        return self.transform(img_a, img_b)


# ===========================================================================
# FUNCION FACTORY: crear_dataloader
# ===========================================================================

def crear_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    modo: Literal["train", "val", "test"] = "train",
    num_workers: int = 2,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """
    Crea un DataLoader optimizado para el entorno Google Colab T4.

    Decide automaticamente los parametros de optimizacion segun el hardware:

    pin_memory:
        True si hay CUDA disponible. Reserva memoria CPU fijada (no paginable
        por el SO) para los tensores del batch. Permite que el controlador CUDA
        use DMA (Direct Memory Access) para la transferencia CPU->GPU sin
        intervención de la CPU, solapando la copia con el computo en GPU.
        Ganancia tipica: 10-15% en throughput con batches grandes.
        En CPU o MPS no tiene efecto util y puede consumir mas RAM.

    persistent_workers:
        True si num_workers > 0. Los procesos worker de carga de datos se
        mantienen vivos entre epocas en lugar de terminarse y recrearse.
        Evita el overhead de fork() que en Python con Colab puede tardar 2-5
        segundos al inicio de cada epoca (especialmente con num_workers=2).

    prefetch_factor:
        Cada worker pre-carga prefetch_factor batches adicionales mientras
        la GPU procesa el batch actual. Reduce la espera cuando __getitem__
        es mas lento que la GPU (tipico con imagenes grandes o augmentation
        compleja). Solo activo cuando num_workers > 0.

    drop_last:
        True en modo train. Descarta el ultimo batch si es menor que batch_size.
        Razon 1: InstanceNorm requiere al menos 1 muestra (funciona con batch=1
                 pero da error con batch=0).
        Razon 2: Un batch mas pequeno produce gradientes de diferente magnitud,
                 lo que puede desestabilizar el paso de actualizacion si no
                 se escala el learning rate.

    Args:
        dataset:     El dataset (DatasetParesSideBySide o DatasetParesCarpetas).
        batch_size:  Tamano de batch. Default: 1 (estandar Pix2Pix).
                     Para T4 con 256x256 y AMP: hasta batch=4 es seguro.
        modo:        'train', 'val' o 'test'. Controla drop_last y shuffle.
        num_workers: Procesos de carga paralela. Default: 2.
                     0 = carga sincrona en el proceso principal (mas lento,
                     pero necesario en Windows con multiprocessing).
        shuffle:     None = True para train, False para val/test.

    Returns:
        DataLoader configurado y listo para iterar.
    """
    if shuffle is None:
        shuffle = (modo == "train")

    usar_pin_memory = torch.cuda.is_available()
    usar_persistent = num_workers > 0

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=usar_pin_memory,
        drop_last=(modo == "train"),
        persistent_workers=usar_persistent,
        prefetch_factor=(2 if num_workers > 0 else None),
    )

    print(
        f"[DataLoader] n={len(dataset)} | batch={batch_size} | "
        f"shuffle={shuffle} | workers={num_workers} | "
        f"pin_memory={usar_pin_memory} | drop_last={modo=='train'}"
    )
    return loader


# ===========================================================================
# UTILIDADES DE ANALISIS Y DIAGNOSTICO
# ===========================================================================

def estimar_memoria_dataset(
    n_imagenes: int,
    tamanio_px: int = 256,
    canales: int = 3,
    bytes_por_pixel: int = 4,   # float32
) -> dict:
    """
    Estima el consumo de memoria del dataset en RAM y VRAM.

    Desglose:
      - RAM (cache completo): todos los tensores del dataset en float32.
      - RAM (un batch): memoria necesaria para un solo batch durante el forward.
      - VRAM (un batch): lo mismo pero en GPU.

    Args:
        n_imagenes:       Numero total de pares en el dataset.
        tamanio_px:       Resolucion de cada imagen (cuadrada). Default: 256.
        canales:          Canales por imagen. Default: 3 (RGB).
        bytes_por_pixel:  Bytes por valor. 4=float32, 2=float16 (AMP).

    Returns:
        Diccionario con estimaciones en MB.
    """
    pixels_imagen = tamanio_px * tamanio_px * canales
    bytes_imagen  = pixels_imagen * bytes_por_pixel
    bytes_par     = bytes_imagen * 2   # A + B

    return {
        "bytes_por_imagen_MB":   bytes_imagen / 1e6,
        "bytes_por_par_MB":      bytes_par / 1e6,
        "cache_completo_MB":     n_imagenes * bytes_par / 1e6,
        "batch1_MB":             bytes_par / 1e6,
        "batch4_MB":             4 * bytes_par / 1e6,
        "batch8_MB":             8 * bytes_par / 1e6,
        "n_imagenes":            n_imagenes,
        "dtype":                 "float32" if bytes_por_pixel == 4 else "float16",
    }


def estadisticas_batch(
    dataloader: DataLoader,
    n_batches: int = 5,
) -> dict:
    """
    Calcula estadisticas de normalizacion sobre los primeros n_batches.

    Util para verificar que la normalizacion es correcta:
    - Media: debe ser cercana a 0 (si la normalizacion es correcta).
    - Std:   debe ser cercana a 1.
    - Rango: debe estar en [-1, 1].

    Args:
        dataloader: DataLoader configurado.
        n_batches:  Numero de batches a muestrear.

    Returns:
        Diccionario con estadisticas de los tensores A y B.
    """
    medias_a, stds_a = [], []
    medias_b, stds_b = [], []
    mins_a, maxs_a   = [], []

    for i, (batch_a, batch_b) in enumerate(dataloader):
        if i >= n_batches:
            break
        medias_a.append(batch_a.mean().item())
        stds_a.append(batch_a.std().item())
        medias_b.append(batch_b.mean().item())
        stds_b.append(batch_b.std().item())
        mins_a.append(batch_a.min().item())
        maxs_a.append(batch_a.max().item())

    return {
        "A_media":    np.mean(medias_a),
        "A_std":      np.mean(stds_a),
        "A_min":      np.min(mins_a),
        "A_max":      np.max(maxs_a),
        "B_media":    np.mean(medias_b),
        "B_std":      np.mean(stds_b),
        "batches":    len(medias_a),
    }


def benchmark_velocidad(
    dataloader: DataLoader,
    n_batches: int = 20,
) -> dict:
    """
    Mide la velocidad de carga del DataLoader en batches/segundo e imagenes/segundo.

    En Google Colab, la velocidad tipica es:
      - Sin cache (disco SSD Colab): ~20-50 items/s para 256x256.
      - Con cache en RAM:            ~500-2000 items/s.

    La GPU T4 puede procesar ~10-30 batches/s para batch=1 con U-Net 256.
    Si el DataLoader es mas lento que la GPU, la GPU estara esperando datos
    (cuello de botella en I/O). El cache elimina este cuello de botella.

    Args:
        dataloader: DataLoader a medir.
        n_batches:  Numero de batches para el benchmark.

    Returns:
        Diccionario con metricas de velocidad.
    """
    batch_size  = dataloader.batch_size or 1
    tiempos     = []

    # Iterar sin procesar los datos (solo medir la carga)
    iterador = iter(dataloader)
    for i in range(min(n_batches, len(dataloader))):
        inicio = time.perf_counter()
        try:
            _ = next(iterador)
        except StopIteration:
            break
        tiempos.append(time.perf_counter() - inicio)

    if not tiempos:
        return {"error": "No se pudo iterar el dataloader"}

    tiempo_medio  = np.mean(tiempos)
    items_por_seg = batch_size / tiempo_medio

    return {
        "batches_medidos":   len(tiempos),
        "tiempo_medio_ms":   tiempo_medio * 1000,
        "tiempo_min_ms":     np.min(tiempos) * 1000,
        "tiempo_max_ms":     np.max(tiempos) * 1000,
        "items_por_segundo": items_por_seg,
        "batches_por_hora":  3600 / tiempo_medio,
    }


# ===========================================================================
# BLOQUE DE VERIFICACION LOCAL
# Ejecutar: python src/data/dataset_loader.py
#
# Crea un dataset temporal con imagenes dummy y verifica:
#   1. Formato side-by-side (DatasetParesSideBySide)
#   2. Formato carpetas separadas (DatasetParesCarpetas)
#   3. Normalizacion correcta (media ~0, rango [-1,1])
#   4. Consistencia del augmentation (A y B reciben el mismo crop y flip)
#   5. Cache en RAM (velocidad vs sin cache)
#   6. Estimacion de memoria para Google Colab
#   7. Benchmark de velocidad del DataLoader
# ===========================================================================
if __name__ == "__main__":
    import tempfile

    torch.manual_seed(0)
    SEP = "=" * 65

    print(SEP)
    print("  Verificacion local: dataset_loader.py")
    print(SEP)

    # ------------------------------------------------------------------
    # Crear datos de prueba en directorio temporal
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Generar 12 imagenes side-by-side 512x256 con patrones distintos
        # para A (gradiente horizontal) y B (gradiente vertical)
        dir_sbs    = tmpdir / "side_by_side"
        dir_a      = tmpdir / "dominio_A"
        dir_b      = tmpdir / "dominio_B"
        dir_sbs.mkdir(); dir_a.mkdir(); dir_b.mkdir()

        N_IMAGENES = 12
        for i in range(N_IMAGENES):
            # Dominio A: gradiente horizontal (simula mapa/boceto)
            arr_a = np.zeros((300, 300, 3), dtype=np.uint8)
            arr_a[:, :, 0] = np.linspace(0, 255, 300, dtype=np.uint8)[np.newaxis, :]
            arr_a[:, :, 1] = (i * 20) % 256
            # Dominio B: gradiente vertical (simula satelite)
            arr_b = np.zeros((300, 300, 3), dtype=np.uint8)
            arr_b[:, :, 2] = np.linspace(0, 255, 300, dtype=np.uint8)[:, np.newaxis]
            arr_b[:, :, 0] = (i * 15) % 256

            img_a = Image.fromarray(arr_a)
            img_b = Image.fromarray(arr_b)

            # Guardar formato side-by-side (ancho total = 600, cada mitad = 300)
            img_sbs = Image.new("RGB", (600, 300))
            img_sbs.paste(img_a, (0, 0))
            img_sbs.paste(img_b, (300, 0))
            img_sbs.save(dir_sbs / f"par_{i:03d}.jpg")

            # Guardar formato carpetas separadas
            img_a.save(dir_a / f"imagen_{i:03d}.jpg")
            img_b.save(dir_b / f"imagen_{i:03d}.jpg")

        print(f"\n  Datos de prueba: {N_IMAGENES} pares en '{tmpdir}'")

        # ==============================================================
        # 1. FORMATO SIDE-BY-SIDE
        # ==============================================================
        print(f"\n[1] DatasetParesSideBySide")
        print("-" * 55)

        for modo in ("train", "val"):
            ds = DatasetParesSideBySide(str(dir_sbs), direction="AtoB", modo=modo)
            a, b = ds[0]

            assert a.shape == torch.Size([3, 256, 256]), f"Forma incorrecta: {a.shape}"
            assert b.shape == torch.Size([3, 256, 256])
            assert a.min() >= -1.01 and a.max() <= 1.01, f"Fuera de rango: [{a.min():.3f}, {a.max():.3f}]"

            print(f"  modo={modo:5s} | A={tuple(a.shape)} | "
                  f"rango=[{a.min():.3f}, {a.max():.3f}] | [OK]")

        # Probar direction BtoA
        ds_bto = DatasetParesSideBySide(str(dir_sbs), direction="BtoA", modo="val")
        ds_ato = DatasetParesSideBySide(str(dir_sbs), direction="AtoB", modo="val")
        a_ato, b_ato = ds_ato[0]
        a_bto, b_bto = ds_bto[0]
        # Con BtoA: la entrada y el objetivo se intercambian
        assert not torch.allclose(a_ato, a_bto), "AtoB y BtoA deben tener entradas distintas"
        print(f"  direction AtoB vs BtoA: entradas distintas | [OK]")

        # ==============================================================
        # 2. FORMATO CARPETAS SEPARADAS
        # ==============================================================
        print(f"\n[2] DatasetParesCarpetas")
        print("-" * 55)

        ds_carpetas = DatasetParesCarpetas(
            str(dir_a), str(dir_b), modo="train", emparejar_por_nombre=True
        )
        a, b = ds_carpetas[0]

        assert a.shape == torch.Size([3, 256, 256])
        assert b.shape == torch.Size([3, 256, 256])
        assert a.min() >= -1.01 and a.max() <= 1.01

        print(f"  {len(ds_carpetas)} pares cargados por nombre")
        print(f"  A={tuple(a.shape)} | rango=[{a.min():.3f}, {a.max():.3f}] | [OK]")

        # ==============================================================
        # 3. VERIFICACION DE NORMALIZACION
        # ==============================================================
        print(f"\n[3] Verificacion de normalizacion [-1, 1]")
        print("-" * 55)

        loader_val = crear_dataloader(
            DatasetParesSideBySide(str(dir_sbs), modo="val"),
            batch_size=4, modo="val", num_workers=0
        )
        stats = estadisticas_batch(loader_val, n_batches=3)

        print(f"  A media  = {stats['A_media']:+.4f}  (esperado: ~0.0)")
        print(f"  A std    = {stats['A_std']:.4f}   (esperado: ~0.5-1.0)")
        print(f"  A rango  = [{stats['A_min']:.3f}, {stats['A_max']:.3f}]  (esperado: [-1, 1])")
        print(f"  B media  = {stats['B_media']:+.4f}")

        assert stats["A_min"] >= -1.05, f"Rango min fuera de limite: {stats['A_min']}"
        assert stats["A_max"] <=  1.05, f"Rango max fuera de limite: {stats['A_max']}"
        print(f"  [OK] Normalizacion correcta")

        # ==============================================================
        # 4. CONSISTENCIA DEL AUGMENTATION (A y B reciben el mismo crop)
        # ==============================================================
        print(f"\n[4] Consistencia del augmentation sincronizado")
        print("-" * 55)

        # Crear un par donde A y B son identicas -> tras el mismo crop, siguen identicas
        img_identica_a = Image.fromarray(
            np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        )
        img_identica_b = img_identica_a.copy()

        transform_train = TransformacionesSincronizadas(modo="train")
        desincronizaciones = 0
        for _ in range(50):
            ta, tb = transform_train(img_identica_a.copy(), img_identica_b.copy())
            if not torch.allclose(ta, tb, atol=1e-5):
                desincronizaciones += 1

        assert desincronizaciones == 0, \
            f"Augmentation desincronizado en {desincronizaciones}/50 pruebas"
        print(f"  50 pruebas con imagenes identicas: A == B en todas | [OK]")
        print(f"  (El mismo crop y flip se aplica a ambas imagenes del par)")

        # ==============================================================
        # 5. CACHE EN RAM (velocidad comparativa)
        # ==============================================================
        print(f"\n[5] Cache en RAM vs sin cache")
        print("-" * 55)

        ds_sin_cache = DatasetParesSideBySide(str(dir_sbs), modo="train", cache_en_ram=False)
        ds_con_cache = DatasetParesSideBySide(str(dir_sbs), modo="train", cache_en_ram=True)

        # Medir tiempo de acceso secuencial (simula una epoca)
        def medir_acceso(ds, nombre):
            inicio = time.perf_counter()
            for i in range(len(ds)):
                _ = ds[i]
            elapsed = time.perf_counter() - inicio
            items_s = len(ds) / elapsed
            print(f"  {nombre:<25}: {elapsed*1000:.1f}ms total | {items_s:.0f} items/s")
            return items_s

        v_sin = medir_acceso(ds_sin_cache, "Sin cache (disco)")
        v_con = medir_acceso(ds_con_cache, "Con cache (RAM)")
        aceleracion = v_con / max(v_sin, 1e-9)
        print(f"  Aceleracion con cache: {aceleracion:.1f}x mas rapido")

        # ==============================================================
        # 6. ESTIMACION DE MEMORIA PARA COLAB
        # ==============================================================
        print(f"\n[6] Estimacion de memoria (dataset Maps ~1096 pares)")
        print("-" * 55)

        for n, label in [(N_IMAGENES, "prueba local"), (1096, "Maps completo")]:
            mem = estimar_memoria_dataset(n, tamanio_px=256)
            print(f"  {label} ({n} pares):")
            print(f"    Cache completo  : {mem['cache_completo_MB']:.1f} MB")
            print(f"    Un batch (bs=1) : {mem['batch1_MB']:.2f} MB")
            print(f"    Un batch (bs=4) : {mem['batch4_MB']:.2f} MB")

        # ==============================================================
        # 7. BENCHMARK DE VELOCIDAD DEL DATALOADER
        # ==============================================================
        print(f"\n[7] Benchmark DataLoader (num_workers=0, sin CUDA)")
        print("-" * 55)

        loader_bench = crear_dataloader(
            DatasetParesSideBySide(str(dir_sbs), modo="train"),
            batch_size=1, modo="train", num_workers=0
        )
        bench = benchmark_velocidad(loader_bench, n_batches=N_IMAGENES - 1)

        print(f"  Batches medidos     : {bench['batches_medidos']}")
        print(f"  Tiempo por batch    : {bench['tiempo_medio_ms']:.1f}ms "
              f"(min={bench['tiempo_min_ms']:.1f}, max={bench['tiempo_max_ms']:.1f})")
        print(f"  Items por segundo   : {bench['items_por_segundo']:.0f}")
        print(f"  Batches por hora    : {bench['batches_por_hora']:.0f}")
        print(f"  Nota: en Colab con disco, esperar ~50-200 items/s sin cache")

        # ==============================================================
        # RESUMEN FINAL
        # ==============================================================
        print(f"\n{SEP}")
        print(f"  RESUMEN DE COMPONENTES")
        print(f"  {'':4} TransformacionesSincronizadas: resize->crop->flip->tensor->norm [-1,1]")
        print(f"  {'':4} DatasetParesSideBySide       : formato A|B en un unico archivo")
        print(f"  {'':4} DatasetParesCarpetas          : formato carpeta_A/ + carpeta_B/")
        print(f"  {'':4} crear_dataloader()            : pin_memory, workers, drop_last")
        print(f"  {'':4} estadisticas_batch()          : verificar normalizacion")
        print(f"  {'':4} benchmark_velocidad()         : medir throughput de carga")
        print(f"  {'':4} estimar_memoria_dataset()     : planificar VRAM y RAM")
        print(SEP)
        print("  [OK] dataset_loader.py verificado correctamente.")
        print(SEP)
