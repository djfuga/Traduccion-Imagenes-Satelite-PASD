"""
predict.py — Inferencia con el generador entrenado
====================================================

Una vez entrenado el modelo, este módulo permite:
1. Cargar solo el generador (no el discriminador, que no se necesita en inferencia).
2. Generar imágenes para imágenes individuales o directorios completos.
3. Crear imágenes de comparación para el blog (entrada | generado | real).

Inferencia vs. Entrenamiento:
-------------------------------
Durante la inferencia:
- Solo se usa el GENERADOR (G), no el discriminador.
- Se desactivan los gradientes (torch.no_grad()) para ahorrar memoria.
- Se pone el modelo en modo eval() para desactivar Dropout e InstanceNorm
  estocástico (se usan estadísticas fijas).
- El batch_size puede ser cualquier valor, ya que InstanceNorm funciona bien
  con cualquier tamaño de batch en inferencia.

Referencia: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from src.models.generator import GeneradorUNet
from src.data.transforms import TransformPar, desnormalizar


def cargar_generador(
    ruta_checkpoint: str,
    dispositivo: torch.device,
    filtros_base: int = 64,
) -> GeneradorUNet:
    """
    Carga el generador desde un checkpoint para inferencia.

    A diferencia de cargar_checkpoint() del módulo de training (que restaura
    el estado completo de entrenamiento), esta función carga SOLO los pesos
    del generador, lo que es más eficiente para inferencia.

    Soporta dos formatos de checkpoint:
    - Checkpoint completo (training): contiene clave 'generador'
    - Checkpoint solo-generador: también contiene clave 'generador'

    Args:
        ruta_checkpoint: Ruta del archivo .pth del checkpoint.
        dispositivo:     Dispositivo donde ejecutar la inferencia.
        filtros_base:    Filtros base del generador (debe coincidir con el entrenamiento).

    Returns:
        GeneradorUNet en modo eval, listo para inferencia.
    """
    if not os.path.exists(ruta_checkpoint):
        raise FileNotFoundError(f"Checkpoint no encontrado: {ruta_checkpoint}")

    generador = GeneradorUNet(filtros_base=filtros_base)

    # map_location permite cargar checkpoints GPU en CPU y viceversa
    estado = torch.load(ruta_checkpoint, map_location=dispositivo)

    # El checkpoint puede contener solo el generador o el estado completo
    if "generador" in estado:
        generador.load_state_dict(estado["generador"])
    else:
        # Compatibilidad: si se guardó directamente el state_dict sin wrapper
        generador.load_state_dict(estado)

    generador = generador.to(dispositivo)
    generador.eval()  # Desactivar Dropout e InstanceNorm estocástico

    print(f"[Inferencia] Generador cargado desde: {ruta_checkpoint}")
    return generador


def predecir_imagen(
    generador: nn.Module,
    ruta_imagen: str,
    dispositivo: torch.device,
    direction: str = "AtoB",
) -> Tuple[Image.Image, Image.Image]:
    """
    Genera la imagen traducida para una imagen de entrada.

    Args:
        generador:    Modelo generador cargado.
        ruta_imagen:  Ruta de la imagen de entrada.
                      Puede ser una imagen simple (256×256) o un par side-by-side (512×256).
        dispositivo:  Dispositivo de inferencia.
        direction:    Si la imagen es side-by-side, qué mitad usar como entrada.

    Returns:
        Tupla (imagen_entrada_pil, imagen_generada_pil).
    """
    transform = TransformPar(modo="val")

    imagen = Image.open(ruta_imagen).convert("RGB")
    ancho, alto = imagen.size

    # Detectar si es imagen side-by-side o imagen simple
    if ancho == alto * 2:  # Formato side-by-side (512×256)
        mitad = ancho // 2
        imagen_a = imagen.crop((0, 0, mitad, alto))
        imagen_b = imagen.crop((mitad, 0, ancho, alto))
        imagen_entrada = imagen_a if direction == "AtoB" else imagen_b
    else:
        # Imagen simple: usar directamente como entrada
        imagen_entrada = imagen

    # Crear un par dummy para el transform (necesita dos imágenes)
    tensor_entrada, _ = transform(imagen_entrada, imagen_entrada)
    tensor_entrada = tensor_entrada.unsqueeze(0).to(dispositivo)  # Añadir dim de batch

    with torch.no_grad():
        tensor_generado = generador(tensor_entrada)

    # Convertir tensores a imágenes PIL
    img_entrada_pil = _tensor_a_pil(tensor_entrada[0].cpu())
    img_generada_pil = _tensor_a_pil(tensor_generado[0].cpu())

    return img_entrada_pil, img_generada_pil


def predecir_directorio(
    generador: nn.Module,
    directorio_entrada: str,
    directorio_salida: str,
    dispositivo: torch.device,
    direction: str = "AtoB",
) -> List[str]:
    """
    Genera imágenes traducidas para todas las imágenes de un directorio.

    Útil para evaluar el modelo sobre el conjunto de test completo o para
    generar resultados en lote para el blog.

    Args:
        generador:           Modelo generador.
        directorio_entrada:  Directorio con imágenes de entrada.
        directorio_salida:   Directorio donde guardar las imágenes generadas.
        dispositivo:         Dispositivo de inferencia.
        direction:           Dirección de traducción.

    Returns:
        Lista de rutas de las imágenes generadas.
    """
    Path(directorio_salida).mkdir(parents=True, exist_ok=True)

    extensiones = {".jpg", ".jpeg", ".png", ".tiff"}
    rutas_entrada = sorted([
        p for p in Path(directorio_entrada).iterdir()
        if p.suffix.lower() in extensiones
    ])

    if not rutas_entrada:
        print(f"[Inferencia] No se encontraron imágenes en: {directorio_entrada}")
        return []

    rutas_salida = []
    print(f"[Inferencia] Procesando {len(rutas_entrada)} imágenes...")

    for i, ruta in enumerate(rutas_entrada):
        _, img_generada = predecir_imagen(generador, str(ruta), dispositivo, direction)

        # Mantener el nombre original del archivo
        ruta_salida = os.path.join(directorio_salida, ruta.name)
        img_generada.save(ruta_salida)
        rutas_salida.append(ruta_salida)

        if (i + 1) % 10 == 0 or (i + 1) == len(rutas_entrada):
            print(f"  [{i+1}/{len(rutas_entrada)}] Procesado: {ruta.name}")

    print(f"[Inferencia] Completado. Imágenes guardadas en: {directorio_salida}")
    return rutas_salida


def crear_grilla_comparacion(
    img_entrada: Image.Image,
    img_generada: Image.Image,
    img_real: Optional[Image.Image] = None,
    ancho_total: int = 768,
) -> Image.Image:
    """
    Crea una imagen de comparación horizontal para el blog.

    Layout:
        Con imagen real:    [Entrada | Generado | Real]
        Sin imagen real:    [Entrada | Generado]

    Esta imagen de 3 columnas (o 2 columnas) es el formato estándar
    para presentar resultados de traducción imagen a imagen en papers y blogs.

    Args:
        img_entrada:   Imagen de entrada.
        img_generada:  Imagen generada por el modelo.
        img_real:      Imagen real objetivo (opcional, para comparación cuantitativa).
        ancho_total:   Ancho total de la imagen de comparación en píxeles.

    Returns:
        Imagen PIL con las imágenes side-by-side.
    """
    n_columnas = 3 if img_real is not None else 2
    ancho_col = ancho_total // n_columnas
    alto = ancho_col  # Asumimos imágenes cuadradas

    imagenes = [img_entrada, img_generada]
    if img_real is not None:
        imagenes.append(img_real)

    # Redimensionar todas al mismo tamaño
    imagenes_redim = [img.resize((ancho_col, alto), Image.BICUBIC) for img in imagenes]

    # Crear imagen final
    comparacion = Image.new("RGB", (ancho_total, alto))
    for i, img in enumerate(imagenes_redim):
        comparacion.paste(img, (i * ancho_col, 0))

    return comparacion


def _tensor_a_pil(tensor: torch.Tensor) -> Image.Image:
    """Convierte tensor (3, H, W) en [-1,1] a imagen PIL."""
    img_np = desnormalizar(tensor).permute(1, 2, 0).numpy()
    img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(img_np)


# ===========================================================================
# BLOQUE DE VERIFICACIÓN LOCAL
# ===========================================================================
if __name__ == "__main__":
    import tempfile
    import sys
    from pathlib import Path

    print("=" * 60)
    print("Verificación local: predict.py")
    print("=" * 60)

    dispositivo = torch.device("cpu")

    # Crear un generador con pesos aleatorios (sin checkpoint real)
    generador = GeneradorUNet(filtros_base=16)  # Pequeño para CPU
    generador.eval()

    # Crear imagen dummy de 256×256
    with tempfile.TemporaryDirectory() as tmpdir:
        # Guardar imagen dummy de entrada (256×256)
        img_dummy = Image.fromarray(
            (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        )
        ruta_entrada = os.path.join(tmpdir, "test_input.png")
        img_dummy.save(ruta_entrada)

        # Verificar predicción individual
        print("\n--- predecir_imagen ---")
        img_ent, img_gen = predecir_imagen(generador, ruta_entrada, dispositivo)
        print(f"  Imagen entrada  : {img_ent.size}")
        print(f"  Imagen generada : {img_gen.size}")
        assert img_gen.size == (256, 256), "Error: tamaño incorrecto"

        # Verificar predicción de directorio
        print("\n--- predecir_directorio ---")
        dir_salida = os.path.join(tmpdir, "output")
        rutas = predecir_directorio(
            generador, tmpdir, dir_salida, dispositivo
        )
        assert len(rutas) == 1

        # Verificar grilla de comparación
        print("\n--- crear_grilla_comparacion ---")
        grilla = crear_grilla_comparacion(img_ent, img_gen, img_ent)
        print(f"  Grilla size: {grilla.size}")
        assert grilla.size == (768, 256)

    print("\n[OK] predict.py verificado correctamente.")
