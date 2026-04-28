"""
dataset.py — Dataset de pares de imágenes alineadas (AlignedDataset)
=====================================================================

Este módulo implementa el cargador de datos para el sistema de traducción
imagen a imagen. El formato de datos estándar de Pix2Pix almacena cada par
(dominio_A, dominio_B) como UNA SOLA IMAGEN en formato "side-by-side":

    ┌─────────────┬─────────────┐
    │  Dominio A  │  Dominio B  │
    │  (256×256)  │  (256×256)  │
    └─────────────┴─────────────┘
          Imagen total: 512×256

¿Por qué este formato?
    - Garantiza que A y B están perfectamente alineados (son capturas del
      mismo lugar geográfico).
    - Simplifica el almacenamiento (un archivo por par en lugar de dos).
    - Es el formato nativo del repositorio pytorch-CycleGAN-and-pix2pix,
      lo que facilita usar sus scripts de preparación de datos.

Para nuestro proyecto:
    - Dominio A: imagen satelital (Sentinel-2, Google Maps, etc.)
    - Dominio B: boceto/mapa cartográfico (tiles OpenStreetMap)
    - Dirección 'AtoB': dado satélite → generar boceto
    - Dirección 'BtoA': dado boceto → generar satélite

Optimizaciones para Google Colab T4:
    - num_workers=2: Colab gratuito tiene 2 CPUs, más workers causa OOM.
    - pin_memory=True solo con CUDA: evita copias innecesarias en CPU.
    - persistent_workers=True: evita recrear workers al inicio de cada época.
    - drop_last=True: evita batches de tamaño 1 que rompen InstanceNorm.

Referencia: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
(ver data/aligned_dataset.py)
Referencia Sketch2Map: https://github.com/PerlMonker303/S2MP
"""

import os
from pathlib import Path
from typing import Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.data.transforms import TransformPar


class AlignedDataset(Dataset):
    """
    Dataset de pares de imágenes alineadas en formato side-by-side (512×256).

    Cada archivo en el directorio raíz es una imagen de 512×256 píxeles donde:
        - La mitad izquierda (columnas 0:256) es el dominio A.
        - La mitad derecha (columnas 256:512) es el dominio B.

    El parámetro `direction` controla cuál mitad es la "entrada" y cuál es
    el "objetivo" durante el entrenamiento:
        - 'AtoB': entrada=A (satélite), objetivo=B (boceto)
        - 'BtoA': entrada=B (boceto), objetivo=A (satélite)

    Esto permite usar el mismo dataset para entrenar en ambas direcciones
    simplemente cambiando un parámetro, sin duplicar los datos.
    """

    def __init__(
        self,
        directorio_raiz: str,
        direction: str = "AtoB",
        modo: str = "train",
    ):
        """
        Args:
            directorio_raiz: Ruta a la carpeta con las imágenes side-by-side.
                             Ej: 'data/processed/train/'
            direction:       'AtoB' (satélite→boceto) o 'BtoA' (boceto→satélite).
            modo:            'train', 'val' o 'test'. Controla la augmentación.
        """
        super().__init__()

        assert direction in ("AtoB", "BtoA"), \
            f"Dirección '{direction}' no válida. Usa 'AtoB' o 'BtoA'."
        assert modo in ("train", "val", "test"), \
            f"Modo '{modo}' no válido. Usa 'train', 'val' o 'test'."

        self.direction = direction
        self.modo = modo
        self.transform = TransformPar(modo=modo)

        # Cargar rutas de todas las imágenes del directorio
        raiz = Path(directorio_raiz)
        if not raiz.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {directorio_raiz}")

        extensiones_validas = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
        self.rutas_imagenes: List[Path] = sorted([
            p for p in raiz.iterdir()
            if p.suffix.lower() in extensiones_validas
        ])

        if len(self.rutas_imagenes) == 0:
            raise ValueError(f"No se encontraron imágenes en: {directorio_raiz}")

        print(f"[Dataset] '{modo}' | {len(self.rutas_imagenes)} pares | dirección: {direction}")

    def __len__(self) -> int:
        return len(self.rutas_imagenes)

    def __getitem__(self, indice: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Carga un par de imágenes y aplica las transformaciones.

        Args:
            indice: Índice del par a cargar.

        Returns:
            Tupla (imagen_entrada, imagen_objetivo), cada tensor de forma (3, 256, 256)
            con valores en [-1, 1].
        """
        ruta = self.rutas_imagenes[indice]

        # Cargar la imagen side-by-side completa (512×256 o 512×256)
        imagen_completa = Image.open(ruta).convert("RGB")
        ancho, alto = imagen_completa.size  # PIL: (ancho, alto)

        # Dividir en dos mitades iguales
        mitad_ancho = ancho // 2
        imagen_a = imagen_completa.crop((0, 0, mitad_ancho, alto))
        imagen_b = imagen_completa.crop((mitad_ancho, 0, ancho, alto))

        # Asignar roles según la dirección
        if self.direction == "AtoB":
            imagen_entrada = imagen_a   # Satélite → entrada
            imagen_objetivo = imagen_b  # Boceto → objetivo
        else:  # BtoA
            imagen_entrada = imagen_b   # Boceto → entrada
            imagen_objetivo = imagen_a  # Satélite → objetivo

        # Aplicar transformaciones sincronizadas (mismo crop y flip para ambas)
        tensor_entrada, tensor_objetivo = self.transform(imagen_entrada, imagen_objetivo)

        return tensor_entrada, tensor_objetivo


def crear_dataloader(
    directorio_raiz: str,
    direction: str = "AtoB",
    modo: str = "train",
    batch_size: int = 1,
    num_workers: int = 2,
    shuffle: bool = None,
) -> DataLoader:
    """
    Crea un DataLoader optimizado para Google Colab T4.

    Parámetros de optimización:
    - num_workers=2: máximo recomendado en Colab gratuito (2 CPUs disponibles).
      Más workers causa OOM por la memoria del proceso de Python.
    - pin_memory=True (solo CUDA): permite transferencia asíncrona CPU→GPU,
      reduciendo el tiempo de espera del DataLoader.
    - persistent_workers=True: los procesos worker no se terminan al finalizar
      una época, eliminando el overhead de re-inicialización.
    - drop_last=True en modo train: elimina el último batch si es incompleto.
      InstanceNorm con batch_size=1 funciona, pero con batch_size<1 falla.
      Más importante: evita estadísticas sesgadas en el último batch pequeño.
    - prefetch_factor=2: cada worker pre-carga 2 batches mientras la GPU procesa,
      reduciendo tiempos de espera IO.

    Args:
        directorio_raiz: Ruta al directorio de imágenes.
        direction:       'AtoB' o 'BtoA'.
        modo:            'train', 'val' o 'test'.
        batch_size:      Tamaño de batch. Default: 1 (estándar Pix2Pix).
        num_workers:     Procesos paralelos de carga. Default: 2 (óptimo Colab).
        shuffle:         Si None, True para train y False para val/test.

    Returns:
        DataLoader configurado.
    """
    dataset = AlignedDataset(
        directorio_raiz=directorio_raiz,
        direction=direction,
        modo=modo,
    )

    if shuffle is None:
        shuffle = (modo == "train")

    # pin_memory solo es útil (y seguro) cuando hay CUDA disponible
    usar_pin_memory = torch.cuda.is_available()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=usar_pin_memory,
        drop_last=(modo == "train"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    print(f"[DataLoader] batch_size={batch_size} | shuffle={shuffle} | "
          f"num_workers={num_workers} | pin_memory={usar_pin_memory}")

    return dataloader


# ===========================================================================
# BLOQUE DE VERIFICACIÓN LOCAL
# Ejecutar: python src/data/dataset.py
# Crea un dataset temporal con imágenes dummy y verifica formas y rangos.
# ===========================================================================
if __name__ == "__main__":
    import tempfile
    import numpy as np
    print("=" * 60)
    print("Verificación local: dataset.py")
    print("=" * 60)

    # Crear directorio temporal con imágenes side-by-side dummy
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nCreando imágenes dummy en: {tmpdir}")

        for i in range(5):
            # Imagen side-by-side: 512 ancho × 256 alto
            imagen_dummy = Image.fromarray(
                (np.random.rand(256, 512, 3) * 255).astype(np.uint8)
            )
            imagen_dummy.save(os.path.join(tmpdir, f"par_{i:03d}.png"))

        # Probar en ambas direcciones y modos
        for direction in ["AtoB", "BtoA"]:
            for modo in ["train", "val"]:
                print(f"\n--- direction={direction}, modo={modo} ---")
                dataset = AlignedDataset(tmpdir, direction=direction, modo=modo)

                entrada, objetivo = dataset[0]
                print(f"  entrada  shape : {entrada.shape}   → Esperado: (3, 256, 256)")
                print(f"  objetivo shape : {objetivo.shape}  → Esperado: (3, 256, 256)")
                print(f"  entrada  rango : [{entrada.min():.3f}, {entrada.max():.3f}]")
                print(f"  objetivo rango : [{objetivo.min():.3f}, {objetivo.max():.3f}]")

                assert entrada.shape  == torch.Size([3, 256, 256])
                assert objetivo.shape == torch.Size([3, 256, 256])

        # Verificar DataLoader (sin CUDA en local, num_workers=0 para evitar errores)
        print("\n--- Verificando DataLoader ---")
        dataloader = crear_dataloader(
            tmpdir, direction="AtoB", modo="train",
            batch_size=2, num_workers=0
        )
        lote_entrada, lote_objetivo = next(iter(dataloader))
        print(f"  Batch entrada  : {lote_entrada.shape}   → (2, 3, 256, 256)")
        print(f"  Batch objetivo : {lote_objetivo.shape}  → (2, 3, 256, 256)")

        assert lote_entrada.shape  == torch.Size([2, 3, 256, 256])
        assert lote_objetivo.shape == torch.Size([2, 3, 256, 256])

    print("\n[OK] dataset.py verificado correctamente.")
