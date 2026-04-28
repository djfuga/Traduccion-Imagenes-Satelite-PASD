"""
transforms.py — Transformaciones de imagen para entrenamiento y validación
===========================================================================

Las transformaciones cumplen dos propósitos:
1. **Preprocesamiento**: Llevar las imágenes al formato que espera la red.
2. **Augmentación** (solo en entrenamiento): Aumentar artificialmente el tamaño
   del dataset y mejorar la generalización del modelo.

Convención de normalización:
------------------------------
Las imágenes se normalizan al rango [-1, 1] usando media=0.5 y std=0.5:
    imagen_normalizada = (imagen_0_1 - 0.5) / 0.5 = imagen_0_1 * 2 - 1

¿Por qué [-1, 1]?
    La capa Tanh del generador produce valores en exactamente este rango.
    Si normalizamos la entrada al mismo rango, el modelo no tiene que
    "deshacer" una escala diferente al compararla con la salida.

Augmentación estándar de Pix2Pix:
-----------------------------------
1. Resize a 286×286 (ligeramente más grande que 256×256).
2. RandomCrop a 256×256 (recorte aleatorio).
3. RandomHorizontalFlip.

Este "jitter" de escala es el estándar del paper original. Expone al modelo
a ligeras variaciones espaciales, reduciendo el sobreajuste.

IMPORTANTE: El mismo recorte y flip debe aplicarse TANTO a la imagen de
entrada (dominio A) COMO a la imagen objetivo (dominio B). Si aplicamos
transformaciones diferentes a cada mitad del par, el modelo aprende pares
incoherentes. Esto se gestiona en dataset.py pasando la misma semilla
aleatoria a ambas transformaciones.

Referencia: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
from PIL import Image


# Parámetros estándar de Pix2Pix
TAMAÑO_CARGA = 286     # Tamaño de resize previo al crop
TAMAÑO_FINAL = 256     # Tamaño final de la imagen
MEDIA_NORM = (0.5, 0.5, 0.5)
STD_NORM   = (0.5, 0.5, 0.5)


class TransformPar:
    """
    Aplica transformaciones sincronizadas a un PAR de imágenes (A, B).

    El problema de aplicar transformaciones independientes a cada imagen:
        - Si aplicamos RandomCrop separado a A y B, el crop puede recortar
          regiones distintas → el par ya no es coherente.
        - Si aplicamos RandomFlip separado, una imagen puede estar volteada
          y la otra no.

    Solución: decidir los parámetros aleatorios UNA VEZ y aplicarlos a las dos.
    """

    def __init__(self, modo: str = "train"):
        """
        Args:
            modo: 'train' para augmentación completa, 'val'/'test' para solo resize.
        """
        assert modo in ("train", "val", "test"), \
            f"Modo '{modo}' no reconocido. Usa 'train', 'val' o 'test'."
        self.modo = modo

    def __call__(self, imagen_a: Image.Image, imagen_b: Image.Image):
        """
        Aplica las transformaciones sincronizadas.

        Args:
            imagen_a: Imagen del dominio A (PIL Image).
            imagen_b: Imagen del dominio B (PIL Image).

        Returns:
            Tupla (tensor_a, tensor_b) con valores en [-1, 1], forma (3, 256, 256).
        """
        # 1. Resize (igual para ambas)
        tamano_resize = TAMAÑO_CARGA if self.modo == "train" else TAMAÑO_FINAL
        imagen_a = TF.resize(imagen_a, [tamano_resize, tamano_resize], Image.BICUBIC)
        imagen_b = TF.resize(imagen_b, [tamano_resize, tamano_resize], Image.BICUBIC)

        if self.modo == "train":
            # 2. RandomCrop: decidir posición del crop UNA vez para ambas imágenes
            i, j, h, w = T.RandomCrop.get_params(
                imagen_a, output_size=(TAMAÑO_FINAL, TAMAÑO_FINAL)
            )
            imagen_a = TF.crop(imagen_a, i, j, h, w)
            imagen_b = TF.crop(imagen_b, i, j, h, w)

            # 3. RandomHorizontalFlip: la misma decisión para ambas
            if random.random() > 0.5:
                imagen_a = TF.hflip(imagen_a)
                imagen_b = TF.hflip(imagen_b)

        # 4. Convertir a tensor [0, 1]
        tensor_a = TF.to_tensor(imagen_a)
        tensor_b = TF.to_tensor(imagen_b)

        # 5. Normalizar a [-1, 1] (compatible con Tanh del generador)
        tensor_a = TF.normalize(tensor_a, mean=MEDIA_NORM, std=STD_NORM)
        tensor_b = TF.normalize(tensor_b, mean=MEDIA_NORM, std=STD_NORM)

        return tensor_a, tensor_b


def desnormalizar(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convierte un tensor de [-1, 1] a [0, 1] para visualización.

    Operación inversa de la normalización:
        imagen_original = (tensor + 1) / 2

    Args:
        tensor: Tensor normalizado en [-1, 1], cualquier forma.

    Returns:
        Tensor con valores en [0, 1], misma forma que la entrada.
    """
    # Clamp para manejar pequeños valores fuera del rango [-1, 1] por imprecisión numérica
    return torch.clamp((tensor + 1.0) / 2.0, min=0.0, max=1.0)


# ===========================================================================
# BLOQUE DE VERIFICACIÓN LOCAL
# Ejecutar: python src/data/transforms.py
# Verifica que las transformaciones producen tensores de forma correcta y
# que los valores están en el rango esperado.
# ===========================================================================
if __name__ == "__main__":
    import numpy as np
    print("=" * 60)
    print("Verificación local: transforms.py")
    print("=" * 60)

    # Crear imágenes dummy de PIL (simular par satelital/boceto)
    imagen_dummy_a = Image.fromarray(
        (np.random.rand(300, 300, 3) * 255).astype(np.uint8)
    )
    imagen_dummy_b = Image.fromarray(
        (np.random.rand(300, 300, 3) * 255).astype(np.uint8)
    )

    for modo in ["train", "val"]:
        transform = TransformPar(modo=modo)
        tensor_a, tensor_b = transform(imagen_dummy_a, imagen_dummy_b)

        print(f"\n--- Modo: {modo} ---")
        print(f"  tensor_a shape : {tensor_a.shape}  → Esperado: (3, 256, 256)")
        print(f"  tensor_b shape : {tensor_b.shape}  → Esperado: (3, 256, 256)")
        print(f"  tensor_a rango : [{tensor_a.min():.3f}, {tensor_a.max():.3f}]  → [-1, 1]")
        print(f"  tensor_b rango : [{tensor_b.min():.3f}, {tensor_b.max():.3f}]  → [-1, 1]")

        # Verificar desnormalización
        tensor_a_desn = desnormalizar(tensor_a)
        print(f"  desnormalizado : [{tensor_a_desn.min():.3f}, {tensor_a_desn.max():.3f}]  → [0, 1]")

        assert tensor_a.shape == torch.Size([3, 256, 256])
        assert tensor_b.shape == torch.Size([3, 256, 256])
        assert tensor_a.min() >= -1.1 and tensor_a.max() <= 1.1
        assert tensor_a_desn.min() >= 0.0 and tensor_a_desn.max() <= 1.0

    print("\n[OK] transforms.py verificado correctamente.")
