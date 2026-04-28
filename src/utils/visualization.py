"""
visualization.py — Visualizaciones para el blog y monitoreo del entrenamiento
==============================================================================

Las visualizaciones cumplen dos roles:
1. **Monitoreo del entrenamiento**: las curvas de pérdida permiten detectar
   problemas (mode collapse, discriminador demasiado fuerte, etc.) sin
   esperar a ver las imágenes generadas.
2. **Contenido del blog**: las grillas de comparación (entrada | generado | real)
   son el formato estándar para presentar resultados de traducción imagen a imagen.

Todas las funciones reciben tensores normalizados en [-1, 1] y aplican
desnormalización internamente antes de mostrar.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend sin pantalla, compatible con Colab y servidores
import matplotlib.pyplot as plt
from PIL import Image

from src.data.transforms import desnormalizar


def mostrar_grilla_muestras(
    real_A: torch.Tensor,
    fake_B: torch.Tensor,
    real_B: torch.Tensor,
    n_muestras: int = 4,
    titulo: str = "Comparación: Entrada | Generado | Real",
    ruta_guardado: Optional[str] = None,
) -> plt.Figure:
    """
    Muestra una grilla de comparación: Entrada | Generado | Real.

    Esta es la visualización más importante del blog. Permite evaluar
    visualmente si el generador está aprendiendo a traducir correctamente.

    Layout para n_muestras=4:
        Fila 1: [A1] [fake_B1] [B1]
        Fila 2: [A2] [fake_B2] [B2]
        Fila 3: [A3] [fake_B3] [B3]
        Fila 4: [A4] [fake_B4] [B4]

    Args:
        real_A:       Batch de imágenes de entrada (dominio A). (N, 3, 256, 256).
        fake_B:       Batch de imágenes generadas. (N, 3, 256, 256).
        real_B:       Batch de imágenes objetivo reales. (N, 3, 256, 256).
        n_muestras:   Número de ejemplos a mostrar (máximo: batch_size).
        titulo:       Título de la figura.
        ruta_guardado: Si se especifica, guarda la figura en esa ruta.

    Returns:
        Figura de matplotlib.
    """
    n = min(n_muestras, real_A.shape[0])
    fig, ejes = plt.subplots(n, 3, figsize=(12, 4 * n))
    fig.suptitle(titulo, fontsize=14, y=1.02)

    if n == 1:
        ejes = ejes[None, :]  # Añadir dimensión de fila para consistencia

    encabezados = ["Entrada (Dominio A)", "Generado (G)", "Real (Dominio B)"]
    imagenes = [real_A, fake_B, real_B]

    for col, (encabezado, tensor) in enumerate(zip(encabezados, imagenes)):
        ejes[0, col].set_title(encabezado, fontsize=11, fontweight="bold")

    for fila in range(n):
        for col, tensor in enumerate(imagenes):
            # Desnormalizar de [-1,1] a [0,1] y convertir a numpy
            img_np = desnormalizar(tensor[fila]).permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)

            ejes[fila, col].imshow(img_np)
            ejes[fila, col].axis("off")

    plt.tight_layout()

    if ruta_guardado:
        Path(ruta_guardado).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta_guardado, dpi=150, bbox_inches="tight")
        print(f"[Visual] Grilla guardada: {ruta_guardado}")

    return fig


def graficar_curvas_perdida(
    losses_historia: Dict[str, List[float]],
    titulo: str = "Curvas de Pérdida del Entrenamiento",
    ruta_guardado: Optional[str] = None,
) -> plt.Figure:
    """
    Grafica la evolución de las pérdidas a lo largo del entrenamiento.

    Qué buscar en las curvas:
    - loss_D_real y loss_D_fake deberían converger cerca de 0.25 (lsgan)
      o 0.69 (vanilla BCE) en un entrenamiento equilibrado.
    - loss_G_total debería decrecer suavemente.
    - Si loss_D cae a ~0 y loss_G_GAN sube, el discriminador es demasiado fuerte.
    - Si loss_G_GAN cae a ~0 rápidamente, puede ser mode collapse.

    Args:
        losses_historia: Dict con claves como 'D_real', 'G_total', etc.
                         Cada valor es una lista de pérdidas promedio por época.
        titulo:          Título de la figura.
        ruta_guardado:   Si se especifica, guarda la figura.

    Returns:
        Figura de matplotlib.
    """
    fig, ejes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(titulo, fontsize=13)

    epocas = list(range(1, len(list(losses_historia.values())[0]) + 1))

    # Subplot izquierdo: pérdidas del discriminador
    ax_d = ejes[0]
    ax_d.set_title("Discriminador", fontweight="bold")
    for nombre in ["D_real", "D_fake"]:
        if nombre in losses_historia:
            ax_d.plot(epocas, losses_historia[nombre], label=nombre, linewidth=1.5)
    ax_d.set_xlabel("Época")
    ax_d.set_ylabel("Pérdida")
    ax_d.legend()
    ax_d.grid(True, alpha=0.3)

    # Subplot derecho: pérdidas del generador
    ax_g = ejes[1]
    ax_g.set_title("Generador", fontweight="bold")
    for nombre in ["G_GAN", "G_L1", "G_total"]:
        if nombre in losses_historia:
            ax_g.plot(epocas, losses_historia[nombre], label=nombre, linewidth=1.5)
    ax_g.set_xlabel("Época")
    ax_g.set_ylabel("Pérdida")
    ax_g.legend()
    ax_g.grid(True, alpha=0.3)

    plt.tight_layout()

    if ruta_guardado:
        Path(ruta_guardado).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta_guardado, dpi=150, bbox_inches="tight")
        print(f"[Visual] Curvas de pérdida guardadas: {ruta_guardado}")

    return fig


def tensor_a_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convierte un tensor (3, H, W) normalizado en [-1, 1] a imagen PIL.

    Args:
        tensor: Tensor (3, H, W) con valores en [-1, 1].

    Returns:
        Imagen PIL en modo RGB.
    """
    img_np = desnormalizar(tensor).permute(1, 2, 0).numpy()
    img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(img_np)


def crear_gif_progreso(
    directorio_imagenes: str,
    ruta_salida: str,
    duracion_frame_ms: int = 200,
) -> None:
    """
    Crea un GIF animado que muestra la progresión del entrenamiento época a época.

    Este GIF es ideal para el blog: visualiza de forma compacta cómo mejora
    la calidad de las imágenes generadas con el entrenamiento.

    Args:
        directorio_imagenes: Carpeta con imágenes de progreso (una por época).
                             Se esperan ordenadas alfabéticamente por nombre.
        ruta_salida:         Ruta del GIF de salida.
        duracion_frame_ms:   Duración de cada frame en milisegundos.
    """
    extensiones = {".png", ".jpg", ".jpeg"}
    rutas = sorted([
        p for p in Path(directorio_imagenes).iterdir()
        if p.suffix.lower() in extensiones
    ])

    if len(rutas) < 2:
        print(f"[Visual] Se necesitan al menos 2 imágenes para crear un GIF. "
              f"Encontradas: {len(rutas)}")
        return

    frames = [Image.open(r).convert("RGB") for r in rutas]
    Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)

    frames[0].save(
        ruta_salida,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=duracion_frame_ms,
        loop=0,  # 0 = loop infinito
    )
    print(f"[Visual] GIF de progreso guardado: {ruta_salida} ({len(frames)} frames)")


# ===========================================================================
# BLOQUE DE VERIFICACIÓN LOCAL
# ===========================================================================
if __name__ == "__main__":
    import tempfile
    print("=" * 60)
    print("Verificación local: visualization.py")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Tensores dummy de imágenes
        batch_A    = torch.randn(4, 3, 256, 256)
        batch_fake = torch.randn(4, 3, 256, 256)
        batch_B    = torch.randn(4, 3, 256, 256)

        # Verificar grilla de comparación
        ruta_grilla = os.path.join(tmpdir, "grilla_test.png")
        fig = mostrar_grilla_muestras(batch_A, batch_fake, batch_B, n_muestras=2,
                                       ruta_guardado=ruta_grilla)
        assert os.path.exists(ruta_grilla), "Error: la grilla no se guardó"
        plt.close(fig)
        print("  mostrar_grilla_muestras: [OK]")

        # Verificar curvas de pérdida
        historia_dummy = {
            "D_real": [0.45, 0.42, 0.40, 0.38],
            "D_fake": [0.51, 0.48, 0.46, 0.43],
            "G_GAN":  [0.88, 0.79, 0.73, 0.68],
            "G_L1":   [0.31, 0.28, 0.25, 0.23],
            "G_total": [31.88, 28.79, 25.73, 23.68],
        }
        ruta_curvas = os.path.join(tmpdir, "curvas_test.png")
        fig = graficar_curvas_perdida(historia_dummy, ruta_guardado=ruta_curvas)
        assert os.path.exists(ruta_curvas), "Error: las curvas no se guardaron"
        plt.close(fig)
        print("  graficar_curvas_perdida: [OK]")

        # Verificar conversión tensor → PIL
        tensor_test = torch.randn(3, 256, 256)
        img_pil = tensor_a_pil(tensor_test)
        assert img_pil.size == (256, 256)
        print("  tensor_a_pil: [OK]")

    print("\n[OK] visualization.py verificado correctamente.")
