"""
metrics.py — Métricas cuantitativas de evaluación
==================================================

Para el blog técnico es necesario evaluar los modelos de forma cuantitativa,
no solo visual. Las métricas estándar para evaluar GANs de imagen a imagen son:

1. **SSIM (Structural Similarity Index)**:
   Mide la similitud perceptual entre dos imágenes considerando luminancia,
   contraste y estructura. Rango: [-1, 1], donde 1 = imágenes idénticas.
   Ref: Wang et al., "Image Quality Assessment: From Error Visibility to
   Structural Similarity", IEEE TIP 2004.

2. **L1 Normalizado (MAE - Mean Absolute Error)**:
   Diferencia media absoluta entre píxeles. Más robusto que MSE ante outliers.
   Rango: [0, 1], donde 0 = imágenes idénticas (para imágenes en [0,1]).

3. **FID (Fréchet Inception Distance)**:
   Mide la distancia entre la distribución de imágenes reales y generadas
   en el espacio de features de InceptionV3. NO compara pares individuales,
   sino distribuciones enteras → requiere múltiples imágenes (mínimo ~50).
   Rango: [0, ∞), donde 0 = distribuciones idénticas.
   Es el estándar de facto para evaluar GANs desde 2017.
   Ref: Heusel et al., "GANs Trained by a Two Time-Scale Update Rule
   Converge to a Local Nash Equilibrium", NeurIPS 2017.

Nota sobre FID: requiere `torchmetrics` (disponible en Colab con pip install).
SSIM y L1 solo requieren `scikit-image` y PyTorch.
"""

import torch
import numpy as np
from typing import List, Optional
from pathlib import Path

from src.data.transforms import desnormalizar


def calcular_ssim(
    tensor_pred: torch.Tensor,
    tensor_real: torch.Tensor,
) -> float:
    """
    Calcula el SSIM entre imagen predicha y real.

    SSIM es preferible al MSE/PSNR porque modela la percepción humana:
    el ojo humano es más sensible a cambios estructurales (bordes, texturas)
    que a diferencias de píxel uniformemente distribuidas.

    Por ejemplo, una imagen ligeramente desplazada 1 píxel tiene un MSE alto
    pero un SSIM casi perfecto, reflejando que perceptualmente son similares.

    Args:
        tensor_pred: Imagen generada, forma (3, H, W) o (1, 3, H, W), valores [-1,1].
        tensor_real: Imagen real, misma forma que tensor_pred.

    Returns:
        Valor SSIM en rango [0, 1] (1 = idénticas).
    """
    from skimage.metrics import structural_similarity as ssim

    # Desnormalizar a [0, 1] para SSIM
    if tensor_pred.dim() == 4:
        tensor_pred = tensor_pred[0]
        tensor_real = tensor_real[0]

    img_pred = desnormalizar(tensor_pred).permute(1, 2, 0).numpy()
    img_real = desnormalizar(tensor_real).permute(1, 2, 0).numpy()

    # channel_axis=2 indica que la dimensión de canales es la última (HxWxC)
    # data_range=1.0 porque las imágenes están en [0, 1]
    valor_ssim = ssim(img_pred, img_real, channel_axis=2, data_range=1.0)
    return float(valor_ssim)


def calcular_l1_normalizado(
    tensor_pred: torch.Tensor,
    tensor_real: torch.Tensor,
) -> float:
    """
    Calcula el error L1 medio normalizado (MAE) entre dos imágenes.

    Con imágenes normalizadas en [-1, 1], el rango máximo de diferencia
    por píxel es 2 (de -1 a 1). Se divide por 2 para normalizar a [0, 1].

    Args:
        tensor_pred: Imagen generada, forma (3, H, W) o (1, 3, H, W).
        tensor_real: Imagen real, misma forma.

    Returns:
        MAE normalizado en rango [0, 1]. Menor es mejor.
    """
    if tensor_pred.dim() == 4:
        tensor_pred = tensor_pred[0]
        tensor_real = tensor_real[0]

    # Calcular diferencia absoluta media y normalizar al rango [0, 1]
    # dividiendo por 2 (rango máximo de diferencia para imágenes en [-1, 1])
    mae = torch.mean(torch.abs(tensor_pred - tensor_real)).item()
    return mae / 2.0


def calcular_fid(
    directorio_real: str,
    directorio_generado: str,
    dispositivo: torch.device,
    tamano_imagen: int = 256,
) -> float:
    """
    Calcula el FID (Fréchet Inception Distance) entre dos conjuntos de imágenes.

    FID requiere un número razonable de imágenes (mínimo ~50, idealmente 1000+)
    para estimar correctamente las distribuciones. Con pocos ejemplos, la
    estimación estadística es ruidosa.

    Internamente usa InceptionV3 para extraer features de cada imagen, luego
    calcula la distancia de Fréchet entre las distribuciones gaussianas ajustadas
    a esos features: FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r·Σ_g)^0.5)

    Requiere: pip install torchmetrics

    Args:
        directorio_real:      Carpeta con imágenes reales (dominio B).
        directorio_generado:  Carpeta con imágenes generadas por G.
        dispositivo:          Dispositivo de cómputo (CUDA recomendado).
        tamano_imagen:        Tamaño de las imágenes.

    Returns:
        Valor FID. Menor es mejor (FID=0 indica distribuciones idénticas).
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from PIL import Image
        import torchvision.transforms as T
    except ImportError:
        print("[Métricas] torchmetrics no disponible. Instala con: pip install torchmetrics")
        return float("nan")

    transform = T.Compose([
        T.Resize((tamano_imagen, tamano_imagen)),
        T.ToTensor(),
        # FID espera imágenes en [0, 1], enteros uint8 o float32
    ])

    def cargar_imagenes_dir(directorio: str) -> torch.Tensor:
        """Carga todas las imágenes de un directorio como tensor (N, 3, H, W)."""
        rutas = sorted(Path(directorio).glob("*.png")) + sorted(Path(directorio).glob("*.jpg"))
        if not rutas:
            raise ValueError(f"No se encontraron imágenes en: {directorio}")
        imagenes = [transform(Image.open(r).convert("RGB")) for r in rutas]
        return torch.stack(imagenes)  # (N, 3, H, W)

    print(f"[Métricas] Calculando FID...")
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(dispositivo)

    imgs_real = cargar_imagenes_dir(directorio_real)
    imgs_fake = cargar_imagenes_dir(directorio_generado)

    print(f"  Imágenes reales    : {len(imgs_real)}")
    print(f"  Imágenes generadas : {len(imgs_fake)}")

    # FID actualiza sus estadísticas en batches para evitar cargar todo en VRAM
    batch_size = 32
    for i in range(0, len(imgs_real), batch_size):
        lote = imgs_real[i:i+batch_size].to(dispositivo)
        fid_metric.update(lote, real=True)

    for i in range(0, len(imgs_fake), batch_size):
        lote = imgs_fake[i:i+batch_size].to(dispositivo)
        fid_metric.update(lote, real=False)

    valor_fid = fid_metric.compute().item()
    print(f"  FID = {valor_fid:.4f}")
    return valor_fid


def evaluar_dataset(
    generador: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    dispositivo: torch.device,
    max_batches: Optional[int] = None,
) -> dict:
    """
    Evalúa el generador sobre un conjunto completo de datos.

    Calcula SSIM y L1 promedio sobre todos los pares del dataloader.
    FID se calcula por separado con calcular_fid() ya que requiere guardar
    imágenes en disco.

    Args:
        generador:   Modelo generador en modo eval.
        dataloader:  DataLoader con pares (A, B) de validación o test.
        dispositivo: Dispositivo de inferencia.
        max_batches: Número máximo de batches a evaluar (None = todos).

    Returns:
        Diccionario con métricas promedio: 'ssim_promedio', 'l1_promedio', 'n_imagenes'.
    """
    generador.eval()
    ssim_total = 0.0
    l1_total = 0.0
    n_imagenes = 0

    print("[Métricas] Evaluando dataset...")

    with torch.no_grad():
        for i, (real_A, real_B) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            real_A = real_A.to(dispositivo)
            fake_B = generador(real_A).cpu()
            real_B = real_B  # Mantener en CPU para las métricas

            batch_size = real_A.shape[0]
            for j in range(batch_size):
                ssim_total += calcular_ssim(fake_B[j], real_B[j])
                l1_total   += calcular_l1_normalizado(fake_B[j], real_B[j])
                n_imagenes += 1

    metricas = {
        "ssim_promedio": ssim_total / max(n_imagenes, 1),
        "l1_promedio":   l1_total   / max(n_imagenes, 1),
        "n_imagenes":    n_imagenes,
    }

    print(f"  Imágenes evaluadas : {n_imagenes}")
    print(f"  SSIM promedio      : {metricas['ssim_promedio']:.4f}  (mayor es mejor, máx=1)")
    print(f"  L1 promedio        : {metricas['l1_promedio']:.4f}  (menor es mejor, mín=0)")

    return metricas


# ===========================================================================
# BLOQUE DE VERIFICACIÓN LOCAL
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Verificación local: metrics.py")
    print("=" * 60)

    # Tensores dummy para probar SSIM y L1
    tensor_pred = torch.randn(3, 256, 256)
    tensor_real = torch.randn(3, 256, 256)

    print("\n--- calcular_ssim ---")
    ssim_val = calcular_ssim(tensor_pred, tensor_real)
    print(f"  SSIM (dos imágenes aleatorias): {ssim_val:.4f}")
    print(f"  SSIM (imagen consigo misma)   : {calcular_ssim(tensor_real, tensor_real):.4f}")
    assert 0.0 <= calcular_ssim(tensor_real, tensor_real) <= 1.0

    print("\n--- calcular_l1_normalizado ---")
    l1_val = calcular_l1_normalizado(tensor_pred, tensor_real)
    l1_identico = calcular_l1_normalizado(tensor_real, tensor_real)
    print(f"  L1 (dos imágenes aleatorias) : {l1_val:.4f}")
    print(f"  L1 (imagen consigo misma)    : {l1_identico:.4f}  → Esperado: 0.0")
    assert l1_identico < 1e-6, "L1 de imagen consigo misma debería ser ~0"
    assert 0.0 <= l1_val <= 1.0

    print("\n[OK] metrics.py verificado correctamente.")
