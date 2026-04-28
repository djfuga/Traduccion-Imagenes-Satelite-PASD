"""
losses.py — Funciones de pérdida para el sistema Pix2Pix cGAN
==============================================================

En una GAN condicional (cGAN) el entrenamiento requiere DOS tipos de pérdida:

1. **Adversarial Loss (GAN Loss)**: El discriminador aprende a distinguir imágenes
   reales de falsas. El generador aprende a engañar al discriminador.

2. **Pérdida L1**: Penaliza la diferencia píxel a píxel entre la imagen generada
   y la imagen objetivo real. Fuerza al generador a producir resultados cercanos
   a la imagen correcta, evitando el "mode collapse" (producir siempre la misma
   imagen genérica).

La pérdida total del generador combina ambas:
    L_total = L_GAN + λ * L_L1      (λ=100 según Isola et al., 2017)

Referencia: Isola et al., "Image-to-Image Translation with Conditional
Adversarial Networks", CVPR 2017.
Código base: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""

import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """
    Pérdida adversarial (GAN Loss) con soporte para dos variantes:

    - 'lsgan': Least Squares GAN (MSELoss). Más estable numéricamente.
      El discriminador predice valores reales en lugar de probabilidades.
      Ref: Mao et al., "Least Squares Generative Adversarial Networks", 2017.

    - 'vanilla': GAN original (BCEWithLogitsLoss). La variante clásica de Goodfellow.
      Más propensa a gradientes saturados durante el entrenamiento.

    Para este proyecto se recomienda 'lsgan' por su estabilidad en la T4 de Colab.

    Los tensores de etiquetas (real=1.0, fake=0.0) se crean dinámicamente del
    mismo tamaño y dispositivo que la predicción del discriminador, evitando
    errores de dispositivo (CPU vs CUDA).
    """

    def __init__(self, gan_mode: str = "lsgan"):
        """
        Args:
            gan_mode: Tipo de pérdida GAN. Opciones: 'lsgan' | 'vanilla'.
        """
        super(GANLoss, self).__init__()

        # Etiquetas para "real" y "fake" (se expanden dinámicamente)
        self.real_label = 1.0
        self.fake_label = 0.0

        self.gan_mode = gan_mode

        if gan_mode == "lsgan":
            # MSELoss: el discriminador regresa valores continuos
            # La pérdida penaliza cuánto se aleja de 1 (real) o 0 (falso)
            self.loss_fn = nn.MSELoss()
        elif gan_mode == "vanilla":
            # BCEWithLogitsLoss: aplica sigmoid internamente (más estable que BCE)
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Modo GAN no reconocido: '{gan_mode}'. Usa 'lsgan' o 'vanilla'.")

    def _crear_tensor_etiqueta(self, prediccion: torch.Tensor, es_real: bool) -> torch.Tensor:
        """
        Crea un tensor de etiquetas del mismo tamaño y dispositivo que la
        predicción del discriminador.

        Por qué esto importa: si el discriminador está en CUDA, el tensor de
        etiquetas también debe estarlo o PyTorch lanzará un error de dispositivo.
        """
        valor = self.real_label if es_real else self.fake_label
        # expand_as replica el escalar al tamaño de la predicción sin copiar memoria
        return torch.full_like(prediccion, fill_value=valor)

    def forward(self, prediccion: torch.Tensor, es_real: bool) -> torch.Tensor:
        """
        Calcula la pérdida GAN.

        Args:
            prediccion: Mapa de salida del discriminador, forma (N, 1, H, W).
                        Para PatchGAN 70x70 con input 256x256: (N, 1, 30, 30).
            es_real:    True si la predicción debería corresponder a imágenes reales,
                        False si corresponde a imágenes falsas (generadas).

        Returns:
            Escalar con el valor de la pérdida.
        """
        etiqueta = self._crear_tensor_etiqueta(prediccion, es_real)
        return self.loss_fn(prediccion, etiqueta)


def perdida_total_generador(
    loss_gan: torch.Tensor,
    loss_l1: torch.Tensor,
    lambda_l1: float = 100.0,
) -> torch.Tensor:
    """
    Combina la pérdida adversarial y la pérdida L1 para el generador.

    Fórmula (Isola et al., 2017, ecuación 4):
        L_cGAN(G, D) + λ * L_L1(G)

    Por qué λ=100:
        - La pérdida L1 está en el rango [0, 2] (imágenes normalizadas a [-1,1]).
        - La pérdida GAN (lsgan) está en el rango [0, 1].
        - Sin escalar, L1 dominaría y el generador ignoraría la señal adversarial.
        - Con λ=100, L1 y GAN contribuyen en magnitudes comparables, produciendo
          imágenes que son a la vez estructuralmente correctas (L1) y visualmente
          realistas (GAN).
        - El valor λ=100 fue determinado experimentalmente en el paper original.

    Args:
        loss_gan:  Pérdida adversarial del generador (escalar).
        loss_l1:   Pérdida L1 entre imagen generada y real (escalar).
        lambda_l1: Factor de ponderación para L1. Default: 100.

    Returns:
        Escalar con la pérdida total del generador.
    """
    return loss_gan + lambda_l1 * loss_l1


# ===========================================================================
# BLOQUE DE VERIFICACIÓN LOCAL
# Ejecutar: python src/models/losses.py
# Verifica que las pérdidas producen escalares positivos sin errores de forma.
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Verificación local: losses.py")
    print("=" * 60)

    # Simular la salida del discriminador PatchGAN: (batch=1, 1ch, 30x30)
    prediccion_real = torch.rand(1, 1, 30, 30)   # Discriminador ve imágenes reales
    prediccion_falsa = torch.rand(1, 1, 30, 30)  # Discriminador ve imágenes falsas

    for modo in ["lsgan", "vanilla"]:
        print(f"\n--- Modo: {modo} ---")
        criterio = GANLoss(gan_mode=modo)

        # Pérdida del discriminador en imágenes reales (debe acercarse a 1)
        loss_d_real = criterio(prediccion_real, es_real=True)
        # Pérdida del discriminador en imágenes falsas (debe acercarse a 0)
        loss_d_falsa = criterio(prediccion_falsa, es_real=False)
        # Pérdida del generador: convencer al discriminador de que es real
        loss_g_gan = criterio(prediccion_falsa, es_real=True)

        # Pérdida L1: diferencia píxel a píxel (imagen generada vs imagen real)
        imagen_generada = torch.rand(1, 3, 256, 256)
        imagen_real = torch.rand(1, 3, 256, 256)
        loss_l1 = nn.L1Loss()(imagen_generada, imagen_real)

        # Pérdida total del generador
        loss_g_total = perdida_total_generador(loss_g_gan, loss_l1, lambda_l1=100.0)

        print(f"  loss_D_real  : {loss_d_real.item():.4f}  (escalar: {loss_d_real.shape})")
        print(f"  loss_D_falsa : {loss_d_falsa.item():.4f}  (escalar: {loss_d_falsa.shape})")
        print(f"  loss_G_GAN   : {loss_g_gan.item():.4f}  (escalar: {loss_g_gan.shape})")
        print(f"  loss_L1      : {loss_l1.item():.4f}  (escalar: {loss_l1.shape})")
        print(f"  loss_G_total : {loss_g_total.item():.4f}  (escalar: {loss_g_total.shape})")

    print("\n[OK] losses.py verificado correctamente.")
