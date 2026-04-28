"""
discriminator.py — Discriminador PatchGAN 70×70
================================================

El discriminador PatchGAN clasifica si PARCHES locales de una imagen son
reales o falsos, en lugar de evaluar la imagen completa como un todo.

¿Por qué parches y no la imagen completa?
------------------------------------------
Un discriminador global (imagen completa → real/falso) tiende a enfocarse
en estructuras de baja frecuencia (colores, composición global) e ignorar
detalles de textura. El PatchGAN, al clasificar parches de ~70×70 píxeles,
penaliza específicamente las frecuencias espaciales ALTAS (texturas, bordes,
detalles finos), produciendo imágenes generadas más nítidas y realistas.

Además, al ser más pequeño que un discriminador global, requiere menos VRAM
y parámetros, lo que es crítico para la GPU T4 de Google Colab.

Arquitectura:
-------------
4 capas convolucionales con stride=2 (downsampling progresivo):
    Input: (N, 6, 256, 256)  ← imagen_condicion + imagen_target concatenadas
    Conv1: (N, 64, 128, 128)
    Conv2: (N, 128, 64, 64)
    Conv3: (N, 256, 32, 32)
    Conv4: (N, 512, 31, 31)   ← stride=1 en la penúltima capa
    Salida: (N, 1, 30, 30)    ← mapa de parches, sin activación final

Cada neurona de salida "ve" un campo receptivo de ~70×70 píxeles de la imagen
de entrada (de ahí el nombre PatchGAN 70×70).

La entrada tiene 6 canales porque el discriminador recibe SIEMPRE la imagen
condición (dominio A) concatenada con la imagen objetivo (dominio B o generada).
Esto es fundamental en una cGAN: el discriminador aprende a evaluar si el par
(condición, imagen) es coherente, no solo si la imagen parece realista de forma
aislada.

Referencia: Isola et al., "Image-to-Image Translation with Conditional
Adversarial Networks", CVPR 2017.
Código base: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
(ver models/networks.py, clase NLayerDiscriminator)
"""

import torch
import torch.nn as nn


class BloqueConvDiscriminador(nn.Module):
    """
    Bloque convolucional básico del discriminador:
        Conv2d → (BatchNorm) → LeakyReLU

    Se usa LeakyReLU (pendiente=0.2) en lugar de ReLU porque:
    - ReLU mata gradientes en neuronas con activación negativa (gradiente=0).
    - LeakyReLU permite un pequeño gradiente negativo (0.2), lo que estabiliza
      el entrenamiento del discriminador al evitar el "dying neuron" problem.
    - Esto es especialmente importante en GANs donde el equilibrio de gradientes
      entre G y D es delicado.
    """

    def __init__(
        self,
        canales_entrada: int,
        canales_salida: int,
        stride: int = 2,
        usar_norm: bool = True,
    ):
        """
        Args:
            canales_entrada: Número de canales de entrada.
            canales_salida:  Número de canales de salida (filtros).
            stride:          Paso de la convolución. stride=2 reduce el espacial a la mitad.
            usar_norm:       Si True, aplica InstanceNorm2d después de la convolución.
                             La primera capa NO usa normalización (convención de Pix2Pix).
        """
        super(BloqueConvDiscriminador, self).__init__()

        capas = [
            # kernel_size=4, padding=1: mantiene la resolución espacial consistente
            # bias=False cuando se usa normalización (la norma absorbe el sesgo)
            nn.Conv2d(
                canales_entrada,
                canales_salida,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=not usar_norm,
            )
        ]

        if usar_norm:
            # InstanceNorm normaliza CADA muestra individualmente, sin depender
            # del tamaño del batch. Esto la hace compatible con batch_size=1,
            # que es el tamaño estándar de Pix2Pix en Colab con poca VRAM.
            capas.append(nn.InstanceNorm2d(canales_salida))

        # LeakyReLU con pendiente negativa 0.2 (estándar en discriminadores GAN)
        capas.append(nn.LeakyReLU(0.2, inplace=True))

        self.bloque = nn.Sequential(*capas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bloque(x)


class DiscriminadorPatchGAN(nn.Module):
    """
    Discriminador PatchGAN de 70×70 píxeles.

    Recibe como entrada la concatenación de:
        - Imagen condición (dominio A): 3 canales
        - Imagen objetivo (dominio B, real o generada): 3 canales
    Total: 6 canales de entrada.

    Produce un mapa de salida de forma (N, 1, 30, 30) donde:
        - Cada valor corresponde a la "probabilidad de ser real" de un parche.
        - No se aplica sigmoid al final (la GANLoss lo gestiona internamente).
    """

    def __init__(self, canales_entrada: int = 6, filtros_base: int = 64):
        """
        Args:
            canales_entrada: Canales de entrada. Default 6 (3 condición + 3 objetivo).
            filtros_base:    Número de filtros en la primera capa. Se duplica en
                             cada capa siguiente hasta llegar a 8× (512 filtros).
        """
        super(DiscriminadorPatchGAN, self).__init__()

        # Capa 1: sin normalización (convención del paper original)
        # Input: (N, 6, 256, 256) → Output: (N, 64, 128, 128)
        self.capa1 = BloqueConvDiscriminador(
            canales_entrada, filtros_base, stride=2, usar_norm=False
        )

        # Capa 2: con normalización
        # Input: (N, 64, 128, 128) → Output: (N, 128, 64, 64)
        self.capa2 = BloqueConvDiscriminador(
            filtros_base, filtros_base * 2, stride=2, usar_norm=True
        )

        # Capa 3: con normalización
        # Input: (N, 128, 64, 64) → Output: (N, 256, 32, 32)
        self.capa3 = BloqueConvDiscriminador(
            filtros_base * 2, filtros_base * 4, stride=2, usar_norm=True
        )

        # Capa 4: stride=1 (no reduce más la resolución espacial)
        # Input: (N, 256, 32, 32) → Output: (N, 512, 31, 31)
        # El stride=1 con kernel=4 y padding=1 reduce de 32 a 31: (32-4+2)/1+1=31
        self.capa4 = BloqueConvDiscriminador(
            filtros_base * 4, filtros_base * 8, stride=1, usar_norm=True
        )

        # Capa de salida: proyecta a 1 canal (mapa de parches real/falso)
        # Input: (N, 512, 31, 31) → Output: (N, 1, 30, 30)
        # kernel=4, stride=1, padding=1: (31-4+2)/1+1=30
        self.capa_salida = nn.Conv2d(
            filtros_base * 8, 1, kernel_size=4, stride=1, padding=1
        )
        # NO sigmoid aquí: GANLoss maneja eso internamente según el modo (lsgan/vanilla)

    def forward(self, condicion: torch.Tensor, objetivo: torch.Tensor) -> torch.Tensor:
        """
        Pasa el par (condición, objetivo) por el discriminador.

        Args:
            condicion: Imagen de entrada/condición (dominio A). Forma: (N, 3, 256, 256).
            objetivo:  Imagen objetivo (real o generada, dominio B). Forma: (N, 3, 256, 256).

        Returns:
            Mapa de parches. Forma: (N, 1, 30, 30).
            Valores altos → el discriminador cree que el par es real.
            Valores bajos → el discriminador cree que el par es falso.
        """
        # Concatenar a lo largo de la dimensión de canales (dim=1)
        # El discriminador ve SIEMPRE el par completo para evaluar coherencia
        x = torch.cat([condicion, objetivo], dim=1)  # → (N, 6, 256, 256)

        x = self.capa1(x)    # → (N, 64, 128, 128)
        x = self.capa2(x)    # → (N, 128, 64, 64)
        x = self.capa3(x)    # → (N, 256, 32, 32)
        x = self.capa4(x)    # → (N, 512, 31, 31)
        x = self.capa_salida(x)  # → (N, 1, 30, 30)

        return x


# ===========================================================================
# BLOQUE DE VERIFICACIÓN LOCAL
# Ejecutar: python src/models/discriminator.py
# Verifica que la salida tiene forma (1, 1, 30, 30) con entradas 256×256.
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Verificación local: discriminator.py")
    print("=" * 60)

    # Tensores dummy que simulan un batch de 1 imagen 256×256 RGB
    condicion_dummy = torch.randn(1, 3, 256, 256)
    objetivo_dummy  = torch.randn(1, 3, 256, 256)

    discriminador = DiscriminadorPatchGAN(canales_entrada=6, filtros_base=64)
    discriminador.eval()  # Modo evaluación para desactivar dropout/batchnorm estocástico

    with torch.no_grad():
        salida = discriminador(condicion_dummy, objetivo_dummy)

    print(f"\nEntrada condición : {condicion_dummy.shape}")
    print(f"Entrada objetivo  : {objetivo_dummy.shape}")
    print(f"Salida (mapa PatchGAN): {salida.shape}")
    print(f"  → Esperado: torch.Size([1, 1, 30, 30])")

    # Contar parámetros
    total_params = sum(p.numel() for p in discriminador.parameters())
    print(f"\nParámetros totales del discriminador: {total_params:,}")
    print(f"  (~{total_params / 1e6:.2f}M parámetros)")

    assert salida.shape == torch.Size([1, 1, 30, 30]), \
        f"Error de dimensión: se esperaba (1,1,30,30), se obtuvo {salida.shape}"
    print("\n[OK] discriminator.py verificado correctamente.")
