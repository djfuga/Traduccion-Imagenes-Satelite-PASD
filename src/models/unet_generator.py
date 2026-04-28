"""
unet_generator.py — Generador U-Net 256 (implementación explícita por capas)
=============================================================================

Este archivo implementa la misma arquitectura U-Net 256 del paper Pix2Pix pero
de forma EXPLÍCITA: cada una de las 8 capas del encoder y las 8 del decoder
tiene su propio nombre y está definida de manera independiente.

¿Por qué esta versión en lugar del diseño recursivo de generator.py?
----------------------------------------------------------------------
- generator.py usa bloques anidados recursivamente → compacto pero opaco.
- Este archivo nombra cada capa individualmente → verbose pero transparente.
- Para el blog educativo, esta versión permite mostrar el flujo de datos
  paso a paso, imprimiendo las dimensiones intermedias en cada etapa.
- Ambas versiones producen exactamente la misma arquitectura.

Mapa completo de la arquitectura U-Net 256:
--------------------------------------------

               ENCODER (downsampling, stride=2)
  ┌──────────────────────────────────────────────────────┐
  │  E1:  Conv(3→64)           256×256 → 128×128 │ sin InstanceNorm
  │  E2:  Conv(64→128)         128×128 →  64×64  │ + InstanceNorm
  │  E3:  Conv(128→256)         64×64  →  32×32  │ + InstanceNorm
  │  E4:  Conv(256→512)         32×32  →  16×16  │ + InstanceNorm
  │  E5:  Conv(512→512)         16×16  →   8×8   │ + InstanceNorm
  │  E6:  Conv(512→512)          8×8   →   4×4   │ + InstanceNorm
  │  E7:  Conv(512→512)          4×4   →   2×2   │ + InstanceNorm
  │  Bottleneck: Conv(512→512)   2×2   →   1×1   │ sin InstanceNorm
  └──────────────────────────────────────────────────────┘

               DECODER (upsampling, stride=2) + skip connections
  ┌──────────────────────────────────────────────────────┐
  │  D8: ConvT(512→512)          1×1   →   2×2   │ + InstanceNorm + Dropout ← bottleneck
  │  D7: ConvT(1024→512)         2×2   →   4×4   │ + InstanceNorm + Dropout ← cat(D8, E7)
  │  D6: ConvT(1024→512)         4×4   →   8×8   │ + InstanceNorm + Dropout ← cat(D7, E6)
  │  D5: ConvT(1024→512)         8×8   →  16×16  │ + InstanceNorm           ← cat(D6, E5)
  │  D4: ConvT(1024→256)        16×16  →  32×32  │ + InstanceNorm           ← cat(D5, E4)
  │  D3: ConvT(512→128)         32×32  →  64×64  │ + InstanceNorm           ← cat(D4, E3)
  │  D2: ConvT(256→64)          64×64  → 128×128 │ + InstanceNorm           ← cat(D3, E2)
  │  D1: ConvT(128→3)          128×128 → 256×256 │ + Tanh                   ← cat(D2, E1)
  └──────────────────────────────────────────────────────┘

Regla de canales para el decoder (¿por qué siempre se duplican?):
------------------------------------------------------------------
Cada bloque del decoder recibe la concatenación de:
    [salida del bloque decoder anterior] + [salida del encoder simétrico]
                 N canales                +          N canales
                                = 2N canales de entrada

Ejemplo D7: recibe cat(D8_out=512ch, E7_out=512ch) = 1024 canales de entrada.

Esta concatenación (y NO suma) es la diferencia clave entre U-Net y
las arquitecturas ResNet: preserva TODA la información del encoder,
mientras que la suma de una ResNet puede cancelar características.

Referencia: Ronneberger et al., "U-Net: Convolutional Networks for
Biomedical Image Segmentation", MICCAI 2015.
Adaptación Pix2Pix: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""

import torch
import torch.nn as nn


# ===========================================================================
# BLOQUES REUTILIZABLES
# ===========================================================================

def bloque_encoder(
    ch_entrada: int,
    ch_salida: int,
    usar_norm: bool = True,
) -> nn.Sequential:
    """
    Bloque encoder: reduce la resolución espacial a la mitad.

    Secuencia:
        LeakyReLU(0.2) → Conv2d(stride=2) → [InstanceNorm2d]

    ¿Por qué LeakyReLU ANTES de la convolución?
    En el diseño recursivo de Pix2Pix, la activación precede la convolución
    en todas las capas intermedias del encoder. Esto sigue el principio
    "pre-activation" que mejora el flujo de gradientes durante el backward.

    ¿Por qué NO hay normalización en E1 (primera capa)?
    La primera capa del encoder recibe la imagen directamente (valores en [-1,1]).
    Aplicar InstanceNorm aquí distorsionaría la distribución de entrada que
    el modelo necesita procesar. La normalización solo se aplica a
    las activaciones intermedias.

    ¿Por qué LeakyReLU(0.2) y no ReLU?
    LeakyReLU permite un gradiente pequeño (0.2) para valores negativos.
    En el encoder, muchas activaciones pueden ser negativas, y ReLU las
    "mataría" (gradiente = 0), dificultando el aprendizaje. Esta pendiente
    negativa del 20% mantiene vivo el flujo de gradientes.

    Args:
        ch_entrada:  Canales de entrada.
        ch_salida:   Canales de salida (número de filtros a aprender).
        usar_norm:   Si True, añade InstanceNorm2d tras la convolución.
                     Debe ser False para la primera capa del encoder (E1).

    Returns:
        nn.Sequential con la secuencia de operaciones del bloque encoder.
    """
    capas = []

    # La primera capa (E1) no tiene LeakyReLU previo ni normalización
    if usar_norm:
        capas.append(nn.LeakyReLU(0.2, inplace=True))

    # Conv2d con kernel=4, stride=2, padding=1:
    # Fórmula de dimensión de salida: floor((H - 4 + 2*1) / 2) + 1 = H/2
    # El kernel de tamaño 4 con stride=2 captura contexto de 4 píxeles
    # y lo comprime en 1, lo que es más informativo que kernel=2.
    # bias=False porque InstanceNorm tiene sus propios parámetros de escala/bias
    capas.append(
        nn.Conv2d(ch_entrada, ch_salida, kernel_size=4, stride=2, padding=1, bias=False)
    )

    if usar_norm:
        # InstanceNorm2d normaliza cada canal de cada muestra individualmente.
        # A diferencia de BatchNorm, NO depende del tamaño del batch,
        # por eso funciona correctamente con batch_size=1 (estándar en Pix2Pix).
        capas.append(nn.InstanceNorm2d(ch_salida, affine=False))

    return nn.Sequential(*capas)


def bloque_decoder(
    ch_entrada: int,
    ch_salida: int,
    usar_dropout: bool = False,
    activacion_final: bool = False,
) -> nn.Sequential:
    """
    Bloque decoder: duplica la resolución espacial.

    Secuencia:
        ReLU → ConvTranspose2d(stride=2) → InstanceNorm2d → [Dropout(0.5)]

    ¿Por qué ReLU en el decoder y LeakyReLU en el encoder?
    El paper Pix2Pix sigue la convención de usar ReLU en el decoder porque:
    - El decoder produce representaciones positivas que se acumulan hacia la imagen.
    - En el decoder la "muerte" de neuronas (ReLU=0) es menos problemática
      porque hay múltiples caminos de información gracias a las skip connections.
    - El contrato de signos entre encoder (LeakyReLU, puede ser negativo) y
      decoder (ReLU, positivo) refleja sus roles diferentes en la arquitectura.

    ¿Por qué ConvTranspose2d y no Upsample + Conv?
    ConvTranspose2d aprende sus propios pesos de upsampling, adaptados a los
    datos del problema. Upsample + Conv usa interpolación fija (bilineal/nearest)
    seguida de una convolución — a veces produce artefactos en damero
    ("checkerboard artifacts"), aunque en algunos papers modernos se prefiere.
    Pix2Pix usa ConvTranspose2d en su implementación original.

    ¿Por qué Dropout(0.5) solo en D8, D7, D6?
    Solo se aplica en los 3 bloques más cercanos al bottleneck.
    El ruido del Dropout en estas capas se propaga a través de las skip
    connections hacia el decoder, añadiendo variabilidad estocástica a las
    imágenes generadas. Esto previene que el generador produzca siempre
    la misma salida determinista (modo "promedio").

    Args:
        ch_entrada:        Canales de entrada (incluye los del skip concatenado).
        ch_salida:         Canales de salida.
        usar_dropout:      Si True, añade Dropout(0.5) al final.
        activacion_final:  Si True, usa Tanh en lugar de InstanceNorm.
                           Solo para D1 (capa de salida final).

    Returns:
        nn.Sequential con la secuencia de operaciones del bloque decoder.
    """
    capas = []

    if activacion_final:
        # Capa de salida (D1): ReLU → ConvTranspose → Tanh
        # Tanh comprime la salida al rango [-1, 1], consistente con la
        # normalización de las imágenes de entrada.
        capas += [
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ch_entrada, ch_salida, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        ]
    else:
        # Capas intermedias del decoder: ReLU → ConvTranspose → InstanceNorm → [Dropout]
        capas += [
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ch_entrada, ch_salida, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ch_salida, affine=False),
        ]
        if usar_dropout:
            capas.append(nn.Dropout(0.5))

    return nn.Sequential(*capas)


# ===========================================================================
# ARQUITECTURA PRINCIPAL
# ===========================================================================

class UNet256(nn.Module):
    """
    Generador U-Net 256 con 8 niveles de encoder y 8 de decoder.

    Características:
    - Entrada: imagen RGB 256×256 normalizada a [-1, 1].
    - Salida: imagen RGB 256×256 en [-1, 1] (activación Tanh).
    - Skip connections: concatenación (no suma) de features del encoder.
    - Dropout(0.5) en los 3 bloques del decoder más cercanos al bottleneck.
    - InstanceNorm2d en todas las capas (excepto E1 y la salida final).

    El número base de filtros (nf=64) se duplica en cada nivel del encoder
    hasta alcanzar el máximo de 512 (= 64 × 8), que se mantiene constante
    desde E4 hasta el bottleneck:
        E1:  64
        E2: 128
        E3: 256
        E4: 512  ← se alcanza el máximo
        E5: 512  (igual que E4)
        E6: 512
        E7: 512
        Bottleneck: 512
    """

    def __init__(
        self,
        ch_entrada: int = 3,
        ch_salida: int = 3,
        nf: int = 64,          # Número base de filtros (se duplica hasta 512)
    ):
        """
        Args:
            ch_entrada: Canales de la imagen de entrada. Default: 3 (RGB).
            ch_salida:  Canales de la imagen de salida. Default: 3 (RGB).
            nf:         Filtros base de la primera capa (E1).
                        El estándar de Pix2Pix es nf=64.
        """
        super(UNet256, self).__init__()

        # ------------------------------------------------------------------ #
        #  ENCODER                                                             #
        #  Convierte la imagen (3ch, 256×256) en un feature map (512ch, 1×1) #
        # ------------------------------------------------------------------ #

        # E1: 3 → 64, resolución 256 → 128
        # Primera capa: SIN LeakyReLU previo ni InstanceNorm
        # (la imagen de entrada ya está normalizada y no debe normalizarse de nuevo)
        self.enc1 = bloque_encoder(ch_entrada, nf, usar_norm=False)

        # E2: 64 → 128, resolución 128 → 64
        self.enc2 = bloque_encoder(nf, nf * 2)

        # E3: 128 → 256, resolución 64 → 32
        self.enc3 = bloque_encoder(nf * 2, nf * 4)

        # E4: 256 → 512, resolución 32 → 16
        # A partir de aquí los canales se mantienen en 512 (= nf * 8)
        self.enc4 = bloque_encoder(nf * 4, nf * 8)

        # E5: 512 → 512, resolución 16 → 8
        self.enc5 = bloque_encoder(nf * 8, nf * 8)

        # E6: 512 → 512, resolución 8 → 4
        self.enc6 = bloque_encoder(nf * 8, nf * 8)

        # E7: 512 → 512, resolución 4 → 2
        self.enc7 = bloque_encoder(nf * 8, nf * 8)

        # Bottleneck: 512 → 512, resolución 2 → 1
        # La capa más profunda: comprime toda la imagen en un vector 1×1.
        # SIN InstanceNorm: este cuello de botella es el representación más
        # comprimida de la imagen; normalizar aquí perdería información global.
        # SIN LeakyReLU previo: sigue la convención del código de Pix2Pix.
        self.bottleneck = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 8, nf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            # No hay InstanceNorm en el bottleneck
        )

        # ------------------------------------------------------------------ #
        #  DECODER                                                             #
        #  Reconstruye la imagen desde el feature map comprimido.             #
        #  Cada bloque recibe la salida del bloque anterior CONCATENADA con   #
        #  la salida del bloque encoder simétrico (skip connection).          #
        # ------------------------------------------------------------------ #

        # D8: 512 → 512, resolución 1 → 2
        # Recibe: bottleneck (512ch)
        # ch_entrada = 512 (solo del bottleneck, sin skip en este nivel)
        # Con Dropout: añade variabilidad estocástica en la generación
        self.dec8 = bloque_decoder(nf * 8, nf * 8, usar_dropout=True)

        # D7: cat(D8=512, E7=512) = 1024 → 512, resolución 2 → 4
        # Recibe: D8_out (512ch) concatenado con E7_out (512ch) = 1024ch
        self.dec7 = bloque_decoder(nf * 8 * 2, nf * 8, usar_dropout=True)

        # D6: cat(D7=512, E6=512) = 1024 → 512, resolución 4 → 8
        self.dec6 = bloque_decoder(nf * 8 * 2, nf * 8, usar_dropout=True)

        # D5: cat(D6=512, E5=512) = 1024 → 512, resolución 8 → 16
        # A partir de aquí ya no hay Dropout
        self.dec5 = bloque_decoder(nf * 8 * 2, nf * 8)

        # D4: cat(D5=512, E4=512) = 1024 → 256, resolución 16 → 32
        # El número de canales empieza a decrecer simétricamente al encoder
        self.dec4 = bloque_decoder(nf * 8 * 2, nf * 4)

        # D3: cat(D4=256, E3=256) = 512 → 128, resolución 32 → 64
        self.dec3 = bloque_decoder(nf * 4 * 2, nf * 2)

        # D2: cat(D3=128, E2=128) = 256 → 64, resolución 64 → 128
        self.dec2 = bloque_decoder(nf * 2 * 2, nf)

        # D1 (salida): cat(D2=64, E1=64) = 128 → ch_salida, resolución 128 → 256
        # Activación Tanh: la imagen de salida debe estar en [-1, 1]
        self.dec1 = bloque_decoder(nf * 2, ch_salida, activacion_final=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la U-Net 256.

        El flujo de datos sigue una forma de "U":
            - Bajando: el encoder comprime progresivamente la imagen.
            - Subiendo: el decoder la reconstruye, recibiendo ayuda de cada
              capa del encoder a través de las skip connections.

        Las skip connections se implementan con torch.cat() a lo largo de
        la dimensión de canales (dim=1), NO con suma (+). La concatenación
        preserva AMBAS representaciones (detalle del encoder + contexto del
        decoder) sin interferencia. La suma en cambio podría cancelar señales
        opuestas en signo.

        Args:
            x: Tensor de entrada, forma (N, 3, 256, 256), valores en [-1, 1].

        Returns:
            Tensor de salida, forma (N, 3, 256, 256), valores en [-1, 1].
        """
        # ---- Encoder: calcular y guardar todos los feature maps ----
        # Cada salida se guarda porque la necesitaremos en el decoder.

        e1 = self.enc1(x)          # (N,  64, 128, 128) — features de borde y color
        e2 = self.enc2(e1)         # (N, 128,  64,  64) — features de textura simple
        e3 = self.enc3(e2)         # (N, 256,  32,  32) — features de patrón
        e4 = self.enc4(e3)         # (N, 512,  16,  16) — features de estructura
        e5 = self.enc5(e4)         # (N, 512,   8,   8) — features de región
        e6 = self.enc6(e5)         # (N, 512,   4,   4) — features globales
        e7 = self.enc7(e6)         # (N, 512,   2,   2) — features muy globales
        btn = self.bottleneck(e7)  # (N, 512,   1,   1) — representación comprimida

        # ---- Decoder: reconstruir con skip connections ----
        # En cada paso:
        #   1. Subir resolución con el bloque decoder.
        #   2. Concatenar con el feature map del encoder simétrico.
        #   El resultado tiene el doble de canales, que el siguiente bloque consume.

        d8 = self.dec8(btn)                        # (N, 512,   2,   2)
        d7 = self.dec7(torch.cat([d8, e7], dim=1)) # (N, 512,   4,   4) ← skip e7
        d6 = self.dec6(torch.cat([d7, e6], dim=1)) # (N, 512,   8,   8) ← skip e6
        d5 = self.dec5(torch.cat([d6, e5], dim=1)) # (N, 512,  16,  16) ← skip e5
        d4 = self.dec4(torch.cat([d5, e4], dim=1)) # (N, 256,  32,  32) ← skip e4
        d3 = self.dec3(torch.cat([d4, e3], dim=1)) # (N, 128,  64,  64) ← skip e3
        d2 = self.dec2(torch.cat([d3, e2], dim=1)) # (N,  64, 128, 128) ← skip e2
        out = self.dec1(torch.cat([d2, e1], dim=1)) # (N,   3, 256, 256) ← skip e1

        return out


# ===========================================================================
# BLOQUE DE VERIFICACIÓN LOCAL
# Ejecutar: python src/models/unet_generator.py
#
# Verifica:
#   1. Las dimensiones de salida del modelo completo.
#   2. Las dimensiones de cada feature map interno (encoder + decoder).
#   3. El rango de valores de salida (debe ser [-1, 1] por Tanh).
#   4. El conteo total de parámetros.
# ===========================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  Verificación local: unet_generator.py")
    print("=" * 65)

    # Tensor dummy: simula un batch de 1 imagen satelital 256×256 RGB
    x_dummy = torch.randn(1, 3, 256, 256)
    print(f"\nEntrada: {x_dummy.shape}  (batch=1, 3ch RGB, 256×256)\n")

    modelo = UNet256(ch_entrada=3, ch_salida=3, nf=64)
    modelo.eval()  # Desactiva Dropout para inferencia determinista

    # ---- Forward pass con hooks para capturar dimensiones intermedias ----
    activaciones: dict = {}

    def registrar_hook(nombre):
        def hook(modulo, entrada, salida):
            activaciones[nombre] = salida.shape
        return hook

    # Registrar hooks en cada sub-módulo nombrado
    modelo.enc1.register_forward_hook(registrar_hook("E1"))
    modelo.enc2.register_forward_hook(registrar_hook("E2"))
    modelo.enc3.register_forward_hook(registrar_hook("E3"))
    modelo.enc4.register_forward_hook(registrar_hook("E4"))
    modelo.enc5.register_forward_hook(registrar_hook("E5"))
    modelo.enc6.register_forward_hook(registrar_hook("E6"))
    modelo.enc7.register_forward_hook(registrar_hook("E7"))
    modelo.bottleneck.register_forward_hook(registrar_hook("Bottleneck"))
    modelo.dec8.register_forward_hook(registrar_hook("D8"))
    modelo.dec7.register_forward_hook(registrar_hook("D7"))
    modelo.dec6.register_forward_hook(registrar_hook("D6"))
    modelo.dec5.register_forward_hook(registrar_hook("D5"))
    modelo.dec4.register_forward_hook(registrar_hook("D4"))
    modelo.dec3.register_forward_hook(registrar_hook("D3"))
    modelo.dec2.register_forward_hook(registrar_hook("D2"))
    modelo.dec1.register_forward_hook(registrar_hook("D1 (salida)"))

    with torch.no_grad():
        salida = modelo(x_dummy)

    # ---- Imprimir tabla de dimensiones por capa ----
    print("  Dimensiones por capa (N=1, formato: canales × alto × ancho):")
    print("  " + "-" * 55)
    print(f"  {'Capa':<20} {'Forma de salida':<30} {'Tipo'}")
    print("  " + "-" * 55)

    for nombre, forma in activaciones.items():
        es_decoder = nombre.startswith("D")
        tipo = "Decoder  ^" if es_decoder else "Encoder  v" if nombre != "Bottleneck" else "Bottleneck"
        print(f"  {nombre:<20} {str(tuple(forma)):<30} {tipo}")

    print("  " + "-" * 55)
    print(f"  {'Salida final':<20} {str(tuple(salida.shape)):<30} Imagen generada")
    print()

    # ---- Verificaciones de correctitud ----
    print("  Verificaciones:")
    assert salida.shape == torch.Size([1, 3, 256, 256]), \
        f"  [FALLO] Forma incorrecta: {salida.shape}"
    print(f"  [OK] Forma de salida: {salida.shape}  (esperado: [1, 3, 256, 256])")

    assert salida.min().item() >= -1.0 and salida.max().item() <= 1.0, \
        f"  [FALLO] Rango de salida fuera de [-1, 1]"
    print(f"  [OK] Rango de salida: [{salida.min().item():.4f}, {salida.max().item():.4f}]  (Tanh en [-1, 1])")

    assert activaciones["Bottleneck"] == torch.Size([1, 512, 1, 1]), \
        f"  [FALLO] Bottleneck incorrecto: {activaciones['Bottleneck']}"
    print(f"  [OK] Bottleneck: {activaciones['Bottleneck']}  (representación 1×1)")

    # ---- Conteo de parámetros ----
    total_params = sum(p.numel() for p in modelo.parameters())
    params_entren = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"\n  Parámetros totales     : {total_params:>12,}")
    print(f"  Parámetros entrenables : {params_entren:>12,}")
    print(f"  Tamaño aprox. (float32): {total_params * 4 / 1e6:>8.1f} MB")

    # ---- Verificación con batch_size=2 ----
    print("\n  Verificación con batch_size=2:")
    x_batch2 = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        salida_b2 = modelo(x_batch2)
    assert salida_b2.shape == torch.Size([2, 3, 256, 256])
    print(f"  [OK] Entrada {x_batch2.shape} -> Salida {salida_b2.shape}")

    print("\n" + "=" * 65)
    print("  [OK] unet_generator.py verificado correctamente.")
    print("=" * 65)
