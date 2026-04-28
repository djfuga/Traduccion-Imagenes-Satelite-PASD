"""
generator.py — Generador U-Net 256
====================================

La U-Net es una arquitectura encoder-decoder con conexiones de salto (skip
connections) que conectan directamente cada capa del encoder con su capa
simétrica en el decoder.

¿Por qué U-Net y no un encoder-decoder simple?
-----------------------------------------------
Un encoder-decoder sin skip connections pasa la información a través del
cuello de botella (bottleneck), perdiendo detalles espaciales de alta frecuencia
(bordes, texturas). Las skip connections permiten que el decoder recupere esta
información directamente, lo que es crítico para tareas de traducción imagen a
imagen donde la estructura espacial debe preservarse (ej: los bordes de una
carretera en el boceto deben corresponderse con la misma carretera en el satélite).

Arquitectura U-Net 256 (8 niveles):
------------------------------------
Dado que la imagen de entrada es 256×256, y cada nivel divide la resolución
a la mitad, necesitamos log2(256) = 8 niveles para llegar al bottleneck de 1×1.

Estructura:
    Encoder (downsampling):
        256×256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1×1 (bottleneck)
    Decoder (upsampling):
        1×1 → 2 → 4 → 8 → 16 → 32 → 64 → 128 → 256×256

En el decoder, cada capa recibe:
    [salida_decoder_anterior] + [salida_encoder_simétrico] (concatenados)
    → canales_decoder * 2

Diseño recursivo:
-----------------
Se implementa mediante un bloque recursivo `BloqueUNetSalto` donde cada bloque
contiene a su submódulo (el siguiente nivel más profundo). El bloque más interno
es el bottleneck; los bloques se "envuelven" uno dentro de otro.

Dropout:
--------
Se aplica Dropout(p=0.5) en los 3 bloques del decoder más cercanos al bottleneck.
El ruido del dropout actúa como regularización Y como fuente de variabilidad
estocástica en la generación (similar al ruido de entrada en GANs no condicionales).

Referencia: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
Image Segmentation", MICCAI 2015.
Adaptación Pix2Pix: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""

import torch
import torch.nn as nn


class BloqueUNetSalto(nn.Module):
    """
    Bloque recursivo de la U-Net. Cada bloque representa UN nivel de la jerarquía.

    Estructura de un bloque:
        Encoder (downsampling):
            x → Conv → Norm → LeakyReLU → [submodulo] → ...

        Decoder (upsampling):
            ... → [submodulo] → ConvTranspose → Norm → ReLU → Dropout(opcional)

        Skip connection:
            La salida final es: torch.cat([x_encoder, x_decoder], dim=1)
            donde x_encoder es la entrada original al bloque y x_decoder
            es lo que el decoder produjo después de procesar el submodulo.

    El diseño recursivo permite construir toda la U-Net anidando bloques:
        BloqueUNetSalto(más_externo, ..., submodulo=
            BloqueUNetSalto(..., submodulo=
                BloqueUNetSalto(bottleneck)
            )
        )
    """

    def __init__(
        self,
        canales_externos: int,
        canales_internos: int,
        canales_entrada: int = None,
        submodulo: nn.Module = None,
        es_capa_mas_externa: bool = False,
        es_bottleneck: bool = False,
        usar_dropout: bool = False,
    ):
        """
        Args:
            canales_externos: Número de canales en el lado externo del bloque
                              (entrada al encoder y salida del decoder de este nivel).
            canales_internos: Número de canales en el lado interno (salida del encoder
                              y entrada al decoder de este nivel).
            canales_entrada:  Canales reales de la imagen de entrada. Solo difiere de
                              canales_externos en la capa más externa (ej: 3 canales RGB).
            submodulo:        El siguiente bloque más interno (recursión).
            es_capa_mas_externa: Si True, este es el bloque más externo de la U-Net.
                              La capa de salida usa Tanh (rango [-1, 1]).
            es_bottleneck:    Si True, este es el bloque más interno (cuello de botella).
                              No tiene skip connection; solo encoder → decoder.
            usar_dropout:     Si True, aplica Dropout(0.5) en el decoder.
                              Se activa en los 3 niveles más cercanos al bottleneck.
        """
        super(BloqueUNetSalto, self).__init__()

        self.es_capa_mas_externa = es_capa_mas_externa
        self.es_bottleneck = es_bottleneck  # necesario para el forward del bloque interno

        # Si no se especifica, los canales de entrada = canales externos
        if canales_entrada is None:
            canales_entrada = canales_externos

        # ---- ENCODER (downsampling) ----
        # Conv2d con stride=2: reduce la resolución espacial a la mitad
        # kernel=4, padding=1: (H-4+2)/2+1 = H/2 exactamente
        # Sin normalización en la capa más externa (convención Pix2Pix)
        # LeakyReLU(0.2) para estabilidad del gradiente en el encoder
        conv_encoder = nn.Conv2d(
            canales_entrada, canales_internos,
            kernel_size=4, stride=2, padding=1, bias=False
        )

        if es_bottleneck:
            # El bottleneck no usa normalización ni activación previa
            encoder = [nn.LeakyReLU(0.2, inplace=True), conv_encoder]
        elif es_capa_mas_externa:
            # Capa más externa: no hay normalización
            encoder = [conv_encoder]
        else:
            # Capas intermedias: LeakyReLU + Conv + InstanceNorm
            encoder = [
                nn.LeakyReLU(0.2, inplace=True),
                conv_encoder,
                nn.InstanceNorm2d(canales_internos),
            ]

        # ---- DECODER (upsampling) ----
        # ConvTranspose2d con stride=2: duplica la resolución espacial
        # La entrada tiene el doble de canales por el skip connection
        # (salvo en el bottleneck que no tiene skip)
        conv_decoder = nn.ConvTranspose2d(
            canales_internos * (1 if es_bottleneck else 2),
            canales_externos,
            kernel_size=4, stride=2, padding=1, bias=False
        )

        if es_capa_mas_externa:
            # La capa más externa produce la imagen final con Tanh
            # Tanh mapea a [-1, 1], compatible con la normalización de entrada
            decoder = [nn.ReLU(inplace=True), conv_decoder, nn.Tanh()]
        elif es_bottleneck:
            # El bottleneck: ReLU + ConvTranspose + Norm
            decoder = [
                nn.ReLU(inplace=True),
                conv_decoder,
                nn.InstanceNorm2d(canales_externos),
            ]
        else:
            # Capas intermedias: ReLU + ConvTranspose + Norm + (Dropout opcional)
            decoder = [
                nn.ReLU(inplace=True),
                conv_decoder,
                nn.InstanceNorm2d(canales_externos),
            ]
            if usar_dropout:
                # Dropout(0.5) añade variabilidad y regularización
                # Solo en los 3 bloques más cercanos al bottleneck
                decoder.append(nn.Dropout(0.5))

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        self.submodulo = submodulo

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.es_capa_mas_externa:
            # La capa más externa NO tiene skip connection.
            # El submodulo ya es un bloque completo U-Net; su salida alimenta el decoder.
            return self.decoder(self.submodulo(self.encoder(x)))
        elif self.es_bottleneck:
            # El bottleneck es el bloque más interno: no tiene submodulo.
            # Su decoder toma canales_internos * 1 (sin cat previo al decoder).
            # PERO sí hace skip cat a su salida, igual que los bloques intermedios:
            #   cat([x, decoder(encoder(x))]) = canales_externos * 2 channels
            # Esto es necesario para que el bloque padre (que wrappea al bottleneck)
            # reciba la cantidad de canales que espera su decoder (canales_internos * 2).
            return torch.cat([x, self.decoder(self.encoder(x))], dim=1)
        else:
            # Capas intermedias: tienen submodulo y skip connection.
            # 1. Codificar la entrada (downsampling × 2)
            x_codificado = self.encoder(x)
            # 2. Propagar por el submodulo (recursión hacia el bottleneck)
            x_submodulo = self.submodulo(x_codificado)
            # 3. Decodificar (upsampling × 2); la entrada ya viene con canales × 2
            x_decodificado = self.decoder(x_submodulo)
            # 4. Skip connection: concatenar entrada original con salida del decoder.
            #    Esto restaura los detalles espaciales de alta frecuencia que el
            #    encoder comprimió al bajar la resolución.
            return torch.cat([x, x_decodificado], dim=1)


class GeneradorUNet(nn.Module):
    """
    Generador U-Net completo para imágenes 256×256.

    Construye la U-Net de 8 niveles de forma recursiva, desde el bottleneck
    hacia afuera:

        Nivel 8 (bottleneck, 1×1):   BloqueUNetSalto(512→512, bottleneck)
        Nivel 7 (2×2):               BloqueUNetSalto(512→512, dropout, wrapping nivel 8)
        Nivel 6 (4×4):               BloqueUNetSalto(512→512, dropout, wrapping nivel 7)
        Nivel 5 (8×8):               BloqueUNetSalto(512→512, dropout, wrapping nivel 6)
        Nivel 4 (16×16):             BloqueUNetSalto(256→512, wrapping nivel 5)
        Nivel 3 (32×32):             BloqueUNetSalto(128→256, wrapping nivel 4)
        Nivel 2 (64×64):             BloqueUNetSalto(64→128, wrapping nivel 3)
        Nivel 1 (128×128, exterior): BloqueUNetSalto(3→64, capa_más_externa, wrapping nivel 2)

    Nota: Los niveles 5, 6, 7 usan Dropout(0.5) como indica el paper Pix2Pix.
    """

    def __init__(
        self,
        canales_entrada: int = 3,
        canales_salida: int = 3,
        filtros_base: int = 64,
        num_niveles: int = 8,
    ):
        """
        Args:
            canales_entrada: Canales de la imagen de entrada (3 para RGB).
            canales_salida:  Canales de la imagen generada (3 para RGB).
            filtros_base:    Filtros en la primera capa. Se duplica hasta llegar
                             a filtros_base * 8 = 512 en el bottleneck.
            num_niveles:     Número de niveles de la U-Net. Para 256×256: 8.
        """
        super(GeneradorUNet, self).__init__()

        # Número máximo de filtros (cap en 512 para evitar explosión de parámetros)
        filtros_max = filtros_base * 8  # 512

        # ---- Construir la U-Net de adentro hacia afuera ----

        # Nivel más interno: el bottleneck (sin submodulo)
        # Produce features de 1×1 para imagen de 256×256
        unet = BloqueUNetSalto(
            filtros_max, filtros_max,
            submodulo=None,
            es_bottleneck=True,
        )

        # Niveles intermedios con filtros_max (con Dropout en los 3 primeros tras el bottleneck)
        # num_niveles - 5 = 3 niveles de bottleneck + dropout
        for i in range(num_niveles - 5):
            unet = BloqueUNetSalto(
                filtros_max, filtros_max,
                submodulo=unet,
                usar_dropout=True,   # Dropout en niveles 5, 6, 7
            )

        # Niveles intermedios con filtros decrecientes (sin dropout)
        # Los filtros se reducen: 512 → 256 → 128 → 64
        multiplicadores_filtros = [4, 2, 1]
        for mult in multiplicadores_filtros:
            unet = BloqueUNetSalto(
                filtros_base * mult,
                filtros_base * min(mult * 2, 8),  # min para no superar filtros_max
                submodulo=unet,
            )

        # Nivel más externo: toma la imagen de entrada (3 canales) y produce la
        # imagen generada (3 canales). Usa Tanh como activación final.
        self.modelo = BloqueUNetSalto(
            canales_salida,
            filtros_base,
            canales_entrada=canales_entrada,
            submodulo=unet,
            es_capa_mas_externa=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del generador.

        Args:
            x: Imagen de entrada, forma (N, 3, 256, 256), normalizada a [-1, 1].

        Returns:
            Imagen generada, forma (N, 3, 256, 256), valores en [-1, 1] (Tanh).
        """
        return self.modelo(x)


# ===========================================================================
# BLOQUE DE VERIFICACIÓN LOCAL
# Ejecutar: python src/models/generator.py
# Verifica que la salida tiene forma (1, 3, 256, 256) y que no hay errores
# matemáticos de dimensión en ningún nivel de la U-Net.
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Verificación local: generator.py")
    print("=" * 60)

    # Tensor dummy: batch=1, 3 canales RGB, 256×256 píxeles
    x_dummy = torch.randn(1, 3, 256, 256)

    generador = GeneradorUNet(canales_entrada=3, canales_salida=3, filtros_base=64)
    generador.eval()

    with torch.no_grad():
        salida = generador(x_dummy)

    print(f"\nEntrada  : {x_dummy.shape}")
    print(f"Salida   : {salida.shape}")
    print(f"  → Esperado: torch.Size([1, 3, 256, 256])")
    print(f"\nRango de valores de salida (Tanh):")
    print(f"  min={salida.min().item():.4f}, max={salida.max().item():.4f}")
    print(f"  → Esperado: valores entre -1.0 y 1.0")

    # Contar parámetros
    total_params = sum(p.numel() for p in generador.parameters())
    print(f"\nParámetros totales del generador: {total_params:,}")
    print(f"  (~{total_params / 1e6:.1f}M parámetros)")

    assert salida.shape == torch.Size([1, 3, 256, 256]), \
        f"Error de dimensión: se esperaba (1,3,256,256), se obtuvo {salida.shape}"
    assert salida.min().item() >= -1.0 and salida.max().item() <= 1.0, \
        "Error: la salida Tanh debería estar en el rango [-1, 1]"

    print("\n[OK] generator.py verificado correctamente.")
