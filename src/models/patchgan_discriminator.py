"""
patchgan_discriminator.py — Discriminador PatchGAN 70x70
=========================================================

CONCEPTO CENTRAL: ¿Qué es un PatchGAN?
----------------------------------------
Un discriminador GAN clásico toma una imagen completa y devuelve UN ÚNICO
escalar: "real" (1) o "falso" (0). El problema es que para hacerlo necesita
ver toda la imagen a la vez, lo que:
  - Requiere muchos parámetros (costoso en VRAM).
  - Tiende a ignorar los detalles locales de textura (bordes, patrones finos).
  - No localiza QUÉ parte de la imagen es falsa.

El PatchGAN (Li & Wand, 2016; popularizado en Pix2Pix, Isola et al., 2017)
resuelve esto clasificando PARCHES independientes de la imagen:
  - La red aplica convoluciones con stride hasta producir un MAPA de salida,
    no un escalar.
  - Cada neurona del mapa de salida "ve" solo un parche local de la imagen
    original (campo receptivo).
  - La pérdida promedia las clasificaciones de todos los parches.

Ventajas del PatchGAN sobre el discriminador global:
  1. MENOS PARÁMETROS: la red es completamente convolucional (FCN), sin
     capas densas. Para 256x256 con nf=64: solo ~2.8M parámetros.
  2. MEJOR TEXTURA: penaliza específicamente las frecuencias espaciales ALTAS
     (texturas, bordes), produciendo imágenes más nítidas.
  3. INDEPENDENCIA DE POSICIÓN: el mismo conjunto de filtros se aplica a
     cada parche, lo que funciona como regularización implícita.
  4. EFICIENCIA EN VRAM: crítico para la GPU T4 de Google Colab.

DERIVACIÓN DEL CAMPO RECEPTIVO (por qué "70x70"):
---------------------------------------------------
El campo receptivo (RF) es la región de la imagen de entrada que "ve"
cada neurona de la capa de salida. Se calcula recursivamente:

    RF_n  = RF_{n-1} + (kernel - 1) * jump_{n-1}
    jump_n = jump_{n-1} * stride_n

Donde jump es cuántos píxeles de entrada corresponden a 1 píxel en esa capa.

Aplicando a las 5 capas de la arquitectura estándar (kernel=4 en todas):

  Capa     stride  RF antes  RF despues  jump antes  jump despues
  ------   ------  --------  ----------  ----------  ------------
  Entrada    —        1          1            1            1
  Conv1      2        1          4            1            2
  Conv2      2        4         10            2            4
  Conv3      2       10         22            4            8
  Conv4      1       22         46            8            8
  Conv5      1       46         70            8            8
  (salida)

Resultado: RF = 70 pixeles. De ahí el nombre "PatchGAN 70x70".

Nota: el RF de 70px es suficiente para capturar texturas y patrones locales
(el ancho de una calle, la trama de un tejado, la textura de vegetación)
sin necesitar ver toda la escena.

ENTRADA DE 6 CANALES:
----------------------
El discriminador recibe la concatenacion del par (imagen_condicion, imagen_objetivo)
a lo largo de la dimension de canales: [A; B] -> 3+3 = 6 canales.

Esto es fundamental en una cGAN (GAN CONDICIONAL): el discriminador NO evalua
si una imagen parece realista de forma aislada, sino si el PAR (condicion,
imagen) es coherente. Sin la condicion, el discriminador podria aprender que
cualquier imagen satelital realista es "real", sin importar si corresponde
al boceto cartografico que se le dio como entrada.

Referencias:
  - Isola et al., "Image-to-Image Translation with Conditional Adversarial
    Networks", CVPR 2017. https://arxiv.org/abs/1611.07004
  - Li & Wand, "Precomputed Real-Time Texture Synthesis with Markovian
    Generative Adversarial Networks", ECCV 2016.
  - Codigo base: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    (ver models/networks.py, clase NLayerDiscriminator)
"""

import torch
import torch.nn as nn
from typing import List, Tuple


# ===========================================================================
# CALCULO ANALITICO DEL CAMPO RECEPTIVO
# ===========================================================================

def calcular_campo_receptivo(
    capas: List[Tuple[int, int]],
) -> List[dict]:
    """
    Calcula el campo receptivo acumulado capa por capa de forma analitica.

    Usa las formulas estandar de receptive field para redes convolucionales:
        RF_n  = RF_{n-1} + (kernel_n - 1) * jump_{n-1}
        jump_n = jump_{n-1} * stride_n

    donde:
        RF   = campo receptivo acumulado hasta esa capa (en pixeles de entrada)
        jump = cuantos pixeles de entrada corresponden a 1 pixel en esa capa
               (equivalente al stride acumulado total)

    Referencia: A guide to receptive field arithmetic for Convolutional Neural
    Networks. Dang Ha The Hien, 2017.

    Args:
        capas: Lista de tuplas (kernel_size, stride) de cada capa convolucional,
               en orden de la entrada hacia la salida.

    Returns:
        Lista de diccionarios con el estado tras cada capa:
        {'capa', 'kernel', 'stride', 'jump', 'rf'}
    """
    rf   = 1   # campo receptivo inicial (1 pixel = la propia entrada)
    jump = 1   # salto inicial (1 pixel de entrada = 1 pixel en la capa 0)
    historial = [{"capa": "entrada", "kernel": 1, "stride": 1, "jump": jump, "rf": rf}]

    for idx, (kernel, stride) in enumerate(capas, start=1):
        # El campo receptivo crece en (kernel-1)*jump por cada nueva capa:
        # cada filtro "ve" kernel pixeles en la representacion actual, pero
        # cada uno de esos pixeles corresponde a jump pixeles de la entrada.
        rf   = rf + (kernel - 1) * jump
        jump = jump * stride
        historial.append({
            "capa": f"Conv{idx}",
            "kernel": kernel,
            "stride": stride,
            "jump": jump,
            "rf": rf,
        })

    return historial


# ===========================================================================
# BLOQUE CONVOLUCIONAL DEL DISCRIMINADOR
# ===========================================================================

def _bloque_conv(
    ch_entrada: int,
    ch_salida: int,
    stride: int = 2,
    usar_norm: bool = True,
    usar_spectral_norm: bool = False,
) -> nn.Sequential:
    """
    Bloque basico del PatchGAN:
        Conv2d -> [Normalizacion] -> LeakyReLU(0.2)

    Decisiones de diseno:
    ---------------------
    kernel=4, padding=1:
        Con stride=2: H_out = floor((H - 4 + 2) / 2) + 1 = H/2   (exacto)
        Con stride=1: H_out = floor((H - 4 + 2) / 1) + 1 = H - 1
        El kernel de tamano 4 es el minimo que garantiza downsampling exacto
        a la mitad con stride=2 y padding=1. Kernels de tamano impar (3, 5)
        con stride=2 producen dimensiones de salida que no son exactamente H/2
        para todos los valores de H.

    bias=False cuando se usa normalizacion:
        La normalizacion (InstanceNorm/SpectralNorm) ya incluye un termino de
        sesgo (bias) como parte de sus parametros. Incluir ademas el bias de
        Conv2d seria redundante y aumentaria innecesariamente el numero de
        parametros.

    LeakyReLU(0.2) en lugar de ReLU:
        ReLU(x) = max(0, x): gradiente = 0 para x < 0 ("neuronas muertas").
        LeakyReLU(x) = x si x>=0, 0.2*x si x<0: gradiente != 0 siempre.
        En el discriminador, activaciones negativas son frecuentes (la red
        aun no ha aprendido a distinguir). Con ReLU, esas neuronas nunca
        se actualizarian. La pendiente 0.2 mantiene el flujo de gradientes
        sin saturar.

    InstanceNorm vs SpectralNorm:
        InstanceNorm (estandar Pix2Pix): normaliza cada canal de cada muestra
        individualmente. Estabiliza el entrenamiento pero puede suavizar
        demasiado los features del discriminador.

        SpectralNorm (Miyato et al., 2018): normaliza los pesos de la capa
        por su valor singular maximo, garantizando que la constante de
        Lipschitz de cada capa sea <= 1. Esto hace al discriminador
        1-Lipschitz sin restringir su capacidad expresiva como InstanceNorm.
        Se recomienda para WGAN o cuando el entrenamiento es inestable.

    Args:
        ch_entrada:          Canales de entrada.
        ch_salida:           Canales de salida.
        stride:              Stride de la convolucion (2 o 1).
        usar_norm:           Si True, aplica InstanceNorm2d.
        usar_spectral_norm:  Si True, envuelve Conv2d con SpectralNorm
                             (alternativa a InstanceNorm, no se usan juntas).

    Returns:
        nn.Sequential con Conv -> [Norm] -> LeakyReLU.
    """
    # bias=False si usamos InstanceNorm (la norma tiene su propio bias)
    # bias=True si usamos SpectralNorm (que no tiene parametro de bias propio)
    usar_bias = usar_spectral_norm or not usar_norm

    conv = nn.Conv2d(
        ch_entrada, ch_salida,
        kernel_size=4, stride=stride, padding=1,
        bias=usar_bias,
    )

    # SpectralNorm envuelve la conv ANTES de construir el Sequential
    if usar_spectral_norm:
        conv = nn.utils.spectral_norm(conv)

    capas: List[nn.Module] = [conv]

    # Solo una de las dos normalizaciones (no se pueden usar juntas)
    if usar_norm and not usar_spectral_norm:
        # affine=True: la norma aprende parametros de escala y traslacion
        # track_running_stats=False: InstanceNorm no mantiene estadisticas
        # entre batches (a diferencia de BatchNorm)
        capas.append(nn.InstanceNorm2d(ch_salida, affine=True))

    # LeakyReLU siempre al final del bloque
    capas.append(nn.LeakyReLU(0.2, inplace=True))

    return nn.Sequential(*capas)


# ===========================================================================
# ARQUITECTURA PRINCIPAL
# ===========================================================================

class PatchGANDiscriminador(nn.Module):
    """
    Discriminador PatchGAN 70x70.

    Recibe directamente el tensor concatenado [condicion; objetivo] de 6 canales
    y produce un mapa de clasificacion por parches de forma (N, 1, 30, 30).

    Cada valor del mapa corresponde a la "puntuacion de realismo" de un parche
    de 70x70 pixeles de la imagen de entrada. Valores altos = parche real,
    valores bajos = parche falso.

    La perdida final del discriminador promedia las puntuaciones de todos los
    parches (900 parches para una imagen 256x256), lo que equivale a entrenar
    el discriminador como si tuviera 900 muestras independientes por imagen.

    Flujo de datos:
        (N, 6, 256, 256)   <- entrada: cat([imagen_A, imagen_B], dim=1)
        (N, 64, 128, 128)  <- Conv1: stride=2
        (N, 128, 64, 64)   <- Conv2: stride=2
        (N, 256, 32, 32)   <- Conv3: stride=2
        (N, 512, 31, 31)   <- Conv4: stride=1  <- sin downsampling aqui
        (N, 1,   30, 30)   <- Conv5 (salida): stride=1, sin activacion
    """

    def __init__(
        self,
        ch_entrada: int = 6,
        nf: int = 64,
        usar_spectral_norm: bool = False,
    ):
        """
        Args:
            ch_entrada:          Canales de entrada. Default: 6 (3+3 concatenados).
            nf:                  Filtros base (numero de filtros de Conv1).
                                 Se duplica en Conv2 y Conv3 hasta el maximo de
                                 nf*8=512. Default: 64.
            usar_spectral_norm:  Si True, usa SpectralNorm en lugar de InstanceNorm.
                                 Alternativa mas moderna y estable para WGAN.
        """
        super(PatchGANDiscriminador, self).__init__()

        self.usar_spectral_norm = usar_spectral_norm

        # ---- Conv1: (N, 6, 256, 256) -> (N, 64, 128, 128) ----
        # Primera capa: SIN normalizacion (convencion Pix2Pix).
        # Razon: la imagen de entrada ya esta normalizada a [-1,1]. Si
        # normalizaramos de nuevo, distorsionariamos la distribucion de entrada
        # que el discriminador necesita para distinguir real de falso.
        self.conv1 = _bloque_conv(
            ch_entrada, nf,
            stride=2, usar_norm=False,
            usar_spectral_norm=False,   # nunca norm en la primera capa
        )

        # ---- Conv2: (N, 64, 128, 128) -> (N, 128, 64, 64) ----
        self.conv2 = _bloque_conv(
            nf, nf * 2,
            stride=2, usar_norm=not usar_spectral_norm,
            usar_spectral_norm=usar_spectral_norm,
        )

        # ---- Conv3: (N, 128, 64, 64) -> (N, 256, 32, 32) ----
        self.conv3 = _bloque_conv(
            nf * 2, nf * 4,
            stride=2, usar_norm=not usar_spectral_norm,
            usar_spectral_norm=usar_spectral_norm,
        )

        # ---- Conv4: (N, 256, 32, 32) -> (N, 512, 31, 31) ----
        # stride=1 deliberado: el campo receptivo sigue creciendo (de 22 a 46px)
        # pero la resolucion espacial no se reduce a la mitad.
        # Con stride=2 la salida seria 16x16 (muy pequeña, solo 256 parches).
        # Con stride=1 la salida es 31x31 -> la conv de salida la reduce a 30x30
        # (900 parches), un numero mas denso que cubre mejor la imagen.
        # Calculo: H_out = floor((32 - 4 + 2*1) / 1) + 1 = 31
        self.conv4 = _bloque_conv(
            nf * 4, nf * 8,
            stride=1, usar_norm=not usar_spectral_norm,
            usar_spectral_norm=usar_spectral_norm,
        )

        # ---- Conv5 (salida): (N, 512, 31, 31) -> (N, 1, 30, 30) ----
        # Una unica neurona por posicion espacial: la puntuacion del parche.
        # Calculo: H_out = floor((31 - 4 + 2*1) / 1) + 1 = 30
        #
        # SIN activacion final (ni Sigmoid ni Tanh):
        #   - Con GANLoss modo 'lsgan': MSELoss compara contra 0 o 1 directamente.
        #   - Con GANLoss modo 'vanilla': BCEWithLogitsLoss aplica sigmoid internamente.
        #   - Aplicar sigmoid aqui y luego BCELoss (en lugar de BCEWithLogitsLoss)
        #     es numericamente inestable por el problema de underflow/overflow.
        conv_salida = nn.Conv2d(nf * 8, 1, kernel_size=4, stride=1, padding=1)
        if usar_spectral_norm:
            conv_salida = nn.utils.spectral_norm(conv_salida)
        self.conv5_salida = conv_salida

    def forward(self, imagen_condicion: torch.Tensor, imagen_objetivo: torch.Tensor) -> torch.Tensor:
        """
        Clasifica si el par (condicion, objetivo) es real o falso por parches.

        La concatenacion de la condicion con el objetivo es el nucleo del
        mecanismo condicional: el discriminador NO puede aprobar una imagen
        objetivamente bonita si no es coherente con la condicion dada.

        Ejemplo en nuestro proyecto:
          - imagen_condicion: boceto cartografico OSM (dominio A)
          - imagen_objetivo:  imagen satelital, real o generada (dominio B)
          El discriminador aprendera que una imagen satelital es "real" solo
          si ADEMAS corresponde geograficamente al boceto dado.

        Args:
            imagen_condicion: Tensor (N, 3, H, W), imagen de entrada/condicion.
            imagen_objetivo:  Tensor (N, 3, H, W), imagen objetivo (real o generada).
                              Para H=W=256: produce salida (N, 1, 30, 30).

        Returns:
            Mapa de parches (N, 1, 30, 30). Sin activacion final.
            Valores logit: positivos = parche tiende a ser real,
                           negativos = parche tiende a ser falso.
        """
        # Concatenar en la dimension de canales: [A; B] -> 6 canales
        # Esta es la condicionalizacion: D(A, B) en lugar de D(B)
        x = torch.cat([imagen_condicion, imagen_objetivo], dim=1)  # (N, 6, H, W)

        x = self.conv1(x)          # (N,  64, H/2,   W/2)   = (N,  64, 128, 128)
        x = self.conv2(x)          # (N, 128, H/4,   W/4)   = (N, 128,  64,  64)
        x = self.conv3(x)          # (N, 256, H/8,   W/8)   = (N, 256,  32,  32)
        x = self.conv4(x)          # (N, 512, H/8-1, W/8-1) = (N, 512,  31,  31)
        x = self.conv5_salida(x)   # (N,   1, H/8-2, W/8-2) = (N,   1,  30,  30)

        return x

    @staticmethod
    def campo_receptivo() -> int:
        """
        Calcula y retorna el campo receptivo teorico en pixeles.

        El campo receptivo de 70px se obtiene de las 5 capas convolucionales
        con kernel=4: RF = 1 + 3*1 + 3*2 + 3*4 + 3*8 + 3*8 = 70.

        Returns:
            Campo receptivo en pixeles de la imagen de entrada.
        """
        especificaciones = [
            (4, 2),  # Conv1: kernel=4, stride=2
            (4, 2),  # Conv2: kernel=4, stride=2
            (4, 2),  # Conv3: kernel=4, stride=2
            (4, 1),  # Conv4: kernel=4, stride=1
            (4, 1),  # Conv5 (salida): kernel=4, stride=1
        ]
        historial = calcular_campo_receptivo(especificaciones)
        return historial[-1]["rf"]


# ===========================================================================
# BLOQUE DE VERIFICACION LOCAL
# Ejecutar: python src/models/patchgan_discriminator.py
#
# Verifica:
#   1. Forma de salida correcta: (1, 1, 30, 30)
#   2. Derivacion analitica del campo receptivo = 70px
#   3. Flujo de gradientes (backward pass sin NaN)
#   4. Comportamiento con pares reales vs falsos
#   5. Comparacion InstanceNorm vs SpectralNorm
# ===========================================================================
if __name__ == "__main__":

    SEP = "=" * 65

    print(SEP)
    print("  Verificacion local: patchgan_discriminator.py")
    print(SEP)

    # ------------------------------------------------------------------
    # 1. DERIVACION DEL CAMPO RECEPTIVO
    # ------------------------------------------------------------------
    print("\n[1] Derivacion analitica del campo receptivo")
    print("-" * 55)
    print(f"  {'Capa':<12} {'kernel':<8} {'stride':<8} {'jump':<8} {'RF (px)'}")
    print(f"  {'-'*11} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")

    especificaciones = [(4, 2), (4, 2), (4, 2), (4, 1), (4, 1)]
    historial_rf = calcular_campo_receptivo(especificaciones)

    for fila in historial_rf:
        marca = " <-- RF final" if fila["capa"] == "Conv5" else ""
        print(f"  {fila['capa']:<12} {fila['kernel']:<8} {fila['stride']:<8} "
              f"{fila['jump']:<8} {fila['rf']}{marca}")

    rf_final = historial_rf[-1]["rf"]
    assert rf_final == 70, f"Campo receptivo incorrecto: {rf_final}, esperado 70"
    print(f"\n  [OK] Campo receptivo = {rf_final}px  -> PatchGAN {rf_final}x{rf_final}")

    # ------------------------------------------------------------------
    # 2. FORWARD PASS Y FORMA DE SALIDA
    # ------------------------------------------------------------------
    print(f"\n[2] Forward pass con entrada 256x256")
    print("-" * 55)

    discriminador = PatchGANDiscriminador(ch_entrada=6, nf=64)
    discriminador.eval()

    condicion = torch.randn(1, 3, 256, 256)   # imagen boceto / satelite
    objetivo  = torch.randn(1, 3, 256, 256)   # imagen objetivo (real o generada)

    with torch.no_grad():
        mapa_parches = discriminador(condicion, objetivo)

    print(f"  Entrada condicion : {condicion.shape}")
    print(f"  Entrada objetivo  : {objetivo.shape}")
    print(f"  Tensor concatenado: {torch.cat([condicion, objetivo], dim=1).shape}")
    print(f"  Salida (mapa)     : {mapa_parches.shape}")
    print(f"  -> Esperado       : torch.Size([1, 1, 30, 30])")
    print(f"  -> Parches totales: {mapa_parches.shape[2] * mapa_parches.shape[3]} "
          f"({mapa_parches.shape[2]}x{mapa_parches.shape[3]})")

    assert mapa_parches.shape == torch.Size([1, 1, 30, 30]), \
        f"Forma incorrecta: {mapa_parches.shape}"
    print(f"  [OK] Forma de salida verificada")

    # ------------------------------------------------------------------
    # 3. FLUJO DE DIMENSIONES CAPA POR CAPA (con hooks)
    # ------------------------------------------------------------------
    print(f"\n[3] Dimensiones intermedias por capa")
    print("-" * 55)
    print(f"  {'Capa':<18} {'Forma de salida':<28} {'stride'}")
    print(f"  {'-'*17} {'-'*27} {'-'*6}")

    dims: dict = {}
    def _hook(nombre):
        def fn(mod, inp, out):
            dims[nombre] = tuple(out.shape)
        return fn

    discriminador.conv1.register_forward_hook(_hook("Conv1 (s=2)"))
    discriminador.conv2.register_forward_hook(_hook("Conv2 (s=2)"))
    discriminador.conv3.register_forward_hook(_hook("Conv3 (s=2)"))
    discriminador.conv4.register_forward_hook(_hook("Conv4 (s=1)"))
    discriminador.conv5_salida.register_forward_hook(_hook("Conv5/salida (s=1)"))

    with torch.no_grad():
        _ = discriminador(condicion, objetivo)

    strides = [2, 2, 2, 1, 1]
    for (nombre, forma), stride in zip(dims.items(), strides):
        print(f"  {nombre:<18} {str(forma):<28} {stride}")

    # ------------------------------------------------------------------
    # 4. FLUJO DE GRADIENTES (backward pass)
    # ------------------------------------------------------------------
    print(f"\n[4] Verificacion de flujo de gradientes")
    print("-" * 55)

    discriminador.train()
    condicion_grad = torch.randn(1, 3, 256, 256)
    objetivo_grad  = torch.randn(1, 3, 256, 256)

    salida_grad = discriminador(condicion_grad, objetivo_grad)

    # Simular la perdida del discriminador con un par "real" (etiqueta = 1)
    etiqueta_real = torch.ones_like(salida_grad)
    perdida = nn.MSELoss()(salida_grad, etiqueta_real)
    perdida.backward()

    # Verificar que todos los parametros tienen gradientes y no son NaN
    params_con_grad = 0
    params_con_nan  = 0
    for nombre_p, param in discriminador.named_parameters():
        if param.grad is not None:
            params_con_grad += 1
            if torch.isnan(param.grad).any():
                params_con_nan += 1

    total_params = sum(p.numel() for p in discriminador.parameters())
    print(f"  Perdida de prueba     : {perdida.item():.4f}")
    print(f"  Parametros con grad   : {params_con_grad}")
    print(f"  Parametros con NaN    : {params_con_nan}  (debe ser 0)")
    print(f"  Total parametros      : {total_params:,}  (~{total_params/1e6:.2f}M)")
    assert params_con_nan == 0, "Hay gradientes NaN: posible inestabilidad numerica"
    print(f"  [OK] Gradientes fluyen correctamente sin NaN")

    # ------------------------------------------------------------------
    # 5. SCORES: par REAL vs par FALSO
    # ------------------------------------------------------------------
    print(f"\n[5] Comportamiento inicial: par real vs par falso")
    print("-" * 55)
    print("  (Con pesos aleatorios, los scores deberian ser similares)")

    discriminador.eval()
    with torch.no_grad():
        # Par REAL: (condicion, imagen_real)
        imagen_real = torch.randn(1, 3, 256, 256)
        score_real = discriminador(condicion, imagen_real)

        # Par FALSO: (condicion, imagen_generada_aleatoria)
        imagen_falsa = torch.randn(1, 3, 256, 256)
        score_falso = discriminador(condicion, imagen_falsa)

    print(f"  Score par real   : media={score_real.mean().item():+.4f}, "
          f"min={score_real.min().item():+.4f}, max={score_real.max().item():+.4f}")
    print(f"  Score par falso  : media={score_falso.mean().item():+.4f}, "
          f"min={score_falso.min().item():+.4f}, max={score_falso.max().item():+.4f}")
    print(f"  Nota: tras el entrenamiento, score_real >> score_falso")
    print(f"  [OK] Forward pass para par real y falso correcto")

    # ------------------------------------------------------------------
    # 6. VARIANTE CON SPECTRAL NORMALIZATION
    # ------------------------------------------------------------------
    print(f"\n[6] Variante: SpectralNorm en lugar de InstanceNorm")
    print("-" * 55)

    disc_spectral = PatchGANDiscriminador(ch_entrada=6, nf=64, usar_spectral_norm=True)
    disc_spectral.eval()

    with torch.no_grad():
        salida_sn = disc_spectral(condicion, objetivo)

    params_sn = sum(p.numel() for p in disc_spectral.parameters())
    print(f"  Forma de salida     : {salida_sn.shape}   (identica a InstanceNorm)")
    print(f"  Parametros (SN)     : {params_sn:,}")
    print(f"  Parametros (IN)     : {total_params:,}")
    print(f"  Diferencia          : {abs(params_sn - total_params):,} parametros")
    assert salida_sn.shape == torch.Size([1, 1, 30, 30])
    print(f"  [OK] SpectralNorm produce la misma forma de salida")

    # ------------------------------------------------------------------
    # 7. BATCH SIZE > 1
    # ------------------------------------------------------------------
    print(f"\n[7] Verificacion con batch_size=4")
    print("-" * 55)

    discriminador.eval()
    condicion_b4 = torch.randn(4, 3, 256, 256)
    objetivo_b4  = torch.randn(4, 3, 256, 256)

    with torch.no_grad():
        salida_b4 = discriminador(condicion_b4, objetivo_b4)

    print(f"  Entrada : {condicion_b4.shape}")
    print(f"  Salida  : {salida_b4.shape}    -> Esperado: (4, 1, 30, 30)")
    assert salida_b4.shape == torch.Size([4, 1, 30, 30])
    print(f"  [OK] Funciona con batch_size=4")

    # ------------------------------------------------------------------
    # RESUMEN FINAL
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print(f"  RESUMEN DE LA ARQUITECTURA PatchGAN 70x70")
    print(f"  {'':4} Entrada : (N, 6, 256, 256)  [3ch condicion + 3ch objetivo]")
    print(f"  {'':4} Salida  : (N, 1,  30,  30)  [900 parches, 1 score por parche]")
    print(f"  {'':4} RF      : 70x70 px por parche")
    print(f"  {'':4} Params  : {total_params:,}  (~{total_params/1e6:.2f}M)")
    print(f"  {'':4} Modo    : InstanceNorm (default) | SpectralNorm (opcional)")
    print(SEP)
    print("  [OK] patchgan_discriminator.py verificado correctamente.")
    print(SEP)
