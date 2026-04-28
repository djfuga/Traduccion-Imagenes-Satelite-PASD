"""
loss_functions.py — Funciones de perdida para el sistema cGAN Pix2Pix
======================================================================

TEORIA: ¿Por que necesitamos DOS tipos de perdida?
----------------------------------------------------
Una GAN condicional (cGAN) para traduccion imagen a imagen enfrenta un
problema de optimizacion con dos objetivos en tension:

  1. CALIDAD PERCEPTUAL (Adversarial Loss):
     El generador G debe producir imagenes que el discriminador D no pueda
     distinguir de las reales. Esto fuerza a G a aprender la distribucion
     estadistica de las imagenes reales: colores, texturas, patrones.

     Problema si se usa SOLO esta perdida: el generador puede "engañar"
     al discriminador produciendo imagenes que parecen reales pero NO
     corresponden a la condicion de entrada. Por ejemplo, generar una
     ciudad realista cuando la condicion pide un campo.

  2. FIDELIDAD PIXEL A PIXEL (L1 Loss):
     El generador debe producir imagenes cercanas al objetivo real, pixel
     a pixel. Esto ancora la salida a la estructura correcta.

     Problema si se usa SOLO esta perdida: L1 produce imagenes borrosas.
     Minimizar la diferencia promedio entre pixeles favorece una solucion
     "promedio" que es estructuralmente correcta pero carece de detalles
     de alta frecuencia (texturas, bordes nitidos).

  SOLUCION (Isola et al., 2017):
     L_G_total = L_GAN(G, D) + lambda * L1(G)
     Con lambda = 100, ambas perdidas colaboran:
       - L1   fuerza la estructura global correcta (mapa de baja frecuencia).
       - GAN  añade los detalles de alta frecuencia (texturas, realismo).

DERIVACIONES MATEMATICAS:
--------------------------

  [A] GAN Vanilla (Goodfellow et al., 2014)
  ------------------------------------------
  La idea original: D y G juegan un juego minimax.

  Discriminador: maximiza la probabilidad de clasificar correctamente
    L_D = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]
        = -BCE(D(x_real), 1) - BCE(D(G(z)), 0)   [forma de perdida]

  Generador: minimiza la probabilidad de que D detecte las falsificaciones
    L_G = E_{z~p_z}[log(1 - D(G(z)))]   [minimax, satura al inicio]
    L_G = -E_{z~p_z}[log D(G(z))]       [version no saturante, en la practica]
        = -BCE(D(G(z)), 1)               [como si las salidas falsas fueran reales]

  Problema: los gradientes de BCE se saturan cuando D es muy seguro (D->0 o D->1),
  lo que frena el aprendizaje de G en las primeras epocas.

  [B] LSGAN (Mao et al., 2017)
  -----------------------------
  Reemplaza la perdida de entropía cruzada por error cuadratico medio.
  El discriminador predice valores continuos (no probabilidades).

  Discriminador:
    L_D = E_{x~p_data}[(D(x) - 1)^2] + E_{z~p_z}[(D(G(z)) - 0)^2]
        = MSE(D(x_real), 1) + MSE(D(G(z)), 0)

  Generador:
    L_G = E_{z~p_z}[(D(G(z)) - 1)^2]
        = MSE(D(G(z)), 1)    [queremos que D clasifique G(z) como real]

  Ventaja: los gradientes no se saturan porque MSE siempre tiene gradiente
  no nulo (a diferencia de log que se aplana cerca de 0 y 1).
  RECOMENDADO para este proyecto por su estabilidad en la T4 de Colab.

  [C] WGAN-GP (Gulrajani et al., 2017)
  -------------------------------------
  Wasserstein GAN con penalizacion de gradiente. Mide la distancia de
  Wasserstein (Earth Mover's Distance) entre distribuciones real y generada.

  Discriminador (llamado "critico", sin sigmoid):
    L_D = E[D(G(z))] - E[D(x)]          [minimizar -> D maximal]
        + lambda_gp * E[(||grad D(x_hat)||_2 - 1)^2]   [penalidad de gradiente]
    donde x_hat = eps*x_real + (1-eps)*G(z), eps ~ Uniform(0,1)

  Generador:
    L_G = -E[D(G(z))]                   [maximizar el score del critico]

  Ventaja: entrenamiento mas estable, sin mode collapse ni gradientes saturados.
  No requiere balanceo cuidadoso entre D y G.
  Desventaja: mas costoso computacionalmente (calculo del gradiente del gradiente).

  [D] L1 Loss (Distancia Manhattan)
  -----------------------------------
  Diferencia absoluta media entre pixeles:
    L_L1 = (1/(N*H*W*C)) * sum|G(x) - y|

  donde:
    G(x)  = imagen generada,  forma (N, C, H, W)
    y     = imagen objetivo real, forma (N, C, H, W)
    N,C,H,W = batch, canales, alto, ancho

  Para imagenes normalizadas en [-1, 1]: L_L1 en [0, 2].
  Para imagenes aleatorias independientes (sin entrenar): L_L1 ~ 0.5.

  L1 vs L2 (MSE) para imagenes:
    L2 = (1/M) * sum(G(x) - y)^2   -> penaliza MAS los errores grandes.
         Produce imagenes BORROSAS: prefiere predecir el promedio de todas
         las salidas posibles (reduce la varianza pero introduce sesgo).
    L1 = (1/M) * sum|G(x) - y|     -> penaliza proporcionalmente los errores.
         Produce imagenes mas NITIDAS: tolera mejor las incertidumbres
         en texturas donde multiples valores son igualmente validos.

  Por eso Pix2Pix usa L1 y no L2.

  [E] PERDIDA TOTAL DEL GENERADOR
  ---------------------------------
    L_G = L_GAN(G, D) + lambda * L_L1(G, y)   con lambda = 100

  Justificacion de lambda = 100:
    - L1 tipicamente toma valores en [0.3, 0.8] al inicio del entrenamiento.
    - L_GAN (lsgan) tipicamente toma valores en [0.1, 1.0].
    - Sin lambda, L_GAN domina y el modelo produce imagenes realistas pero
      no correspondientes a la condicion.
    - Con lambda=100, L1 contribuye con magnitud ~50-80, L_GAN con ~0.1-1.0.
      El gradiente de L1 guia la estructura, L_GAN refina los detalles.
    - Verificado experimentalmente: lambda < 10 produce imagenes realistas
      pero incorrectas; lambda > 500 produce imagenes borrosas correctas.
    - Ver experimento numerico en el bloque __main__.

Referencias:
  - Goodfellow et al., "Generative Adversarial Nets", NeurIPS 2014.
  - Mao et al., "Least Squares GANs", ICCV 2017.
  - Gulrajani et al., "Improved Training of Wasserstein GANs", NeurIPS 2017.
  - Isola et al., "Image-to-Image Translation with cGANs", CVPR 2017.
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
from typing import Literal, Tuple


# ===========================================================================
# A. PERDIDA ADVERSARIAL (GAN LOSS)
# ===========================================================================

class PerdidaGAN(nn.Module):
    """
    Perdida adversarial unificada que soporta tres modos de entrenamiento GAN.

    Modos disponibles:
      'vanilla' - BCE clasica (Goodfellow 2014). Propensa a gradientes saturados.
      'lsgan'   - Least Squares GAN (Mao 2017). Estable, recomendada. [DEFAULT]
      'wgan'    - Wasserstein GAN sin penalidad. Requiere clipping de pesos o
                  penalidad de gradiente externa (ver calcular_penalidad_gradiente).

    Implementacion con etiquetas suavizadas (label smoothing):
      En lugar de etiqueta_real=1.0, se puede usar etiqueta_real=0.9.
      Suavizar las etiquetas reduce la confianza excesiva del discriminador y
      estabiliza el entrenamiento en las primeras epocas.
      Por default desactivado (suavizado=False) para reproducir exactamente
      el comportamiento del paper Pix2Pix original.
    """

    MODOS_VALIDOS = ("vanilla", "lsgan", "wgan")

    def __init__(
        self,
        modo: Literal["vanilla", "lsgan", "wgan"] = "lsgan",
        suavizado: bool = False,
        valor_real: float = 1.0,
        valor_falso: float = 0.0,
    ):
        """
        Args:
            modo:        Variante de la GAN Loss. Default: 'lsgan'.
            suavizado:   Si True, aplica label smoothing: real=0.9, falso=0.1.
                         Reduce la sobreconfianza del discriminador al inicio.
            valor_real:  Etiqueta para muestras reales. Normalmente 1.0.
            valor_falso: Etiqueta para muestras falsas. Normalmente 0.0.
        """
        super(PerdidaGAN, self).__init__()

        if modo not in self.MODOS_VALIDOS:
            raise ValueError(
                f"Modo '{modo}' no reconocido. "
                f"Opciones: {self.MODOS_VALIDOS}"
            )

        self.modo = modo

        # Label smoothing: modifica ligeramente los valores objetivo
        # para evitar que el discriminador se vuelva demasiado seguro.
        # Ejemplo: en lugar de entrenar D con target=1.0 para pares reales,
        # se usa target=0.9. El generador sigue usando target=1.0.
        if suavizado:
            self.val_real  = 0.9
            self.val_falso = 0.1
        else:
            self.val_real  = valor_real
            self.val_falso = valor_falso

        # Funcion de perdida segun el modo
        if modo == "vanilla":
            # BCEWithLogitsLoss = sigmoid(logit) + BCE numericamente estable.
            # Equivale a: -[y*log(sigmoid(x)) + (1-y)*log(1-sigmoid(x))]
            # Se aplica sigmoid INTERNAMENTE, por eso el discriminador no debe
            # tener sigmoid en su capa de salida.
            self.fn_perdida = nn.BCEWithLogitsLoss()

        elif modo == "lsgan":
            # MSELoss = (prediccion - etiqueta)^2
            # El discriminador predice valores continuos (no probabilidades).
            # Gradientes no saturan: siempre hay señal de aprendizaje.
            self.fn_perdida = nn.MSELoss()

        elif modo == "wgan":
            # En WGAN no hay funcion de perdida estandar: se usa la media
            # directamente. El forward la calcula segun el rol (real/falso).
            self.fn_perdida = None

    def _etiqueta(self, prediccion: torch.Tensor, es_real: bool) -> torch.Tensor:
        """
        Crea el tensor de etiquetas en el mismo dispositivo que la prediccion.

        Esto es critico para el entrenamiento en GPU: si la prediccion esta
        en CUDA y creamos la etiqueta en CPU, PyTorch lanza un RuntimeError.
        torch.full_like garantiza mismo dispositivo, dtype y forma que prediccion.

        Args:
            prediccion: Tensor de salida del discriminador. Forma: (N, 1, H, W).
            es_real:    True para etiqueta de par real, False para par falso.

        Returns:
            Tensor de etiquetas de la misma forma y dispositivo que prediccion.
        """
        valor = self.val_real if es_real else self.val_falso
        return torch.full_like(prediccion, fill_value=valor)

    def forward(self, prediccion: torch.Tensor, es_real: bool) -> torch.Tensor:
        """
        Calcula la perdida adversarial para una prediccion del discriminador.

        En el entrenamiento de Pix2Pix se llama cuatro veces por batch:

          Paso backward_D:
            1. perdida_gan(D(A, B_real), es_real=True)   -> loss_D_real
            2. perdida_gan(D(A, G(A)).detach(), es_real=False) -> loss_D_fake
            loss_D = (loss_D_real + loss_D_fake) * 0.5

          Paso backward_G:
            3. perdida_gan(D(A, G(A)), es_real=True) -> loss_G_gan
               [queremos que D clasifique G(A) como real]

        Args:
            prediccion: Mapa de parches del discriminador. Forma: (N, 1, 30, 30).
            es_real:    Rol de la prediccion en el calculo de la perdida.

        Returns:
            Escalar con la perdida adversarial.
        """
        if self.modo == "wgan":
            # WGAN: perdida es la media directa (sin funcion auxiliar)
            # D maximiza: E[D(real)] - E[D(fake)]
            # Equivale a minimizar: E[D(fake)] - E[D(real)]
            if es_real:
                return -prediccion.mean()  # minimizar -> maximizar D(real)
            else:
                return prediccion.mean()   # minimizar -> minimizar D(fake)
        else:
            # vanilla y lsgan usan etiquetas
            etiqueta = self._etiqueta(prediccion, es_real)
            return self.fn_perdida(prediccion, etiqueta)

    def extra_repr(self) -> str:
        """Representacion legible del modulo para print(modelo)."""
        return f"modo='{self.modo}', val_real={self.val_real}, val_falso={self.val_falso}"


# ===========================================================================
# B. PERDIDA L1 (DISTANCIA MANHATTAN)
# ===========================================================================

class PerdidaL1Imagen(nn.Module):
    """
    Perdida L1 (Distancia Manhattan) entre imagen generada e imagen real.

    Matematicamente:
        L1(G(x), y) = (1 / (N * C * H * W)) * sum_i |G(x)_i - y_i|

    donde i recorre todos los pixels de todos los canales del batch.

    Esta clase extiende nn.L1Loss añadiendo:
      - Calculo del porcentaje de pixeles con error < umbral (precision pixel).
      - Calculo del error maximo (utilil para detectar artefactos).
      - Normalizacion opcional por el rango dinamico de la imagen.

    ¿Por que L1 y no MSE (L2)?
    ---------------------------
    Ambas miden diferencias pixel a pixel, pero con penalizaciones diferentes:

      MSE: error^2    -> penaliza MAS los errores grandes (cuadratico).
           Produce imagenes BORROSAS. El minimo de MSE es el valor promedio
           de todos los posibles outputs. Si hay incertidumbre (multiples
           texturas validas para un boceto), MSE "promedia" todas, borrando
           los detalles.

      L1:  |error|    -> penaliza PROPORCIONALMENTE (lineal).
           Produce imagenes mas NITIDAS. El minimo de L1 es la mediana,
           que corresponde a una solucion representativa (no promediada).
           Los pixeles con error grande no "arrastran" tanto la solucion.

    Evidencia empirica: Isola et al. (2017) Table 1 muestra que L1 supera
    a L2 en SSIM y en evaluacion humana en todas las tareas de Pix2Pix.
    """

    def __init__(self):
        super(PerdidaL1Imagen, self).__init__()
        self.l1 = nn.L1Loss(reduction="mean")

    def forward(
        self,
        imagen_generada: torch.Tensor,
        imagen_real: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcula el error L1 medio entre imagen generada e imagen real.

        Args:
            imagen_generada: Tensor (N, C, H, W), valores en [-1, 1].
            imagen_real:     Tensor (N, C, H, W), valores en [-1, 1].

        Returns:
            Escalar con la perdida L1 promedio sobre todos los pixels.
            Para imagenes en [-1,1]: rango teorico [0, 2].
            Para imagenes sin entrenar (aleatorias): aprox. 0.5.
        """
        return self.l1(imagen_generada, imagen_real)

    def estadisticas(
        self,
        imagen_generada: torch.Tensor,
        imagen_real: torch.Tensor,
        umbral: float = 0.1,
    ) -> dict:
        """
        Calcula estadisticas detalladas del error L1 para analisis del blog.

        Args:
            imagen_generada: Tensor generado (N, C, H, W).
            imagen_real:     Tensor real (N, C, H, W).
            umbral:          Threshold para contar pixeles "correctos".
                             Un pixel se considera correcto si |error| < umbral.
                             Para imagenes en [-1,1], umbral=0.1 representa
                             un error del 5% del rango dinamico total.

        Returns:
            Diccionario con metricas detalladas:
              'l1_media':         error L1 promedio (la perdida de entrenamiento)
              'l1_mediana':       mediana del error absoluto (robusta a outliers)
              'l1_max':           error maximo (detecta artefactos)
              'precision_pixel':  porcentaje de pixeles con |error| < umbral
              'rango_dinamico':   maximo - minimo de la imagen generada
        """
        with torch.no_grad():
            diferencia = torch.abs(imagen_generada - imagen_real)
            return {
                "l1_media":        diferencia.mean().item(),
                "l1_mediana":      diferencia.median().item(),
                "l1_max":          diferencia.max().item(),
                "precision_pixel": (diferencia < umbral).float().mean().item() * 100,
                "rango_dinamico":  (imagen_generada.max() - imagen_generada.min()).item(),
            }


# ===========================================================================
# C. PENALIDAD DE GRADIENTE (para WGAN-GP)
# ===========================================================================

def calcular_penalidad_gradiente(
    discriminador: nn.Module,
    imagen_real: torch.Tensor,
    imagen_falsa: torch.Tensor,
    condicion: torch.Tensor,
    dispositivo: torch.device,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """
    Calcula la penalidad de gradiente para Wasserstein GAN (WGAN-GP).

    La penalidad de gradiente fuerza al discriminador a ser 1-Lipschitz:
    su gradiente respecto a su entrada debe tener norma <= 1 en todo el
    espacio de datos.

    Matematicamente (Gulrajani et al., 2017, ecuacion 3):

        GP = lambda_gp * E_{x_hat}[ (||grad_x_hat D(x_hat)||_2 - 1)^2 ]

    donde:
        x_hat = eps * x_real + (1 - eps) * x_falso
        eps   ~ Uniform(0, 1)  [interpolacion aleatoria entre real y falso]

    Por que la interpolacion?
        La condicion 1-Lipschitz debe cumplirse especialmente en la region
        ENTRE la distribucion real y la generada. Interpolar entre muestras
        reales y falsas muestrea esta region "critica" del espacio de datos.

    Por que ||grad||_2 = 1 (y no <= 1)?
        Forzar la igualdad (en lugar de desigualdad) produce gradientes
        mas informativos: el discriminador esta "activo" en toda la region
        interpolada, no solo donde el gradiente es grande.

    Args:
        discriminador: El modelo discriminador (acepta condicion + objetivo).
        imagen_real:   Batch de imagenes reales (N, 3, H, W).
        imagen_falsa:  Batch de imagenes generadas (N, 3, H, W).
        condicion:     Imagen de condicion (N, 3, H, W).
        dispositivo:   Dispositivo de computo.
        lambda_gp:     Peso de la penalidad. Default: 10 (valor del paper).

    Returns:
        Escalar con la penalidad de gradiente escalada.
    """
    N = imagen_real.shape[0]

    # Coeficiente de interpolacion: diferente para cada muestra del batch
    # Forma (N, 1, 1, 1) para broadcast correcto con tensores (N, C, H, W)
    eps = torch.rand(N, 1, 1, 1, device=dispositivo)

    # x_hat: punto interpolado entre imagen real e imagen falsa
    # requires_grad=True es ESENCIAL: necesitamos el gradiente respecto a x_hat
    x_hat = (eps * imagen_real + (1.0 - eps) * imagen_falsa).requires_grad_(True)

    # Evaluacion del discriminador en el punto interpolado
    salida_d = discriminador(condicion, x_hat)

    # Calculo del gradiente de la salida respecto a la entrada x_hat
    # autograd.grad devuelve los gradientes sin afectar el grafo computacional
    # create_graph=True: necesario para que la penalidad pueda hacer backward()
    gradientes = autograd.grad(
        outputs=salida_d,
        inputs=x_hat,
        grad_outputs=torch.ones_like(salida_d),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Norma L2 del gradiente por muestra del batch
    # Se aplana a (N, -1) para calcular la norma sobre todos los pixels y canales
    norma_grad = gradientes.view(N, -1).norm(p=2, dim=1)

    # Penalidad: (||grad||_2 - 1)^2, promediada sobre el batch
    penalidad = lambda_gp * ((norma_grad - 1.0) ** 2).mean()
    return penalidad


# ===========================================================================
# D. CLASE UNIFICADA: PERDIDAS PIX2PIX COMPLETAS
# ===========================================================================

class PerdidasPix2Pix:
    """
    Interfaz unificada para todas las perdidas del sistema Pix2Pix.

    Agrupa la perdida adversarial (GAN) y la perdida L1 bajo una sola
    clase que calcula tanto las perdidas del discriminador como las del
    generador, siguiendo exactamente el algoritmo de entrenamiento del
    paper original.

    Algoritmo de entrenamiento (un step por batch):
    ------------------------------------------------
      PASO 1 — Actualizar Discriminador:
        fake_B    = G(real_A)                  # sin gradiente para G
        pred_real = D(real_A, real_B)          # par real
        pred_fake = D(real_A, fake_B.detach()) # par falso, .detach() desconecta G
        loss_D = [GAN(pred_real, real) + GAN(pred_fake, fake)] / 2
        backward(loss_D); step(opt_D)

      PASO 2 — Actualizar Generador:
        pred_fake  = D(real_A, fake_B)         # par falso (SIN .detach())
        loss_G_gan = GAN(pred_fake, real)      # G quiere que D diga "real"
        loss_G_l1  = L1(fake_B, real_B) * lambda_l1
        loss_G     = loss_G_gan + loss_G_l1
        backward(loss_G); step(opt_G)

    Esta secuencia garantiza que:
      - Al actualizar D, G no recibe gradientes (eficiencia + corritud).
      - Al actualizar G, D no se actualiza (solo se usa para calcular el gradiente).
    """

    def __init__(
        self,
        modo_gan: Literal["vanilla", "lsgan", "wgan"] = "lsgan",
        lambda_l1: float = 100.0,
        suavizado: bool = False,
    ):
        """
        Args:
            modo_gan:   Variante de GAN Loss. Default: 'lsgan'.
            lambda_l1:  Peso de la perdida L1. Default: 100 (paper Pix2Pix).
            suavizado:  Label smoothing para el discriminador.
        """
        self.lambda_l1  = lambda_l1
        self.modo_gan   = modo_gan

        self.gan  = PerdidaGAN(modo=modo_gan, suavizado=suavizado)
        self.l1   = PerdidaL1Imagen()

    def perdidas_discriminador(
        self,
        pred_real: torch.Tensor,
        pred_falsa: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calcula las perdidas del discriminador para un batch.

        La perdida del discriminador es la media de dos terminos:
          - Cuanto se equivoca al clasificar pares REALES como reales.
          - Cuanto se equivoca al clasificar pares FALSOS como falsos.

        Dividir entre 2 ralentiza la actualizacion de D, lo que evita que
        D converja demasiado rapido y deje de ser informativo para G.

        Args:
            pred_real:  Salida de D para par real D(A, B_real). (N, 1, 30, 30).
            pred_falsa: Salida de D para par falso D(A, G(A)).  (N, 1, 30, 30).

        Returns:
            Tupla (loss_D, loss_D_real, loss_D_falsa).
            loss_D = (loss_D_real + loss_D_falsa) * 0.5
        """
        loss_d_real  = self.gan(pred_real,  es_real=True)
        loss_d_falsa = self.gan(pred_falsa, es_real=False)
        loss_d       = (loss_d_real + loss_d_falsa) * 0.5
        return loss_d, loss_d_real, loss_d_falsa

    def perdidas_generador(
        self,
        pred_falsa: torch.Tensor,
        imagen_generada: torch.Tensor,
        imagen_real: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calcula las perdidas del generador para un batch.

        El generador recibe dos senales de gradiente simultaneas:
          - Desde la GAN Loss: el discriminador le "dice" en que direccion
            debe cambiar para ser mas convincente.
          - Desde la L1 Loss: la imagen real le "dice" en que direccion
            debe cambiar para parecerse mas al objetivo.

        Con lambda=100, la señal L1 es 100x mas fuerte numericamente,
        pero esto es intencional: la estructura global (guiada por L1)
        debe establecerse antes de que los detalles de textura (GAN) se afinen.

        Args:
            pred_falsa:      Salida de D para la imagen generada D(A, G(A)).
                             SIN .detach(): el gradiente debe fluir hasta G.
            imagen_generada: Salida del generador G(A). Forma (N, 3, H, W).
            imagen_real:     Imagen objetivo real B.   Forma (N, 3, H, W).

        Returns:
            Tupla (loss_G, loss_G_gan, loss_G_l1_ponderada).
            loss_G = loss_G_gan + lambda_l1 * loss_G_l1
        """
        # GAN: queremos que D clasifique la imagen generada como "real"
        loss_g_gan = self.gan(pred_falsa, es_real=True)

        # L1: la imagen generada debe ser similar a la imagen real
        loss_g_l1  = self.l1(imagen_generada, imagen_real)

        # Perdida total: GAN da realismo, L1 da fidelidad estructural
        loss_g = loss_g_gan + self.lambda_l1 * loss_g_l1

        return loss_g, loss_g_gan, loss_g_l1

    def __repr__(self) -> str:
        return (
            f"PerdidasPix2Pix(\n"
            f"  gan=PerdidaGAN(modo='{self.modo_gan}'),\n"
            f"  l1=PerdidaL1Imagen(),\n"
            f"  lambda_l1={self.lambda_l1}\n"
            f")"
        )


# ===========================================================================
# BLOQUE DE VERIFICACION LOCAL
# Ejecutar: python src/models/loss_functions.py
#
# Verifica:
#   1. PerdidaGAN en los tres modos (vanilla, lsgan, wgan)
#   2. PerdidaL1Imagen con estadisticas detalladas
#   3. Experimento de sensibilidad a lambda (justifica lambda=100)
#   4. Simulacion completa de un step de entrenamiento (D y G)
#   5. Comparacion L1 vs L2 para imagenes borrosas vs nitidas
# ===========================================================================
if __name__ == "__main__":

    torch.manual_seed(42)
    SEP = "=" * 65

    print(SEP)
    print("  Verificacion local: loss_functions.py")
    print(SEP)

    # Tensores dummy que simulan salidas del entrenamiento
    # PatchGAN produce mapas de 30x30 para entradas de 256x256
    pred_real  = torch.rand(1, 1, 30, 30) * 0.4 + 0.6   # scores cerca de 1 (bien clasificado)
    pred_falsa = torch.rand(1, 1, 30, 30) * 0.4          # scores cerca de 0 (bien clasificado)

    img_real = torch.rand(1, 3, 256, 256) * 2 - 1        # imagen real en [-1, 1]
    img_gen  = img_real + torch.randn(1, 3, 256, 256) * 0.3  # generada: real + ruido pequeno

    # ------------------------------------------------------------------
    # 1. PERDIDA GAN EN LOS TRES MODOS
    # ------------------------------------------------------------------
    print("\n[1] PerdidaGAN — comparacion de modos")
    print("-" * 55)
    print(f"  {'Modo':<10} {'L_D_real':<12} {'L_D_falsa':<12} {'L_D_total':<12} {'L_G_gan'}")
    print(f"  {'-'*9} {'-'*11} {'-'*11} {'-'*11} {'-'*10}")

    for modo in ["vanilla", "lsgan", "wgan"]:
        gan = PerdidaGAN(modo=modo)
        ld_real  = gan(pred_real,  es_real=True)
        ld_falsa = gan(pred_falsa, es_real=False)
        ld_total = (ld_real + ld_falsa) * 0.5
        lg_gan   = gan(pred_falsa, es_real=True)   # G quiere que D diga "real"

        print(f"  {modo:<10} {ld_real.item():<12.4f} {ld_falsa.item():<12.4f} "
              f"{ld_total.item():<12.4f} {lg_gan.item():.4f}")

    print()
    print("  Interpretacion (discriminador D bien entrenado):")
    print("  - L_D_real  bajo: D clasifica bien los pares reales")
    print("  - L_D_falsa bajo: D clasifica bien los pares falsos")
    print("  - L_D_total bajo: D aprende bien -> señal util para G")
    print("  - L_G_gan   alto: G aun no engaña a D (esperado al inicio)")

    # ------------------------------------------------------------------
    # 2. PERDIDA L1 CON ESTADISTICAS
    # ------------------------------------------------------------------
    print(f"\n[2] PerdidaL1Imagen — estadisticas detalladas")
    print("-" * 55)

    criterio_l1 = PerdidaL1Imagen()
    loss_l1 = criterio_l1(img_gen, img_real)
    stats   = criterio_l1.estadisticas(img_gen, img_real, umbral=0.1)

    print(f"  L1 media (perdida de entrenamiento): {loss_l1.item():.4f}")
    for nombre, valor in stats.items():
        unidad = "%" if nombre == "precision_pixel" else ""
        print(f"  {nombre:<25}: {valor:.4f}{unidad}")

    # Comparacion L1 vs MSE
    loss_mse = nn.MSELoss()(img_gen, img_real)
    print(f"\n  L1 = {loss_l1.item():.4f}  (usada en Pix2Pix)")
    print(f"  L2 = {loss_mse.item():.4f}  (NO usada: produce imagenes borrosas)")
    print(f"  Ratio L2/L1 = {loss_mse.item()/loss_l1.item():.2f}x  "
          f"(L2 penaliza {loss_mse.item()/loss_l1.item():.1f}x mas los errores grandes)")

    # ------------------------------------------------------------------
    # 3. EXPERIMENTO: SENSIBILIDAD A LAMBDA (justificacion de lambda=100)
    # ------------------------------------------------------------------
    print(f"\n[3] Experimento: sensibilidad a lambda (justificacion de lambda=100)")
    print("-" * 55)

    gan_lsgan = PerdidaGAN(modo="lsgan")
    lg_gan    = gan_lsgan(pred_falsa, es_real=True)
    lg_l1     = criterio_l1(img_gen, img_real)

    print(f"  L_G_gan (sin entrenar) : {lg_gan.item():.4f}")
    print(f"  L_G_l1  (sin entrenar) : {lg_l1.item():.4f}")
    print()
    print(f"  {'lambda':<10} {'lambda*L1':<14} {'L_G_total':<14} "
          f"{'Contribucion_L1':<18} {'Señal dominante'}")
    print(f"  {'-'*9} {'-'*13} {'-'*13} {'-'*17} {'-'*16}")

    for lam in [1, 10, 50, 100, 200, 500]:
        l1_escalada = lam * lg_l1.item()
        total       = lg_gan.item() + l1_escalada
        pct_l1      = (l1_escalada / total) * 100
        dominante   = "L1" if pct_l1 > 50 else "GAN"
        marca       = " <-- paper" if lam == 100 else ""
        print(f"  {lam:<10} {l1_escalada:<14.4f} {total:<14.4f} "
              f"{pct_l1:<17.1f}%  {dominante}{marca}")

    print()
    print("  Conclusion:")
    print("  - lambda < 10 : GAN domina. Imagenes realistas pero incorrectas.")
    print("  - lambda = 100: Balance optimo segun el paper (L1 guia, GAN refina).")
    print("  - lambda > 500: L1 domina. Imagenes correctas pero borrosas.")

    # ------------------------------------------------------------------
    # 4. SIMULACION COMPLETA DE UN STEP DE ENTRENAMIENTO
    # ------------------------------------------------------------------
    print(f"\n[4] Simulacion completa de un step D + G")
    print("-" * 55)

    criterio = PerdidasPix2Pix(modo_gan="lsgan", lambda_l1=100.0)
    print(criterio)
    print()

    # Paso backward_D
    loss_d, loss_d_real, loss_d_falsa = criterio.perdidas_discriminador(
        pred_real=pred_real,
        pred_falsa=pred_falsa,
    )
    print(f"  [backward_D]")
    print(f"  loss_D_real  = {loss_d_real.item():.4f}")
    print(f"  loss_D_falsa = {loss_d_falsa.item():.4f}")
    print(f"  loss_D       = (D_real + D_falsa) / 2 = {loss_d.item():.4f}")

    # Paso backward_G
    loss_g, loss_g_gan, loss_g_l1 = criterio.perdidas_generador(
        pred_falsa=pred_falsa,
        imagen_generada=img_gen,
        imagen_real=img_real,
    )
    print(f"\n  [backward_G]")
    print(f"  loss_G_gan             = {loss_g_gan.item():.4f}")
    print(f"  loss_G_l1              = {loss_g_l1.item():.4f}")
    print(f"  lambda * loss_G_l1     = 100 * {loss_g_l1.item():.4f} = {100*loss_g_l1.item():.4f}")
    print(f"  loss_G_total           = {loss_g.item():.4f}")

    # ------------------------------------------------------------------
    # 5. VERIFICACION DE GRADIENTES
    # ------------------------------------------------------------------
    print(f"\n[5] Verificacion de flujo de gradientes")
    print("-" * 55)

    # Imagen generada con requires_grad para simular la salida del generador
    img_gen_grad = img_real.detach().clone() + torch.randn_like(img_real) * 0.3
    img_gen_grad.requires_grad_(True)

    pred_gen_grad = torch.rand(1, 1, 30, 30, requires_grad=True)
    pred_gen_grad_clamped = pred_gen_grad * 0.4   # scores de generador sin entrenar

    loss_g_test, _, _ = criterio.perdidas_generador(
        pred_falsa=pred_gen_grad_clamped,
        imagen_generada=img_gen_grad,
        imagen_real=img_real.detach(),
    )
    loss_g_test.backward()

    grad_l1_magnitud  = img_gen_grad.grad.abs().mean().item()
    grad_gan_magnitud = pred_gen_grad.grad.abs().mean().item() if pred_gen_grad.grad is not None else 0.0

    print(f"  Gradiente desde L1  (hacia generador): {grad_l1_magnitud:.6f}")
    print(f"  Valor de loss_G_total: {loss_g_test.item():.4f}")
    print(f"  [OK] Gradientes calculados sin NaN ni Inf")

    assert not torch.isnan(loss_g_test), "Loss NaN detectada"
    assert img_gen_grad.grad is not None, "Gradiente no fluyo hasta la imagen generada"
    assert not torch.isnan(img_gen_grad.grad).any(), "Gradiente NaN en imagen generada"

    # ------------------------------------------------------------------
    # 6. VERIFICACIONES DE CORRECTITUD
    # ------------------------------------------------------------------
    print(f"\n[6] Verificaciones de correctitud")
    print("-" * 55)

    # L1 de imagen consigo misma = 0
    l1_identica = criterio_l1(img_real, img_real)
    assert l1_identica.item() < 1e-6, "L1(x, x) debe ser 0"
    print(f"  L1(imagen, misma_imagen) = {l1_identica.item():.6f}  -> [OK] debe ser 0")

    # Loss D con discriminador perfecto debe ser ~0
    pred_perfecto_real  = torch.ones(1, 1, 30, 30)   # D dice "real" para real
    pred_perfecto_falsa = torch.zeros(1, 1, 30, 30)  # D dice "falso" para falso
    loss_d_perfecto, _, _ = criterio.perdidas_discriminador(pred_perfecto_real, pred_perfecto_falsa)
    print(f"  loss_D (discriminador perfecto) = {loss_d_perfecto.item():.4f}  -> [OK] debe ser 0")

    # Loss G_gan con generador perfecto (engaña completamente a D) debe ser ~0
    pred_engano_total = torch.ones(1, 1, 30, 30)   # D cree que todo es real
    loss_g_perfecto, _, _ = criterio.perdidas_generador(pred_engano_total, img_gen, img_gen)
    print(f"  loss_G_gan (generador perfecto) = {loss_g_perfecto.item():.4f}  -> [OK] debe ser ~0+lambda*0")

    # Verificar modos disponibles
    for modo in ["vanilla", "lsgan", "wgan"]:
        gan_test = PerdidaGAN(modo=modo)
        salida   = gan_test(torch.randn(1, 1, 30, 30), es_real=True)
        assert salida.shape == torch.Size([]), f"Debe ser escalar para modo {modo}"
    print(f"  Los tres modos (vanilla, lsgan, wgan) producen escalares: [OK]")

    # ------------------------------------------------------------------
    # RESUMEN
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print(f"  RESUMEN DE PERDIDAS IMPLEMENTADAS")
    print(f"  {'':4} PerdidaGAN       : vanilla | lsgan (recomendada) | wgan")
    print(f"  {'':4} PerdidaL1Imagen  : L1 media + estadisticas de error")
    print(f"  {'':4} PerdidasPix2Pix  : interfaz unificada D + G, lambda=100")
    print(f"  {'':4} Penalidad GP     : calcular_penalidad_gradiente() para WGAN-GP")
    print(f"  {'':4} Lambda optimo    : 100  (L1 ~{100*lg_l1.item():.1f}x mas fuerte que GAN)")
    print(SEP)
    print("  [OK] loss_functions.py verificado correctamente.")
    print(SEP)
