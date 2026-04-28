"""
trainer.py — Loop de entrenamiento Pix2Pix
==========================================

Este es el módulo central del entrenamiento. Implementa el ciclo de
actualización alternado entre discriminador (D) y generador (G) que
define la dinámica de entrenamiento de una GAN condicional.

Flujo de entrenamiento por cada batch:
----------------------------------------
1. **backward_D** (actualizar discriminador):
   a. Congelar G (sus parámetros no deben cambiar en este paso).
   b. Forward con par REAL: D(condicion, real) → prediccion_real → loss_D_real
   c. Forward con par FALSO: D(condicion, fake.detach()) → prediccion_fake → loss_D_fake
      (.detach() separa el grafo de G para que sus gradientes no se calculen aquí)
   d. loss_D = (loss_D_real + loss_D_fake) / 2
   e. backward(loss_D) → step(opt_D)

2. **backward_G** (actualizar generador):
   a. Descongelar G.
   b. Forward: fake_B = G(real_A)
   c. El discriminador evalúa: D(condicion, fake) → queremos que crea que es real
   d. loss_G_GAN = GANLoss(D(A, fake_B), es_real=True)  ← G "engaña" a D
   e. loss_G_L1 = L1Loss(fake_B, real_B)  ← G se parece a la imagen real
   f. loss_G = loss_G_GAN + λ * loss_G_L1
   g. backward(loss_G) → step(opt_G)

Mixed Precision Training (AMP):
---------------------------------
Con AMP, las operaciones de forward y backward se realizan en float16,
reduciendo el uso de VRAM y acelerando el entrenamiento en GPUs con
Tensor Cores (como la T4 de Colab).

El GradScaler escala los gradientes para evitar "underflow" numérico:
en float16, valores pequeños de gradiente pueden redondearse a 0.
El scaler multiplica el loss por un factor grande antes del backward,
y luego divide el gradiente real antes del optimizer.step().

Gradient Accumulation:
-----------------------
Con grad_accum_steps=4, actualizamos los pesos cada 4 batches en lugar de
cada 1. Esto simula un batch_size efectivo de 4 sin aumentar el uso de VRAM
(que sería proporcional al batch_size real).

Scheduler de Learning Rate:
-----------------------------
Pix2Pix usa un scheduler lineal:
    - Primeras n_epochs épocas: lr constante.
    - Siguientes n_epochs_decay épocas: lr decrece linealmente hasta 0.

Esto permite una convergencia más suave en las últimas épocas.

Referencia: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
(ver models/pix2pix_model.py y train.py)
"""

import time
from collections import defaultdict
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast

from src.models.generator import GeneradorUNet
from src.models.discriminator import DiscriminadorPatchGAN
from src.models.losses import GANLoss, perdida_total_generador
from src.models.networks import congelar_modelo, descongelar_modelo
from src.training.config import ConfigEntrenamiento


class EntrenadorPix2Pix:
    """
    Encapsula todo el estado y la lógica del entrenamiento Pix2Pix.

    Responsabilidades:
    - Inicializar G, D, optimizadores, schedulers y scaler AMP.
    - Implementar los pasos backward_D y backward_G.
    - Gestionar la acumulación de gradientes.
    - Registrar las pérdidas para logging y visualización.
    - Guardar y cargar checkpoints.
    """

    def __init__(self, config: ConfigEntrenamiento):
        self.config = config
        self.dispositivo = self._obtener_dispositivo()

        # ---- Modelos ----
        self.G = GeneradorUNet(
            canales_entrada=config.canales_imagen,
            canales_salida=config.canales_imagen,
            filtros_base=config.filtros_generador,
        ).to(self.dispositivo)

        self.D = DiscriminadorPatchGAN(
            canales_entrada=config.canales_imagen * 2,  # 6 = 3 condición + 3 objetivo
            filtros_base=config.filtros_discriminador,
        ).to(self.dispositivo)

        # ---- Funciones de pérdida ----
        self.criterio_gan = GANLoss(gan_mode=config.gan_mode).to(self.dispositivo)
        self.criterio_l1 = nn.L1Loss()

        # ---- Optimizadores Adam ----
        # beta1=0.5 en lugar del default 0.9: reduce el momentum para GANs
        # Esto evita que el optimizador acumule demasiado "ímpetu" en una
        # dirección cuando las pérdidas oscilan (comportamiento típico de GANs)
        self.opt_G = Adam(
            self.G.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
        )
        self.opt_D = Adam(
            self.D.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
        )

        # ---- Schedulers de Learning Rate ----
        # Decaimiento lineal: lr constante por n_epochs, luego decrece a 0
        total_epocas = config.n_epochs + config.n_epochs_decay

        def regla_decay(epoca_actual):
            # epoca_actual es 0-indexed en LambdaLR
            if epoca_actual < config.n_epochs:
                return 1.0  # lr constante
            else:
                # Fracción lineal de decaimiento
                fraccion = 1.0 - (epoca_actual - config.n_epochs) / max(1, config.n_epochs_decay)
                return max(0.0, fraccion)

        self.scheduler_G = LambdaLR(self.opt_G, lr_lambda=regla_decay)
        self.scheduler_D = LambdaLR(self.opt_D, lr_lambda=regla_decay)

        # ---- Mixed Precision Training ----
        # GradScaler escala los gradientes para evitar underflow en float16
        # enabled=False si no hay CUDA (CPU no soporta AMP)
        amp_disponible = config.use_amp and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=amp_disponible)
        self.usar_amp = amp_disponible

        if self.usar_amp:
            print("[AMP] Mixed Precision Training activado (float16)")
        else:
            print("[AMP] Mixed Precision Training desactivado (float32)")

        # ---- Estado interno ----
        self.epoca_actual = 0
        self.paso_global = 0
        # Acumulador de pérdidas por época (listas de valores por batch)
        self.losses_acumuladas: Dict[str, list] = defaultdict(list)
        # Historia completa (promedio por época)
        self.losses_historia: Dict[str, list] = defaultdict(list)

    def _obtener_dispositivo(self) -> torch.device:
        """Detecta el dispositivo disponible."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _paso_discriminador(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        fake_B: torch.Tensor,
    ) -> torch.Tensor:
        """
        Actualiza el discriminador D.

        D debe aprender a:
            - Dar valores ALTOS cuando el par (A, B_real) es real.
            - Dar valores BAJOS cuando el par (A, B_fake) es falso.

        El .detach() en fake_B es fundamental: evita que los gradientes
        del paso backward del discriminador fluyan hacia los pesos del
        generador. Sin .detach(), estaríamos modificando G mientras
        intentamos actualizar D, corrompiendo el entrenamiento.

        Args:
            real_A: Imagen condición (dominio A). Forma: (N, 3, 256, 256).
            real_B: Imagen objetivo real (dominio B). Forma: (N, 3, 256, 256).
            fake_B: Imagen generada por G. Forma: (N, 3, 256, 256).

        Returns:
            Pérdida total del discriminador.
        """
        # Congelar G: D no debe afectar los parámetros de G en este paso
        congelar_modelo(self.G)

        self.opt_D.zero_grad()

        with autocast(enabled=self.usar_amp):
            # Par real: D evalúa (condición, imagen_real)
            pred_real = self.D(real_A, real_B)
            loss_D_real = self.criterio_gan(pred_real, es_real=True)

            # Par falso: D evalúa (condición, imagen_generada)
            # .detach() separa fake_B del grafo computacional de G
            pred_fake = self.D(real_A, fake_B.detach())
            loss_D_fake = self.criterio_gan(pred_fake, es_real=False)

            # Pérdida total del discriminador (promedio entre real y falso)
            # El promedio estabiliza el entrenamiento: si D solo ve reales o
            # solo falsos en un batch, el gradiente sería sesgado
            loss_D = (loss_D_real + loss_D_fake) * 0.5

        self.scaler.scale(loss_D).backward()
        self.scaler.step(self.opt_D)
        self.scaler.update()

        # Registrar pérdidas individuales para logging detallado
        self.losses_acumuladas["D_real"].append(loss_D_real.item())
        self.losses_acumuladas["D_fake"].append(loss_D_fake.item())

        return loss_D

    def _paso_generador(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        fake_B: torch.Tensor,
        paso_acum: int,
    ) -> torch.Tensor:
        """
        Actualiza el generador G.

        G debe aprender a:
            - Generar imágenes que engañen al discriminador (GAN Loss).
            - Generar imágenes parecidas a la imagen real (L1 Loss).

        La acumulación de gradientes (grad_accum_steps) permite simular
        un batch más grande dividiendo la actualización en múltiples pasos.

        Args:
            real_A:    Imagen condición.
            real_B:    Imagen objetivo real.
            fake_B:    Imagen generada (ya calculada en el paso del discriminador).
            paso_acum: Número de paso dentro del ciclo de acumulación (0-indexed).

        Returns:
            Pérdida total del generador.
        """
        # Descongelar G para que sus gradientes se calculen
        descongelar_modelo(self.G)

        # Solo hacer zero_grad al INICIO del ciclo de acumulación
        if paso_acum == 0:
            self.opt_G.zero_grad()

        with autocast(enabled=self.usar_amp):
            # Evaluar el par falso: D(condición, imagen_generada)
            # Ahora queremos que D crea que es real (es_real=True)
            # El gradiente fluye por D → fake_B → G (sin .detach() aquí)
            pred_fake = self.D(real_A, fake_B)
            loss_G_GAN = self.criterio_gan(pred_fake, es_real=True)

            # Pérdida L1: cuán diferente es fake_B de real_B píxel a píxel
            loss_G_L1 = self.criterio_l1(fake_B, real_B)

            # Combinar pérdidas con lambda=100 para L1
            loss_G = perdida_total_generador(loss_G_GAN, loss_G_L1, self.config.lambda_l1)

            # Escalar la pérdida para la acumulación de gradientes:
            # Al dividir por grad_accum_steps, el gradiente acumulado equivale
            # al gradiente de un batch de tamaño: batch_size * grad_accum_steps
            loss_G_escalada = loss_G / self.config.grad_accum_steps

        self.scaler.scale(loss_G_escalada).backward()

        # Solo actualizar pesos al FINAL del ciclo de acumulación
        es_ultimo_paso_acum = (paso_acum + 1) == self.config.grad_accum_steps
        if es_ultimo_paso_acum:
            self.scaler.step(self.opt_G)
            self.scaler.update()

        self.losses_acumuladas["G_GAN"].append(loss_G_GAN.item())
        self.losses_acumuladas["G_L1"].append(loss_G_L1.item())
        self.losses_acumuladas["G_total"].append(loss_G.item())

        return loss_G

    def paso_entrenamiento(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        paso_acum: int = 0,
    ) -> Dict[str, float]:
        """
        Ejecuta un paso completo de entrenamiento (D + G) para un batch.

        Args:
            real_A:    Batch de imágenes del dominio A. Forma: (N, 3, 256, 256).
            real_B:    Batch de imágenes del dominio B. Forma: (N, 3, 256, 256).
            paso_acum: Índice del paso de acumulación de gradientes (0-indexed).

        Returns:
            Diccionario con los valores de pérdida del paso actual.
        """
        real_A = real_A.to(self.dispositivo)
        real_B = real_B.to(self.dispositivo)

        # 1. Forward: G genera la imagen falsa
        with autocast(enabled=self.usar_amp):
            fake_B = self.G(real_A)

        # 2. Actualizar discriminador
        loss_D = self._paso_discriminador(real_A, real_B, fake_B)

        # 3. Actualizar generador
        loss_G = self._paso_generador(real_A, real_B, fake_B, paso_acum)

        self.paso_global += 1

        return {
            "D_real": self.losses_acumuladas["D_real"][-1],
            "D_fake": self.losses_acumuladas["D_fake"][-1],
            "G_GAN":  self.losses_acumuladas["G_GAN"][-1],
            "G_L1":   self.losses_acumuladas["G_L1"][-1],
            "G_total": self.losses_acumuladas["G_total"][-1],
        }

    def entrenar_epoca(self, dataloader) -> Dict[str, float]:
        """
        Entrena el modelo por una época completa.

        Args:
            dataloader: DataLoader con los pares de imágenes de entrenamiento.

        Returns:
            Diccionario con las pérdidas promedio de la época.
        """
        self.G.train()
        self.D.train()

        # Reiniciar acumuladores de pérdida
        self.losses_acumuladas = defaultdict(list)

        inicio_epoca = time.time()
        paso_acum = 0

        for i, (real_A, real_B) in enumerate(dataloader):
            losses = self.paso_entrenamiento(real_A, real_B, paso_acum)

            paso_acum = (paso_acum + 1) % self.config.grad_accum_steps

        # Calcular promedios de la época
        losses_promedio = {
            nombre: sum(valores) / len(valores)
            for nombre, valores in self.losses_acumuladas.items()
            if valores
        }

        # Guardar en historia
        for nombre, valor in losses_promedio.items():
            self.losses_historia[nombre].append(valor)

        # Actualizar scheduler al final de la época
        self.scheduler_G.step()
        self.scheduler_D.step()

        self.epoca_actual += 1

        # Liberar caché VRAM al final de cada época (previene fragmentación)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tiempo_epoca = time.time() - inicio_epoca
        losses_promedio["tiempo_seg"] = tiempo_epoca

        return losses_promedio

    def obtener_lr_actual(self) -> float:
        """Retorna el learning rate actual del optimizador G."""
        return self.opt_G.param_groups[0]["lr"]

    def generar_imagen(self, real_A: torch.Tensor) -> torch.Tensor:
        """
        Genera una imagen (modo evaluación, sin gradientes).

        Args:
            real_A: Imagen condición. Forma: (N, 3, 256, 256).

        Returns:
            Imagen generada en CPU, forma (N, 3, 256, 256).
        """
        self.G.eval()
        with torch.no_grad():
            real_A = real_A.to(self.dispositivo)
            fake_B = self.G(real_A)
        return fake_B.cpu()


# ===========================================================================
# BLOQUE DE VERIFICACIÓN LOCAL
# Ejecutar: python src/training/trainer.py
# Verifica que el loop de entrenamiento funciona sin errores en CPU.
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Verificación local: trainer.py")
    print("=" * 60)

    # Configuración mínima para prueba rápida
    config = ConfigEntrenamiento(
        n_epochs=1,
        n_epochs_decay=1,
        batch_size=1,
        use_amp=False,       # AMP requiere CUDA, desactivar para prueba CPU
        grad_accum_steps=2,
        filtros_generador=16,       # Reducir para que corra más rápido en CPU
        filtros_discriminador=16,
    )

    print(f"\nDispositivo: CPU (prueba local sin GPU)")
    entrenador = EntrenadorPix2Pix(config)

    # Simular 3 pasos de entrenamiento con tensores dummy
    print("\n--- Simulando 3 pasos de entrenamiento ---")
    for paso in range(3):
        real_A = torch.randn(1, 3, 256, 256)
        real_B = torch.randn(1, 3, 256, 256)
        idx_acum = paso % config.grad_accum_steps
        losses = entrenador.paso_entrenamiento(real_A, real_B, idx_acum)
        print(f"  Paso {paso+1}: D_real={losses['D_real']:.4f} | "
              f"G_GAN={losses['G_GAN']:.4f} | G_L1={losses['G_L1']:.4f}")

    # Verificar que todos los valores son escalares finitos
    for nombre, valor in losses.items():
        assert isinstance(valor, float), f"{nombre} no es float"
        assert valor == valor, f"{nombre} es NaN"  # NaN != NaN

    # Verificar inferencia
    print("\n--- Verificando generación de imagen ---")
    imagen_test = torch.randn(1, 3, 256, 256)
    imagen_generada = entrenador.generar_imagen(imagen_test)
    print(f"  Imagen generada: {imagen_generada.shape}")
    assert imagen_generada.shape == torch.Size([1, 3, 256, 256])

    print("\n[OK] trainer.py verificado correctamente.")
