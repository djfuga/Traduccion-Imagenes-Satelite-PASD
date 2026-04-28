"""
logger.py — Sistema de logging estructurado para el entrenamiento
=================================================================

El logging durante el entrenamiento de una GAN es especialmente importante
porque permite detectar problemas comunes:
    - Mode collapse: la pérdida del generador cae a 0 repentinamente.
    - Discriminador demasiado fuerte: loss_D cerca de 0, loss_G muy alta.
    - Discriminador demasiado débil: loss_D cerca de 0.7 (igual que azar).
    - NaN en pérdidas: indica overflow numérico (frecuente sin AMP correctamente).

Este logger escribe simultáneamente a:
    1. Consola (stdout): para seguimiento en tiempo real en Colab.
    2. Archivo .log: para análisis posterior y debugging.
    3. TensorBoard (opcional): para visualización de curvas en tiempo real.

El formato de log incluye timestamp, época, batch y todas las pérdidas
relevantes del entrenamiento Pix2Pix.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class LoggerEntrenamiento:
    """
    Logger estructurado para el entrenamiento Pix2Pix.

    Combina logging a consola y archivo con un formato consistente que
    facilita el análisis posterior del entrenamiento.
    """

    def __init__(
        self,
        nombre_experimento: str,
        directorio_logs: str = "logs",
        usar_tensorboard: bool = False,
    ):
        """
        Args:
            nombre_experimento: Identificador único del experimento
                                (ej: 'sat2sketch_lsgan_batch1').
            directorio_logs:    Directorio donde guardar los archivos .log.
            usar_tensorboard:   Si True, también escribe a TensorBoard.
        """
        self.nombre = nombre_experimento
        self.usar_tensorboard = usar_tensorboard
        self.writer_tb = None

        # Crear directorio de logs si no existe
        Path(directorio_logs).mkdir(parents=True, exist_ok=True)

        # Nombre del archivo: incluye timestamp para no sobrescribir logs anteriores
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta_log = os.path.join(directorio_logs, f"{nombre_experimento}_{timestamp}.log")

        # Configurar el logger de Python
        self.logger = logging.getLogger(nombre_experimento)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()  # Limpiar handlers previos si existe

        # Handler para archivo (guarda todo, nivel DEBUG y superior)
        handler_archivo = logging.FileHandler(ruta_log, encoding="utf-8")
        handler_archivo.setLevel(logging.DEBUG)

        # Handler para consola (solo INFO y superior, más limpio)
        handler_consola = logging.StreamHandler(sys.stdout)
        handler_consola.setLevel(logging.INFO)

        # Formato: [TIMESTAMP] NIVEL - Mensaje
        formato = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)-8s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler_archivo.setFormatter(formato)
        handler_consola.setFormatter(formato)

        self.logger.addHandler(handler_archivo)
        self.logger.addHandler(handler_consola)

        # Inicializar TensorBoard si se solicita
        if usar_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                directorio_tb = os.path.join(directorio_logs, "tensorboard", nombre_experimento)
                self.writer_tb = SummaryWriter(log_dir=directorio_tb)
                self.logger.info(f"TensorBoard iniciado en: {directorio_tb}")
            except ImportError:
                self.logger.warning("TensorBoard no disponible. Instala con: pip install tensorboard")

        self.logger.info(f"Logger iniciado para experimento: '{nombre_experimento}'")
        self.logger.info(f"Archivo de log: {ruta_log}")

    def log_inicio_entrenamiento(self, config_dict: dict) -> None:
        """Registra los hiperparámetros al inicio del entrenamiento."""
        self.logger.info("=" * 60)
        self.logger.info("INICIO DE ENTRENAMIENTO")
        self.logger.info("=" * 60)
        for clave, valor in config_dict.items():
            self.logger.info(f"  {clave:30s}: {valor}")
        self.logger.info("=" * 60)

    def log_epoca(
        self,
        epoca: int,
        total_epocas: int,
        losses: Dict[str, float],
        tiempo_epoch_seg: float,
        lr_actual: float,
    ) -> None:
        """
        Registra el resumen de una época completa.

        Args:
            epoca:            Número de época actual (base 1).
            total_epocas:     Total de épocas del entrenamiento.
            losses:           Diccionario con pérdidas promedio de la época.
                              Esperado: {'D_real', 'D_fake', 'G_GAN', 'G_L1', 'G_total'}
            tiempo_epoch_seg: Duración de la época en segundos.
            lr_actual:        Learning rate actual (puede cambiar con el scheduler).
        """
        # Formatear losses para el log
        losses_str = " | ".join([f"{k}={v:.4f}" for k, v in losses.items()])
        tiempo_min = tiempo_epoch_seg / 60

        self.logger.info(
            f"Época [{epoca:4d}/{total_epocas}] | "
            f"Tiempo: {tiempo_min:.1f}min | "
            f"LR: {lr_actual:.6f} | "
            f"{losses_str}"
        )

        # Enviar a TensorBoard si está disponible
        if self.writer_tb is not None:
            for nombre_loss, valor in losses.items():
                self.writer_tb.add_scalar(f"Loss/{nombre_loss}", valor, epoca)
            self.writer_tb.add_scalar("LR/generador", lr_actual, epoca)

    def log_batch(
        self,
        epoca: int,
        batch: int,
        total_batches: int,
        losses: Dict[str, float],
    ) -> None:
        """
        Registra las pérdidas de un batch individual (a nivel DEBUG).

        Solo escribe al archivo, no a consola (para no saturar la salida).

        Args:
            epoca:         Época actual.
            batch:         Índice del batch actual.
            total_batches: Total de batches por época.
            losses:        Diccionario con pérdidas del batch.
        """
        losses_str = " | ".join([f"{k}={v:.4f}" for k, v in losses.items()])
        self.logger.debug(
            f"  Época {epoca} | Batch [{batch:4d}/{total_batches}] | {losses_str}"
        )

    def log_checkpoint_guardado(self, epoca: int, ruta: str) -> None:
        """Registra que se guardó un checkpoint."""
        self.logger.info(f"Checkpoint guardado — Época {epoca} → {ruta}")

    def log_advertencia(self, mensaje: str) -> None:
        """Registra una advertencia."""
        self.logger.warning(mensaje)

    def log_error(self, mensaje: str) -> None:
        """Registra un error."""
        self.logger.error(mensaje)

    def cerrar(self) -> None:
        """Cierra el writer de TensorBoard si está activo."""
        if self.writer_tb is not None:
            self.writer_tb.close()
            self.logger.info("TensorBoard writer cerrado.")


# ===========================================================================
# BLOQUE DE VERIFICACIÓN LOCAL
# ===========================================================================
if __name__ == "__main__":
    import tempfile, time
    print("=" * 60)
    print("Verificación local: logger.py")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = LoggerEntrenamiento(
            nombre_experimento="test_sat2sketch",
            directorio_logs=tmpdir,
            usar_tensorboard=False,
        )

        logger.log_inicio_entrenamiento({
            "direction": "AtoB",
            "batch_size": 1,
            "lr": 2e-4,
            "n_epochs": 200,
        })

        # Simular una época
        for batch in range(1, 4):
            logger.log_batch(
                epoca=1, batch=batch, total_batches=100,
                losses={"D_real": 0.45, "D_fake": 0.52, "G_GAN": 0.88, "G_L1": 0.31}
            )

        logger.log_epoca(
            epoca=1, total_epocas=200,
            losses={"D_real": 0.45, "D_fake": 0.52, "G_GAN": 0.88, "G_L1": 0.31, "G_total": 31.88},
            tiempo_epoch_seg=125.3,
            lr_actual=2e-4,
        )

        logger.log_checkpoint_guardado(epoca=1, ruta=f"{tmpdir}/epoch_001.pth")

        # Verificar que el archivo de log existe
        archivos_log = list(Path(tmpdir).glob("*.log"))
        assert len(archivos_log) == 1, "Debería existir exactamente un archivo .log"
        print(f"\nArchivo de log creado: {archivos_log[0].name}")

        logger.cerrar()

    print("\n[OK] logger.py verificado correctamente.")
