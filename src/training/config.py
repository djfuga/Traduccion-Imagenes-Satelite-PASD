"""
config.py — Configuración centralizada de hiperparámetros
==========================================================

Centralizar todos los hiperparámetros en un solo lugar tiene varias ventajas:
1. **Reproducibilidad**: guardar la configuración junto con los checkpoints
   permite recrear exactamente las mismas condiciones de entrenamiento.
2. **Experimentación**: cambiar un hiperparámetro solo requiere modificar
   un archivo, no buscar en múltiples scripts.
3. **Colab-friendly**: la configuración se puede pasar como argumento de
   línea de comandos o modificar directamente en la celda del notebook.

Hiperparámetros clave y sus justificaciones:
---------------------------------------------
- batch_size=1:     InstanceNorm funciona con batch=1 (BatchNorm no).
                    El paper Pix2Pix original usó batch=1.
                    En T4 con 256×256, batch>4 puede agotar VRAM.

- lr=2e-4:          Tasa de aprendizaje estándar para Adam en GANs.
                    Valores mayores desestabilizan el entrenamiento;
                    menores hacen el aprendizaje excesivamente lento.

- beta1=0.5:        El paper DCGAN y Pix2Pix recomiendan beta1=0.5 en lugar
                    del default 0.9 de Adam. Con 0.9, el momentum acumulado
                    puede desestabilizar las oscilaciones típicas de GANs.

- lambda_l1=100:    Ver justificación en losses.py.

- use_amp=True:     Mixed Precision Training. Reduce VRAM un 40% y acelera
                    el entrenamiento en GPUs con Tensor Cores (T4 los tiene).

- grad_accum_steps=4: Simula batch_size efectivo de 4 sin usar más VRAM.
                    El gradiente se acumula 4 pasos antes de actualizar pesos.

Referencia: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ConfigEntrenamiento:
    """
    Configuración completa para el entrenamiento Pix2Pix.
    Usa dataclass para serialización automática y documentación de tipos.
    """

    # ---- Datos ----
    directorio_datos: str = "data/processed"
    direction: str = "AtoB"        # 'AtoB' (sat→boceto) o 'BtoA' (boceto→sat)

    # ---- Arquitectura ----
    canales_imagen: int = 3        # RGB
    filtros_generador: int = 64    # Filtros base de la U-Net
    filtros_discriminador: int = 64  # Filtros base del PatchGAN

    # ---- Entrenamiento ----
    n_epochs: int = 100            # Épocas con learning rate constante
    n_epochs_decay: int = 100      # Épocas con lr decay lineal hasta 0
    # Total = n_epochs + n_epochs_decay = 200 épocas (estándar Pix2Pix)

    batch_size: int = 1            # Ver justificación arriba
    lr: float = 2e-4               # Tasa de aprendizaje para Adam
    beta1: float = 0.5             # Momentum Adam (ver justificación arriba)
    beta2: float = 0.999           # Segundo momento Adam (default estándar)

    lambda_l1: float = 100.0       # Peso de la pérdida L1
    gan_mode: str = "lsgan"        # 'lsgan' o 'vanilla'

    # ---- Optimizaciones VRAM (crítico para Colab T4) ----
    use_amp: bool = True           # Mixed Precision Training (float16)
    grad_accum_steps: int = 4      # Acumulación de gradientes (batch efectivo: 4)

    # ---- DataLoader ----
    num_workers: int = 2           # Máximo recomendado en Colab gratuito

    # ---- Checkpointing ----
    directorio_checkpoints: str = "checkpoints"
    frecuencia_checkpoint: int = 10  # Guardar cada N épocas
    continuar_desde_checkpoint: bool = False
    ruta_checkpoint_inicio: Optional[str] = None

    # ---- Google Drive (Colab) ----
    usar_drive: bool = False
    ruta_drive: str = "/content/drive/MyDrive/satelite-blog/checkpoints"

    # ---- Logging ----
    frecuencia_log: int = 100      # Logear loss cada N batches
    frecuencia_muestra: int = 5    # Guardar imágenes de muestra cada N épocas
    usar_tensorboard: bool = False

    # ---- Reproducibilidad ----
    semilla: int = 42

    def guardar(self, ruta: str) -> None:
        """
        Serializa la configuración a un archivo JSON.

        Guardar la configuración junto a los checkpoints garantiza que
        siempre podemos reproducir un experimento específico.

        Args:
            ruta: Ruta del archivo JSON de salida.
        """
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        print(f"[Config] Configuración guardada en: {ruta}")

    @classmethod
    def cargar(cls, ruta: str) -> "ConfigEntrenamiento":
        """
        Carga la configuración desde un archivo JSON.

        Args:
            ruta: Ruta del archivo JSON.

        Returns:
            Instancia de ConfigEntrenamiento con los valores del archivo.
        """
        with open(ruta, "r", encoding="utf-8") as f:
            datos = json.load(f)
        print(f"[Config] Configuración cargada desde: {ruta}")
        return cls(**datos)

    def imprimir_resumen(self) -> None:
        """Imprime un resumen de los parámetros más importantes."""
        print("\n" + "=" * 50)
        print("  CONFIGURACIÓN DE ENTRENAMIENTO")
        print("=" * 50)
        print(f"  Dirección      : {self.direction}")
        print(f"  Épocas totales : {self.n_epochs + self.n_epochs_decay}")
        print(f"  Batch size     : {self.batch_size}")
        print(f"  Batch efectivo : {self.batch_size * self.grad_accum_steps}")
        print(f"  Learning rate  : {self.lr}")
        print(f"  Lambda L1      : {self.lambda_l1}")
        print(f"  Modo GAN       : {self.gan_mode}")
        print(f"  Mixed Precision: {self.use_amp}")
        print(f"  Grad Accum     : {self.grad_accum_steps} pasos")
        print("=" * 50 + "\n")


# Configuración predeterminada para uso rápido
CONFIGURACION_DEFAULT = ConfigEntrenamiento()


# ===========================================================================
# BLOQUE DE VERIFICACIÓN LOCAL
# ===========================================================================
if __name__ == "__main__":
    import tempfile, os
    print("=" * 60)
    print("Verificación local: config.py")
    print("=" * 60)

    config = ConfigEntrenamiento(
        direction="AtoB",
        n_epochs=50,
        use_amp=True,
    )
    config.imprimir_resumen()

    # Probar serialización
    with tempfile.TemporaryDirectory() as tmpdir:
        ruta_json = os.path.join(tmpdir, "config_test.json")
        config.guardar(ruta_json)
        config_cargada = ConfigEntrenamiento.cargar(ruta_json)

        assert config_cargada.direction == "AtoB"
        assert config_cargada.n_epochs == 50
        assert config_cargada.use_amp == True
        print("  Serialización JSON: [OK]")

    print("\n[OK] config.py verificado correctamente.")
