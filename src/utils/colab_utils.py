"""
colab_utils.py — Utilidades específicas para Google Colab
==========================================================

Google Colab tiene particularidades importantes que este módulo gestiona:

1. **Memoria volátil**: Colab reinicia el runtime después de ~12h de inactividad
   o si se desconecta. TODO el trabajo en /content/ se pierde.
   Solución: sincronizar checkpoints y resultados con Google Drive.

2. **VRAM limitada**: La T4 gratuita tiene ~15GB nominales, pero el sistema
   puede dejar disponibles solo 12-13GB. Después de varias épocas, la
   fragmentación de memoria puede causar OOM aunque haya "espacio libre".
   Solución: limpiar caché periódicamente con clear_vram().

3. **CPU limitada**: Colab gratuito tiene 2 CPUs. Usar más de 2 workers
   en el DataLoader puede causar OOM en la memoria de sistema (no VRAM).

4. **Instalaciones no persistentes**: Las librerías instaladas con pip se
   pierden al reiniciar el runtime.

Este módulo centraliza estas preocupaciones para que los notebooks sean
más limpios y reproducibles.

Referencia map-sat (inspiración para gestión de datos en Colab):
https://github.com/miquel-espinosa/map-sat
"""

import gc
import os
import shutil
from pathlib import Path
from typing import Optional


def montar_drive(punto_montaje: str = "/content/drive") -> bool:
    """
    Monta Google Drive en el entorno Colab.

    Debe ejecutarse al inicio de cada sesión de Colab antes de guardar
    o cargar checkpoints desde Drive.

    Args:
        punto_montaje: Ruta donde montar el Drive. Default: '/content/drive'.

    Returns:
        True si el montaje fue exitoso, False si falló o no estamos en Colab.
    """
    try:
        from google.colab import drive  # type: ignore
        drive.mount(punto_montaje, force_remount=False)
        print(f"[Drive] Google Drive montado en: {punto_montaje}")
        return True
    except ImportError:
        print("[Drive] No estamos en Google Colab. Drive no disponible.")
        return False
    except Exception as e:
        print(f"[Drive] Error al montar Drive: {e}")
        return False


def verificar_gpu() -> dict:
    """
    Imprime información detallada sobre la GPU disponible y retorna un
    diccionario con los datos para uso programático.

    Returns:
        Diccionario con 'disponible', 'nombre', 'vram_total_gb', 'vram_libre_gb'.
    """
    import torch

    info = {"disponible": False, "nombre": "CPU", "vram_total_gb": 0, "vram_libre_gb": 0}

    print("\n" + "=" * 50)
    print("  INFORMACIÓN DE GPU")
    print("=" * 50)

    if torch.cuda.is_available():
        nombre = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_total = props.total_memory / 1e9
        vram_libre = (props.total_memory - torch.cuda.memory_allocated(0)) / 1e9
        vram_reservada = torch.cuda.memory_reserved(0) / 1e9

        info.update({
            "disponible": True,
            "nombre": nombre,
            "vram_total_gb": round(vram_total, 2),
            "vram_libre_gb": round(vram_libre, 2),
        })

        print(f"  GPU            : {nombre}")
        print(f"  VRAM Total     : {vram_total:.1f} GB")
        print(f"  VRAM Libre     : {vram_libre:.1f} GB")
        print(f"  VRAM Reservada : {vram_reservada:.1f} GB")
        print(f"  CUDA Version   : {torch.version.cuda}")
        print(f"  PyTorch Version: {torch.__version__}")

        # Avisar si hay poca VRAM disponible
        if vram_libre < 3.0:
            print(f"\n  [AVISO] Menos de 3GB de VRAM libre. Considera ejecutar clear_vram().")
    else:
        print("  GPU: No disponible (usando CPU)")
        print(f"  PyTorch Version: {torch.__version__}")

    print("=" * 50 + "\n")
    return info


def limpiar_vram() -> None:
    """
    Libera la VRAM no utilizada y fuerza la recolección de basura de Python.

    ¿Por qué es necesario?
    PyTorch mantiene un caché de memoria CUDA asignada pero no usada
    ('reserved but not allocated'). Después de varias épocas, este caché
    puede crecer varios GB y causar OOM aunque el modelo en sí sea pequeño.

    La combinación torch.cuda.empty_cache() + gc.collect() libera tanto
    el caché de CUDA como los objetos Python que referencian tensores en GPU.

    Cuándo llamar a esta función:
    - Al final de cada época de entrenamiento.
    - Antes de evaluar en el conjunto de validación.
    - Si se recibe un error de OOM (Out of Memory).
    - Al cambiar de dirección de entrenamiento (AtoB → BtoA).
    """
    import torch

    # Primero liberar objetos Python (puede liberar referencias a tensores)
    gc.collect()

    # Luego vaciar el caché de CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        vram_libre = (
            torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_allocated(0)
        ) / 1e9
        print(f"[VRAM] Caché limpiado. VRAM libre ahora: {vram_libre:.1f} GB")
    else:
        print("[VRAM] No hay GPU disponible. Solo se ejecutó gc.collect().")


def estimar_uso_vram(
    batch_size: int = 1,
    tamano_imagen: int = 256,
    usar_amp: bool = True,
) -> None:
    """
    Estima el uso de VRAM para los parámetros dados.

    Esta estimación es aproximada y basada en las características de la
    arquitectura U-Net 256 + PatchGAN 70×70.

    Los componentes principales del uso de VRAM son:
    1. Parámetros del modelo (G + D): ~64MB en float32, ~32MB en float16.
    2. Gradientes: igual que los parámetros (~64MB).
    3. Estados del optimizador Adam (x2 parámetros): ~128MB.
    4. Activaciones del forward pass (el mayor costo):
       - Proporcional a batch_size y tamano_imagen^2.
       - La U-Net con skip connections guarda TODAS las activaciones del
         encoder durante el forward para usarlas en el backward.

    Args:
        batch_size:    Tamaño del batch.
        tamano_imagen: Resolución de las imágenes (NxN).
        usar_amp:      Si True, asume float16 para activaciones.
    """
    # Estimaciones empíricas basadas en U-Net 256 + PatchGAN
    bytes_por_pixel = 2 if usar_amp else 4  # float16 vs float32

    # Parámetros del modelo
    params_G_mb = 54  # ~54MB para U-Net 256
    params_D_mb = 10  # ~10MB para PatchGAN
    gradientes_mb = params_G_mb + params_D_mb
    optimizer_mb = (params_G_mb + params_D_mb) * 2  # Adam: 2 momentos

    # Activaciones (dominante): proporcional a batch × pixels × capas
    factor_escala = (tamano_imagen / 256) ** 2
    activaciones_base_mb = 800  # ~800MB para batch=1, 256×256, float32
    activaciones_mb = (
        activaciones_base_mb
        * batch_size
        * factor_escala
        * (1 if not usar_amp else 0.5)
    )

    total_mb = params_G_mb + params_D_mb + gradientes_mb + optimizer_mb + activaciones_mb
    total_gb = total_mb / 1024

    print("\n" + "=" * 50)
    print("  ESTIMACIÓN DE USO DE VRAM")
    print("=" * 50)
    print(f"  Configuración  : batch={batch_size}, img={tamano_imagen}x{tamano_imagen}, AMP={usar_amp}")
    print(f"  Parámetros G   : ~{params_G_mb} MB")
    print(f"  Parámetros D   : ~{params_D_mb} MB")
    print(f"  Gradientes     : ~{gradientes_mb} MB")
    print(f"  Optimizadores  : ~{optimizer_mb} MB")
    print(f"  Activaciones   : ~{activaciones_mb:.0f} MB  ← componente dominante")
    print(f"  TOTAL estimado : ~{total_gb:.1f} GB")
    print(f"  T4 disponible  : ~13 GB  → {'[OK] Suficiente' if total_gb < 12 else '[AVISO] Puede ser justo'}")
    print("=" * 50 + "\n")


def instalar_dependencias_colab() -> None:
    """
    Instala las dependencias extras que no vienen preinstaladas en Colab.

    Las librerías base (torch, torchvision, PIL, numpy, matplotlib) ya están
    disponibles. Esta función instala solo lo adicional.

    Llamar una vez por sesión, al inicio del notebook de setup.
    """
    import subprocess
    dependencias = [
        "torchmetrics>=1.0.0",       # FID y otras métricas
        "scikit-image>=0.20.0",       # SSIM
        "tqdm>=4.65.0",               # Barras de progreso
    ]
    print("[Colab] Instalando dependencias adicionales...")
    for dep in dependencias:
        resultado = subprocess.run(
            ["pip", "install", "-q", dep],
            capture_output=True, text=True
        )
        if resultado.returncode == 0:
            print(f"  [OK] {dep}")
        else:
            print(f"  [FALLO] {dep}: {resultado.stderr[:100]}")


def copiar_a_drive(ruta_local: str, ruta_drive: str) -> bool:
    """
    Copia un archivo del filesystem local de Colab a Google Drive.

    Esencial para no perder checkpoints cuando el runtime se desconecta.

    Args:
        ruta_local: Ruta del archivo en el runtime de Colab (/content/...).
        ruta_drive: Ruta destino en Drive (/content/drive/MyDrive/...).

    Returns:
        True si la copia fue exitosa.
    """
    try:
        Path(ruta_drive).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ruta_local, ruta_drive)
        size_mb = os.path.getsize(ruta_drive) / 1e6
        print(f"[Drive] Copiado a Drive: {ruta_drive} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"[Drive] Error al copiar a Drive: {e}")
        return False


# ===========================================================================
# BLOQUE DE VERIFICACIÓN LOCAL
# (No podemos probar Drive o CUDA sin el entorno Colab, pero sí las funciones
# que no dependen de ellos)
# ===========================================================================
if __name__ == "__main__":
    import tempfile
    print("=" * 60)
    print("Verificación local: colab_utils.py")
    print("=" * 60)

    # verificar_gpu() funciona sin CUDA (informa que no hay GPU)
    info = verificar_gpu()
    print(f"GPU disponible: {info['disponible']}")

    # limpiar_vram() funciona sin CUDA
    limpiar_vram()

    # estimar_uso_vram() es puramente computacional
    for bs, amp in [(1, True), (1, False), (4, True)]:
        estimar_uso_vram(batch_size=bs, tamano_imagen=256, usar_amp=amp)

    # Probar copia de archivos
    with tempfile.TemporaryDirectory() as tmpdir:
        origen = os.path.join(tmpdir, "checkpoint.pth")
        destino = os.path.join(tmpdir, "drive_backup", "checkpoint.pth")

        # Crear archivo dummy
        with open(origen, "w") as f:
            f.write("checkpoint_dummy")

        exito = copiar_a_drive(origen, destino)
        assert exito and os.path.exists(destino), "Error: la copia a Drive falló"
        print("  copiar_a_drive: [OK]")

    print("\n[OK] colab_utils.py verificado correctamente.")
