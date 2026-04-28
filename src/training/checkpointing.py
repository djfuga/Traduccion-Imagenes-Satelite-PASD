"""
checkpointing.py — Guardado y carga de checkpoints del modelo
==============================================================

En Google Colab, los checkpoints son críticos porque:
1. El runtime se reinicia automáticamente después de ~12h de inactividad.
2. Perder 10 horas de entrenamiento por una desconexión es un riesgo real.
3. Los checkpoints permiten también retomar el entrenamiento desde una época
   específica si se quiere comparar diferentes configuraciones.

¿Qué guardar en un checkpoint?
--------------------------------
Un checkpoint completo debe contener TODO lo necesario para reproducir
exactamente el estado del entrenamiento:
    - state_dict del generador (pesos y sesgos)
    - state_dict del discriminador
    - state_dict del optimizador G (incluye momentos de Adam)
    - state_dict del optimizador D
    - state_dict del GradScaler (para Mixed Precision Training)
    - Número de época actual
    - Historial de pérdidas

Sin los state_dicts de los optimizadores, Adam comenzaría desde cero y
los primeros batches tendrían gradientes erróneos (sin historia de momentos).

Referencia: https://pytorch.org/tutorials/beginner/saving_loading_models.html
"""

import os
import glob
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.optim import Optimizer


def guardar_checkpoint(
    epoca: int,
    generador: nn.Module,
    discriminador: nn.Module,
    optimizador_g: Optimizer,
    optimizador_d: Optimizer,
    losses_historia: Dict[str, list],
    directorio: str,
    prefijo: str = "checkpoint",
    grad_scaler: Optional[Any] = None,
) -> str:
    """
    Guarda el estado completo del entrenamiento en un archivo .pth.

    El nombre del archivo incluye el número de época para poder identificar
    y comparar checkpoints de diferentes momentos del entrenamiento.

    Args:
        epoca:            Número de época actual.
        generador:        Modelo generador.
        discriminador:    Modelo discriminador.
        optimizador_g:    Optimizador del generador.
        optimizador_d:    Optimizador del discriminador.
        losses_historia:  Diccionario con listas de pérdidas por época.
        directorio:       Carpeta donde guardar el checkpoint.
        prefijo:          Prefijo del nombre del archivo.
        grad_scaler:      GradScaler para AMP (opcional).

    Returns:
        Ruta completa del checkpoint guardado.
    """
    Path(directorio).mkdir(parents=True, exist_ok=True)
    ruta = os.path.join(directorio, f"{prefijo}_epoca_{epoca:04d}.pth")

    estado = {
        "epoca": epoca,
        "generador": generador.state_dict(),
        "discriminador": discriminador.state_dict(),
        "optimizador_g": optimizador_g.state_dict(),
        "optimizador_d": optimizador_d.state_dict(),
        "losses_historia": losses_historia,
    }

    # El GradScaler de AMP tiene su propio state_dict con el factor de escala
    # actual. Sin él, AMP comenzaría con escala por defecto (2^16), lo que
    # puede causar overflow en los primeros pasos del entrenamiento retomado.
    if grad_scaler is not None:
        estado["grad_scaler"] = grad_scaler.state_dict()

    torch.save(estado, ruta)
    size_mb = os.path.getsize(ruta) / 1e6
    print(f"[Checkpoint] Guardado: {ruta} ({size_mb:.1f} MB)")

    return ruta


def cargar_checkpoint(
    ruta: str,
    generador: nn.Module,
    discriminador: nn.Module,
    optimizador_g: Optimizer,
    optimizador_d: Optimizer,
    dispositivo: torch.device,
    grad_scaler: Optional[Any] = None,
    solo_pesos: bool = False,
) -> Dict[str, Any]:
    """
    Carga un checkpoint y restaura el estado de todos los componentes.

    El parámetro `map_location` es esencial: permite cargar un checkpoint
    guardado en GPU (Colab) en CPU (desarrollo local) y viceversa.
    Sin él, torch.load intentaría cargar los tensores en el mismo dispositivo
    donde fueron guardados, fallando si ese dispositivo no está disponible.

    Args:
        ruta:          Ruta del archivo .pth del checkpoint.
        generador:     Modelo generador (se modificará in-place).
        discriminador: Modelo discriminador (se modificará in-place).
        optimizador_g: Optimizador G (se modificará in-place).
        optimizador_d: Optimizador D (se modificará in-place).
        dispositivo:   Dispositivo destino (CPU, CUDA, MPS).
        grad_scaler:   GradScaler de AMP (opcional, se modifica in-place).
        solo_pesos:    Si True, solo carga los pesos del generador
                       (útil para inferencia, sin necesidad de D ni optimizadores).

    Returns:
        Diccionario con metadatos del checkpoint ('epoca', 'losses_historia').
    """
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"Checkpoint no encontrado: {ruta}")

    # map_location garantiza que los tensores se carguen en el dispositivo correcto
    estado = torch.load(ruta, map_location=dispositivo)

    generador.load_state_dict(estado["generador"])

    if not solo_pesos:
        discriminador.load_state_dict(estado["discriminador"])
        optimizador_g.load_state_dict(estado["optimizador_g"])
        optimizador_d.load_state_dict(estado["optimizador_d"])

        if grad_scaler is not None and "grad_scaler" in estado:
            grad_scaler.load_state_dict(estado["grad_scaler"])

    epoca = estado.get("epoca", 0)
    losses = estado.get("losses_historia", {})

    print(f"[Checkpoint] Cargado: {ruta}")
    print(f"  → Época: {epoca}")
    if losses:
        ultima_loss_g = losses.get("G_total", [None])[-1]
        print(f"  → Última loss_G_total: {ultima_loss_g}")

    return {"epoca": epoca, "losses_historia": losses}


def obtener_ultimo_checkpoint(directorio: str, prefijo: str = "checkpoint") -> Optional[str]:
    """
    Encuentra el checkpoint más reciente en un directorio.

    Busca archivos con el patrón '{prefijo}_epoca_NNNN.pth' y devuelve
    el que tenga el número de época más alto.

    Args:
        directorio: Carpeta donde buscar checkpoints.
        prefijo:    Prefijo del nombre de los archivos.

    Returns:
        Ruta del checkpoint más reciente, o None si no hay ninguno.
    """
    patron = os.path.join(directorio, f"{prefijo}_epoca_*.pth")
    checkpoints = sorted(glob.glob(patron))

    if not checkpoints:
        print(f"[Checkpoint] No se encontraron checkpoints en: {directorio}")
        return None

    ultimo = checkpoints[-1]
    print(f"[Checkpoint] Último checkpoint encontrado: {ultimo}")
    return ultimo


def guardar_solo_generador(
    generador: nn.Module,
    ruta: str,
) -> None:
    """
    Guarda solo los pesos del generador (para inferencia o transferencia).

    Útil cuando solo se necesita el generador para generar imágenes,
    sin guardar el estado completo del entrenamiento (discriminador,
    optimizadores, etc.), reduciendo el tamaño del archivo ~80%.

    Args:
        generador: Modelo generador a guardar.
        ruta:      Ruta del archivo de salida.
    """
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"generador": generador.state_dict()}, ruta)
    size_mb = os.path.getsize(ruta) / 1e6
    print(f"[Checkpoint] Generador guardado: {ruta} ({size_mb:.1f} MB)")


# ===========================================================================
# BLOQUE DE VERIFICACIÓN LOCAL
# ===========================================================================
if __name__ == "__main__":
    import tempfile
    print("=" * 60)
    print("Verificación local: checkpointing.py")
    print("=" * 60)

    # Importar modelos para la prueba
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.models.generator import GeneradorUNet
    from src.models.discriminator import DiscriminadorPatchGAN

    dispositivo = torch.device("cpu")
    G = GeneradorUNet()
    D = DiscriminadorPatchGAN()
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    losses_historicas = {
        "D_real": [0.45, 0.42],
        "D_fake": [0.51, 0.48],
        "G_GAN": [0.88, 0.79],
        "G_L1": [0.31, 0.28],
        "G_total": [31.88, 28.79],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        # Guardar checkpoint completo
        ruta_ckpt = guardar_checkpoint(
            epoca=2,
            generador=G, discriminador=D,
            optimizador_g=opt_G, optimizador_d=opt_D,
            losses_historia=losses_historicas,
            directorio=tmpdir,
        )

        # Cargar checkpoint
        G2 = GeneradorUNet()
        D2 = DiscriminadorPatchGAN()
        opt_G2 = torch.optim.Adam(G2.parameters(), lr=2e-4, betas=(0.5, 0.999))
        opt_D2 = torch.optim.Adam(D2.parameters(), lr=2e-4, betas=(0.5, 0.999))

        info = cargar_checkpoint(
            ruta_ckpt, G2, D2, opt_G2, opt_D2, dispositivo
        )
        assert info["epoca"] == 2, "Error: la época no se restauró correctamente"

        # Obtener último checkpoint
        ultimo = obtener_ultimo_checkpoint(tmpdir)
        assert ultimo == ruta_ckpt, "Error: no se encontró el checkpoint correcto"

        # Guardar solo generador
        ruta_solo_g = os.path.join(tmpdir, "generador.pth")
        guardar_solo_generador(G, ruta_solo_g)
        assert os.path.exists(ruta_solo_g)

    print("\n[OK] checkpointing.py verificado correctamente.")
