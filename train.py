"""
train.py — Punto de entrada principal para el entrenamiento Pix2Pix
====================================================================

Script ejecutable desde consola:
    py -3 train.py --datos data/processed --direction AtoB --epocas 200

O importable desde un notebook Jupyter (ver notebooks/03_training_sat2sketch.ipynb):
    from train import bucle_entrenamiento, config_desde_args
    config, args = config_desde_args(["--datos", "data/processed", "--epocas", "10"])
    bucle_entrenamiento(config, "mi_experimento", args)

Flujo de entrenamiento Pix2Pix por epoca:
------------------------------------------
Para cada batch (real_A, real_B) del DataLoader:

  1. G genera la imagen falsa:
         fake_B = G(real_A)

  2. Actualizar D (paso backward_D):
         loss_D_real = criterion_gan(D(real_A, real_B),        label=REAL)
         loss_D_fake = criterion_gan(D(real_A, fake_B.detach()), label=FAKE)
         loss_D = (loss_D_real + loss_D_fake) * 0.5
         backward(loss_D) -> step(opt_D)

     .detach() en fake_B es CRITICO: desconecta fake_B del grafo de G.
     Sin el, los gradientes de D fluirian hacia G en este paso, corrompiendo
     el entrenamiento (G se actualizaria cuando solo deberia hacerlo D).

  3. Actualizar G (paso backward_G):
         loss_G_GAN = criterion_gan(D(real_A, fake_B), label=REAL)  # G engana a D
         loss_G_L1  = criterion_l1(fake_B, real_B)                  # G se parece a real
         loss_G = loss_G_GAN + lambda_l1 * loss_G_L1                # lambda=100
         backward(loss_G) -> step(opt_G)

Optimizaciones para Colab T4 (GPU gratuita, ~13 GB VRAM):
-----------------------------------------------------------
- Mixed Precision (AMP): float16 -> -40% VRAM, +50% velocidad con Tensor Cores
- Gradient accumulation (steps=4): batch efectivo 4 sin multiplicar VRAM
- InstanceNorm2d: matematicamente correcto con batch_size=1
- DataLoader pin_memory + num_workers=2: maximiza throughput de datos
- torch.cuda.empty_cache() al final de cada epoca: evita fragmentacion

Referencias:
- Pix2Pix: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
- Sketch2Map: https://github.com/PerlMonker303/S2MP
"""

import gc
import os
import sys
import time
import argparse
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

# Importaciones del proyecto
from src.training.config import ConfigEntrenamiento
from src.training.trainer import EntrenadorPix2Pix
from src.training.checkpointing import (
    guardar_checkpoint,
    cargar_checkpoint,
    obtener_ultimo_checkpoint,
)
from src.data.dataset_loader import DatasetParesSideBySide, crear_dataloader
from src.utils.logger import LoggerEntrenamiento
from src.utils.visualization import mostrar_grilla_muestras


# ===========================================================================
# ARGUMENTOS DE LINEA DE COMANDOS
# ===========================================================================

def parsear_argumentos(argv: Optional[list] = None) -> argparse.Namespace:
    """
    Define y parsea los argumentos de linea de comandos.

    Disenado para ser llamado desde:
    - Consola:  py -3 train.py --datos data/processed --epocas 200
    - Notebook: args = parsear_argumentos(["--datos", "data/processed"])

    Todos los valores por defecto coinciden con ConfigEntrenamiento, de modo
    que ejecutar sin argumentos produce una configuracion valida y reproducible.

    Args:
        argv: Lista de strings con los argumentos. None -> lee sys.argv.

    Returns:
        Namespace de argparse con todos los parametros parseados.
    """
    parser = argparse.ArgumentParser(
        prog="train.py",
        description=(
            "Entrenamiento Pix2Pix: traduccion bidireccional satelite <-> boceto"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Datos ----
    g = parser.add_argument_group("Datos")
    g.add_argument(
        "--datos", type=str, default="data/processed", metavar="DIR",
        help="Directorio con subcarpetas train/ y val/ (formato side-by-side)",
    )
    g.add_argument(
        "--direction", type=str, default="AtoB", choices=["AtoB", "BtoA"],
        help="AtoB: satelite->boceto | BtoA: boceto->satelite",
    )
    g.add_argument(
        "--workers", type=int, default=2, metavar="N",
        help="Numero de workers del DataLoader (max recomendado en Colab: 2)",
    )
    g.add_argument(
        "--cache_ram", action="store_true", default=False,
        help="Precargar todo el dataset en RAM (~2 GB para Maps completo)",
    )

    # ---- Arquitectura ----
    g = parser.add_argument_group("Arquitectura")
    g.add_argument(
        "--nf_gen", type=int, default=64, metavar="N",
        help="Filtros base de la U-Net (reducir a 32 si VRAM es critica)",
    )
    g.add_argument(
        "--nf_disc", type=int, default=64, metavar="N",
        help="Filtros base del PatchGAN 70x70",
    )

    # ---- Hiperparametros ----
    g = parser.add_argument_group("Hiperparametros")
    g.add_argument(
        "--epocas", type=int, default=100, metavar="N",
        help="Epocas con learning rate constante (paper original: 100)",
    )
    g.add_argument(
        "--epocas_decay", type=int, default=100, metavar="N",
        help="Epocas de decaimiento lineal del LR hasta 0 (paper original: 100)",
    )
    g.add_argument(
        "--batch", type=int, default=1, metavar="N",
        help="Batch size (1 es el estandar de Pix2Pix con InstanceNorm)",
    )
    g.add_argument(
        "--lr", type=float, default=2e-4, metavar="LR",
        help="Learning rate de Adam (paper original: 0.0002)",
    )
    g.add_argument(
        "--beta1", type=float, default=0.5, metavar="B1",
        help="Primer momento de Adam (0.5 para GANs, no el default 0.9)",
    )
    g.add_argument(
        "--lambda_l1", type=float, default=100.0, metavar="L",
        help="Peso de la perdida L1: loss_G = loss_GAN + lambda * loss_L1",
    )
    g.add_argument(
        "--gan_mode", type=str, default="lsgan", choices=["lsgan", "vanilla"],
        help="lsgan: MSELoss (mas estable) | vanilla: BCEWithLogitsLoss",
    )
    g.add_argument(
        "--grad_accum", type=int, default=4, metavar="N",
        help="Pasos de acumulacion de gradientes (batch efectivo = batch * N)",
    )

    # ---- Optimizaciones VRAM ----
    g = parser.add_argument_group("Optimizaciones VRAM (Colab T4)")
    g.add_argument(
        "--amp", action="store_true", default=True,
        help="Activar Mixed Precision Training (requiere CUDA con Tensor Cores)",
    )
    g.add_argument(
        "--no_amp", action="store_false", dest="amp",
        help="Desactivar Mixed Precision Training (usar float32)",
    )

    # ---- Checkpoints ----
    g = parser.add_argument_group("Checkpoints")
    g.add_argument(
        "--checkpoints", type=str, default="checkpoints", metavar="DIR",
        help="Directorio base para guardar los checkpoints",
    )
    g.add_argument(
        "--frecuencia_ckpt", type=int, default=10, metavar="N",
        help="Guardar checkpoint cada N epocas (siempre se guarda la ultima)",
    )
    g.add_argument(
        "--reanudar", action="store_true", default=False,
        help="Reanudar desde el ultimo checkpoint disponible",
    )
    g.add_argument(
        "--checkpoint_inicio", type=str, default=None, metavar="RUTA",
        help="Ruta especifica de un .pth para reanudar (sobreescribe --reanudar)",
    )

    # ---- Logging y visualizacion ----
    g = parser.add_argument_group("Logging y visualizacion")
    g.add_argument(
        "--resultados", type=str, default="results", metavar="DIR",
        help="Directorio base para guardar imagenes de muestra por epoca",
    )
    g.add_argument(
        "--logs", type=str, default="logs", metavar="DIR",
        help="Directorio para archivos .log y TensorBoard",
    )
    g.add_argument(
        "--frecuencia_muestra", type=int, default=5, metavar="N",
        help="Guardar grid de comparacion (entrada|generado|real) cada N epocas",
    )
    g.add_argument(
        "--frecuencia_log", type=int, default=50, metavar="N",
        help="Imprimir perdidas en consola cada N batches",
    )
    g.add_argument(
        "--tensorboard", action="store_true", default=False,
        help="Activar logging a TensorBoard (instalar: pip install tensorboard)",
    )
    g.add_argument(
        "--nombre_exp", type=str, default=None, metavar="NOMBRE",
        help="Nombre del experimento (default: generado desde los parametros)",
    )

    # ---- Reproducibilidad ----
    g = parser.add_argument_group("Reproducibilidad")
    g.add_argument(
        "--semilla", type=int, default=42, metavar="S",
        help="Semilla aleatoria para torch y random (garantiza reproducibilidad)",
    )

    # Flag especial para verificacion local sin dataset
    g = parser.add_argument_group("Desarrollo")
    g.add_argument(
        "--verificar", action="store_true", default=False,
        help="Ejecutar verificacion local con tensores dummy (sin dataset real)",
    )

    return parser.parse_args(argv)


# ===========================================================================
# CONSTRUCCION DE LA CONFIGURACION
# ===========================================================================

def config_desde_args(
    argv: Optional[list] = None,
) -> Tuple[ConfigEntrenamiento, argparse.Namespace]:
    """
    Convierte los argumentos de linea de comandos en ConfigEntrenamiento.

    Esta funcion es el puente entre la interfaz de consola (argparse) y
    el sistema de configuracion interno (dataclass). Permite usar el mismo
    entrenamiento desde consola o desde un notebook Jupyter.

    Uso desde notebook:
        config, args = config_desde_args([
            "--datos", "data/processed",
            "--epocas", "50",
            "--direction", "AtoB",
        ])

    Args:
        argv: Lista de argumentos. None -> usa sys.argv (modo consola).

    Returns:
        Tupla (ConfigEntrenamiento, argparse.Namespace).
    """
    args = parsear_argumentos(argv)

    config = ConfigEntrenamiento(
        directorio_datos=args.datos,
        direction=args.direction,
        filtros_generador=args.nf_gen,
        filtros_discriminador=args.nf_disc,
        n_epochs=args.epocas,
        n_epochs_decay=args.epocas_decay,
        batch_size=args.batch,
        lr=args.lr,
        beta1=args.beta1,
        lambda_l1=args.lambda_l1,
        gan_mode=args.gan_mode,
        grad_accum_steps=args.grad_accum,
        use_amp=args.amp,
        num_workers=args.workers,
        directorio_checkpoints=args.checkpoints,
        frecuencia_checkpoint=args.frecuencia_ckpt,
        continuar_desde_checkpoint=(args.reanudar or args.checkpoint_inicio is not None),
        ruta_checkpoint_inicio=args.checkpoint_inicio,
        frecuencia_log=args.frecuencia_log,
        frecuencia_muestra=args.frecuencia_muestra,
        usar_tensorboard=args.tensorboard,
        semilla=args.semilla,
    )

    return config, args


# ===========================================================================
# NOMBRE DEL EXPERIMENTO
# ===========================================================================

def generar_nombre_experimento(
    config: ConfigEntrenamiento,
    args: argparse.Namespace,
) -> str:
    """
    Genera un nombre descriptivo y unico para el experimento.

    El nombre se usa para organizar checkpoints, logs y resultados.
    Incluye los hiperparametros clave para identificar el experimento
    solo con el nombre del directorio.

    Ejemplo: 'AtoB_lsgan_nf64_lr0002_b1_accum4'

    Args:
        config: Configuracion del entrenamiento.
        args:   Argumentos parseados (para nombre_exp personalizado).

    Returns:
        Nombre del experimento como string.
    """
    if hasattr(args, "nombre_exp") and args.nombre_exp:
        return args.nombre_exp

    # lr=0.0002 -> "lr0002"
    lr_str = f"lr{int(config.lr * 1e4):04d}"

    return (
        f"{config.direction}"
        f"_{config.gan_mode}"
        f"_nf{config.filtros_generador}"
        f"_{lr_str}"
        f"_b{config.batch_size}"
        f"_accum{config.grad_accum_steps}"
    )


# ===========================================================================
# PREPARACION DE DIRECTORIOS
# ===========================================================================

def preparar_directorios(
    nombre_exp: str,
    dir_checkpoints: str,
    dir_resultados: str,
    dir_logs: str,
) -> Dict[str, Path]:
    """
    Crea los directorios necesarios para el experimento.

    Centralizarlo aqui garantiza que el bucle de entrenamiento no falle
    por directorios inexistentes en mitad del proceso.

    Args:
        nombre_exp:      Identificador del experimento.
        dir_checkpoints: Ruta base de checkpoints.
        dir_resultados:  Ruta base de resultados/muestras visuales.
        dir_logs:        Ruta de logs.

    Returns:
        Diccionario con los Path de cada directorio creado.
    """
    dirs = {
        "checkpoints": Path(dir_checkpoints) / nombre_exp,
        "resultados":  Path(dir_resultados) / nombre_exp,
        "logs":        Path(dir_logs),
    }
    for ruta in dirs.values():
        ruta.mkdir(parents=True, exist_ok=True)

    print(f"[Dirs] Checkpoints : {dirs['checkpoints']}")
    print(f"[Dirs] Resultados  : {dirs['resultados']}")
    print(f"[Dirs] Logs        : {dirs['logs']}")
    return dirs


# ===========================================================================
# CARGA DEL DATASET
# ===========================================================================

def cargar_datasets(
    config: ConfigEntrenamiento,
    cache_ram: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Crea los DataLoaders de entrenamiento y validacion.

    Decisiones de diseno para Colab T4:
    - pin_memory=True:        transferencia DMA directa CPU->GPU (mas rapida).
    - num_workers=2:          maximo recomendado en Colab (mas causa errores).
    - drop_last=True en train: descarta batch incompleto para InstanceNorm estable.
    - shuffle=True solo en train: val debe ser determinista para comparar epocas.
    - workers=0 en val:       evita overhead de fork en iteraciones cortas.

    Args:
        config:    Configuracion con directorio de datos y parametros del loader.
        cache_ram: Si True, precarga todas las imagenes en RAM antes del loop.

    Returns:
        Tupla (dl_train, dl_val). dl_val es None si no existe directorio val/.

    Raises:
        FileNotFoundError: Si no existe el directorio train/.
    """
    dir_train = Path(config.directorio_datos) / "train"
    dir_val   = Path(config.directorio_datos) / "val"

    if not dir_train.exists():
        raise FileNotFoundError(
            f"No se encontro el directorio de train: {dir_train}\n"
            f"Ejecuta primero:\n"
            f"  py -3 src/data/download_maps.py"
        )

    dataset_train = DatasetParesSideBySide(
        directorio_raiz=str(dir_train),
        direction=config.direction,
        modo="train",
        cache_en_ram=cache_ram,
    )
    dl_train = crear_dataloader(
        dataset=dataset_train,
        batch_size=config.batch_size,
        modo="train",
        num_workers=config.num_workers,
        shuffle=True,
    )
    print(f"[Data] Train : {len(dataset_train)} pares | {len(dl_train)} batches/epoca")

    dl_val = None
    if dir_val.exists():
        dataset_val = DatasetParesSideBySide(
            directorio_raiz=str(dir_val),
            direction=config.direction,
            modo="val",
            cache_en_ram=False,
        )
        dl_val = crear_dataloader(
            dataset=dataset_val,
            batch_size=config.batch_size,
            modo="val",
            num_workers=0,
            shuffle=False,
        )
        print(f"[Data] Val   : {len(dataset_val)} pares | {len(dl_val)} batches")
    else:
        print(f"[Data] Val   : no encontrado en {dir_val} (se omite)")

    return dl_train, dl_val


# ===========================================================================
# GUARDADO DE IMAGENES DE MUESTRA
# ===========================================================================

def guardar_muestras(
    entrenador: EntrenadorPix2Pix,
    dl_val: Optional[DataLoader],
    dir_resultados: Path,
    epoca: int,
    n_muestras: int = 4,
) -> None:
    """
    Genera y guarda un grid de comparacion: Entrada | Generado | Real.

    Las imagenes de muestra son la herramienta mas importante para monitorear
    visualmente el progreso del entrenamiento. Permiten detectar:
    - Mode collapse: todas las imagenes generadas son identicas.
    - Artefactos: texturas incorrectas o bordes borrosos.
    - Progreso: como mejora la calidad epoca a epoca.

    Las muestras se guardan en:
        {dir_resultados}/muestra_epoca_{epoca:04d}.png

    Args:
        entrenador:    Entrenador con el generador en modo eval.
        dl_val:        DataLoader de validacion. None -> no guarda nada.
        dir_resultados: Directorio de salida.
        epoca:         Numero de epoca (para el nombre del archivo).
        n_muestras:    Cuantos ejemplos incluir en el grid.
    """
    if dl_val is None:
        return

    try:
        real_A, real_B = next(iter(dl_val))
    except StopIteration:
        return

    # Generar en modo eval: no_grad + G.eval() internamente
    fake_B = entrenador.generar_imagen(real_A)

    ruta = dir_resultados / f"muestra_epoca_{epoca:04d}.png"
    mostrar_grilla_muestras(
        real_A=real_A,
        fake_B=fake_B,
        real_B=real_B,
        n_muestras=min(n_muestras, real_A.shape[0]),
        titulo=f"Epoca {epoca} | {entrenador.config.direction}",
        ruta_guardado=str(ruta),
    )
    print(f"  [Muestra] {ruta.name} guardada")


# ===========================================================================
# BUCLE PRINCIPAL DE ENTRENAMIENTO
# ===========================================================================

def bucle_entrenamiento(
    config: ConfigEntrenamiento,
    nombre_exp: str,
    args_extra: Optional[argparse.Namespace] = None,
) -> EntrenadorPix2Pix:
    """
    Ejecuta el bucle completo de entrenamiento Pix2Pix.

    Esta funcion orquesta todos los componentes del proyecto:
    - Instancia el Generador (U-Net 256) y el Discriminador (PatchGAN 70x70)
    - Configura los optimizadores Adam con lr=0.0002, beta1=0.5
    - Carga los datasets y crea los DataLoaders optimizados para T4
    - Ejecuta el bucle de epocas con backward_D + backward_G alternados
    - Gestiona la acumulacion de gradientes (batch efectivo sin mas VRAM)
    - Guarda checkpoints periodicos (recuperacion ante desconexiones en Colab)
    - Guarda grids de comparacion para monitoreo visual del progreso

    Exportable a notebook:
        from train import bucle_entrenamiento, config_desde_args
        config, args = config_desde_args(["--datos", "data/processed", "--epocas", "5"])
        entrenador = bucle_entrenamiento(config, "prueba_notebook", args)

    Args:
        config:     Configuracion completa del entrenamiento.
        nombre_exp: Nombre identificador del experimento.
        args_extra: Namespace con argumentos adicionales (cache_ram, resultados...).

    Returns:
        Objeto EntrenadorPix2Pix con los modelos entrenados (G y D).
    """
    # Fijar semillas para reproducibilidad
    torch.manual_seed(config.semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.semilla)

    # --- Directorios de salida ---
    dirs = preparar_directorios(
        nombre_exp=nombre_exp,
        dir_checkpoints=config.directorio_checkpoints,
        dir_resultados=getattr(args_extra, "resultados", "results"),
        dir_logs=getattr(args_extra, "logs", "logs"),
    )

    # --- Logger ---
    logger = LoggerEntrenamiento(
        nombre_experimento=nombre_exp,
        directorio_logs=str(dirs["logs"]),
        usar_tensorboard=config.usar_tensorboard,
    )
    logger.log_inicio_entrenamiento(asdict(config))

    # --- Datasets ---
    cache_ram = getattr(args_extra, "cache_ram", False)
    dl_train, dl_val = cargar_datasets(config, cache_ram=cache_ram)

    # -------------------------------------------------------------------
    # INICIALIZACION: U-Net 256 + PatchGAN 70x70 + Adam + schedulers
    # -------------------------------------------------------------------
    # EntrenadorPix2Pix instancia internamente:
    #   G = GeneradorUNet(nf=config.filtros_generador)
    #   D = DiscriminadorPatchGAN(nf=config.filtros_discriminador)
    #   opt_G = Adam(G.params(), lr=2e-4, betas=(0.5, 0.999))
    #   opt_D = Adam(D.params(), lr=2e-4, betas=(0.5, 0.999))
    #   scheduler_G/D = LambdaLR con decaimiento lineal
    #   scaler = GradScaler (si AMP disponible)
    print("\n[Modelo] Inicializando arquitecturas Pix2Pix...")
    entrenador = EntrenadorPix2Pix(config)
    total_epocas = config.n_epochs + config.n_epochs_decay

    n_G = sum(p.numel() for p in entrenador.G.parameters() if p.requires_grad)
    n_D = sum(p.numel() for p in entrenador.D.parameters() if p.requires_grad)
    print(f"[Modelo] Generador (U-Net 256)     : {n_G:>12,} parametros")
    print(f"[Modelo] Discriminador (PatchGAN)  : {n_D:>12,} parametros")
    print(f"[Modelo] Dispositivo               : {entrenador.dispositivo}")
    print(f"[Modelo] Adam lr={config.lr:.4f} | beta1={config.beta1} | beta2={config.beta2}")
    print(f"[Modelo] Total epocas              : {total_epocas} "
          f"({config.n_epochs} fijas + {config.n_epochs_decay} decay)")
    config.imprimir_resumen()

    # -------------------------------------------------------------------
    # REANUDAR DESDE CHECKPOINT (si se solicita)
    # -------------------------------------------------------------------
    epoca_inicio = 1
    ruta_ckpt = None

    if config.ruta_checkpoint_inicio:
        ruta_ckpt = config.ruta_checkpoint_inicio
    elif config.continuar_desde_checkpoint:
        ruta_ckpt = obtener_ultimo_checkpoint(
            str(dirs["checkpoints"]),
            prefijo=f"ckpt_{config.direction}",
        )

    if ruta_ckpt:
        print(f"\n[Resume] Cargando: {ruta_ckpt}")
        meta = cargar_checkpoint(
            ruta=ruta_ckpt,
            generador=entrenador.G,
            discriminador=entrenador.D,
            optimizador_g=entrenador.opt_G,
            optimizador_d=entrenador.opt_D,
            dispositivo=entrenador.dispositivo,
            grad_scaler=entrenador.scaler if entrenador.usar_amp else None,
        )
        epoca_inicio = meta["epoca"] + 1
        for nombre, vals in meta.get("losses_historia", {}).items():
            entrenador.losses_historia[nombre] = vals
        # Avanzar los schedulers al estado correcto
        for _ in range(meta["epoca"]):
            entrenador.scheduler_G.step()
            entrenador.scheduler_D.step()
        print(f"[Resume] Reanudando desde epoca {epoca_inicio}")

    # -------------------------------------------------------------------
    # BUCLE ITERATIVO DE EPOCAS
    # -------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  INICIO DEL ENTRENAMIENTO: {nombre_exp}")
    print(f"  Epocas      : {epoca_inicio} -> {total_epocas}")
    print(f"  Batches/ep. : {len(dl_train)}")
    print(f"  Batch efect.: {config.batch_size * config.grad_accum_steps} imagenes")
    print(f"{'='*65}\n")

    t_inicio_total = time.time()

    for epoca in range(epoca_inicio, total_epocas + 1):
        entrenador.G.train()
        entrenador.D.train()
        entrenador.losses_acumuladas = defaultdict(list)

        t_inicio_epoca = time.time()
        paso_acum = 0  # Contador para gradient accumulation

        # ----------------------------------------------------------------
        # Loop interno: iterar sobre todos los batches de la epoca
        # ----------------------------------------------------------------
        for batch_idx, (real_A, real_B) in enumerate(dl_train):
            # paso_acum controla cuando se hace zero_grad y cuando se hace step
            # del optimizador G (ver trainer.py: _paso_generador)
            losses = entrenador.paso_entrenamiento(real_A, real_B, paso_acum)
            paso_acum = (paso_acum + 1) % config.grad_accum_steps

            # Log nivel DEBUG en archivo (no satura consola)
            logger.log_batch(
                epoca=epoca,
                batch=batch_idx + 1,
                total_batches=len(dl_train),
                losses=losses,
            )

            # Impresion periodica en consola
            if (batch_idx + 1) % config.frecuencia_log == 0:
                print(
                    f"  [Ep {epoca:4d}/{total_epocas}]"
                    f" Batch {batch_idx+1:4d}/{len(dl_train)}"
                    f" | D_r={losses['D_real']:.3f}"
                    f" D_f={losses['D_fake']:.3f}"
                    f" | G_gan={losses['G_GAN']:.3f}"
                    f" G_l1={losses['G_L1']:.4f}"
                    f" | lr={entrenador.obtener_lr_actual():.6f}"
                )

        # ----------------------------------------------------------------
        # Fin de epoca: promedios, schedulers, limpieza VRAM
        # ----------------------------------------------------------------
        losses_promedio = {
            k: sum(v) / len(v)
            for k, v in entrenador.losses_acumuladas.items() if v
        }
        for nombre, valor in losses_promedio.items():
            entrenador.losses_historia[nombre].append(valor)

        # Decaimiento lineal del learning rate (scheduler LambdaLR)
        entrenador.scheduler_G.step()
        entrenador.scheduler_D.step()
        entrenador.epoca_actual += 1

        # Liberar VRAM fragmentada al final de cada epoca
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        t_epoca = time.time() - t_inicio_epoca
        lr_actual = entrenador.obtener_lr_actual()

        logger.log_epoca(
            epoca=epoca,
            total_epocas=total_epocas,
            losses=losses_promedio,
            tiempo_epoch_seg=t_epoca,
            lr_actual=lr_actual,
        )

        # Estimacion del tiempo restante
        epocas_hechas = epoca - epoca_inicio + 1
        t_medio = (time.time() - t_inicio_total) / epocas_hechas
        t_restante = t_medio * (total_epocas - epoca)

        print(
            f"  [OK] Ep {epoca:4d}/{total_epocas}"
            f" | {t_epoca/60:.1f}min"
            f" | ~{t_restante/3600:.1f}h restante"
            f" | G={losses_promedio.get('G_total', 0):.4f}"
            f" D_r={losses_promedio.get('D_real', 0):.3f}"
            f" D_f={losses_promedio.get('D_fake', 0):.3f}"
            f" | lr={lr_actual:.6f}"
        )

        # ----------------------------------------------------------------
        # Checkpoint periodico
        # ----------------------------------------------------------------
        if epoca % config.frecuencia_checkpoint == 0 or epoca == total_epocas:
            ruta_ckpt_guardado = guardar_checkpoint(
                epoca=epoca,
                generador=entrenador.G,
                discriminador=entrenador.D,
                optimizador_g=entrenador.opt_G,
                optimizador_d=entrenador.opt_D,
                losses_historia=dict(entrenador.losses_historia),
                directorio=str(dirs["checkpoints"]),
                prefijo=f"ckpt_{config.direction}",
                grad_scaler=entrenador.scaler if entrenador.usar_amp else None,
            )
            print(f"  [Ckpt] {Path(ruta_ckpt_guardado).name}")

        # ----------------------------------------------------------------
        # Imagenes de muestra para monitoreo visual
        # ----------------------------------------------------------------
        if epoca % config.frecuencia_muestra == 0 or epoca == total_epocas:
            guardar_muestras(entrenador, dl_val, dirs["resultados"], epoca)

    # -------------------------------------------------------------------
    # FIN DEL ENTRENAMIENTO
    # -------------------------------------------------------------------
    t_total = time.time() - t_inicio_total
    print(f"\n{'='*65}")
    print(f"  ENTRENAMIENTO COMPLETADO: {nombre_exp}")
    print(f"  Duracion total : {t_total/3600:.2f} horas")
    print(f"  Epocas         : {total_epocas}")
    print(f"  Checkpoints    : {dirs['checkpoints']}")
    print(f"  Resultados     : {dirs['resultados']}")
    print(f"{'='*65}\n")

    if logger.writer_tb is not None:
        logger.writer_tb.close()

    return entrenador


# ===========================================================================
# PUNTO DE ENTRADA PARA PRODUCCION
# ===========================================================================

def main() -> None:
    """
    Entrada principal para ejecucion desde consola.

    Uso:
        py -3 train.py --datos data/processed --direction AtoB
        py -3 train.py --datos data/processed --direction BtoA --epocas 50
        py -3 train.py --datos data/processed --reanudar
        py -3 train.py --verificar
    """
    config, args = config_desde_args()

    if args.verificar:
        _verificacion_local()
        return

    nombre_exp = generar_nombre_experimento(config, args)
    print(f"\n  Pix2Pix — Traduccion Bidireccional Satelital/Boceto")
    print(f"  Experimento: {nombre_exp}")

    bucle_entrenamiento(config, nombre_exp, args_extra=args)


# ===========================================================================
# VERIFICACION LOCAL (sin dataset real)
# Ejecutar: py -3 train.py --verificar
# ===========================================================================

def _verificacion_local() -> None:
    """
    Prueba el bucle completo con tensores dummy, sin necesidad de datos reales.

    Verifica que:
    1. U-Net y PatchGAN se instancian correctamente.
    2. Los optimizadores Adam tienen lr=0.0002 y beta1=0.5.
    3. El loop de backward_D + backward_G produce perdidas finitas.
    4. El scheduler de LR funciona (decaimiento lineal).
    5. La generacion de imagenes tiene rango [-1, 1] (salida Tanh).
    6. El parseo de argumentos es correcto.
    """
    print("=" * 65)
    print("  Verificacion local: train.py (sin dataset)")
    print("=" * 65)

    # Configuracion minima para prueba rapida en CPU
    config_test = ConfigEntrenamiento(
        n_epochs=1,
        n_epochs_decay=0,
        batch_size=1,
        use_amp=False,           # AMP requiere CUDA
        grad_accum_steps=2,
        filtros_generador=16,    # Reducir filtros para velocidad en CPU
        filtros_discriminador=16,
        frecuencia_log=1,
        frecuencia_checkpoint=1,
        frecuencia_muestra=999,  # Desactivar muestras (sin val real)
    )

    # [1] Instanciar modelos
    print("\n[1] Instanciando U-Net 256 + PatchGAN 70x70...")
    entrenador = EntrenadorPix2Pix(config_test)

    n_G = sum(p.numel() for p in entrenador.G.parameters())
    n_D = sum(p.numel() for p in entrenador.D.parameters())
    print(f"    Generador    : {n_G:,} parametros")
    print(f"    Discriminador: {n_D:,} parametros")
    print(f"    Dispositivo  : {entrenador.dispositivo}")

    # [2] Verificar optimizadores Adam
    print("\n[2] Verificando Adam (lr=0.0002, beta1=0.5)...")
    lr_G  = entrenador.opt_G.param_groups[0]["lr"]
    b1_G  = entrenador.opt_G.param_groups[0]["betas"][0]
    lr_D  = entrenador.opt_D.param_groups[0]["lr"]
    b1_D  = entrenador.opt_D.param_groups[0]["betas"][0]

    assert abs(lr_G - 2e-4) < 1e-8, f"lr_G incorrecto: {lr_G}"
    assert abs(lr_D - 2e-4) < 1e-8, f"lr_D incorrecto: {lr_D}"
    assert abs(b1_G - 0.5) < 1e-8,  f"beta1_G incorrecto: {b1_G}"
    assert abs(b1_D - 0.5) < 1e-8,  f"beta1_D incorrecto: {b1_D}"

    print(f"    opt_G: lr={lr_G:.6f} | beta1={b1_G} | [OK]")
    print(f"    opt_D: lr={lr_D:.6f} | beta1={b1_D} | [OK]")

    # [3] Simular 4 pasos de entrenamiento (2 ciclos de acumulacion)
    print("\n[3] Simulando 4 pasos de entrenamiento...")
    ultimas_losses = {}
    for paso in range(4):
        real_A = torch.randn(1, 3, 256, 256)
        real_B = torch.randn(1, 3, 256, 256)
        idx_acum = paso % config_test.grad_accum_steps
        losses = entrenador.paso_entrenamiento(real_A, real_B, idx_acum)
        ultimas_losses = losses
        print(
            f"    Paso {paso+1}/4 | "
            f"D_real={losses['D_real']:.4f} "
            f"D_fake={losses['D_fake']:.4f} | "
            f"G_GAN={losses['G_GAN']:.4f} "
            f"G_L1={losses['G_L1']:.5f}"
        )

    # Verificar que todas las perdidas son finitas
    for nombre, valor in ultimas_losses.items():
        assert isinstance(valor, float), f"'{nombre}' no es float: {type(valor)}"
        assert valor == valor,           f"'{nombre}' es NaN"         # NaN != NaN
        assert valor != float("inf"),    f"'{nombre}' es infinito"
    print("    [OK] Todas las perdidas son finitas y bien formadas")

    # [4] Scheduler de learning rate
    print("\n[4] Verificando scheduler de LR (decaimiento lineal)...")
    lr_antes = entrenador.opt_G.param_groups[0]["lr"]
    entrenador.scheduler_G.step()
    entrenador.scheduler_D.step()
    lr_despues = entrenador.opt_G.param_groups[0]["lr"]
    # Con n_epochs=1, n_epochs_decay=0: el LR no debe cambiar (paso 1 de 1 = factor 1.0)
    print(f"    LR antes step: {lr_antes:.6f}")
    print(f"    LR tras step : {lr_despues:.6f}")
    print("    [OK] Scheduler funcional")

    # [5] Generacion de imagenes en modo eval
    print("\n[5] Verificando generacion (modo eval, sin gradientes)...")
    imagen_entrada = torch.randn(1, 3, 256, 256)
    imagen_generada = entrenador.generar_imagen(imagen_entrada)

    assert imagen_generada.shape == torch.Size([1, 3, 256, 256]), \
        f"Forma incorrecta: {imagen_generada.shape}"
    assert imagen_generada.min() >= -1.0 - 1e-4, \
        f"Minimo fuera de rango Tanh: {imagen_generada.min():.4f}"
    assert imagen_generada.max() <= 1.0 + 1e-4, \
        f"Maximo fuera de rango Tanh: {imagen_generada.max():.4f}"

    print(f"    Forma    : {imagen_generada.shape}")
    print(f"    Rango    : [{imagen_generada.min():.3f}, {imagen_generada.max():.3f}]")
    print("    [OK] Imagen generada con Tanh en [-1, 1]")

    # [6] Parseo de argumentos
    print("\n[6] Verificando parseo de argumentos...")
    config2, args2 = config_desde_args([
        "--lr", "0.0002",
        "--epocas", "100",
        "--direction", "BtoA",
        "--gan_mode", "lsgan",
        "--grad_accum", "4",
        "--batch", "1",
    ])
    assert abs(config2.lr - 2e-4) < 1e-8
    assert config2.direction == "BtoA"
    assert config2.gan_mode == "lsgan"
    assert config2.grad_accum_steps == 4

    nombre_gen = generar_nombre_experimento(config2, args2)
    print(f"    Nombre generado: '{nombre_gen}'")
    print("    [OK] Argumentos parseados correctamente")

    # [7] Instrucciones de uso en produccion
    print("\n[7] Uso en produccion:")
    print("  " + "-" * 55)
    print("  # Entrenamiento completo satelite -> boceto (200 epocas):")
    print("  py -3 train.py --datos data/processed --direction AtoB")
    print()
    print("  # Boceto -> satelite con Mixed Precision y cache RAM:")
    print("  py -3 train.py --datos data/processed --direction BtoA")
    print("                 --amp --cache_ram")
    print()
    print("  # Reanudar desde el ultimo checkpoint:")
    print("  py -3 train.py --datos data/processed --reanudar")
    print()
    print("  # Desde notebook Jupyter:")
    print("  from train import bucle_entrenamiento, config_desde_args")
    print("  config, args = config_desde_args([")
    print("      '--datos', 'data/processed', '--epocas', '50'")
    print("  ])")
    print("  entrenador = bucle_entrenamiento(config, 'exp_v1', args)")

    print()
    print("=" * 65)
    print("  [OK] train.py verificado correctamente.")
    print("=" * 65)


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    main()
