"""
networks.py — Funciones factory y utilidades de inicialización
===============================================================

Este módulo centraliza la creación e inicialización de los modelos, siguiendo
el patrón del repositorio original de Pix2Pix. Actúa como punto de entrada
único para instanciar el generador y el discriminador listos para entrenar.

¿Por qué inicializar los pesos explícitamente?
----------------------------------------------
PyTorch inicializa los pesos de Conv2d y ConvTranspose2d con la distribución
de Kaiming/He por defecto, que está optimizada para redes con ReLU y sin
normalización. Sin embargo, en una GAN:

1. Los pesos iniciales afectan crítica mente al equilibrio D vs G en las
   primeras iteraciones. Kaiming puede producir activaciones demasiado grandes.
2. El paper Pix2Pix usa inicialización Normal con std=0.02, que produce
   activaciones más pequeñas y un arranque más suave del entrenamiento.
3. La inicialización de BatchNorm/InstanceNorm con weight=1 y bias=0 garantiza
   que la normalización empiece como una identidad (no perturba el flujo inicial).

Referencia: Isola et al., CVPR 2017.
Código base: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
(ver models/networks.py, función init_weights)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from src.models.generator import GeneradorUNet
from src.models.discriminator import DiscriminadorPatchGAN


def obtener_dispositivo() -> torch.device:
    """
    Detecta automáticamente el mejor dispositivo disponible:
        1. CUDA (GPU NVIDIA) — ideal para Google Colab con T4
        2. MPS (GPU Apple Silicon) — para desarrollo local en Mac
        3. CPU — fallback siempre disponible

    Returns:
        torch.device con el dispositivo seleccionado.
    """
    if torch.cuda.is_available():
        dispositivo = torch.device("cuda")
        nombre_gpu = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] Usando CUDA: {nombre_gpu} ({vram_total:.1f} GB VRAM)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dispositivo = torch.device("mps")
        print("[GPU] Usando Apple Silicon MPS")
    else:
        dispositivo = torch.device("cpu")
        print("[CPU] CUDA no disponible. Usando CPU (entrenamiento será lento).")
    return dispositivo


def inicializar_pesos(modelo: nn.Module, ganancia: float = 0.02) -> None:
    """
    Inicializa los pesos de capas Conv2d, ConvTranspose2d e InstanceNorm2d.

    Estrategia:
    - Conv2d y ConvTranspose2d: distribución Normal con media=0 y std=ganancia.
      Pesos pequeños (0.02) evitan saturación prematura en las primeras épocas.
    - InstanceNorm2d: weight=1, bias=0 (inicialización identidad).
      Esto asegura que al inicio la normalización no distorsiona las activaciones.

    La función se aplica recursivamente a todos los submódulos del modelo
    usando model.apply(fn), que visita cada módulo en profundidad.

    Args:
        modelo:   El modelo de red neuronal (generador o discriminador).
        ganancia: Desviación estándar de la distribución Normal. Default: 0.02.
    """
    def _inicializar_capa(capa: nn.Module) -> None:
        nombre_clase = capa.__class__.__name__

        if nombre_clase in ("Conv2d", "ConvTranspose2d"):
            # Inicialización Normal truncada: mejora estabilidad vs Normal pura
            nn.init.normal_(capa.weight.data, mean=0.0, std=ganancia)
            if capa.bias is not None:
                nn.init.constant_(capa.bias.data, 0.0)

        elif nombre_clase == "InstanceNorm2d":
            if capa.weight is not None:
                # weight=1: la normalización empieza como identidad (sin escala)
                nn.init.constant_(capa.weight.data, 1.0)
            if capa.bias is not None:
                # bias=0: sin desplazamiento inicial
                nn.init.constant_(capa.bias.data, 0.0)

    # apply() visita recursivamente todos los submódulos y aplica _inicializar_capa
    modelo.apply(_inicializar_capa)


def construir_generador(
    dispositivo: Optional[torch.device] = None,
    canales_entrada: int = 3,
    canales_salida: int = 3,
    filtros_base: int = 64,
) -> GeneradorUNet:
    """
    Construye, inicializa y mueve el generador U-Net al dispositivo indicado.

    Args:
        dispositivo:    Dispositivo de cómputo. Si None, se detecta automáticamente.
        canales_entrada: Canales de la imagen de entrada. Default: 3 (RGB).
        canales_salida:  Canales de la imagen generada. Default: 3 (RGB).
        filtros_base:    Filtros de la primera capa. Default: 64.

    Returns:
        GeneradorUNet inicializado y listo para entrenar en el dispositivo.
    """
    if dispositivo is None:
        dispositivo = obtener_dispositivo()

    generador = GeneradorUNet(
        canales_entrada=canales_entrada,
        canales_salida=canales_salida,
        filtros_base=filtros_base,
    )

    # Inicializar pesos antes de mover al dispositivo (más eficiente en CPU)
    inicializar_pesos(generador, ganancia=0.02)

    generador = generador.to(dispositivo)

    total_params = sum(p.numel() for p in generador.parameters())
    print(f"[Generador] Inicializado. Parámetros: {total_params:,} (~{total_params/1e6:.1f}M)")

    return generador


def construir_discriminador(
    dispositivo: Optional[torch.device] = None,
    canales_entrada: int = 6,
    filtros_base: int = 64,
) -> DiscriminadorPatchGAN:
    """
    Construye, inicializa y mueve el discriminador PatchGAN al dispositivo indicado.

    Args:
        dispositivo:    Dispositivo de cómputo. Si None, se detecta automáticamente.
        canales_entrada: Canales de entrada. Default: 6 (3 condición + 3 objetivo).
        filtros_base:    Filtros de la primera capa. Default: 64.

    Returns:
        DiscriminadorPatchGAN inicializado y listo para entrenar en el dispositivo.
    """
    if dispositivo is None:
        dispositivo = obtener_dispositivo()

    discriminador = DiscriminadorPatchGAN(
        canales_entrada=canales_entrada,
        filtros_base=filtros_base,
    )

    inicializar_pesos(discriminador, ganancia=0.02)
    discriminador = discriminador.to(dispositivo)

    total_params = sum(p.numel() for p in discriminador.parameters())
    print(f"[Discriminador] Inicializado. Parámetros: {total_params:,} (~{total_params/1e6:.1f}M)")

    return discriminador


def congelar_modelo(modelo: nn.Module) -> None:
    """
    Congela todos los parámetros del modelo (requires_grad=False).

    Uso: congelar el generador mientras se actualiza el discriminador,
    y viceversa. Esto previene que los gradientes fluyan hacia el modelo
    que NO debe actualizarse en ese paso.

    Referencia: Esta técnica se usa en el backward_D de Pix2Pix para
    evitar calcular gradientes del generador al actualizar el discriminador.
    """
    for param in modelo.parameters():
        param.requires_grad = False


def descongelar_modelo(modelo: nn.Module) -> None:
    """
    Descongela todos los parámetros del modelo (requires_grad=True).
    """
    for param in modelo.parameters():
        param.requires_grad = True


# ===========================================================================
# BLOQUE DE VERIFICACIÓN LOCAL
# Ejecutar: python src/models/networks.py
# Verifica que G y D se construyen correctamente y los pesos están inicializados.
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Verificación local: networks.py")
    print("=" * 60)

    # Usar CPU para verificación local (no requiere GPU)
    dispositivo = torch.device("cpu")
    print(f"\nDispositivo seleccionado: {dispositivo}")

    print("\n--- Construyendo Generador ---")
    G = construir_generador(dispositivo=dispositivo)

    print("\n--- Construyendo Discriminador ---")
    D = construir_discriminador(dispositivo=dispositivo)

    # Verificar que los modelos están en el dispositivo correcto
    g_device = next(G.parameters()).device
    d_device = next(D.parameters()).device
    print(f"\n[Verificación de dispositivo]")
    print(f"  Generador en    : {g_device}")
    print(f"  Discriminador en: {d_device}")

    # Verificar forward pass completo G → D
    print("\n--- Forward pass completo: G → D ---")
    imagen_entrada = torch.randn(1, 3, 256, 256).to(dispositivo)

    G.eval()
    D.eval()
    with torch.no_grad():
        imagen_generada = G(imagen_entrada)
        salida_d = D(imagen_entrada, imagen_generada)

    print(f"  imagen_entrada  : {imagen_entrada.shape}")
    print(f"  imagen_generada : {imagen_generada.shape}")
    print(f"  salida_D        : {salida_d.shape}")

    # Verificar congelar/descongelar
    print("\n--- Verificando congelar/descongelar ---")
    congelar_modelo(G)
    assert not any(p.requires_grad for p in G.parameters()), "Error: G debería estar congelado"
    descongelar_modelo(G)
    assert all(p.requires_grad for p in G.parameters()), "Error: G debería estar descongelado"
    print("  congelar_modelo / descongelar_modelo: [OK]")

    assert imagen_generada.shape == torch.Size([1, 3, 256, 256])
    assert salida_d.shape == torch.Size([1, 1, 30, 30])

    print("\n[OK] networks.py verificado correctamente.")
