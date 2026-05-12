# Traducción Bidireccional de Imágenes Satelitales — Pix2Pix

Implementación de una red cGAN (Pix2Pix) para traducción imagen a imagen entre fotografía aérea/satelital y mapas cartográficos (OpenStreetMap). El modelo funciona en ambas direcciones:

- **A→B**: Satelital → Mapa de carreteras
- **B→A**: Mapa de carreteras → Satelital (síntesis de textura)

Toda la arquitectura está documentada en español con explicaciones matemáticas en los notebooks.

---

## Arquitectura

| Componente | Descripción |
|---|---|
| **Generador** | U-Net 256 con 8 niveles de skip connections, Dropout en 3 bloques centrales, activación Tanh |
| **Discriminador** | PatchGAN 70×70 — clasifica parches locales en lugar de la imagen completa |
| **Pérdida** | LS-GAN + L1: `L = L_GAN + λ·L_L1` con λ=100 (AtoB) / λ=10 (BtoA) |
| **Entrenamiento** | Adam (lr=2e-4, β₁=0.5), Mixed Precision (AMP), gradient accumulation ×4 |

---

## Estructura del Repositorio

```
├── src/
│   ├── models/          # Generador U-Net, Discriminador PatchGAN, funciones de pérdida
│   ├── data/            # Dataset loader, transforms, descarga del dataset
│   ├── training/        # Trainer, configuración, checkpointing
│   ├── inference/       # Predicción single/batch, métricas SSIM/L1
│   └── utils/           # Logger, visualización, utilidades Colab
├── notebooks/
│   ├── 00_setup_colab.ipynb            # Verificar entorno e instalar dependencias
│   ├── 01_data_exploration.ipynb       # EDA: visualizar pares satelital/mapa
│   ├── 02_architecture_demo.ipynb      # Forward pass didáctico con tensores dummy
│   ├── 03_training_sat2sketch.ipynb    # Entrenamiento A→B
│   ├── 04_training_sketch2sat.ipynb    # Entrenamiento B→A
│   ├── 05_inference_demo.ipynb         # Demo interactivo con imágenes propias
│   └── 06_results_analysis.ipynb       # Métricas, curvas de pérdida, conclusiones
├── train.py             # Script principal de entrenamiento (CLI)
├── requirements.txt
└── .gitignore
```

Los directorios `data/`, `checkpoints/` y `results/` están excluidos del repositorio por tamaño. Se generan automáticamente al ejecutar el código.

---

## Opción A — Google Colab (recomendado, GPU gratuita)

La forma más sencilla de ejecutar el proyecto sin instalar nada localmente. Colab proporciona una GPU T4 gratuita suficiente para entrenar el modelo completo.

### Pasos

**1. Abrir el notebook de configuración**

En Google Colab, abre `notebooks/00_setup_colab.ipynb`. Este notebook:
- Verifica que hay GPU disponible y estima los tiempos de entrenamiento
- Monta Google Drive (para guardar checkpoints de forma persistente)
- Clona el repositorio y sitúa el entorno correctamente
- Instala todas las dependencias automáticamente

Para clonar el repositorio desde dentro de Colab:

```python
!git clone https://github.com/djfuga/Traduccion-Imagenes-Satelite-PASD.git
%cd Traduccion-Imagenes-Satelite-PASD
```

**2. Descargar el dataset**

Ejecuta la celda de descarga del notebook 00, o directamente:

```python
!python src/data/download_maps.py
```

Esto descarga el dataset Berkeley Maps (~255 MB) y lo coloca en `data/processed/train/` y `data/processed/val/` (1096 + 1098 pares de imágenes).

**3. Ejecutar los notebooks en orden**

| Notebook | Qué hace | Duración estimada |
|---|---|---|
| `00_setup_colab.ipynb` | Configura el entorno | 2-3 min |
| `01_data_exploration.ipynb` | Explora el dataset | < 1 min |
| `02_architecture_demo.ipynb` | Verifica la arquitectura | < 1 min |
| `03_training_sat2sketch.ipynb` | Entrena A→B (satelital→mapa) | 2.5 h aprox. en T4 |
| `04_training_sketch2sat.ipynb` | Entrena B→A (mapa→satelital) | 2.5 h aprox. en T4 |
| `05_inference_demo.ipynb` | Genera traducciones con el modelo | < 5 min |
| `06_results_analysis.ipynb` | Métricas SSIM/L1, curvas de pérdida | < 5 min |

> **Consejo para sesiones cortas**: Colab puede desconectarse. El entrenamiento guarda un checkpoint cada 10 épocas en `checkpoints/`. Para reanudar, descomenta la línea `# '--reanudar'` en la celda de configuración del notebook 03 o 04 antes de relanzar.

> **Persistencia con Drive**: Los checkpoints se guardan localmente en la sesión de Colab. Para no perderlos si la sesión expira, monta Drive desde el notebook 00 y copia los checkpoints allí periódicamente, o usa la opción de guardar en Drive del trainer.

---

## Opción B — Ejecución Local

Requiere Python 3.10+ y una GPU con al menos 4 GB de VRAM (recomendado), aunque también funciona en CPU para inferencia.

### Requisitos previos

- Python 3.10 o superior
- CUDA 11.8+ (opcional, para entrenamiento con GPU)
- ~3 GB de espacio en disco (dataset + checkpoints)

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/djfuga/Traduccion-Imagenes-Satelite-PASD.git
cd Traduccion-Imagenes-Satelite-PASD

# 2. Crear entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

> **Solo CPU (sin CUDA)**: El código detecta automáticamente si hay GPU disponible y usa CPU como fallback. El entrenamiento en CPU es viable para pruebas rápidas pero no para las 200 épocas completas.

### Descargar el dataset

```bash
python src/data/download_maps.py
```

Descarga el dataset Berkeley Maps desde el servidor oficial de Pix2Pix y lo sitúa en `data/processed/`.

### Verificar que todo funciona

Antes de entrenar, comprueba que los modelos y el pipeline de datos son correctos:

```bash
python src/models/generator.py       # Debe imprimir: (1, 3, 256, 256)
python src/models/discriminator.py   # Debe imprimir: (1, 1, 30, 30)
python train.py --verificar          # Forward pass completo sin entrenamiento
```

### Entrenamiento

```bash
# Dirección A→B: satelital → mapa de carreteras
python train.py \
    --datos data/processed \
    --direction AtoB \
    --epocas 100 \
    --epocas_decay 100 \
    --amp

# Dirección B→A: mapa → satelital  (lambda_l1=10 evita el mode collapse)
python train.py \
    --datos data/processed \
    --direction BtoA \
    --epocas 100 \
    --epocas_decay 100 \
    --lambda_l1 10 \
    --amp

# Reanudar desde el último checkpoint
python train.py --datos data/processed --direction AtoB --reanudar
```

Parámetros disponibles:

| Argumento | Valor por defecto | Descripción |
|---|---|---|
| `--datos` | `data/processed` | Directorio con `train/` y `val/` |
| `--direction` | `AtoB` | Dirección de traducción (`AtoB` o `BtoA`) |
| `--epocas` | `100` | Épocas con LR constante |
| `--epocas_decay` | `100` | Épocas con LR decreciente hasta 0 |
| `--batch` | `1` | Tamaño de batch (1 es el estándar del paper) |
| `--lr` | `0.0002` | Learning rate inicial |
| `--lambda_l1` | `100` | Peso de la pérdida L1 (usar `10` para B→A) |
| `--gan_mode` | `lsgan` | Tipo de GAN loss (`lsgan` o `vanilla`) |
| `--grad_accum` | `4` | Pasos de acumulación de gradiente |
| `--amp` | `False` | Activar Mixed Precision (requiere CUDA) |
| `--frecuencia_ckpt` | `10` | Guardar checkpoint cada N épocas |
| `--reanudar` | `False` | Continuar desde el último checkpoint |

### Inferencia sobre imágenes propias

Una vez entrenado el modelo, usa el notebook `05_inference_demo.ipynb` o directamente:

```python
from src.inference.predict import cargar_generador, predecir_imagen

G = cargar_generador('checkpoints/sat2sketch/ckpt_AtoB_epoca_0200.pth')
imagen_generada = predecir_imagen('mi_imagen_satelital.jpg', G)
```

---

## Opción C — Solo Inferencia (sin reentrenar)

Si solo quieres probar el modelo con tus propias imágenes sin entrenar desde cero, descarga los pesos preentrenados desde la página de [Releases](https://github.com/djfuga/Traduccion-Imagenes-Satelite-PASD/releases/latest):

| Archivo | Dirección | Tamaño |
|---------|-----------|--------|
| `generador_AtoB_ep200.pth` | Satelital → Mapa | ~208 MB |
| `generador_BtoA_ep200.pth` | Mapa → Satelital | ~208 MB |

Son checkpoints reducidos (solo pesos del generador, sin optimizador ni discriminador).

1. Descarga uno o ambos archivos y colócalos en:
   - `checkpoints/sat2sketch/generador_AtoB_ep200.pth`
   - `checkpoints/sketch2sat/generador_BtoA_ep200.pth`
2. Abre `notebooks/05_inference_demo.ipynb`
3. Ejecuta todas las celdas — el notebook detecta automáticamente el checkpoint más reciente

El notebook acepta cualquier imagen satelital en formato JPG/PNG y genera su mapa correspondiente.

---

## Resultados

Entrenamiento de 200 épocas (100 LR constante + 100 LR decay) sobre el dataset Berkeley Maps con GPU T4 en Google Colab.

| Dirección | SSIM ↑ | L1 ↓ | PSNR ↑ | N |
|-----------|--------|------|--------|---|
| **A→B** Satelital → Mapa | 0.794 ± 0.057 | 0.026 ± 0.012 | 27.60 ± 3.71 dB | 200 |
| **B→A** Mapa → Satelital | 0.218 ± 0.070 | 0.135 ± 0.026 | 15.15 ± 1.94 dB | 200 |

> Referencia Isola et al. (2017): SSIM ≈ 0.49 en A→B sobre este mismo dataset.

**Nota sobre B→A**: las métricas pixel-wise (SSIM/PSNR) son intencionalmente bajas. Usar `λ_L1=100` en esta dirección produce *mode collapse* (imágenes uniformemente oscuras) que infla artificialmente el SSIM. Con `λ_L1=10` el generador aprende texturas satelitales reales a costa de mayor distorsión pixel a pixel — la calidad perceptual es superior a pesar de las métricas numéricas peores (*perception-distortion tradeoff*, Blau & Michaeli 2018).

---

## Dataset

**Berkeley Maps** — Isola et al. (2017)

- 1 096 pares de entrenamiento + 1 098 pares de validación
- Formato side-by-side: cada archivo es una imagen de 1200×600 px (mitad izquierda = satelital, mitad derecha = mapa OSM)
- Ciudades de Estados Unidos capturadas con Google Maps

El script `src/data/download_maps.py` descarga y procesa el dataset automáticamente.

---

## Requisitos del sistema

| Componente | Mínimo | Recomendado |
|---|---|---|
| Python | 3.10 | 3.11 |
| PyTorch | 2.0 | 2.1+ |
| GPU VRAM | 4 GB (con AMP) | 8 GB |
| RAM | 8 GB | 16 GB |
| Disco | 3 GB | 5 GB |
| CUDA | 11.8 | 12.1 |

En Google Colab con T4 gratuita (15 GB VRAM) el modelo usa ~2.5 GB con Mixed Precision activado.

---

## Referencias

- **Pix2Pix** — Isola et al. (2017): *Image-to-Image Translation with Conditional Adversarial Networks* · [Paper](https://arxiv.org/abs/1611.07004) · [Código original](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- **Sketch2Map** — [PerlMonker303/S2MP](https://github.com/PerlMonker303/S2MP)
- **Map-Sat** (difusión) — [miquel-espinosa/map-sat](https://github.com/miquel-espinosa/map-sat)
- **Seg2Sat** (difusión) — [RubenGres/Seg2Sat](https://github.com/RubenGres/Seg2Sat)
