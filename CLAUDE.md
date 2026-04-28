Contexto del Proyecto:
Desarrollo de código para un blog técnico universitario sobre "Traducción Bidireccional entre Imágenes Satelitales y Bocetos Cartográficos".
La arquitectura principal es una U-Net 256 combinada con un discriminador PatchGAN (basado en Pix2Pix).

Limitaciones de Infraestructura (Crítico):
Este proyecto se está programando localmente, pero TODO el entrenamiento y ejecución pesada se hará en Google Colab (GPU T4 gratuita, baja VRAM).

El código debe estar optimizado para bajo consumo de memoria gráfica (usar DataLoader eficientes de PyTorch, limpiar cachés, etc.).

Priorizar la creación de código modular en Python y cuadernos Jupyter (.ipynb) listos para subir a Colab.

Referencias Obligatorias (Estado del Arte):
Cuando escribas la arquitectura o utilices funciones de otros repositorios, DEBES añadir comentarios referenciando la fuente original. Las bases del proyecto son:

Arquitectura U-Net/Pix2Pix: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

Referencia de mapas a satélite: https://github.com/PerlMonker303/S2MP (Sketch2Map)

Modelos de difusión (para futuras discusiones): https://github.com/miquel-espinosa/map-sat y https://github.com/RubenGres/Seg2Sat

Directrices de Código y Estilo:
Stack: Python 3.10+ y PyTorch 2.0+ (compatible con el entorno actual de Colab).

Entrada/Salida: Tensores de imagen de tamaño 256x256x3.

Funciones de pérdida (Losses): Al implementar la cGAN, incluye siempre la pérdida L1 (L1 Loss) combinada con la Adversarial Loss (GAN Loss), utilizando típicamente lambda = 100 para la pérdida L1.

Idioma de los comentarios: Estrictamente en español y muy didácticos (explicando el "por qué" de cada bloque matemático), ya que el código acompañará a un blog educativo.

Reglas de Verificación Local (Feedback Loop)
Antes de dar un script de red neuronal por válido, DEBES incluir un bloque if __name__ == "__main__": al final.

Este bloque debe generar tensores "dummy" (aleatorios) de tamaño  y pasarlos por el modelo (forward pass) para imprimir las dimensiones de salida. Esto permite comprobar localmente que no hay errores matemáticos de dimensión antes de subir el código a Google Colab.