# TFM — Optimización de modelos de IA mediante técnicas de Edge AI para hardware de bajo consumo

Trabajo Final de Máster — Máster Universitario en Inteligencia Artificial (UNIR)
Autor: Pablo Federico Martín Luna
Director: María Aurora Martínez Rey
Tipo de trabajo: Comparativa de soluciones

## Descripción

Comparativa sistemática de tres técnicas de optimización Edge AI —cuantización INT8,
podado no estructurado por magnitud y destilación de conocimiento— aplicadas sobre un
autoencoder convolucional (Conv-AE) entrenado para la detección preliminar de patrones
anómalos en series temporales de constantes vitales, con validación final sobre el
microcontrolador NodeMCU ESP8266.

Este trabajo es la continuación directa del Trabajo Final de Grado *Terminal Punto de
Salud* (Martín Luna, 2024, Universidad Francisco de Vitoria), que desarrolló un
dispositivo de monitorización de constantes vitales sobre NodeMCU ESP8266 sin capacidad
de análisis local inteligente.

**Nota importante:** el módulo desarrollado es un sistema experimental de detección
preliminar de patrones anómalos. No ha sido validado clínicamente y no debe emplearse
para la toma de decisiones médicas reales.

## Resultados principales

| Técnica | Tamaño TFLite (KB) | RAM estimada (KB) | F1 | Cumple restricciones |
|---|---|---|---|---|
| Baseline (float32) | 8,89 | 17,79 | 0,5942 | — |
| Cuantización INT8 | 24,65 | 49,30 | 0,5942 | Sí |
| Podado 50% + INT8 | 24,65 | 49,30 | 0,5816 | Sí |
| **Destilación + INT8** | **17,92** | **35,84** | **0,5496** | **Sí (técnica seleccionada)** |

Validación real en hardware físico (NodeMCU ESP8266, técnica de destilación + INT8):

- **Latencia media de inferencia:** 30,80 ms (límite: 500 ms)
- **RAM libre tras carga del modelo:** 2.688 bytes
- **Resolver:** `MicroMutableOpResolver<11>` (11 operadores reales del modelo)

Detalle completo de metodología y resultados en la memoria del TFM (no incluida en este
repositorio por contener datos derivados de MIMIC-III bajo licencia restringida de cita).

## Dataset

Este trabajo emplea **MIMIC-III Clinical Database Demo v1.4**, un subconjunto público de
100 pacientes de la base de datos MIMIC-III.

- **Acceso:** requiere registro y acreditación en PhysioNet:
  https://physionet.org/content/mimiciii-demo/1.4/
- **Licencia:** ODC Open Database License (ODbL) v1.0
- **Importante:** en cumplimiento de la licencia, este repositorio **no redistribuye**
  ningún fichero original de MIMIC-III. Solo se incluye código de procesamiento y
  artefactos derivados (modelos entrenados, ficheros `.tflite`) que no permiten
  reconstruir los datos clínicos originales.

Para reproducir este trabajo, cada usuario debe descargar el dataset directamente desde
PhysioNet tras completar el proceso de acreditación (CITI Data or Specimens Only
Research course).

## Estructura del repositorio

```
tfm-edge-ai-esp8266/
├── notebooks/
│   └── codigo_fase1_fase2.ipynb        # Notebook unificado (EDA, Fase 1, Fase 2)
├── modelos/
│   ├── conv_ae_baseline.keras          # Modelo base entrenado (formato Keras)
│   ├── conv_ae_baseline_f32.tflite     # Baseline exportado a TFLite float32
│   ├── tecnica1_int8.tflite            # Cuantización INT8
│   ├── tecnica2_pruning_int8.tflite    # Podado no estructurado 50% + INT8
│   └── tecnica3_distillation_int8.tflite  # Destilación + INT8 (técnica seleccionada)
├── esp8266/
│   ├── esp8266_benchmark.ino           # Sketch de benchmark de latencia
│   └── conv_ae_model_data.h            # Modelo serializado como array C (sin PROGMEM)
├── resultados/
│   ├── metricas_baseline.json
│   ├── config_modelo_base.json
│   ├── B9_comparativa_arquitecturas.json
│   └── resultados_fase2.json
├── figuras/                             # Gráficas generadas (.png)
├── requirements.txt
├── LICENSE
└── README.md
```

## Requisitos de instalación

### Entorno Python (notebook)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Ver `requirements.txt` para versiones exactas. Entrenado y probado originalmente en
Kaggle (Python 3.12, TensorFlow 2.18.0).

### Entorno Arduino (despliegue ESP8266)

1. Arduino IDE 2.x
2. Board manager: paquete `esp8266` (Espressif/Arduino community), versión ≥ 3.1.2
3. Placa: `Generic ESP8266 Module` (o `NodeMCU 1.0`, `ESP-12E Module`)
4. Librería: `Chirale_TensorFlowLite`, instalable desde el gestor de librerías de
   Arduino IDE

**Nota técnica importante:** la librería oficial `esp-tflite-micro` de Espressif **no es
compatible con ESP8266** (requiere ESP-IDF y solo soporta arquitecturas ESP32/S/C). Este
proyecto emplea `Chirale_TensorFlowLite` como alternativa viable, con las adaptaciones
descritas en el Anexo D de la memoria (resolver de operadores acotado, ausencia de
`PROGMEM`, gestión del *Soft WDT reset*).

## Cómo reproducir el trabajo

1. **Obtener el dataset.** Registrarse en PhysioNet y descargar
   `mimic-iii-clinical-database-demo-1.4`.
2. **Ejecutar el notebook.** Abrir `notebooks/codigo_fase1_fase2.ipynb` en Kaggle o
   Jupyter, añadiendo el dataset como input. Ejecutar todas las celdas en orden con
   `SEED=42` (ya configurado). La Fase 1 entrena el modelo base; la Fase 2 aplica las
   tres técnicas y genera todos los artefactos, incluidos los `.tflite` y el sketch de
   Arduino.
3. **Verificar resultados.** Comparar `resultados/resultados_fase2.json` con los valores
   de la Tabla 3 de la memoria.
4. **Desplegar en ESP8266.**
   - Copiar `esp8266/esp8266_benchmark.ino` y `esp8266/conv_ae_model_data.h` en una
     carpeta local llamada `esp8266_benchmark` (mismo nombre que el `.ino` principal,
     requisito de Arduino IDE).
   - Abrir con Arduino IDE, seleccionar la placa y compilar.
   - Si el auto-reset falla (frecuente si hay periféricos conectados en el bus serie),
     usar el modo manual: mantener pulsado `FLASH`, pulsar y soltar `RST`, soltar
     `FLASH`, e iniciar la subida inmediatamente.
   - Abrir el monitor serie a 115.200 baudios para ver la latencia medida.

## Reproducibilidad

Todas las operaciones estocásticas (inicialización de pesos, orden de entrenamiento,
selección del conjunto de calibración para cuantización, generación de anomalías
sintéticas) usan la semilla fija `SEED=42`, configurada globalmente al inicio del
notebook:

```python
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

La reproducibilidad exacta está garantizada dentro del mismo entorno Kaggle. En entornos
locales pueden existir diferencias menores derivadas de versión de CUDA u operaciones no
deterministas de TensorFlow con GPU.

## Limitaciones conocidas

- Dataset reducido (100 pacientes, todos fallecidos): ver Sección 6.4 de la memoria.
- Ausencia de etiquetas de anomalía reales; evaluación mediante inyección de fallos
  sintéticos (ver Anexo C).
- El proxy de deterioro clínico (DOD) es un análisis exploratorio, no una validación
  clínica (ver Sección 6.4).
- `tensorflow_model_optimization` (TFMOT) es incompatible con Keras 3 en el entorno
  usado; el podado se implementó mediante enmascaramiento manual (ver Sección 5.1.2).

## Licencia del código

El código de este repositorio se distribuye bajo licencia MIT (ver `LICENSE`). Esto no
afecta a la licencia propia de MIMIC-III (ODbL), que rige el dataset y no se redistribuye
aquí.

## Citar este trabajo

```
Martín Luna, P. F. (2026). Optimización de modelos de Inteligencia Artificial mediante
técnicas de Edge AI para su ejecución en hardware de bajo consumo: una comparativa de
soluciones [Trabajo Final de Máster, Universidad Internacional de La Rioja].
```

## Contacto

Pablo Federico Martín Luna — [correo de contacto]
