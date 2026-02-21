# Sistema de Reconocimiento Automático de Matrículas Vehiculares Argentinas Mediante Visión por Computadora y OCR en Python

Instituto Superior De Formación Técnica N°130.
Tecnicatura Superior En Análisis De Sistemas.
Algoritmos y Estructuras De Datos III.


## 1. Resumen

El presente proyecto desarrolla un sistema de Reconocimiento Automático de Matrículas mediante técnicas de Visión por Computador.

El sistema implementa un pipeline completo que abarca:

* Detección de matrículas mediante un modelo YOLOv8 entrenado.
* Recorte y preprocesamiento de la región detectada.
* Reconocimiento de caracteres utilizando Fast Plate OCR.
* Postprocesado de texto.
* Evaluación cuantitativa mediante métricas de detección y reconocimiento.
* Interfaz de menú interactiva para facilitar el uso del sistema.
* Procesamiento en tiempo real desde webcam.
* Visualización de resultados con anotaciones sobre las imágenes.

Se garantiza reproducibilidad mediante scripts organizados y entorno controlado.


## 2. Objetivo

Diseñar e implementar un sistema capaz de:

1. Detectar automáticamente la región de la matrícula en imágenes o webcam.
2. Extraer el texto completo de la placa.
3. Evaluar el desempeño utilizando métricas detección y OCR.
4. Presentar resultados cuantificables y analizables.


## 3. Arquitectura del Sistema

El flujo lógico implementado es el siguiente:

```
Imagen / Webcam
      ↓
Detección (YOLOv8)
      ↓
Recorte de región
      ↓
Preprocesamiento
      ↓
OCR (Fast Plate OCR)
      ↓
Postprocesamiento
      ↓
Texto final
```


## 4. Tecnologías Utilizadas

* Python 3.10+
* Ultralytics YOLOv8
* Fast Plate OCR (ONNX Runtime)
* OpenCV
* NumPy
* Tkinter (interfaz gráfica)
* Python-Levenshtein (métricas de evaluación)
* PyTorch (soporte GPU)
* psutil (monitoreo de recursos)


## 5. Estructura del Proyecto

```
project/
├── data/
│	├── ocr/
│	├── train/
│	├── val/
│	└── test/
├── models/
│   └── weights/
│   	└── best.pt
├── src/
│   ├── train.py
│   ├── detect.py
│   ├── ocr.py
│   ├── evaluate.py
│   ├── main.py
│   └── dataset.py
├── requirements.txt
└── README.md

```


## 6. Instalación y Reproducibilidad

Clonar repositorio:

```bash
git clone https://github.com/STS130/reconocimiento-matriculas.git
cd reconocimiento-matriculas
```

Crear entorno virtual (Python 3.10/11 recomendado):

```bash
python -m venv venv
```

Activar entorno:

```bash
venv\Scripts\activate
```

Instalar dependencias:

```bash
python -m pip install -r requirements.txt
```


## 7. Ejecución

### Interfaz principal (recomendado)

El sistema incluye una interfaz de menú interactiva que permite acceder a todas las funcionalidades:

```bash
python src/main.py
```

El menú ofrece las siguientes opciones:

1. **Procesar imagen**: Selecciona una imagen individual para detectar y reconocer matrículas.
2. **Procesar carpeta**: Procesa todas las imágenes en una carpeta seleccionada.
3. **Webcam**: Procesa en tiempo real desde la cámara web (presiona ESC para salir).
4. **Evaluar modelo**: Ejecuta la evaluación completa del modelo con todas las métricas.
5. **Salir**: Cierra la aplicación.


## 8. Métricas Implementadas

### Detección

* Precision
* Recall
* F1-score
* mAP@0.5
* IoU promedio

### Reconocimiento OCR

* CER (Character Error Rate)
* Accuracy por matrícula completa

### Rendimiento

* Latencia promedio por imagen (lote)
* Latencia promedio por frame (webcam)
* Uso de memoria RAM
* Uso de memoria GPU (si está disponible)

## 9. Autores

* S. Tomás Leira, Simon Ortiz, Santiago Villeres
* Instituto Superior De Formación Técnica N°130
* Tecnicatura Superior En Análisis De Sistemas
* Algoritmos y Estructuras De Datos III
* Docente: Pablo Letier
* 23 de febrero de 2026