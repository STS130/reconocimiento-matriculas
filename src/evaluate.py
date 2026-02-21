'''
Este módulo calcula las métricas de rendimiento del sistema completo.
Evalúa tanto la calidad de la detección (precision, recall, F1, IoU, mAP@0.5) como la exactitud del OCR (CER y tasa de acierto).
Además, mide latencias y consumo de recursos, proporcionando una visión integral del desempeño.
'''

from ultralytics import YOLO
from ocr import OCR
import cv2
import os
import time
import psutil
import torch
import numpy as np
from Levenshtein import distance as distancia_levenshtein

RUTA_MODELO = "models/weights/best.pt"
NOMBRE_MODELO_OCR = "cct-xs-v1-global-model"

RUTA_DATOS = "data/data.yaml"
RUTA_IMAGENES_TEST = "data/test/images"

RUTA_LABELS_TEST = "data/test/labels"

RUTA_IMAGENES_OCR = "data/ocr/images"
RUTA_LABELS_OCR = "data/ocr/labels.txt"

class Evaluador:
    '''
    Esta clase encapsula la lógica para evaluar el rendimiento del sistema de detección y OCR,
    incluyendo la carga de modelos, el cálculo de métricas y la medición de latencias y recursos.
    '''
    def __init__(self):
        self.modelo = YOLO(RUTA_MODELO)
        self.modelo_ocr = OCR(NOMBRE_MODELO_OCR)

    def evaluar(self):
        '''
        Este método ejecuta la evaluación completa del sistema, llamando a métodos privados para calcular métricas de detección, exactitud de OCR, latencias y consumo de recursos, y mostrando los resultados de manera clara y estructurada.
        '''
        precision, recall, f1, mapa50 = self._evaluar_deteccion()
        print(f"mAP@0.5: {mapa50:.4f}")
        print(f"Precisión: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")
        
        iou_media = self._calcular_iou_media()
        print(f"IoU media: {iou_media:.4f}")

        error_caracter, precision_ocr, contador = self._evaluar_ocr()
        print(f"OCR - imágenes procesadas: {contador}")
        print(f"CER: {error_caracter:.4f}")
        print(f"Precisión: {precision_ocr:.4f}")

        latencia = self._medir_latencia_lote()
        print(f"Latencia promedio por imagen: {latencia:.2f} ms")

        latencia_webcam = self._medir_latencia_webcam()

        if latencia_webcam:
            print(f"Latencia promedio webcam: {latencia_webcam:.2f} ms/frame")
        else:
            print("No se pudo medir latencia webcam")

        ram, gpu = self._medir_recursos()
        print(f"RAM usada: {ram:.2f} MB")
        if gpu:
            print(f"Memoria GPU usada: {gpu:.2f} MB")
        else:
            print("GPU no disponible")

    def _evaluar_deteccion(self):
        '''
        Este método utiliza la funcion .val del modelo YOLO para evaluar el rendimiento de detección en el conjunto de prueba, calculando métricas como precision, recall, F1 y mAP@0.5.
        '''
        resultados = self.modelo.val(data=RUTA_DATOS, split='test', verbose=False)
        precision = resultados.box.mp
        recall = resultados.box.mr
        f1 = 2 * precision * recall / (precision + recall)
        mapa50 = resultados.box.map50

        return precision, recall, f1, mapa50

    def _calcular_iou_media(self):
        '''
        Este método calcula la IoU media entre las predicciones del modelo de detección y las anotaciones de ground truth en el conjunto de prueba, proporcionando una medida adicional de la calidad de las detecciones.
        El cálculo se realiza de la siguiente manera:
        - Se itera sobre cada imagen del conjunto de prueba, cargando la imagen y sus etiquetas correspondientes.
        - Para cada imagen, se extraen las cajas de ground truth a partir de las etiquetas, y las cajas de predicción a partir de los resultados del modelo.
        - Para cada caja de predicción, se calcula la IoU con cada caja de ground truth, guardando la mejor IoU obtenida.
        - Finalmente, se calcula la media de las mejores IoU obtenidas para todas las predicciones, devolviendo este valor como resultado.
        '''        
        ious = []

        for imagen in sorted(os.listdir(RUTA_IMAGENES_TEST)):

            ruta_imagen = RUTA_IMAGENES_TEST + "/" + imagen
            ruta_label = RUTA_LABELS_TEST + "/" + imagen.rsplit('.', 1)[0] + '.txt'

            imagen = cv2.imread(ruta_imagen)

            altura, ancho = imagen.shape[:2]

            with open(ruta_label, 'r') as archivo:
                ground_truth = [[float(p[1])*ancho - float(p[3])*ancho/2, float(p[2])*altura - float(p[4])*altura/2,
                       float(p[1])*ancho + float(p[3])*ancho/2, float(p[2])*altura + float(p[4])*altura/2]
                      for p in [l.strip().split() for l in archivo] if len(p) >= 5]
                
            predicciones = [[*box.xyxy[0].cpu().numpy()] for r in self.modelo(ruta_imagen, verbose=False)
                    for box in r.boxes if box.conf[0] > 0.25]
            
            for p in predicciones:
                mejor = max([self._iou(p, g) for g in ground_truth], default=0)
                if mejor > 0:
                    ious.append(mejor)

        return np.mean(ious) if ious else 0.0
    
    def _iou(self, b1, b2):
        '''
        Este método calcula la Intersección sobre Unión (IoU) entre dos cajas delimitadoras, representadas por sus coordenadas (x1, y1, x2, y2). La IoU se calcula como el área de intersección entre las dos cajas dividido por el área de unión de las cajas. Este valor proporciona una medida de la superposición entre las cajas, siendo 1.0 una coincidencia perfecta y 0.0 sin superposición.    
        '''
        xi1, yi1, xi2, yi2 = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
        interseccion = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - interseccion
        return interseccion / union if union > 0 else 0.0

    def _evaluar_ocr(self):
        '''
        Este método evalúa la exactitud del modelo OCR utilizando el conjunto de datos específico para OCR, calculando el Character Error Rate (CER) y la tasa de acierto (precisión) del OCR. El proceso se realiza de la siguiente manera:
        - Se cargan las etiquetas de ground truth desde un archivo de texto, creando un diccionario que mapea el nombre de cada imagen con su texto correspondiente.
        - Se itera sobre cada imagen del conjunto de datos OCR, cargando la imagen y utilizando el modelo OCR para obtener el texto reconocido.
        - Para cada imagen, se calcula el CER utilizando la distancia de Levenshtein entre el texto reconocido y el texto de ground truth, normalizado por la longitud del texto de ground truth.
        - Se acumulan los CER y la cantidad de aciertos (cuando el texto reconocido coincide exactamente con el texto de ground truth) para calcular las métricas finales.
        - Finalmente, se devuelve el CER promedio, la precisión del OCR y el número total de imágenes
        '''
        labels = {}

        with open(RUTA_LABELS_OCR, 'r', encoding='utf-8-sig') as archivo:
            for linea in archivo:
                linea = linea.strip()
                partes = linea.split('\t')
                labels[partes[0]] = partes[1]

        lista_imagenes = sorted(f for f in os.listdir(RUTA_IMAGENES_OCR))

        total_cer = 0.0
        total_precision = 0
        contador = 0

        for nombre_imagen in lista_imagenes:

            ruta_imagen = RUTA_IMAGENES_OCR + "/" + nombre_imagen

            imagen = cv2.imread(ruta_imagen)

            texto_pred = self.modelo_ocr.reconocer(imagen)

            texto_gt = labels[nombre_imagen]

            cer = distancia_levenshtein(texto_pred, texto_gt) / max(len(texto_gt), 1)
            total_cer += cer

            if texto_pred == texto_gt:
                total_precision += 1

            contador += 1

        return total_cer / contador, total_precision / contador, contador


    def _medir_latencia_lote(self):
        '''
        Este método mide la latencia promedio por imagen al procesar un lote de imágenes del conjunto de prueba, ejecutando el modelo de detección en cada imagen y calculando el tiempo total para obtener una medida representativa del rendimiento del modelo en condiciones de uso real.
        '''
        lista_imagenes = [os.path.join(RUTA_IMAGENES_TEST, f) for f in os.listdir(RUTA_IMAGENES_TEST)[:10]]
        tiempo_inicio = time.time()

        for ruta_imagen in lista_imagenes:
            imagen = cv2.imread(ruta_imagen)
            self.modelo(imagen, verbose=False)

        tiempo_fin = time.time()

        return (tiempo_fin - tiempo_inicio) / len(lista_imagenes) * 1000


    def _medir_latencia_webcam(self):
        '''
        Este método mide la latencia promedio por frame al procesar video en tiempo real desde la webcam, ejecutando el modelo de detección en cada frame capturado durante un período de tiempo determinado, y calculando el tiempo total para obtener una medida representativa del rendimiento del modelo en condiciones de uso real con video en vivo.
        '''
        captura = cv2.VideoCapture(0)

        if not captura.isOpened():
            return None

        lista_tiempos = []

        for _ in range(100):
            ret, frame = captura.read()

            if not ret:
                break

            inicio = time.time()

            self.modelo(frame, verbose=False)

            fin = time.time()
            lista_tiempos.append((fin - inicio) * 1000)

        captura.release()

        return sum(lista_tiempos) / len(lista_tiempos) if lista_tiempos else None

    def _medir_recursos(self):
        '''
        Este método mide el consumo de recursos del sistema durante la ejecución del modelo, incluyendo el uso de RAM y memoria GPU (si está disponible), proporcionando una visión integral del impacto del modelo en los recursos del sistema.
        '''
        proceso = psutil.Process()
        ram_mb = proceso.memory_info().rss / 1024 / 1024

        if torch.cuda.is_available():
            memoria_gpu = torch.cuda.memory_allocated() / 1024 / 1024
            return ram_mb, memoria_gpu

        return ram_mb, None