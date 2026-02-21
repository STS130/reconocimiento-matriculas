'''
Este módulo contiene utilidades para la organización y preparación del dataset. 
Renombra archivos y etiquetas, y prepara la porción del dataset que se utiliza para evaluar el OCR.
'''

import os
import cv2
from ultralytics import YOLO

CARPETA_IMAGENES = "data/images"
CARPETA_LABELS = "data/labels"

RUTA_MODELO = "models/weights/best.pt"
INPUT = "data/images"
OUTPUT = "ocr/images"
UMBRAL_CONFIANZA = 0.4


def renombrar():
    '''
    Esta función renombra las imágenes y sus etiquetas correspondientes en orden numérico secuencial.
    '''
    imagenes = sorted([f for f in os.listdir(CARPETA_IMAGENES) if f.endswith(".jpg")])
    contador = 1

    for img in imagenes:
        nombre_base = os.path.splitext(img)[0]
        label = nombre_base + ".txt"

        ruta_imagen = os.path.join(CARPETA_IMAGENES, img)
        ruta_label = os.path.join(CARPETA_LABELS, label)

        if os.path.exists(ruta_label):
            nuevo_nombre = f"{contador:03d}"

            os.rename(ruta_imagen, os.path.join(CARPETA_IMAGENES, nuevo_nombre + ".jpg"))
            os.rename(ruta_label, os.path.join(CARPETA_LABELS, nuevo_nombre + ".txt"))

            contador += 1


def recortar():
    
    '''
    Esta función recorta las matrículas de las imagenes utilizando YOLO 
    y las guarda en la ruta seleccionada para el posterior evaluación del OCR.
    '''
    os.makedirs(OUTPUT, exist_ok=True)

    modelo = YOLO(RUTA_MODELO)

    id = 1

    for nombre_imagen in sorted(os.listdir(INPUT)):
        if not nombre_imagen.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        ruta_imagen = os.path.join(INPUT, nombre_imagen)
        imagen = cv2.imread(ruta_imagen)

        if imagen is None:
            continue

        resultados = modelo(imagen, conf=UMBRAL_CONFIANZA)

        for resultado in resultados:
            for box in resultado.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                imagen_matricula = imagen[y1:y2, x1:x2]

                if imagen_matricula.size == 0:
                    continue

                nuevo_nombre = f"{id:06d}.jpg"
                cv2.imwrite(os.path.join(OUTPUT, nuevo_nombre), imagen_matricula)

                print(f"Guardado: {nuevo_nombre}")

                id += 1

if __name__ == "__main__":
    renombrar()
    recortar()