'''
Este script se encarga del entrenamiento del modelo de detección basado en YOLOv8, 
utilizando un dataset personalizado definido en data.yaml.

Este archivo define constantes para asegurar consistencia y coherencia.

Se utiliza una función que encapsula la lógica.
'''

from ultralytics import YOLO

MODELO = "yolov8s.pt"
RUTA_DATOS = "data/data.yaml"
EPOCHS = 100
TAMANIO_IMAGEN = 832
PATIENCE = 20
BATCH = 16,
PROYECTO = "runs"
NOMBRE_MODELO = "detector"


def entrenar_modelo():

    '''
    Esta función tiene el objetivo de entrenar el modelo YOLO preentrenado 
    en la versión elegida, donde se establecen distintos parámetros como las epocas,
    y tamaño de imagen.
    Retorna el modelo best.pt que se utilizará en producción.
    '''
    modelo = YOLO(MODELO)

    resultados = modelo.train(data=RUTA_DATOS, epochs=EPOCHS, imgsz=TAMANIO_IMAGEN, batch=BATCH, patience=PATIENCE, project=PROYECTO,name=NOMBRE_MODELO, lr0=0.005,lrf=0.01, degrees=5, translate=0.1, scale=0.5, fliplr=0.5, patience=20)

    return resultados

if __name__ == "__main__":
    entrenar_modelo()