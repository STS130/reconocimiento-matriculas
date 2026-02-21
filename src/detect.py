'''
Este módulo implementa el núcleo de inferencia del sistema. 
Su función es recibir imágenes (individuales, por lote o desde webcam), 
ejecutar el modelo de detección y obtener las regiones de interés correspondientes a las matrículas. 
Además, gestiona la visualización de los resultados y la integración con el módulo OCR.
'''

from ultralytics import YOLO
import cv2
from ocr import OCR
import os

UMBRAL_CONFIANZA = 0.65
EXTENSIONES_IMAGEN = (".jpg", ".jpeg", ".png")

class Detector:
    '''
    Esta clase encapsula la funcionalidad de detección de matrículas, incluyendo la carga del modelo, el procesamiento de imágenes y la integración con OCR.
    '''
    def __init__(self, ruta_modelo, nombre_modelo_ocr, umbral_confianza=UMBRAL_CONFIANZA):
        self.modelo = YOLO(ruta_modelo)
        self.ocr = OCR(nombre_modelo_ocr)
        self.umbral_confianza = umbral_confianza


    def procesar_imagen(self, ruta):
        '''
        Este método procesa una imagen o un directorio de imágenes, ejecutando el modelo de detección y mostrando los resultados.
        Parte de esta lógica se ha extraído a métodos privados para mejorar la legibilidad y modularidad del código.
        '''
        if os.path.isdir(ruta):
            for archivo in os.listdir(ruta):
                if archivo.lower().endswith(EXTENSIONES_IMAGEN):
                    ruta_completa = os.path.join(ruta, archivo)
                    salir = self._procesar_archivo(ruta_completa)
                    if salir:
                        break
        else:
            self._procesar_archivo(ruta)


    def procesar_webcam(self):
        '''
        Este método accede a la webcam, procesa el video en tiempo real y muestra los resultados de detección.
        La ejecución se detiene al presionar la tecla ESC.
        '''
        captura = cv2.VideoCapture(0)

        if not captura.isOpened():
            print("No se pudo acceder a la webcam")
            return

        while True:
            ret, cuadro = captura.read()
            if not ret:
                break

            resultados = self.modelo(cuadro, verbose=False)

            if resultados[0].boxes:
                self._procesar_detecciones(cuadro, resultados)

            cv2.imshow("Webcam", cuadro)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        captura.release()
        cv2.destroyAllWindows()


    def _procesar_archivo(self, ruta_imagen):
        '''
        Este método procesa un archivo de imagen específico, ejecutando el modelo de detección y mostrando los resultados.
        Devuelve True si el usuario decide salir durante la visualización de los resultados, False en caso contrario.
        '''
        imagen = self._cargar_imagen(ruta_imagen)

        resultados = self.modelo(imagen, verbose=False)

        if resultados[0].boxes:
            self._procesar_detecciones(imagen, resultados)
        else:
            print("No se detectaron matrículas en la imagen")

        cv2.imshow("Imagen", self._redimensionar(imagen, ancho=960))

        tecla = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if tecla == 27:
            return True

        return False


    def _cargar_imagen(self, ruta):
        '''
        Este método carga una imagen desde la ruta especificada, asegurándose de que el archivo es válido y se puede procesar.
        '''
        imagen = cv2.imread(ruta)

        if imagen is None:
            raise ValueError("No se pudo cargar la imagen")
        return imagen


    def _procesar_detecciones(self, imagen, resultados):
        '''
        Este método procesa los resultados de detección, extrayendo las regiones de interés correspondientes a las matrículas, ejecutando OCR y mostrando los resultados.
        Una vez reconocida la matrícula, se llama a un método privado para dibujar los resultados en la imagen.
        '''
        for resultado in resultados:

            indices_matriculas = (resultado.boxes.cls == 0).nonzero(as_tuple=True)[0]

            for idx in indices_matriculas:

                confianza_yolo = resultado.boxes.conf[idx].item()

                if confianza_yolo > self.umbral_confianza:

                    xyxy = resultado.boxes.xyxy[idx].squeeze().tolist()
                    x1, y1 = int(xyxy[0]), int(xyxy[1])
                    x2, y2 = int(xyxy[2]), int(xyxy[3])

                    imagen_matricula = imagen[y1:y2, x1:x2]
                    
                    texto = self.ocr.reconocer(imagen_matricula)

                    print(f"Matricula detectada: {texto}")
                    print(f"Confianza YOLO: {confianza_yolo:.2f}")

                    self._dibujar_resultados(imagen, x1, y1, x2, y2, texto)


    def _dibujar_resultados(self, imagen, x1, y1, x2, y2, texto):
        '''
        Este método dibuja un rectángulo alrededor de la matrícula detectada y muestra el texto reconocido por OCR.
        Se establecen parámetros de fuente, escala y grosor para asegurar una visualización clara y legible de los resultados.
        '''
        fuente = cv2.FONT_HERSHEY_SIMPLEX
        escala = 1.0
        grosor = 2

        (ancho_texto, alto_texto), _ = cv2.getTextSize(texto, fuente, escala, grosor)

        x_inicio = x1
        y_inicio = y1 - alto_texto - 10
        x_fin = x1 + ancho_texto + 10
        y_fin = y1

        cv2.rectangle(imagen, (x_inicio, y_inicio), (x_fin, y_fin), (0, 255, 0), -1)

        cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(imagen,texto, (x1 + 5, y1 - 5),fuente, escala, (0, 0, 0),grosor)


    def _redimensionar(self, imagen, ancho=960):
        '''
        Este método redimensiona la imagen manteniendo la relación de aspecto, ajustando el ancho a un valor específico y calculando el alto correspondiente.
        '''
        h, w = imagen.shape[:2]
        escala = ancho / w
        nuevo_alto = int(h * escala)
        return cv2.resize(imagen, (ancho, nuevo_alto))