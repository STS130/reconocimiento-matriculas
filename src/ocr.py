'''
Este archivo encapsula el uso del motor Fast Plate OCR. Se encarga de recibir la imagen recortada de la matrícula, aplicar el pre y postprocesamiento necesario y devolver el texto reconocido.
Contiene la clase OCR, que se inicializa con el nombre del modelo OCR a utilizar. Sus funciones son:
- reconocer(imagen_matricula)
-_preprocesar(imagen)
- _postprocesar(texto)
'''
from fast_plate_ocr import LicensePlateRecognizer
import cv2

ANCHO_OCR = 128
ALTO_OCR = 64

class OCR:

    def __init__(self, nombre_modelo):
        self.ocr = LicensePlateRecognizer(nombre_modelo)

    def reconocer(self, imagen_matricula):
        '''
        Este método recibe la imagen recortada de la matrícula, la preprocesa, la pasa por el modelo OCR y luego postprocesa el texto resultante para devolverlo limpio. Estas acciones se derivan a funciones privadas.
        '''
        imagen_matricula_preprocesada = self._preprocesar(imagen_matricula)
        texto = self.ocr.run(imagen_matricula_preprocesada)
        return self._postprocesar(texto)


    def _preprocesar(self, imagen):
        '''
        Este método se encarga de redimensionar la imagen de la matrícula al tamaño esperado por el modelo OCR (128x64, según la documentación del modelo).
        '''
        imagen_preprocesada= cv2.resize(imagen, (ANCHO_OCR, ALTO_OCR))
        return imagen_preprocesada
    
    
    def _postprocesar(self, texto):
        '''
        Este método se encarga de limpiar el texto reconocido por el modelo OCR. En este caso, al ser un modelo global, el modelo completa los espacios vacios con guiones bajos. Por lo tanto, se reemplazan los guiones bajos por espacios vacíos para obtener el texto final de la matrícula.
        '''
        texto_postprocesado = texto[0].replace("_", "")
        return texto_postprocesado