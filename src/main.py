'''
Este archivo actúa como entrada a la aplicación, gestiona la interacción usuario-sistema, 
permitiendo al operador seleccionar e inicializar los módulos necesarios.
'''

from detect import Detector
import tkinter
from tkinter import filedialog
from evaluate import Evaluador

RUTA_MODELO = "models/weights/best.pt"
NOMBRE_MODELO_OCR = "cct-xs-v1-global-model"

IMAGEN = "1"
CARPETA = "2"
WEBCAM = "3"
EVALUAR = "4"
SALIR = "5"

def main():
    '''
    Esta función es el punto de entrada a la aplicación, gestiona la interacción usuario-sistema, 
    permitiendo al usuario seleccionar los módulos necesarios.
    '''
    detector = Detector(ruta_modelo=RUTA_MODELO, nombre_modelo_ocr=NOMBRE_MODELO_OCR)

    while True:

        print("Seleccione una opción:")
        print("1 : Procesar imagen")
        print("2 : Procesar carpeta")
        print("3 : Webcam")
        print("4 : Evaluar modelo")
        print("5 : Salir")

        opcion = input("Opción: ").strip()
        
        if opcion == SALIR:
            break

        root = tkinter.Tk()
        root.attributes('-topmost', True)
        
        root.withdraw()

        if opcion == IMAGEN:
            ruta = filedialog.askopenfilename(title="Seleccionar imagen", filetypes=[("Imagenes", "*.jpg *.jpeg *.png")])

        elif opcion == CARPETA:
            ruta = filedialog.askdirectory(title="Seleccionar carpeta")

        elif opcion == WEBCAM:
            detector.procesar_webcam()
            continue
        
        elif opcion == EVALUAR:
            evaluador = Evaluador()
            evaluador.evaluar()
            continue

        elif opcion == SALIR:
            break

        else:
            print("Opción inválida")
            continue

        if ruta:
            detector.procesar_imagen(ruta)

        else:
            print("No se seleccionó nada")

if __name__ == "__main__":
    main()