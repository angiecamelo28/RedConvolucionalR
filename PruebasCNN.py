
import cv2
from Prediccion import Prediccion

clases=["1","2","3","4"]

ancho=256
alto=256

miModeloCNN=Prediccion("models/modeloD.h5",ancho,alto)
imagen=cv2.imread("dataset/val/0/26 (19).jpg")

claseResultado=miModeloCNN.predecir(imagen)
print("La imagen cargada es ",clases[claseResultado])

while True:
    cv2.imshow("imagen",imagen)
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break
cv2.destroyAllWindows()