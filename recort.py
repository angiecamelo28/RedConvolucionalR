import cv2

def recortar_imagen(imagen, x, y, ancho, alto):
    recorte = imagen[y:y+alto, x:x+ancho]
    return recorte

# Cargar imagen
imagen = cv2.imread("dataset/train/1x1 (1).jpg")

# Coordenadas y dimensiones del recorte
x = 100
y = 100
ancho = 100
alto = 100

# Recortar imagen
recorte = recortar_imagen(imagen, x, y, ancho, alto)

# Mostrar imagen original y recorte
cv2.imshow("Imagen Original", imagen)
cv2.imshow("Recorte", recorte)
cv2.waitKey(0)
cv2.destroyAllWindows()