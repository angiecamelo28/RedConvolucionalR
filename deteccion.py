import cv2
import numpy as np
import tensorflow as tf

# Cargar el modelo entrenado
model = tf.keras.models.load_model('models/modeloA.h5')

# Etiquetas de categorías
categorias = ["1","2","3","4"] # Reemplaza con tus etiquetas de categorías

# Función para resaltar polígonos y clasificar categorías en tiempo real
def resaltar_poligonos(image):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar umbralización para obtener una imagen binaria
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Encontrar los contornos en la imagen binaria
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterar sobre los contornos y dibujar polígonos en la imagen original
    for contour in contours:
        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)

        # Extraer el polígono individualmente
        x, y, w, h = cv2.boundingRect(approx)
        polygon_roi = image[y:y + h, x:x + w]

        # Preprocesar el polígono para la clasificación
        preprocessed_polygon = cv2.resize(polygon_roi, (100, 100))
        preprocessed_polygon = preprocessed_polygon / 255.0
        preprocessed_polygon = np.expand_dims(preprocessed_polygon, axis=0)

        # Realizar la clasificación de categorías
        predictions = model.predict(preprocessed_polygon)
        category_index = np.argmax(predictions[0])
        category_label = categorias[category_index]

        # Mostrar la categoría de forma gráfica junto al contorno
        cv2.putText(image, category_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar la imagen con los polígonos resaltados y las categorías clasificadas
    cv2.imshow('Resaltado de polígonos y clasificación', image)

# Inicializar la cámara web
cap = cv2.VideoCapture(0)  # Cambiar a 1 si tienes varias cámaras

while True:
    # Leer un frame desde la cámara
    ret, frame = cap.read()

    # Aplicar la función para resaltar polígonos y clasificar categorías en el frame actual
    resaltar_poligonos(frame)

    # Detener el bucle al presionar la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()