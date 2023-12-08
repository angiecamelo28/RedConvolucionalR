import pandas as pd
import tensorflow as tf
import keras
import numpy as np
import seaborn as sn

import cv2
###Importar componentes de la red neuronal
from keras.models import Sequential
from keras.layers import InputLayer,Input,Conv2D, MaxPool2D,Reshape,Dense,Flatten
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.callbacks import EarlyStopping

##################################

def cargarDatos(rutaOrigen,numeroCategorias,limite,ancho,alto):
    imagenesCargadas=[]
    valorEsperado=[]
    for categoria in range(0,numeroCategorias):
        for idImagen in range(0,limite[categoria]):
            ruta=rutaOrigen+str(categoria)+"/"+str(categoria)+"_"+str(idImagen)+".jpg"
            print(ruta) 
            imagen = cv2.imread(ruta)
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            imagen = cv2.resize(imagen, (ancho, alto))
            imagen = imagen.flatten()
            imagen = imagen / 255 #este paso normaliza
            imagenesCargadas.append(imagen)
            probabilidades = np.zeros(numeroCategorias)
            probabilidades[categoria] = 1
            valorEsperado.append(probabilidades)
    imagenesEntrenamiento = np.array(imagenesCargadas)
    valoresEsperados = np.array(valorEsperado)
    return imagenesEntrenamiento, valoresEsperados

#################################
ancho=256
alto=256
pixeles=ancho*alto
#Imagen RGB -->3
numeroCanales=1
formaImagen=(ancho,alto,numeroCanales)
numeroCategorias=4

cantidaDatosEntrenamiento=[5000,5000,5000,5000]
cantidaDatosPruebas=[1000,1000,1000,1000]

#Cargar las imágenes
imagenes, probabilidades=cargarDatos("dataset/train/",numeroCategorias,cantidaDatosEntrenamiento,ancho,alto)

model=Sequential()
#Capa entrada
model.add(InputLayer(input_shape=(pixeles,)))
model.add(Reshape(formaImagen))

#Capas Ocultas
#Capas convolucionales
model.add(Conv2D(kernel_size=3,strides=2,filters=8,padding="same",activation="relu",name="capa_1"))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(kernel_size=4,strides=1,filters=36,padding="same",activation="relu",name="capa_2"))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(kernel_size=5,strides=2,filters=64,padding="same",activation="relu",name="capa_3"))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(kernel_size=6,strides=1,filters=128,padding="same",activation="relu",name="capa_4"))
model.add(MaxPool2D(pool_size=2,strides=2))

#Aplanamiento
model.add(Flatten())
model.add(Dense(128,activation="relu"))

#Capa de salida
model.add(Dense(numeroCategorias,activation="softmax"))


#Traducir de keras a tensorflow
model.compile(optimizer=tf.keras.optimizers.Adam(),loss="categorical_crossentropy", metrics=["accuracy"])
#Entrenamiento
model.fit(x=imagenes,y=probabilidades,epochs=50,batch_size=60)
#early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Entrenamiento con Early Stopping
#model.fit(x=imagenes, y=probabilidades, epochs=100, batch_size=60, validation_split=0.2, callbacks=[early_stopping])


# Prueba del modelo
imagenesPrueba, probabilidadesPrueba = cargarDatos("dataset/test/", numeroCategorias, cantidaDatosPruebas, ancho, alto)
metricResult = model.evaluate(x=imagenesPrueba, y=probabilidadesPrueba)

# Accuracy
accuracy = metricResult[1]

# Predicciones
predicciones = model.predict(imagenesPrueba)
etiquetas_predichas = np.argmax(predicciones, axis=1)

# Etiquetas reales
etiquetas_reales = np.argmax(probabilidadesPrueba, axis=1)

# Precision, Recall y F1 Score


precision = precision_score(etiquetas_reales, etiquetas_predichas, average='weighted')
recall = recall_score(etiquetas_reales, etiquetas_predichas, average='weighted')
f1 = f1_score(etiquetas_reales, etiquetas_predichas, average='weighted')

# Loss (Pérdida)
loss = metricResult[0]

# Épocas de entrenamiento
epocas = 50  # O el número de épocas que especificaste durante el entrenamiento

# Tiempo de respuesta
# Calcula el tiempo de respuesta aquí según tus necesidades

# Imprimir los metricResult
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Loss:", loss)
print("Épocas de entrenamiento:", epocas)

predicciones = model.predict(imagenesPrueba)
predicciones_etiquetas = np.argmax(predicciones, axis=1)
etiquetas_verdaderas = np.argmax(probabilidadesPrueba, axis=1)
matriz_confusion = confusion_matrix(etiquetas_verdaderas, predicciones_etiquetas)
print("Matriz de confusión:")
print(matriz_confusion)
print('KNN Reports\n',classification_report(etiquetas_verdaderas, predicciones_etiquetas))


# Guardar modelo
ruta="models/modeloR.h5"
model.save(ruta)
# Informe de estructura de la red
model.summary()

metricResult = model.evaluate(x=imagenes, y=probabilidades)
scnn_pred = model.predict(imagenesPrueba, batch_size=60, verbose=1)
scnn_predicted = np.argmax(scnn_pred, axis=1)

# Creamos la matriz de confusión
scnn_cm = confusion_matrix(np.argmax(probabilidadesPrueba, axis=1), scnn_predicted)

# Visualiamos la matriz de confusión
scnn_df_cm = pd.DataFrame(scnn_cm, range(4), range(4))
plt.figure(figsize=(20, 14))
sn.set(font_scale=1.4)  # for label size
sn.heatmap(scnn_df_cm, annot=True, annot_kws={"size": 12})  # font size
plt.show()

scnn_report = classification_report(np.argmax(probabilidadesPrueba, axis=1), scnn_predicted)
print("SCNN REPORT", scnn_report)