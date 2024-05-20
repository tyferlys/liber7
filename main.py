import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist         # загрузка набора данных Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

(x_train, y_train), (x_test, y_test) = mnist.load_data()# загрузка набора данных Mnist в массивы для обучения и тестирования
# нормализация входных данных
x_train = x_train / 255
x_test = x_test / 255
y_train_cat = keras.utils.to_categorical(y_train, 10)#Категорирование числовых меток обучающей части набора Mnist
y_test_cat = keras.utils.to_categorical(y_test, 10) #Категорирование числовых меток тестовой части набора Mnist

# модель СНС
model = keras.Sequential([
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(64, (3,3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10,  activation='softmax')
])
#print(model.summary()) # посмотреть параметры СНС
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
#print( x_train.shape )# Посмотреть входной вектор
#Сборка модели СНС
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])
#Обучние модель СНС
his = model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)
model.evaluate(x_test, y_test_cat)
