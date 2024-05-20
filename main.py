import numpy as np
import matplotlib.pyplot as plt
from keras.src.models import model
from keras.src.saving import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing import image

fashion_category = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

class Network:
    def __init__(self, learn):
        if learn:
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train = x_train / 255
            x_test = x_test / 255
            self.y_train_cat = keras.utils.to_categorical(y_train, 10)
            self.y_test_cat = keras.utils.to_categorical(y_test, 10)

            self.model = keras.Sequential([
                Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
                MaxPooling2D((2, 2), strides=2),
                Conv2D(64, (3, 3), padding='same', activation='relu'),
                MaxPooling2D((2, 2), strides=2),
                Flatten(),
                Dense(128, activation='relu'),
                Dense(10, activation='softmax')
            ])

            self.x_train = np.expand_dims(x_train, axis=3)
            self.x_test = np.expand_dims(x_test, axis=3)

            self.learn()

    def learn(self):
        self.model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.history = self.model.fit(self.x_train, self.y_train_cat, batch_size=32, epochs=5, validation_split=0.2)
        self.model.evaluate(self.x_test, self.y_test_cat)

        self.model.save('fashion_mnist_model.h5')

    def history_show(self):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'bo', label='Тренировочная точность')
        plt.plot(epochs, val_acc, 'b', label='Валидационная точность')
        plt.title('Тренировочная и валидационная точность')
        plt.xlabel('Эпохи')
        plt.ylabel('Точность')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'bo', label='Тренировочные потери')
        plt.plot(epochs, val_loss, 'b', label='Валидационные потери')
        plt.title('Тренировочные и валидационные потери')
        plt.xlabel('Эпохи')
        plt.ylabel('Потери')
        plt.legend()

        plt.show()

    def _load_and_preprocess_image(self, img_path):
        img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self):
        array_images = [
            "./testing/test1.jpg",
            "./testing/test2.jpg",
            "./testing/test3.jpg",
            "./testing/test4.jpg"
        ]

        model = load_model('fashion_mnist_model.h5')

        for image_item in array_images:
            img_array = self._load_and_preprocess_image(image_item)
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)

            img = image.load_img(image_item, target_size=(28, 28), color_mode='grayscale')
            plt.imshow(img, cmap='gray')
            plt.title(f'Предсказанно: {fashion_category[predicted_class]}')
            plt.axis('off')
            plt.show()


network = Network(False)
#network.history_show()
network.predict()
