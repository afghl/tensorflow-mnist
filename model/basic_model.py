import tensorflow as tf
from tensorflow.keras import layers, models


class BasicModel(models.Model):
    def __init__(self):
        super(BasicModel, self).__init__()
        self.flatten = layers.Flatten(input_shape=(28, 28))
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        return self.dense2(x)


def create_basic_model():
    model = BasicModel()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
