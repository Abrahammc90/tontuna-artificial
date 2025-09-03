import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

import tensorflow as tf
import argparse

os.system('clear')  # Clear the console output

keras = tf.keras
layers = keras.layers
cifar10 = keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

#input = keras.Input(shape=(32, 32, 3))
#x = layers.Conv2D(32, 3, padding="valid", activation="relu")(input)
#x = layers.MaxPool2D()(x)
#x = layers.Conv2D(64, 5, padding="same")(x)
#x = layers.BatchNormalization()(x)
#x = keras.activations.relu(x)
#x = layers.Flatten()(x)
#x = layers.Dense(192, activation="relu")(x)
#x = layers.Reshape(target_shape=(8, 8, 3))(x)
#x = layers.Conv2D(128, 3, activation="relu")(x)
#x = layers.MaxPool2D()(x)
#x = layers.Conv2D(64, 3)(x)
#x = layers.BatchNormalization()(x)
#x = keras.activations.relu(x)
#x = layers.Flatten()(x)
#x = layers.Dense(64, activation="relu")(x)
#output = layers.Dense(10, activation="softmax")(x)


#Del tutorial
#input = keras.Input(shape=(32, 32, 3))
#x = layers.Conv2D(32, 3)(input)
#x = layers.BatchNormalization()(x)
#x = keras.activations.relu(x)
#x = layers.MaxPooling2D()(x)
#x = layers.Conv2D(64, 5, padding='same')(x)
#x = layers.BatchNormalization()(x)
#x = keras.activations.relu(x)
#x = layers.Conv2D(128, 3)(x)
#x = layers.BatchNormalization()(x)
#x = keras.activations.relu(x)
#x = layers.Flatten()(x)
#x = layers.Dense(64, activation='relu')(x)
#output = layers.Dense(10)(x)

#model = keras.Model(inputs=input, outputs=output)
#model.compile(
#    loss=keras.losses.SparseCategoricalCrossentropy(),
#    optimizer = keras.optimizers.Adam(lr=1e-3),
#    metrics=["accuracy"]
#)
#model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=2)
#model.evaluate(x_test, y_test, batch_size=64, verbose=2)

input = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, 3)(input)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.MaxPooling2D()(x)
for i in range(10):
    x = layers.Conv2D(32, 3)(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    dense_layer = layers.Flatten()(x)
    dense_layer = layers.Dense(64, activation='relu')(dense_layer)
    output = layers.Dense(10)(dense_layer)
    model = keras.Model(inputs=input, outputs=output)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer = keras.optimizers.Adam(lr=1e-3),
        metrics=["accuracy"]
    )
    model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=2)
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=64, verbose=2)
    print("accuracy is:", accuracy)