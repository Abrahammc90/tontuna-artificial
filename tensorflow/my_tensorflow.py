from html import parser
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

import tensorflow as tf
from typing import cast
import argparse

os.system('clear')  # Clear the console output


parser = argparse.ArgumentParser(description='TensorFlow MNIST Example')
parser.add_argument('--load_from', type=str, dest="h5_load_model", default=None,
                    help='Path to the model file')
parser.add_argument('--save_as', type=str, dest="h5_save_model", default=None,
                    help='Save the model after training')
args = parser.parse_args()

keras = tf.keras
layers = keras.layers
mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0


#if args.h5_load_model:
#    print(f"Loading model from {args.h5_load_model}...")
#    model = cast(keras.Model, keras.models.load_model(args.h5_load_model))
#else:
#    model = keras.Sequential([
#        keras.Input(shape=(28*28,)),
#        layers.Dense(512, activation='relu'),
#        layers.Dense(256, activation='relu'),
#        layers.Dense(10)])
#
#    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
#                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                    metrics=['accuracy'])
#    
#    print("Training the model...")
#    model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=2)#validation_split=0.2)
#
#
#    if args.h5_save_model:
#        model.save(args.h5_save_model)


#model = keras.Sequential()


input_a = keras.Input(shape=(int(28*28/2),))
x = layers.Dense(256, activation='relu', name="x1")(input_a)

input_b = keras.Input(shape=(int(28*28/2),))
y = layers.Dense(256, activation='relu', name="y1")(input_b)

combine = layers.concatenate([x, y], axis=1)
z = layers.Dense(256, activation='relu', name='z1')(combine)

outputs = layers.Dense(10, activation='softmax')(z)

model = keras.Model(inputs=[input_a, input_b], outputs=outputs)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])

#print(model.summary())

print("Training the model...")
model.fit([x_train[:, :392], x_train[:, 392:]], y_train, epochs=5, batch_size=32, verbose=2)

print("Evaluating the model...")
model.evaluate([x_test[:, :392], x_test[:, 392:]], y_test, batch_size=32, verbose=2)