import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# Code for below adapted from the Keras and Tensorflow guides:
# https://keras.io/examples/vision/mnist_convnet/
# https://www.tensorflow.org/datasets/keras_example
# https://medium.com/analytics-vidhya/lenet-with-tensorflow-a35da0d503df

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalise input images to 0.0 - 1.0
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Expand the dimensionality of the input image tensors.
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

# Construct a modified LeNet architecture
model = tf.keras.models.Sequential([
    Conv2D(6, (3,3), padding='valid', activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(8, (3,3), padding='valid', activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.1
)

model.summary()

model.save('models/conv_mnist/model')