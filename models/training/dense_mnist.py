import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Code for below adapted from the Keras and Tensorflow guides:
# https://keras.io/examples/vision/mnist_convnet/
# https://www.tensorflow.org/datasets/keras_example

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalise input images to 0.0 - 1.0
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
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
    epochs=6,
    validation_split=0.1
)