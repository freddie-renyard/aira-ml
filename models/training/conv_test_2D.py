import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# Construct a simple 2D convolutional model for testing.
input_shape = (3, 3, 3)
model = tf.keras.models.Sequential([
    Conv2D(6, (3,3), padding='same', activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2,2))
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.summary()
model.save('models/conv_test_2D/model')