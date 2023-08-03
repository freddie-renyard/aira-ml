import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# Construct a simple single layer Dense model for testing.
model = tf.keras.models.Sequential([
    Dense(10, activation='relu', input_shape=(10,))
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.summary()
model.save('models/single_dense/model')