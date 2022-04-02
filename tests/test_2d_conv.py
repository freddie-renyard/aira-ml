import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("models/conv_test_2D/model")
model.summary()

# Create a test tensor of all ones so that all
# values in each output filter map will be the same.
input_shape = model.get_config()['layers'][0]['config']['batch_input_shape']
in_tensor = np.ones(input_shape[1:])

# Extend the dims of the input tensor to make it compatible with the
# TensorFlow model.
in_tensor = in_tensor[tf.newaxis, :]

inference = model.predict(in_tensor)

for i in range(np.shape(inference)[3]):
    print("Filter {} value: {}".format(i, inference[0, :, :, i][0,0]))