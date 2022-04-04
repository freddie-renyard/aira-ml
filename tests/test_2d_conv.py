import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("models/conv_test_2D/model")
model.summary()

# Create a test tensor of all ones so that all
# values in each output filter map will be the same.

input_shape = model.get_config()['layers'][0]['config']['batch_input_shape']
in_tensor = np.ones(input_shape[1:])
print(input_shape)

# Extend the dims of the input tensor to make it compatible with the
# TensorFlow model.
in_tensor = in_tensor[tf.newaxis, :]

inference = model.predict(in_tensor)

filter = np.array(model.layers[0].get_weights()[0])[:,:,0,0].flatten()
print(filter)
print(np.sum(filter[0:8]))

for i in range(np.shape(inference)[3]):
    print("Filter {} value: {}".format(i, inference[0, :, :, i][0,0]))