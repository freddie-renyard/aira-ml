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

print(model.layers[0].get_weights()[0][:,:,0,0])
print(np.shape(model.layers[0].get_weights()[0]))

print("Bias: ", model.layers[0].get_weights()[1])
print(inference)

for i in range(np.shape(inference)[3]):
    total = 0
    print("First z conv: ", end=" ")
    for j in range(3):
        filter = np.array(model.layers[0].get_weights()[0])[:,:,j,i].flatten()
        print("{:.3f}".format(np.sum(np.multiply(filter, [0,0,0,0,1,1,0,1,1]))), end=" ")
        total += np.sum(filter)

    print("Sum total: {:.4f}".format(total))

for i in range(np.shape(inference)[3]):
    print("Filter {} value: {:.5f}".format(i, float(inference[0, :, :, i][0,0])))

