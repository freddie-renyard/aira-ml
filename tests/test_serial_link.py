from cgi import test
import tensorflow as tf
from aira_ml.serial_link import SerialLink
import numpy as np
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt

def send_image_internal():
    """ Input an image into the serial link and decode it to
    test the serial link's encoder and decoder, in the absence
    of an actual serial port.
    """

    (_, _), (x_test, _) = mnist.load_data()

    link = SerialLink()
    test_data = x_test[4] / 256.0

    uart_data = link.send_data(test_data)
    out_data = link.receive_data(uart_data)

    out_img_shape = np.shape(test_data)
    out_img = np.reshape(out_data, out_img_shape)

    plt.subplot(1,2,1)
    plt.imshow(test_data)
    plt.axis('off')
    plt.title("Image Before Serial")
    plt.title

    plt.subplot(1,2,2)
    plt.imshow(out_img)
    plt.title("Image After Serial")
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    send_image_internal()