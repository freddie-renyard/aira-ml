from cgi import test
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

    for i in range(1):
        test_data = x_test[i] / 256.0
        out_img = link.get_inference(test_data)
        print("Got image!")

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