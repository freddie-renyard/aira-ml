import tensorflow as tf
from aira_ml.serial_link import SerialLink
import numpy as np

def send_image_internal():
    """ Input an image into the serial link and decode it to
    test the serial link's encoder and decoder, in the absence
    of an actual serial port.
    """

    link = SerialLink()
    test_data = np.arange(0,784).reshape(1,28,28) / 784.0

    uart_data = link.send_data(test_data)
    link.receive_data(uart_data)