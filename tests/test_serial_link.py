from cgi import test
from timeit import repeat
from tokenize import Triple
from aira_ml.serial_link import SerialLink
import numpy as np
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from aira_ml.tools.binary_tools import BinCompiler
import time

import os
from subprocess import check_call


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

def compute_mse(arr_1, arr_2):
    """Compute the MSE of two input tensors.
    """

    # Ensure both tensors are flat.
    elements = np.prod(np.shape(arr_1))
    arr_1 = np.reshape(arr_1, elements)

    elements = np.prod(np.shape(arr_2))
    arr_2 = np.reshape(arr_2, elements)

    output = np.subtract(arr_2, arr_1)
    output = np.square(output)

    return np.sum(output) / np.shape(output)[0]

def evaluate_inference(trials, show_img=False):

    print("\nDELTA: Testing MNIST inference...\n")
    
    # Load the actual ML model
    path_to_model = "models/dense_mnist/model"
    model = load_model(path_to_model)

    # Load testing data
    (_, _), (x_test, y_test) = mnist.load_data()

    link = SerialLink()

    aira_correct = 0
    tf_correct = 0
    concordance = 0

    for i in range(trials):

        #┬áNormalise input image.
        test_data = (x_test[i:i+1] / 256.0) + np.expand_dims(np.random.rand(28,28), axis=0)*0

        # Compute Tensorflow output.
        tf_inference = model.predict(test_data)

        # Get inference from the FPGA.
        aira_inference = link.get_inference(test_data)

        # Threshold the floating point errors
        aira_inference = aira_inference * (aira_inference > (10**-8))

        if show_img:
            plt.imshow(test_data[0])
            plt.axis('off')
            plt.show()

        actual_number = y_test[i]
        tf_number = int(np.argmax(tf_inference))
        aira_number = int(np.argmax(aira_inference))

        if tf_number == actual_number:
            tf_correct += 1
        if aira_number == actual_number:
            aira_correct += 1
        if aira_number == tf_number:
            concordance += 1
        
        print(
            "TRIAL {} Actual Output {} FPGA {} TensorFlow {} MSE: {}"
            .format(
                i,
                actual_number, 
                aira_number,
                tf_number,
                round(compute_mse(tf_inference, aira_inference), 2)
            )
        )

    print(
        "End of test: \n\t FPGA accuracy {} \n\t TensorFlow Accuracy {} \n\t FPGA-Tensorflow concordance {}"
        .format(
            (aira_correct / trials),
            (tf_correct / trials),
            (concordance / trials)
        )
    )

def evaluate_uart_speed(trials):
    
    (_, _), (x_test, _) = mnist.load_data()
    link = SerialLink()
    start_time = time.time()

    for i in range(trials):
        test_data = (x_test[i:i+1] / 256.0)
        aira_inference = link.get_inference(test_data)
    
    end_time = time.time()
    print("Complete: {} ms per inference".format((end_time - start_time)*1000/trials))

if __name__ == "__main__":

    print("\nDELTA: Loading the configuration file onto the FPGA...\n")

    # Load the bitstream file onto the FPGA.
    cwd = os.getcwd()
    script_path = cwd + "/aira_ml/fpga_load.sh"
    check_call(script_path, shell=True)

    try:
        evaluate_inference(10, show_img=True)
    except:
        while True:
            try:
                check_call(script_path, shell=True)
                break
            except:
                print("FPGA not found.")
                
        evaluate_inference(10, show_img=True)