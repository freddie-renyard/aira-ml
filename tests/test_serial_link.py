from cgi import test
from timeit import repeat
from aira_ml.serial_link import SerialLink
import numpy as np
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

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

def evaluate_inference(trials):
    
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

        # Normalise input image.
        test_data = x_test[i:i+1] / 256.0

        # Compute Tensorflow output.
        tf_inference = model.predict(test_data)

        # Get inference from the FPGA.
        aira_inference = link.get_inference(test_data)

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
            "Actual Output {} FPGA {} TensorFlow {} MSE: {}"
            .format(
                actual_number, 
                aira_number,
                tf_number,
                round(compute_mse(tf_inference, aira_inference), 2)
            )
        )

    print(
        "End of test: \n\t FPGA accuracy {} \n\t TensorFlow Accuracy {} \n\t FPGA-Tensorflow concordance {}"
        .format(
            round((aira_correct / trials), 2),
            round((tf_correct / trials), 2),
            round((concordance / trials), 2)
        )
    )

if __name__ == "__main__":
    evaluate_inference(20)