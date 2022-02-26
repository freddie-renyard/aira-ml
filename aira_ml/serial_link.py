from lib2to3.pgen2.token import N_TOKENS
import tensorflow as tf
import numpy as np
from keras.layers import Input, Flatten
from keras.models import Model
import json
import time
import serial 

class SerialLink:

    def __init__(self):

        with open("aira_ml/cache/serial_params.json", "r") as file:
            params = json.load(file)

        self.n_tx = params["input_bit_depth"]
        self.n_rx = params["output_bit_depth"]

        self.tx_num = params["input_number"] 
        self.rx_num = params["output_number"]

        with open("aira_ml/config/serial_config.json") as file:
            params = json.load(file)

        self.baud = params["baud_rate"]
        self.port_name = params["serial_port"]

        # Get the single byte codes that signal different data
        # requests/sends to the FPGA.
        self.code_data_in = params["code_data_in"]
        self.code_data_out = params["code_data_out"]

        self.begin_serial(5)
    
    def flatten_tensor(self, in_tensor):
        """Flatten an input tensor to allow for serial transmission 
        to the FPGA.
        """
        in_tensor = np.array(in_tensor)
        return np.reshape(in_tensor, (self.tx_num), order="C")

    def begin_serial(self, timeout):
        """Begin the serial communication to the FPGA.
        """

        print("AIRA: Attempting to open serial port...")
        
        # Define the number of connection attempts
        rest_interval = 1
        max_attempts = timeout // rest_interval
        attempts = 0
        
        while True: 
            try:
                self.serial_link = serial.Serial(self.port, baudrate=self.baud_rate, timeout=0.0005)
                print('AIRA: Opened serial port to device at', self.port_name)
            except:
                print('AIRA: Failed to connect to {}. Reattempting...'.format(self.port_name))
                attempts += 1
                time.sleep(rest_interval)

            if attempts >= max_attempts and max_attempts != 0:
                print("AIRA: ERROR: Serial connection to {} failed.".format(self.port_name))
                break

    def send_data(self):
        """Send data to the FPGA using UART.
        """

    def get_inference(self):
        """Send data to the FPGA and await a response.
        """

    def trial_flatten(self):
        """Test method to allow testing of different input tensor shapes
        to ensure that they are flattened appropriately.
        """
        inputs = Input(shape=(3,3))
        prediction = Flatten()(inputs)
        model = Model(inputs=inputs, outputs=prediction)

        X = np.arange(0,9).reshape(1,3,3)

        print(X)
        print(np.reshape(X, (9)))
        print(model.predict(X))