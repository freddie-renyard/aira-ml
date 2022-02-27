import tensorflow as tf
import numpy as np
from keras.layers import Input, Flatten
from keras.models import Model
import json
import time
import serial
from bitstring import Bits, BitArray
from math import ceil

from aira_ml.tools.aira_exceptions import AiraException
from aira_ml.tools.binary_tools import BinCompiler 

class SerialLink:

    def __init__(self):

        with open("aira_ml/cache/serial_params.json", "r") as file:
            params = json.load(file)

        self.n_tx = params["input_bit_depth"]
        self.n_rx = params["output_bit_depth"]

        self.tx_num = params["input_number"]
        self.rx_num = params["output_number"]

        # Get the input formats as integers for faster comparisons.
        # 0 - int, 1 - float.
        self.in_format_code = self.get_format_code(params["input_format"])
        self.out_format_code = self.get_format_code(params["output_format"])

        # Get the input data parameters.
        self.n_in_man = params["n_input_mantissa"]
        self.n_in_exp = params["n_input_exponent"]

        # Get the output data parameters.
        self.n_out_man = params["n_output_mantissa"]
        self.n_out_exp = params["n_output_exponent"]

        # Get the single byte codes that signal different data
        # requests/sends to the FPGA.
        self.code_data_in = self.get_format_code(params["input_format"])
        self.code_data_out = self.get_format_code(params["output_format"])

        with open("aira_ml/config/serial_config.json") as file:
            params = json.load(file)

        self.baud = params["baud_rate"]
        self.port_name = params["serial_port"]

        if self.code_data_out == 0:
            # ...The output is a signed integer.
            self.n_output = self.n_out_man
            self.bits_to_rx = (self.n_output * self.rx_num)
        elif self.code_data_out == 1:
            # ...The output is a float.
            self.n_output = (1 + self.n_out_man + self.n_out_exp)
            self.bits_to_rx = self.n_output * self.rx_num
        
        self.bytes_to_rx = ceil(self.bits_to_rx / 8)

        #self.begin_serial(5)
    
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

    def send_data(self, tensor):
        """Send the input tensor to the FPGA using UART.
        """

        flat_tensor = self.flatten_tensor(tensor)

        # Serialise the output data.
        # Format: LSBs - index 0 of the flattened tensor

        # Concatenate all the entries together to save bandwidth.
        # The hardware will use an appropriately sized intermediate 
        # register to reconstruct the data.
        bin_to_send = ""
        if self.in_format_code == 0:
            
            # The implicit assumption here is that the range of all integers is -1 to 1.
            # Anything else should be compiled to a floating point format.
            for value in flat_tensor:
                bin_val = BinCompiler.compile_to_signed(
                    value       = value, 
                    n_output    = self.n_in_man, 
                    n_radix     = self.n_in_man
                )

                bin_to_send = bin_val + bin_to_send
            
        elif self.in_format_code == 1:
            for value in flat_tensor:
                bin_val = BinCompiler.compile_to_float(
                    value,
                    self.n_in_man,
                    self.n_in_exp
                ) 

                bin_to_send = bin_val + bin_to_send

        tx_data = Bits(bin=bin_to_send)

        try:
            self.serial_link.write(tx_data.bytes)
        except:
            print("SERIAL: Data write failed.")
            return tx_data.bytes

    def receive_data(self, test_data=None):
        """Receive data from the FPGA over the serial port.
        Returns the model's output tensor.
        """
        if test_data == None:
            rx_bytes = self.serial_link.read(size=self.bytes_to_rx)
        else:
            rx_bytes = test_data
        
        rx_data = BitArray(rx_bytes).bin

        # Truncate the data received to remove leading 0s, as the serial
        # link only transmits bytes.
        self.bits_to_rx = self.rx_num * self.n_rx
        rx_data = rx_data[:self.bits_to_rx]

        # Extract the binary from the serial string.
        # Start at the end of the string, as this is the first value.
        flat_outputs = []
        for i in range(self.bits_to_rx-self.n_rx, -self.n_rx, -self.n_rx): # FOR TESTING

            if self.code_data_out == 1:
                rx_val = BinCompiler.decode_custom_float(
                    rx_data[i:i+self.n_rx],
                    self.n_in_man,
                    self.n_in_exp
                )
            elif self.code_data_out == 0:
                rx_val = BitArray(bin=rx_data[i:i+self.n_rx]).int
                rx_val /= 2 ** (self.n_out_man)
            
            flat_outputs.append(rx_val)

        # TODO Reshape this into the output tensor shape.
        return flat_outputs
            
    def get_inference(self, tensor):
        """Send data to the FPGA and await a response.
        """

        self.send_data(tensor)

    def get_format_code(self, format_str):
        """Get the input format as a code for quicker comparisons when sending data.
        """

        if format_str == 'float':
            return 1
        elif format_str == 'int':
            return 0
        else:
            raise AiraException("Data format not recognised.")

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