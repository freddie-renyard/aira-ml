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

        self.n_tx = ceil(params["input_bit_depth"] / 8) * 8
        self.tx_zero_pad = self.n_tx - params["input_bit_depth"]

        self.n_rx = ceil(params["output_bit_depth"] / 8) * 8
        self.n_output = params["output_bit_depth"]
        self.rx_zero_pad = self.n_rx - self.n_output

        self.tx_num = params["input_number"]
        self.rx_num = params["output_number"]

        self.rx_tensor_shape = params["output_tensor_shape"]

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

        self.bytes_to_rx = int(self.n_rx / 8) * self.rx_num
        self.bits_to_rx = self.bytes_to_rx * 8

        self.begin_serial(5)
    
    def flatten_tensor(self, in_tensor):
        """Flatten an input tensor to allow for serial transmission 
        to the FPGA.
        """
        in_tensor = np.array(in_tensor)
        return np.reshape(in_tensor, (self.tx_num), order="C")

    def reshape_tensor(self, in_flat_tensor):
        """ Reshape the received tensor into it's proper form.
        """

        target_shape = self.rx_tensor_shape
        return np.reshape(in_flat_tensor, target_shape)

    def begin_serial(self, timeout):
        """Begin the serial communication to the FPGA.
        """

        print("AIRA: Attempting to open serial port...")
        
        # Define the number of connection attempts
        rest_interval = 1 
        max_attempts = timeout // rest_interval
        attempts = 0
        
        while True: 
            self.serial_link = serial.Serial(self.port_name, baudrate=self.baud, timeout=1)
            try:
                self.serial_link = serial.Serial(self.port_name, baudrate=self.baud)
                self.reader = ReadLine(self.serial_link)
                print('AIRA: Opened serial port to device at', self.port_name)
                break
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
        self.debug_tx_bin = []

        # Serialise the output data.
        # Format: LSBs - index 0 of the flattened tensor

        if self.in_format_code == 0:
            
            # The implicit assumption here is that the range of all integers is -1 to 1.
            # Anything else should be compiled to a floating point format.
            for value in flat_tensor:
                bin_val = BinCompiler.compile_to_signed(
                    value       = value, 
                    n_output    = self.n_in_man, 
                    n_radix     = self.n_in_man-1
                )
                self.tx_bin_str(bin_val)
                
        elif self.in_format_code == 1:
            for value in flat_tensor:
                bin_val = BinCompiler.compile_to_float(
                    value,
                    self.n_in_man,
                    self.n_in_exp
                )

                self.tx_bin_str(bin_val)

    def tx_bin_str(self, tx_str):
        """Write bits to the serial port.
        """
        
        # TODO Edit the length to support arbitrary lengths.
        zero_padding = "0" * self.tx_zero_pad
        
        tx_data = Bits(bin = zero_padding + tx_str)
        try:
            self.serial_link.write(tx_data.bytes)
        except:
            print("SERIAL: Data write failed.")

    def receive_data(self):
        """Receive data from the FPGA over the serial port.
        Returns the model's output tensor.
        """
        
        rx_bytes = self.reader.readline(self.bytes_to_rx)
        rx_data = BitArray(bytes=rx_bytes).bin

        # Extract the binary from the serial string.
        # Start at the end of the string, as this is the first value.
        flat_outputs = []
        for i in range(0, self.bits_to_rx, self.n_rx): 
            
            data_raw = rx_data[i+self.rx_zero_pad:i+self.n_rx]
            if self.code_data_out == 1:
                rx_val = BinCompiler.decode_custom_float(
                    data_raw,
                    self.n_out_man,
                    self.n_out_exp
                )
            elif self.code_data_out == 0:
                rx_val = BitArray(bin=data_raw).int
                rx_val /= 2 ** (self.n_out_man-1)
        
            flat_outputs.append(rx_val)

        return self.reshape_tensor(flat_outputs)

    def get_inference(self, tensor):
        """Send data to the FPGA and await a response.
        """

        self.send_data(tensor)

        return self.receive_data()

    def get_format_code(self, format_str):
        """Get the input format as a code for quicker comparisons when sending data.
        """

        if format_str == 'float':
            return 1
        elif format_str == 'int':
            return 0
        else:
            raise AiraException("Data format not recognised.")

# This code is inspired by GitHub user skoehler's code from the 
# following pyserial issue: https://github.com/pyserial/pyserial/issues/216
class ReadLine:
    def __init__(self, s):
        self.s = s
    
    def readline(self, bytes_to_read):
        
        buf = bytearray()
        buf_size = 0
        while buf_size < bytes_to_read:
            i = max(1, min(2048, self.s.in_waiting))
            buf += self.s.read(i)
            buf_size += i
        return buf
            