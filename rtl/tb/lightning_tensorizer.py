# compiling a deep neural network into verilog data files that can be synthesized together with the Lightning logic
# authors: Zhizhen Zhong (zhizhenz@mit.edu)

import numpy as np

## DAC is taking 8 samples (128 bits), so no repetition in DAC data

## for 14-bit coding, 0x7FFC is 8191, 0x8000 is -8192
def tohex(val):
  return hex(val * 128) # convert a 8-bit value to 15 bit non-negative value (32767)


## input a list of decimal integer, output a list of hex (16 bits) strings
def Dec2Hex(dec_streams):
    hex_list = []
    hex_stream = ""
    for i in range(len(dec_streams)):
        hex_value = tohex(dec_streams[i])
        if len(hex_value) < 6:
            hex_value = '0x' + '0'*(6-len(hex_value)) + hex_value[2-len(hex_value):]
        # hex_streams.append(bytes(hex_value, encoding="raw_unicode_escape"))
        hex_list.append(hex_value)
        
        hex_stream = hex_value[2:] + hex_stream

        hex_stream = str(hex_stream)

    hex_stream = hex_stream.upper()

    return hex_list, hex_stream


def RescaleData(all_layers, bitwidth, verbose=False):
    maxvalue = 1
    minvalue = 99999

    max_fpga = pow(2, bitwidth) - 1  # maximum integer value under $bitwidth

    for layer in all_layers:
        if maxvalue < np.max(layer):
            maxvalue = np.max(layer)
        if minvalue > np.min(layer):
            minvalue = np.min(layer)

    global_maxavlue = max(maxvalue, -minvalue)

    scale_factor = max_fpga / global_maxavlue
    if verbose:
        print("global_maxavlue", global_maxavlue)
        print("{} bit gives you maximum interger {}".format(bitwidth, max_fpga))
        print("scale_factor", scale_factor)

    rescale_all_layer = []
    for layer in all_layers:
        rescale_layer = np.zeros((layer.shape[0], layer.shape[1]))
        for i in range(layer.shape[0]):
            for j in range(layer.shape[1]):
                if layer[i, j] != 0:
                    rescale_layer[i,j] = round(layer[i, j] * scale_factor)

        rescale_all_layer.append(rescale_layer)

    return rescale_all_layer, scale_factor


def TakeAbsoluteValues(layer):
    absolute_matrix = np.zeros((layer.shape[0], layer.shape[1]))
    sign_matrix = np.zeros((layer.shape[0], layer.shape[1]))

    for i in range(layer.shape[0]):
        for j in range(layer.shape[1]):
            if layer[i,j] >= 0:
                absolute_matrix[i,j] = layer[i,j]
                sign_matrix[i,j] = 1
            
            if layer[i,j] < 0:
                absolute_matrix[i,j] = -layer[i,j]
                sign_matrix[i,j] = 0

    return absolute_matrix, sign_matrix


# here the sequence matters, output converted_data is a list where first value is indexed 0
def GenerateDataStream(data, samples_per_cycle, process_sign, verbose=False):
    converted_data = []
    datalength = 0

    if process_sign == "value":
        for i in range(data.shape[0]):  # rows of matrix
            vector_count = 0
            datalength = 0
            for j in range(data.shape[1]):  # columns of matrix
                if data[i,j] >= 0:  
                    converted_data.append(int(round(data[i,j])))
                    datalength += 1
                    vector_count += 1
                
                if data[i,j] < 0:
                    converted_data.append(int(round(-data[i,j])))
                    datalength += 1
                    vector_count += 1
            
            if (vector_count % samples_per_cycle > 0):
                for i in range(samples_per_cycle - (vector_count % samples_per_cycle)):
                    converted_data.append(0)  # pad 0 if the samples does not occupy a full cycle
                    datalength += 1

    elif process_sign == "sign":
        for i in range(data.shape[0]):  # rows of matrix
            vector_count = 0
            datalength = 0
            for j in range(data.shape[1]):  # columns of matrix
                converted_data.append(int(data[i,j]))
                datalength += 1
                vector_count += 1
            
            if (vector_count % samples_per_cycle > 0):
                for i in range(samples_per_cycle - (vector_count % samples_per_cycle)):
                    converted_data.append(0)
                    datalength += 1
        
    if datalength % samples_per_cycle > 0:
        for i in range(samples_per_cycle - (datalength % samples_per_cycle)):
            converted_data.append(0)
            
    if verbose:
        print("initial data samples: {}, generated data samples:{}, taking {} cycles at {} samples/cycle".format(data.shape[0]*data.shape[1], len(converted_data), len(converted_data)/samples_per_cycle, samples_per_cycle))
    
    return converted_data
