# compiling a deep neural network into verilog data files that can be synthesized together with the Lightning logic
# authors: Zhizhen Zhong (zhizhenz@mit.edu)

import pickle
import numpy as np
from lightning_tensorizer import *
import struct

def FullLeNet(dac_bit, cycle_bit):
    # load MNIST LeNet Model
    print("1. Read LeNet model...")
    layer1 = pickle.load(open("/home/zhizhenzhong/lightning/data/saved_models/lenet/fc_1.p", "rb"))
    layer2 = pickle.load(open("/home/zhizhenzhong/lightning/data/saved_models/lenet/fc_2.p", "rb"))
    layer3 = pickle.load(open("/home/zhizhenzhong/lightning/data/saved_models/lenet/fc_3.p", "rb"))
    
    # # synthetic data  
    # layer1 = np.ones((300, 784))
    # layer2 = np.ones((100, 300))
    # layer3 = np.ones((10, 100))

    # rescale data to 8 bit accuracy
    print("2. Rescale data to map FPGA output range (8 bit)...")
    rescale_multiple_layers, scale_factor = RescaleData([layer1, layer2, layer3], dac_bit)

    # take absolute values and its signs
    print("3. Take absolute values and signs...")
    absolute_layer_1, sign_layer_1 = TakeAbsoluteValues(rescale_multiple_layers[0])
    absolute_layer_2, sign_layer_2 = TakeAbsoluteValues(rescale_multiple_layers[1])
    absolute_layer_3, sign_layer_3 = TakeAbsoluteValues(rescale_multiple_layers[2])

    print("4. [weight] generate DAC data stream...")
    samples_per_cycle = int(cycle_bit/16)
    converted_absolute_data_1 = GenerateDataStream(absolute_layer_1, samples_per_cycle, "value")
    converted_absolute_data_2 = GenerateDataStream(absolute_layer_2, samples_per_cycle, "value")
    converted_absolute_data_3 = GenerateDataStream(absolute_layer_3, samples_per_cycle, "value")

    print("5. [weight] Converting to Hex...")
    _, absolute_hex_stream_1 = Dec2Hex(converted_absolute_data_1)
    _, absolute_hex_stream_2 = Dec2Hex(converted_absolute_data_2)
    _, absolute_hex_stream_3 = Dec2Hex(converted_absolute_data_3)

    ## write sram
    sram_addr = 0
    hex_num = int(cycle_bit/4)
    
    # store SRAM data 
    with open('../sram/lenet/lut/lenet_full_absolute_{}.v'.format(cycle_bit), 'w+') as f:
        f.write("       // layer 1\n")
        for k in range(int(len(absolute_hex_stream_1)/hex_num)):
            f.write("       init_data[{}] = {}'h{};\n".format(sram_addr, int(cycle_bit), absolute_hex_stream_1[len(absolute_hex_stream_1)-hex_num*(k+1): len(absolute_hex_stream_1)-hex_num*k]))
            sram_addr += 1

        f.write("       // layer 2\n")
        for k in range(int(len(absolute_hex_stream_2)/hex_num)):
            f.write("       init_data[{}] = {}'h{};\n".format(sram_addr, int(cycle_bit), absolute_hex_stream_2[len(absolute_hex_stream_2)-hex_num*(k+1): len(absolute_hex_stream_2)-hex_num*k]))
            sram_addr += 1

        f.write("       // layer 3\n")
        for k in range(int(len(absolute_hex_stream_3)/hex_num)):
            f.write("       init_data[{}] = {}'h{};\n".format(sram_addr, int(cycle_bit), absolute_hex_stream_3[len(absolute_hex_stream_3)-hex_num*(k+1): len(absolute_hex_stream_3)-hex_num*k]))
            sram_addr += 1
        
        f.write("       // preamble\n")
        f.write("       init_data[{}] = 256'h80008000800080008000800080008000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;\n".format(sram_addr))
        f.close()
    
    # store DRAM data into bare metal
    dram_cycle_bit = 64
    dram_hex_num = int(dram_cycle_bit/4)
    with open('../../platform/ZCU111/src/lenet_full_absolute_{}.h'.format(dram_cycle_bit), 'w+') as f:
        f.write("// layer 1\n")
        for k in range(int(len(absolute_hex_stream_1)/dram_hex_num)):
            f.write("0x{},\n".format(absolute_hex_stream_1[len(absolute_hex_stream_1)-dram_hex_num*(k+1): len(absolute_hex_stream_1)-dram_hex_num*k]))

        f.write("// layer 2\n")
        for k in range(int(len(absolute_hex_stream_2)/dram_hex_num)):
            f.write("0x{},\n".format(absolute_hex_stream_2[len(absolute_hex_stream_2)-dram_hex_num*(k+1): len(absolute_hex_stream_2)-dram_hex_num*k]))

        f.write("// layer 3\n")
        for k in range(int(len(absolute_hex_stream_3)/dram_hex_num)):
            f.write("0x{},\n".format(absolute_hex_stream_3[len(absolute_hex_stream_3)-dram_hex_num*(k+1): len(absolute_hex_stream_3)-dram_hex_num*k]))

        f.close()
    
    # store DRAM data into binary file
    with open ('../../platform/ZCU111/hardware/src/lenet_full_absolute_{}.bin'.format(dram_cycle_bit), "wb") as f:
        for k in range(int(len(converted_absolute_data_1))):
            f.write(struct.pack("@H", converted_absolute_data_1[k]))
        for k in range(int(len(converted_absolute_data_2))):
            f.write(struct.pack("@H", converted_absolute_data_2[k]))
        for k in range(int(len(converted_absolute_data_3))):
            f.write(struct.pack("@H", converted_absolute_data_3[k]))
   
    ###########################################################################
    # sign streams
    print("6. [sign] Generate sign streams...")
    vector_sign_layer_1 = GenerateDataStream(sign_layer_1, samples_per_cycle, "sign")
    vector_sign_layer_2 = GenerateDataStream(sign_layer_2, samples_per_cycle, "sign")
    vector_sign_layer_3 = GenerateDataStream(sign_layer_3, samples_per_cycle, "sign")

    print("7. [sign] Writing...")
    str_vector_sign_layer_1 = ''
    for i in vector_sign_layer_1:
        str_vector_sign_layer_1 = str(int(i)) + str_vector_sign_layer_1

    str_vector_sign_layer_2 = ''
    for i in vector_sign_layer_2:
        str_vector_sign_layer_2 = str(int(i)) + str_vector_sign_layer_2

    str_vector_sign_layer_3 = ''
    for i in vector_sign_layer_3:
        str_vector_sign_layer_3 = str(int(i)) + str_vector_sign_layer_3

    # write sram
    sram_addr = 0
    with open('../sram/lenet/lut/lenet_full_absolute_sign_{}.v'.format(cycle_bit), 'w+') as f:
        f.write("       // layer 1\n")
        for k in range(int(len(str_vector_sign_layer_1)/samples_per_cycle)):
            f.write("       init_sign[{}] = {}'b{};\n".format(sram_addr, samples_per_cycle, str_vector_sign_layer_1[len(str_vector_sign_layer_1)-samples_per_cycle*(k+1): len(str_vector_sign_layer_1)-samples_per_cycle*k]))
            sram_addr += 1

        f.write("       // layer 2\n")
        for k in range(int(len(str_vector_sign_layer_2)/samples_per_cycle)):
            f.write("       init_sign[{}] = {}'b{};\n".format(sram_addr, samples_per_cycle, str_vector_sign_layer_2[len(str_vector_sign_layer_2)-samples_per_cycle*(k+1): len(str_vector_sign_layer_2)-samples_per_cycle*k]))
            sram_addr += 1
        
        f.write("       // layer 3\n")
        for k in range(int(len(str_vector_sign_layer_3)/samples_per_cycle)):
            f.write("       init_sign[{}] = {}'b{};\n".format(sram_addr, samples_per_cycle, str_vector_sign_layer_3[len(str_vector_sign_layer_3)-samples_per_cycle*(k+1): len(str_vector_sign_layer_3)-samples_per_cycle*k]))
            sram_addr += 1

        f.close()

    ###########################################################################
    print("8. [input] Generate some input image data")
    # MNIST LeNet data
    mnist_data = pickle.load(open("/home/zhizhenzhong/lightning/data/saved_activation/lenet/mnistdata.p", "rb"))

    # for now let us only consider the first 50 pictures for quick processing
    image_list = [np.array(mnist_data[i,:,:]).reshape(1, (mnist_data[i,:,:].shape[0]*mnist_data[i,:,:].shape[1])) for i in range(500)]
    # # synthetic data
    # image_list = [np.ones((1, 784)) for i in range(50)]
    rescale_multiple_images, mnist_scale = RescaleData(image_list, dac_bit)

    for i in range(500):
        locals()["absolute_input_"+str(i)], _= TakeAbsoluteValues(rescale_multiple_images[i])
        locals()["converted_absolute_input_"+str(i)] = GenerateDataStream(locals()["absolute_input_"+str(i)], samples_per_cycle, "value")
        _, locals()["absolute_input_hex_stream_"+str(i)] = Dec2Hex(locals()["converted_absolute_input_"+str(i)])

    # write sram
    sram_addr = 0
    with open('../sram/lenet/lut/mnist_{}.v'.format(cycle_bit), 'w+') as f:
        for i in range(500):
            f.write("       // input image {}\n".format(i))
            for k in range(int(len(locals()["absolute_input_hex_stream_"+str(i)])/hex_num)):
                f.write("       init_data[{}] = {}'h{};\n".format(sram_addr, cycle_bit, locals()["absolute_input_hex_stream_"+str(i)][len(locals()["absolute_input_hex_stream_"+str(i)])-hex_num*(k+1): len(locals()["absolute_input_hex_stream_"+str(i)])-hex_num*k]))
                sram_addr += 1
        
        f.write("       // preamble\n")
        f.write("       init_data[{}] = 256'hFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;\n".format(sram_addr))
        f.close()


if __name__ == "__main__":
    FullLeNet(dac_bit = 8, cycle_bit=256)