# emulating the computation process of a photonic deep neural network
# authors: Zhizhen Zhong (zhizhenz@mit.edu)

import math
import numpy as np
from lightning_tensorizer import *

def identity(inputs): return inputs

def ReLU(inputs):
    outputs = []
    for input in inputs:
        if input > 0: outputs.append(input)
        else: outputs.append(0)

    return outputs

def rtl_normalize(l):
    largest_product = max([abs(n) for n in l])
    highest_set_bit = -1
    normalized_products = []

    while largest_product > 0:
        highest_set_bit += 1
        largest_product >>= 1

    if highest_set_bit == -1: return l

    for product in l:
        if highest_set_bit < 8:
            normalized_products.append(product << (7 - highest_set_bit))
        else: normalized_products.append(product >> (highest_set_bit - 7))

    return normalized_products

def perfect_normalize(l):
    rescaled, _sf = RescaleData([np.array(l).reshape((1,len(l)))], 8)
    return list(rescaled[0][0])

class PhotonicMultiplier:
    def __init__(self, weights, signs, nonlinear_methods, normalize):
        self._weights = weights
        self._signs = signs
        self._nonlinear = nonlinear_methods
        self._normalize = normalize

        assert len(weights) == len(nonlinear_methods), \
          f"passed {len(nonlinear_methods)} for {len(weights)} layers"

        self.reset()

    def reset(self): self._step = 0

    def _8bit_multiply(self, input):
        weight = self._weights[self._step]
        products = []

        for w in range(len(weight)):
            dac0 = math.floor(input[w % len(input)])
            dac1 = math.floor(weight[w])
            products.append((dac0 * dac1) >> 8)
            # print(f"{hex(dac0)} * {hex(dac1)} = {hex(products[-1])}")

        return products

    def _integrate_products(self, products, layersize):
        integrated_products = []
        sign = self._signs[self._step]

        for i in range(0, len(products), layersize):
            unsigned_slice = products[i:i + layersize]
            slice = []
            for j in range(len(unsigned_slice)):
                if sign[i + j]: slice.append(unsigned_slice[j])
                else: slice.append(-unsigned_slice[j])

            integrated_products.append(sum(slice))

        return integrated_products

    def step(self, inputs):
        assert self._step < len(self._weights), \
          f"tried to do more photonic multiplication runs than exist layers!"

        while len(inputs) % 16 != 0: inputs.append(0)

        raw_products = self._8bit_multiply(inputs)
        integrated_products = self._integrate_products(raw_products, len(inputs))
        #print(f"integrated products: {integrated_products}")
        nonlinear_outputs = self._nonlinear[self._step](integrated_products)
        #print(f"nonlinear outputs: {nonlinear_outputs}")
        normalized_products = self._normalize(nonlinear_outputs)
        # print(f"normalized products: {[hex(i) for i in normalized_products]}")

        self._step += 1
        #print(normalized_products)
        return normalized_products

def emulate_lightning(input_vector, weights, signs, perfect_scaling):
    # if perfect scaling is set to true, emulate the lightning RTL but
    # use a perfect floating point rescaling factor through the computation
    # otherwise, use the in-situ normalization method used in our RTL
    if perfect_scaling: normalize = perfect_normalize
    else: normalize = rtl_normalize

    nonlinear_method_by_layer = (ReLU, ReLU, identity)
    multiplier = PhotonicMultiplier(weights, signs, nonlinear_method_by_layer, normalize)

    l1o = multiplier.step(input_vector)
    l2i = ReLU(l1o)

    l2o = multiplier.step(l2i)
    l3i = ReLU(l2o)

    l3o = multiplier.step(l3i)
    return l1o, l2i, l2o, l3i, l3o

# Run inference using purely matrix multiplication and nonlinear
# This is purely computation without framework (tensorflow, pytorch, etc) support
def run_full_precision(input_0, layers):
    input_0 = np.array(input_0)

    # first layer
    output_0 = np.matmul(layers[0], input_0)
    input_1 = ReLU(output_0)

    # second layer
    output_1 = np.matmul(layers[1], input_1)
    input_2 = ReLU(output_1)

    # third layer
    output_2 = np.matmul(layers[2], input_2)

    return output_0, input_1, output_1, input_2, output_2
