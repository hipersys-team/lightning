from typing import Tuple, List, Dict, Set

class Layer():
    '''
    Layer of DNN
    '''
    def __init__(self, id, prereqs, children):
        self.id = id
        self.prereqs = prereqs
        self.children = children


class ConvLayer(Layer):
    '''
    Convolutional layer of DNN
    '''
    def __init__(self, id:int, input_channels:int, kernel_size:int, output_shape:Tuple[int,int,int,int], prereqs=set(), children=[]) -> None:
        '''
        Parameters
        ----------
        input_channels: number of channels of input to layer
        kernel_size: size of kernel edge used in convolution on input in this layer
        output_shape: tuple (batch_num, output_channels, output_height, output_width)
        '''
        super().__init__(id, prereqs, children)
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.output_shape = output_shape


class FCLayer(Layer):
    '''
    Fully-connected layer of DNN
    '''
    def __init__(self, id:int, input_size:int, output_size:int, prereqs=set(), children=[]) -> None:
        '''
        Parameters
        ----------
        input_size: size of vectors being multiplied in layer
        output_size: number of products computed in layer
        '''
        super().__init__(id, prereqs, children)
        self.input_size = input_size
        self.output_size = output_size


class Model():
    '''
    DNN Architecture
    '''
    def __init__(self, input_dims:Tuple[int,int], input_channels:float, layers:List[Layer]) -> None:
        '''
        Parameters
        ----------
        input_dims: height and width of input
        input_channels: number of bytes the channels occupy
        layers: list of layers that make up DNN
        '''
        self.input_size = input_dims[0] * input_dims[1] * input_channels # in bytes
        self.layers = layers


class ReadableModel():
    '''
    DNN in simulation-friendly format
    '''
    def __init__(self, name:str, layer_index:Dict[int,Tuple[int,int,List[int]]], prereqs:Dict[int,Set[int]], independent_layers:Set[int]) -> None:
        '''
        Parameters
        ----------
        name: name of DNN
        layer_index: table of layer id -> (vector_len, VVPs, children)
        prereqs: table of layer id -> layers they're dependent on (that haven't been computed yet)
        independent_layers: set of layers that don't have any prereqs
        '''
        self.name = name
        self.layer_index = layer_index
        self.prereqs = prereqs
        self.independent_layers = independent_layers