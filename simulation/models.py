from dnn_classes import Model, ConvLayer, FCLayer, ReadableModel
from tqdm import tqdm
from typing import List
import math

def make_bert() -> List[FCLayer]:
    '''
    Generates layers for BERT-Large model.

    Returns
    -------
    layers: list of layers for BERT-Large
    '''
    children:List[int] = []
    layers = [
        FCLayer(1, 29056, 1024, children=children), # word embeddings
        # position embeddings (not multiplication)
        # token embeddings (not multiplication)
        # ^ these embeddings are summed to get input embedding
    ]
    prereqs = set([1])
    offset = 1
    # 24 encoders
    for enc_id in range(24):
        concat_prereqs = set()
        # MULTI-HEADED ATTENTION (16x parallel)
        for _ in range(16):
            layers.append(FCLayer(offset+1, 1024, int(512*1024/16), prereqs=prereqs, children=[offset+4])) # linear projection of Q (QW_i^Q)
            layers.append(FCLayer(offset+2, 1024, int(512*1024/16), prereqs=prereqs, children=[offset+4])) # linear projection of K (KW_i^K)
            layers.append(FCLayer(offset+3, 1024, int(512*1024/16), prereqs=prereqs, children=[offset+5])) # linear projection of V (VW_i^V)
            children.extend([offset+1,offset+2,offset+3])
            layers.append(FCLayer(offset+4, int(1024/16), 512*512, prereqs=set([offset+1,offset+2]), children=[offset+5])) # QW_i^Q x (KW_i^K)^T
            # scaling (not matmul)
            # softmax (not matmul)
            layers.append(FCLayer(offset+5, 512, int(512*1024/16), prereqs=set([offset+3,offset+4]), children=[enc_id*(16*5+3)+1+16*5+1])) # softmax x VW_i^V
            concat_prereqs.add(offset+5)
            offset += 5
        
        # concat x W^O
        layers.append(FCLayer(offset+1, 1024, 512*1024, prereqs=concat_prereqs, children=[offset+2]))
        # add+norm (no matmul)
        # FEED-FORWARD
        # 1. Linear Transformation
        layers.append(FCLayer(offset+2, 1024, 512*4096, prereqs=set([offset+1]), children=[offset+3]))
        # 2. Activation Function (no matmul)
        children = []
        # 3. Linear Transformation
        layers.append(FCLayer(offset+3, 4096, 512*1024, prereqs=set([offset+2]), children=children))
        prereqs = set([offset+3])
        # add+norm (no matmul)
        offset += 3
        
    # pooler (not considering)

    return layers

def make_gpt2() -> List[FCLayer]:
    '''
    Generates layers for GPT-2 (Extra-Large) model.

    Returns
    -------
    layers: list of layers for GPT-2 (Extra-Large)
    '''
    children:List[int] = []
    layers = [
        FCLayer(1, 50257, 1600, children=children), # input embeddings
        FCLayer(2, 1024, 1600, children=children) # positional embeddings
    ]
    offset = 2
    prereqs = set([1,2])
    # 48 encoders
    for enc_id in range(48):
        concat_prereqs = set()
        # MULTI-HEADED ATTENTION (25x parallel)
        for _ in range(25):
            layers.append(FCLayer(offset+1, 1600, int(1024*1600/25), prereqs=prereqs, children=[offset+4])) # linear projection of Q (QW_i^Q)
            layers.append(FCLayer(offset+2, 1600, int(1024*1600/25), prereqs=prereqs, children=[offset+4])) # linear projection of K (KW_i^K)
            layers.append(FCLayer(offset+3, 1600, int(1024*1600/25), prereqs=prereqs, children=[offset+5])) # linear projection of V (VW_i^V)
            children.extend([offset+1,offset+2,offset+3])
            layers.append(FCLayer(offset+4, int(1600/25), 1024*1024, prereqs=set([offset+1,offset+2]), children=[offset+5])) # QW_i^Q x (KW_i^K)^T
            # scaling (not matmul)
            # softmax (not matmul)
            layers.append(FCLayer(offset+5, 1024, int(1024*1600/25), prereqs=set([offset+3,offset+4]), children=[enc_id*(25*5+3)+2+25*5+1])) # softmax x VW_i^V
            concat_prereqs.add(offset+5)
            offset += 5
        # concat x W^O
        layers.append(FCLayer(offset+1, 1600, 1024*1600, prereqs=concat_prereqs, children=[offset+2]))
        # add+norm (no matmul)
        # FEED-FORWARD
        # 1. Linear Transformation
        layers.append(FCLayer(offset+2, 1600, 1024*6400, prereqs=set([offset+1]), children=[offset+3]))
        # 2. Activation Function (no matmul)
        children = []
        # 3. Linear Transformation
        layers.append(FCLayer(offset+3, 6400, 1024*1600, prereqs=set([offset+2]), children=children))
        prereqs = set([offset+3])
        # add+norm (no matmul)
        offset += 3
    layers.append(FCLayer(offset+1, 1600, 1024*50257, prereqs=set([offset])))
    children.append(offset+1)
    return layers

MODELS = {
    "LeNet-300-100": Model((28,28), 0.125, [ # 0.125 (or 1/8) channels because only uses 1 bit to encode
        FCLayer(1, 784, 300, children=[2]),
        FCLayer(2, 300, 100, prereqs=set([1]), children=[3]),
        FCLayer(3, 100, 10, prereqs=set([2]))
    ]),
    "AlexNet": Model((224,224), 3, [
        ConvLayer(1, 3, 11, (-1, 64, 55, 55), children=[2]), # layer 1
        ConvLayer(2, 64, 5, (-1, 192, 27, 27), prereqs=set([1]), children=[3]), # layer 4
        ConvLayer(3, 192, 3, (-1, 384, 13, 13), prereqs=set([2]), children=[4]), # layer 7
        ConvLayer(4, 384, 3, (-1, 256, 13, 13), prereqs=set([3]), children=[5]), # layer 9
        ConvLayer(5, 256, 3, (-1, 256, 13, 13), prereqs=set([4]), children=[6]), # layer 11
        FCLayer(6, 256*6*6, 4096, prereqs=set([5]), children=[7]), # layer 16
        FCLayer(7, 4096, 4096, prereqs=set([6]), children=[8]), # layer 19
        FCLayer(8, 4096, 1000, prereqs=set([7])) # layer 21
    ]),
    "ResNet-18": Model((224,224), 3, [
        ConvLayer(1, 3, 7, (-1, 64, 112, 112), children=[2]), # layer 1
        ConvLayer(2, 64, 3, (-1, 64, 56, 56), prereqs=set([1]), children=[3]), # layer 5
        ConvLayer(3, 64, 3, (-1, 64, 56, 56), prereqs=set([2]), children=[4]), # layer 8
        ConvLayer(4, 64, 3, (-1, 64, 56, 56), prereqs=set([3]), children=[5]), # layer 12
        ConvLayer(5, 64, 3, (-1, 64, 56, 56), prereqs=set([4]), children=[6]), # layer 15
        ConvLayer(6, 64, 3, (-1, 128, 28, 28), prereqs=set([5]), children=[7]), # layer 19
        ConvLayer(7, 128, 3, (-1, 128, 28, 28), prereqs=set([6]), children=[8]), # layer 22
        ConvLayer(8, 64, 1, (-1, 128, 28, 28), prereqs=set([7]), children=[9]), # layer 24, projection shortcuts
        ConvLayer(9, 128, 3, (-1, 128, 28, 28), prereqs=set([8]), children=[10]), # layer 28
        ConvLayer(10, 128, 3, (-1, 128, 28, 28), prereqs=set([9]), children=[11]), # layer 31
        ConvLayer(11, 128, 3, (-1, 256, 14, 14), prereqs=set([10]), children=[12]), # layer 35
        ConvLayer(12, 256, 3, (-1, 256, 14, 14), prereqs=set([11]), children=[13]), # layer 38
        ConvLayer(13, 128, 1, (-1, 256, 14, 14), prereqs=set([12]), children=[14]), # layer 40, projection shortcuts
        ConvLayer(14, 256, 3, (-1, 256, 14, 14), prereqs=set([13]), children=[15]), # layer 44
        ConvLayer(15, 256, 3, (-1, 256, 14, 14), prereqs=set([14]), children=[16]), # layer 47
        ConvLayer(16, 256, 3, (-1, 512, 7, 7), prereqs=set([15]), children=[17]), # layer 51
        ConvLayer(17, 512, 3, (-1, 512, 7, 7), prereqs=set([16]), children=[18]), # layer 54
        ConvLayer(18, 256, 1, (-1, 512, 7, 7), prereqs=set([17]), children=[19]), # layer 56, projection shortcuts
        ConvLayer(19, 512, 3, (-1, 512, 7, 7), prereqs=set([18]), children=[20]), # layer 60
        ConvLayer(20, 512, 3, (-1, 512, 7, 7), prereqs=set([19]), children=[21]), # layer 63
        FCLayer(21, 512, 1000, prereqs=set([20])) # layer 68
    ]),
    "VGG-16": Model((224,224), 3, [
        ConvLayer(1, 3, 3, (-1, 64, 224, 224), children=[2]), # layer 1
        ConvLayer(2, 64, 3, (-1, 64, 224, 224), prereqs=set([1]), children=[3]), # layer 3
        ConvLayer(3, 64, 3, (-1, 128, 112, 112), prereqs=set([2]), children=[4]), # layer 6
        ConvLayer(4, 128, 3, (-1, 128, 112, 112), prereqs=set([3]), children=[5]), # layer 8
        ConvLayer(5, 128, 3, (-1, 256, 56, 56), prereqs=set([4]), children=[6]), # layer 11
        ConvLayer(6, 256, 3, (-1, 256, 56, 56), prereqs=set([5]), children=[7]), # layer 13
        ConvLayer(7, 256, 3, (-1, 256, 56, 56), prereqs=set([6]), children=[8]), # layer 15
        ConvLayer(8, 256, 3, (-1, 512, 28, 28), prereqs=set([7]), children=[9]), # layer 18
        ConvLayer(9, 512, 3, (-1, 512, 28, 28), prereqs=set([8]), children=[10]), # layer 20
        ConvLayer(10, 512, 3, (-1, 512, 28, 28), prereqs=set([9]), children=[11]), # layer 22
        ConvLayer(11, 512, 3, (-1, 512, 14, 14), prereqs=set([10]), children=[12]), # layer 25
        ConvLayer(12, 512, 3, (-1, 512, 14, 14), prereqs=set([11]), children=[13]), # layer 27
        ConvLayer(13, 512, 3, (-1, 512, 14, 14), prereqs=set([12]), children=[14]), # layer 29
        FCLayer(14, 25088, 4096, prereqs=set([13]), children=[15]), # layer 33
        FCLayer(15, 4096, 4096, prereqs=set([14]), children=[16]), # layer 36
        FCLayer(16, 4096, 1000, prereqs=set([15])) # layer 39
    ]),
    "VGG-19": Model((224,224), 3, [
        ConvLayer(1, 3, 3, (-1, 64, 224, 224), children=[2]), # layer 1
        ConvLayer(2, 64, 3, (-1, 64, 224, 224), prereqs=set([1]), children=[3]), # layer 3
        ConvLayer(3, 64, 3, (-1, 128, 112, 112), prereqs=set([2]), children=[4]), # layer 6
        ConvLayer(4, 128, 3, (-1, 128, 112, 112), prereqs=set([3]), children=[5]), # layer 8
        ConvLayer(5, 128, 3, (-1, 256, 56, 56), prereqs=set([4]), children=[6]), # layer 11
        ConvLayer(6, 256, 3, (-1, 256, 56, 56), prereqs=set([5]), children=[7]), # layer 13
        ConvLayer(7, 256, 3, (-1, 256, 56, 56), prereqs=set([6]), children=[8]), # layer 15
        ConvLayer(8, 256, 3, (-1, 256, 56, 56), prereqs=set([7]), children=[9]), # layer 17
        ConvLayer(9, 256, 3, (-1, 512, 28, 28), prereqs=set([8]), children=[10]), # layer 20
        ConvLayer(10, 512, 3, (-1, 512, 28, 28), prereqs=set([9]), children=[11]), # layer 22
        ConvLayer(11, 512, 3, (-1, 512, 28, 28), prereqs=set([10]), children=[12]), # layer 24
        ConvLayer(12, 512, 3, (-1, 512, 28, 28), prereqs=set([11]), children=[13]), # layer 26
        ConvLayer(13, 512, 3, (-1, 512, 14, 14), prereqs=set([12]), children=[14]), # layer 29
        ConvLayer(14, 512, 3, (-1, 512, 14, 14), prereqs=set([13]), children=[15]), # layer 31
        ConvLayer(15, 512, 3, (-1, 512, 14, 14), prereqs=set([14]), children=[16]), # layer 33
        ConvLayer(16, 512, 3, (-1, 512, 14, 14), prereqs=set([15]), children=[17]), # layer 35
        FCLayer(17, 25088, 4096, prereqs=set([16]), children=[18]), # layer 39
        FCLayer(18, 4096, 4096, prereqs=set([17]), children=[19]), # layer 42
        FCLayer(19, 4096, 1000, prereqs=set([18])) # layer 45
    ]),
    "BERT": Model((512,1), 10, make_bert()), # 10 bytes per token for simplicity
    "GPT-2": Model((1024,1), 10, make_gpt2()),
    "DLRM": Model((512, 1), 10, [
        # embeddings
        FCLayer(1, 9980333, 64, children=[30]),
        FCLayer(2, 36084, 64, children=[30]),
        FCLayer(3, 17217, 64, children=[30]),
        FCLayer(4, 7378, 64, children=[30]),
        FCLayer(5, 20134, 64, children=[30]),
        FCLayer(6, 3, 64, children=[30]),
        FCLayer(7, 7112, 64, children=[30]),
        FCLayer(8, 1442, 64, children=[30]),
        FCLayer(9, 61, 64, children=[30]),
        FCLayer(10, 9758201, 64, children=[30]),
        FCLayer(11, 1333352, 64, children=[30]),
        FCLayer(12, 313829, 64, children=[30]),
        FCLayer(13, 10, 64, children=[30]),
        FCLayer(14, 2208, 64, children=[30]),
        FCLayer(15, 11156, 64, children=[30]),
        FCLayer(16, 122, 64, children=[30]),
        FCLayer(17, 4, 64, children=[30]),
        FCLayer(18, 970, 64, children=[30]),
        FCLayer(19, 14, 64, children=[30]),
        FCLayer(20, 11156, 64, children=[30]),
        FCLayer(21, 7267859, 64, children=[30]),
        FCLayer(22, 9946608, 64, children=[30]),
        FCLayer(23, 415421, 64, children=[30]),
        FCLayer(24, 12420, 64, children=[30]),
        FCLayer(25, 101, 64, children=[30]),
        FCLayer(26, 36, 64, children=[30]),
        # bottom MLP
        FCLayer(27, 13, 512, children=[28]),
        FCLayer(28, 512, 256, prereqs=set([27]), children=[29]),
        FCLayer(29, 256, 64, prereqs=set([28]), children=[30]),
        # top MLP
        FCLayer(30, 415, 512, prereqs=set([i+1 for i in range(26)]+[29]), children=[31]),
        FCLayer(31, 512, 512, prereqs=set([30]), children=[32]),
        FCLayer(32, 512, 256, prereqs=set([31]), children=[33]),
        FCLayer(33, 256, 1, prereqs=set([32]))
    ])
}   

def build_model(model_name:str, min_vec_size=1000000) -> ReadableModel:
    '''
    Generates layers of DNN model in simulation-friendly format

    Parameters
    ----------
    model_name: name of model to load
    min_vec_size: minimum length of vector multiplication (for granularity)
    
    Returns
    -------
    model: model in simulation-readable format (see ReadableModel spec)
    '''
    m = MODELS[model_name]
    layer_sets = m.layers
    layer_index = {} # layer id -> (vector_len, num_VVPs, children)
    prereqs = {} # layer -> layers they're dependent on (that haven't been computed yet)
    independent_layers = set() # layers that don't have any prereqs
    for layer in tqdm(layer_sets, desc=f'Building {model_name}...'):
        if isinstance(layer, ConvLayer):
            vector_length = layer.kernel_size ** 2
            batch_num, output_channels, output_height, output_width = layer.output_shape
            vvps = layer.input_channels * output_channels * output_height * output_width
            if vector_length >= min_vec_size:
                layer_index[layer.id] = (vector_length, vvps, layer.children.copy()) # to prevent aliasing
            else:
                layer_index[layer.id] = (min_vec_size, math.ceil(vector_length*vvps/min_vec_size), layer.children.copy())
        elif isinstance(layer, FCLayer):
            if layer.input_size >= min_vec_size:
                layer_index[layer.id] = (layer.input_size, layer.output_size, layer.children.copy()) # to prevent aliasing
            else:
                layer_index[layer.id] = (min_vec_size, math.ceil(layer.input_size*layer.output_size/min_vec_size), layer.children.copy())
        if layer.prereqs:
            prereqs[layer.id] = layer.prereqs.copy()
        else:
            independent_layers.add(layer.id)
    return ReadableModel(model_name, layer_index, prereqs, independent_layers)

if __name__ == "__main__":
    filename = "./results/power_utilization.tsv"
    data = ""
    lightning_power = {}
    for name, power, gran, cores, freq in [("Lightning", 91.875, 1, 1024, 97), ("A100", 250, 2048, 6912, 1.41), ("A100X", 300, 2048, 6912, 1.41), ("Brainwave", 125, 2048, 96000, 0.25)]:
        data += f"{name}\n"
        for model in ["AlexNet", "ResNet-18", "VGG-16", "VGG-19", "BERT", "GPT-2", "DLRM"]:
            r_model = build_model(model, gran)
            total_multiplications = 0
            for l_id in r_model.layer_index:
                total_multiplications += r_model.layer_index[l_id][0]*r_model.layer_index[l_id][1]
            print(total_multiplications)
            m_power = total_multiplications*power/cores/freq/(10**9)
            data += f"{model}\t{m_power}"
            if name == "Lightning":
                lightning_power[model] = m_power
            else:
                data += f"\t{m_power/lightning_power[model]}"
            data += "\n"
        data += "\n"
    with open(filename, "w") as file:
        file.write(data)
    print(f"Output accessible at {filename}")