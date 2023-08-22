# lightning model accuracy checker
# authors: Jay Lang (jaytlang@mit.edu), Zhizhen Zhong (zhizhenz@mit.edu)

import getopt
import multiprocessing
import numpy as np
import os
import pickle
import re
import subprocess
import sys
import torch

from lightning_tensorizer import *
from lightning_emulator import *

supported_models = ["lenet"]

specific_image = None
model = None
trials = 50
parallelism = 4
verbose = False

help = f"""
usage: {sys.argv[0]} [-hv] [-i image] [-m model] [-n parallelism] [-t trials]
options:
    -h: show this help message and exit
    -i: only check a single image then exit. -m must be specified.
    -m: only check a single model. the default is to check all models.
    -n: run n simulation processes in parallel. defaults to {parallelism}
    -t: test the first t inputs. defaults to {trials}.
    -v: be verbose during simulation (print when trials are being run)
"""

def usage(why):
    if why is not None: print(why)
    print(help)
    sys.exit(2)

def parse_arguments():
    global specific_image
    global model
    global trials
    global parallelism
    global verbose

    try: args, trailing = getopt.getopt(sys.argv[1:], 'hvi:m:n:t:')
    except getopt.GetoptError as err: usage(err)

    if trailing: usage(f"unrecognized trailing arguments {trailing}")

    for o, a in args:
        if o == '-h':
            print(help)
            sys.exit(0)
        elif o == '-i':
            try: specific_image = int(a)
            except ValueError: usage("-i requires an integer argument")
        elif o == '-m':
            model = a
            if model not in supported_models:
                usage(f"unrecognized model {model}")
        elif o == "-n":
            try: parallelism = int(a)
            except ValueError: usage("-n requires an integer argument")

            if parallelism <= 0: usage("you must have parallelism of at least one")
        elif o == "-t":
            try: trials = int(a)
            except ValueError: usage("-t requires an integer argument")

            if trials <= 0: usage("you must run >= 1 trials")

        elif o == "-v": verbose = True

        else: usage(f"unrecognized option {o}")

    if specific_image is not None:
        if model is None: usage("-m must be specified with -i")
        trials = specific_image + 1
        parallelism = 1

# we expect one file per layer, such that
# when we sort the files alphanumerically the
# first layer comes first. example: fc_1.p, fc_2.p, ...
#
# returns original layers, scaled layers, signs
def load_layers(model):
    matrix_layers = []
    datapath = f"../../data/saved_models/{model}"

    dirents = [f"{datapath}/{d}" for d in sorted(os.listdir(datapath))]
    layerfiles = [d for d in dirents if os.path.isfile(d)]
    print("layerfiles", layerfiles)

    for layerfile in layerfiles:
        with open(layerfile, 'rb') as f: layer = pickle.load(f)
        matrix_layers.append(layer)

    # rescale and derive scale factors
    scaled_layers, _scale_factor = RescaleData(matrix_layers, 8)

    # signing stuff
    sign_layers = []
    for i in range(len(scaled_layers)):
        abs_layer, raw_sign_layer = TakeAbsoluteValues(scaled_layers[i])
        scaled_layers[i] = GenerateDataStream(abs_layer, 16, "value")
        sign_layers.append(GenerateDataStream(raw_sign_layer, 16, "sign"))

    return matrix_layers, scaled_layers, sign_layers

def load_data_common(magic):
    data, _scale_factor = RescaleData(magic, 8)

    i = 0
    images = []

    for image in data:
        abs_data, _signs = TakeAbsoluteValues(image)
        images.append(GenerateDataStream(abs_data, 16, "value"))

        i = i + 1
        if i >= trials: break

    if trials > i:
        if specific_image is None:
            usage(f"more trials specified than images available")
        else: usage(f"image {specific_image} does not exist")

    return images

# due to the existence of the 'magic' line which is spooky
# medieval witchcraft to me, i'm just gonna write a separate
# function for each model's data loader
def load_data_lenet():
    path = "../../data/saved_datasets/lenet/mnistdata.p"
    with open(path, 'rb') as f: data = pickle.load(f)

    # frankly, this is a terrifying line of code
    magic = [np.array(data[i,:,:]).reshape(1, (data[i,:,:].shape[0]*data[i,:,:].shape[1])) for i in range(10000)]
    return load_data_common(magic)

def load_data_iot():
    # spooky black magic
    data = torch.load("../../data/saved_datasets/iot/X_test.pth", map_location=torch.device('cpu'))
    magic = [np.array(data[i,:]).reshape(1, (data[i,:].shape[0])) for i in range(10000)]
    return load_data_common(magic)

def load_labels(model):
    path = f"../../data/saved_datasets/{model}/label.p"
    with open(path, 'rb') as f: return pickle.load(f)

def emulate_full_precision(image, model):
    _o0, _i1, _o1, _i2, output = run_full_precision(image, model)

    maxval = -float("inf")
    maxindex = 0

    for i in range(len(output)):
        if output[i] > maxval:
            maxindex = i
            maxval = output[i]

    return maxindex

# emulating 8-bit lightning
def emulate(image, model, layers, sign_layers, perfect_scaling):
    _o1, _i2, _o2, _i3, output = emulate_lightning(image, layers, sign_layers, perfect_scaling)

    maxval = -float("inf")
    maxindex = 0

    for i in range(len(output)):
        if output[i] > maxval:
            maxindex = i
            maxval = output[i]

    return maxindex

# performing verilator simulation
def simulate(index, model):
    if verbose: print(f"simulating input {index}...")
    p = subprocess.run(f"./obj_dir/V{model}_sim {index} | grep 'Result'",
        shell=True,
        capture_output=True,
        check=True)
    output = str(p.stdout, encoding='ascii')

    i = 0
    maxindex = 0
    maxvalue = -float("inf")

    for line in output.splitlines():
        value = int(re.search(r'-?\d+$', line).group())
        if value > maxvalue:
            maxindex = i
            maxvalue = value

        i = i + 1
        if i >= 10: break

    return maxindex

def accuracy_check(model):
    print(f"starting accuracy checking for {model}")

    matrix_layers, scaled_layers, sign_layers = load_layers(model)
    labels = load_labels(model)

    if model == "iot": images = load_data_iot()
    elif model == "lenet": images = load_data_lenet()

    baseline = f"baseline correctness for emulated {model} under full precision"
    qc = f"correctness of emulated {model} RTL against ground truth"
    sc = f"correctness of emulated {model} RTL if perfect scaling is used"
    lc = f"correctness of simulated {model} against ground truth"
    sa = f"match rate of simulated {model} with emulated RTL with perfect scaling"
    la = f"match rate of simulated {model} with emulated RTL (must be 100)"

    stats = {baseline: 0, qc: 0, sc: 0, lc: 0, sa: 0, la: 0}

    subprocess.run(f"make build-sim-{model}-single-core",
        shell=True,
        check=True)

    start = 0
    if specific_image is not None: start = specific_image

    if parallelism > 1:
        with multiprocessing.Pool(parallelism) as p:
            full_precision_arguments = [(images[i], matrix_layers) for i in range(start, trials)]
            emulate_rtl_arguments = [(images[i], model, scaled_layers, sign_layers, False) for i in range(start, trials)]
            emulate_ps_arguments = [(images[i], model, scaled_layers, sign_layers, True) for i in range(start, trials)]
            simulate_arguments = [(i, model) for i in range(start, trials)]

            full_precision_results = p.starmap(emulate_full_precision, full_precision_arguments)
            # actual RTL emulation
            emulate_rtl_results = p.starmap(emulate, emulate_rtl_arguments)
            # perfect scaling, everything else emulated correctly
            emulate_ps_results = p.starmap(emulate, emulate_ps_arguments)
            simulate_results = p.starmap(simulate, simulate_arguments)

    # github actions has an aneurysm whenever we try to use multiprocessing, so don't use it
    else:
        emulate_rtl_results = []
        emulate_ps_results = []
        simulate_results = []
        full_precision_results = []

        for i in range(start, trials):
            emulate_rtl_results.append(emulate(images[i], model, scaled_layers, sign_layers, False))
            emulate_ps_results.append(emulate(images[i], model, scaled_layers, sign_layers, True))
            simulate_results.append(simulate(i, model))
            full_precision_results.append(emulate_full_precision(images[i], matrix_layers))

    for i in range(len(full_precision_results)):
        index = start + i
        reported = False

        if labels[index] == full_precision_results[i]: stats[baseline] += 1
        if labels[index] == emulate_rtl_results[i]: stats[qc] += 1
        if labels[index] == emulate_ps_results[i]: stats[sc] += 1

        if emulate_ps_results[i] == simulate_results[i]: stats[sa] += 1

        if emulate_rtl_results[i] == simulate_results[i]: stats[la] += 1
        else:
            reported = True
            print(f"model {model} input {index}: MISMATCH (emulated model gives {emulate_rtl_results[i]}, verilator simulation gives {simulate_results[i]}, ground truth gives {labels[index]})")

        if labels[index] == simulate_results[i]: stats[lc] += 1
        elif not reported:
            reported = True
            print(f"model {model} input {index}: INCORRECT (emulated model gives {emulate_rtl_results[i]}, verilator simulation gives {simulate_results[i]}, ground truth gives {labels[index]})")

        if not reported and verbose: print(f"model {model} input {index}: correct")

    def report_accuracy(label):
        print(f"{label}: {stats[label] * 100 / (trials - start)}")

    report_accuracy(baseline)
    report_accuracy(qc)
    report_accuracy(sc)
    report_accuracy(lc)
    report_accuracy(sa)
    report_accuracy(la)

    # if emulation accuracy does not match simulation accuracy
    # one of the two is bugged! we cannot in good conscience
    # call the accuracy test "passed", since these two results
    # should be _exactly_ identical
    if stats[la] != trials - start:
        print(f"BUG: model {model} simulation results don't match emulation")
        sys.exit(1)

if __name__ == "__main__":
    parse_arguments()

    if model is not None: supported_models = [model]
    for model in supported_models: accuracy_check(model)
