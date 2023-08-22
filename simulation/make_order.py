from typing import List
import argparse
import random
import pickle

def gen_random_order(num_reqs:int, possible_models:List[str]) -> List[str]:
    '''
    Generates a random order of DNN requests

    Parameters
    ----------
    num_reqs: number of request in order
    possible_models: names of DNNs to randomly schedule

    Returns
    -------
    o: ordering of `num_reqs` number of DNN requests
    '''
    return [random.choice(possible_models) for _ in range(num_reqs)]

def ParseOpt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_reqs', type=int, help="exact number of requests to simulate")
    parser.add_argument('--pkl_num', type=int, help="request order pickle file identifier")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()

    return opt

if __name__=="__main__":
    opt = ParseOpt()
    possible_models = ["AlexNet", "ResNet-18", "VGG-16", "VGG-19", "BERT", "GPT-2", "DLRM"]
    rand_order = gen_random_order(opt.num_reqs, possible_models)
    with open(f'orders/order_{opt.pkl_num}.pkl', 'wb') as file:
        pickle.dump(rand_order, file)