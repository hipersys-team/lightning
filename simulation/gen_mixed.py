from typing import List, Tuple
from models import MODELS
import numpy as np
import pickle
import argparse

def gen_mixed_arrivals(order:List[str], network_speed:float, poisson=True) -> List[Tuple[str,float]]:
    '''
    Generates a sequence of mixed arrivals

    Parameters
    ----------
    order: names of models in the order to be scheduled
    network_speed: network speed in Gbps
    poisson: whether interarrivals should be randomly distributed (otherwise even)

    Returns
    -------
    schedule: sequence of mixed arrivals (model names and their arrival times in ns)
    '''
    schedule = []
    time = 0
    for model_name in order:
        bit_stream = MODELS[model_name].input_size*8 # bytes => bits
        interarrival_space = bit_stream / network_speed # in ns
        if poisson:
            time += round(np.random.exponential(interarrival_space))
        else:
            time += interarrival_space
        schedule.append((model_name, time))
    return schedule

def ParseOpt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--network_speed', type=float, help="network speed (in Gbps)")
    parser.add_argument('--pkl_num', type=int, help="request schedule pickle file identifier")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()

    return opt

if __name__=="__main__":
    opt = ParseOpt()
    with open(f"./orders/order_{opt.pkl_num}.pkl", "rb") as file:
        rand_order = pickle.load(file)
    sched = gen_mixed_arrivals(rand_order, opt.network_speed)
    sched_filename = f'sim_scheds/mixed_sched_{opt.network_speed}_Gbps_{opt.pkl_num}.pkl'
    with open(sched_filename, 'wb') as file:
        pickle.dump(sched, file)
    print(f"Schedule accessible at {sched_filename}")