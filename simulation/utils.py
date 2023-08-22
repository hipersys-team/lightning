from typing import List, Tuple
from sim_classes import Job
import math

def gen_jobs(layer_set:List[Tuple[int,int,int,List[int]]], curr_time:int, req_id:int, overhead_factor=0.):
    '''
    Generates Jobs given a list of parallel layers

    Parameters
    ----------
    layer_set: list of layer dimensions (layer_id, input_size, num_vvps, children) to be executed in parallel (no dependencies)
    curr_time: simulator's current time (in ts)
    req_id: id of associated Request
    overhead_factor: see spec for Simulator.__init__()

    Returns
    -------
    jobs: list of appropriately-scheduled Jobs corresponding to layer information
    '''
    jobs = []
    for layer_id, input_size, vvps, children in layer_set:
        overhead_time = math.ceil(overhead_factor*input_size)
        jobs.append(Job(curr_time+overhead_time, req_id, layer_id, vvps, input_size))
    return jobs