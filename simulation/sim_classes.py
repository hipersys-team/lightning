from typing import List, Dict
from dnn_classes import ReadableModel
import math

class Processor():
    '''
    Specifications of processor to be simulated
    '''
    def __init__(self, name:str, num_cores:int, freq:float, dpl:Dict[str,int], overhead_factor=0.) -> None:
        '''
        Parameters
        ----------
        name: label for processor
        num_cores: number of cores available for simulation
        freq: clock frequency (in GHz)
        dpl: table of datapath latencies (in ns)
        overhead_factor: coefficient multiplied by layer input size to determine overhead
        '''
        self.name = name
        self.num_cores = num_cores
        self.freq = freq
        self.dpl = dpl.copy()
        for m in self.dpl:
            self.dpl[m] = math.ceil(self.dpl[m] * freq) # ns -> ts (rounded up to nearest timeslot)
        self.overhead_factor = overhead_factor


class Event():
    '''
    Event for scheduling to simulator
    '''
    def __init__(self, start_t:int, req_id:int) -> None:
        '''
        Parameters
        ----------
        start_t: time (in ts of simulator) of event's arrival (non-negative integer)
        req_id: identifier of request that the event is associated with (a natural number)
        '''
        self.start_t = start_t
        self.req_id = req_id


class Task():
    '''
    Represents VVP in a layer of DNN
    '''
    def __init__(self, req_id:int, layer_id:int, size:int) -> None:
        '''
        Parameters
        ----------
        req_id: identifier of request that the event is associated with (a natural number)
        layer_id: id of layer in model this Task is associated with
        size: number of element-element multiplication operations in VVP
        '''
        self.req_id = req_id
        self.layer_id = layer_id
        self.size = size


class Job(Event):
    '''
    Represents a layer in DNN
    '''
    def __init__(self, start_t:int, req_id:int, layer_id: int, vvps:int, input_size:int,task=None,is_first_job=False) -> None:
        '''
        Parameters
        ----------
        start_t: see Event spec
        req_id: see Event spec
        layer_id: id of layer in model this Job represents
        vvps: number of VVPs in layer
        input_size: duration of each VVP (in ts)
        task: Task object associated with VVP
        is_first_job: whether this Job is a first layer in its DNN request
        '''
        super().__init__(start_t, req_id)
        self.layer_id = layer_id
        self.vvps = vvps
        self.input_size:int = input_size
        if task == None:
            self.task = Task(req_id, layer_id, input_size)
        else:
            self.task = task
        self.is_first_job = is_first_job


class JobEnd(Event):
    '''
    Represents end of a Job (DNN layer)
    '''
    def __init__(self, start_t:int, req_id:int, layer_id:int):
        '''
        Parameters
        ----------
        start_t: see Event spec (the event is 0 cycles long)
        req_id: see Event spec
        layer_id: ID of layer in model this Job represents
        '''
        super().__init__(start_t, req_id)
        self.layer_id = layer_id


class Request(Event):
    '''
    Represents DNN with only fully-connected layers
    '''
    def __init__(self, start_t:int, model:ReadableModel, req_id:int) -> None:
        '''
        Parameters
        ----------
        start_t: see Event spec
        layers: list of tuples that outline the input size and number of VVPs for each layer
        req_id: see Event spec
        '''
        super().__init__(start_t, req_id)
        self.model = ReadableModel(model.name, model.layer_index, model.prereqs, model.independent_layers) # copied to prevent aliasing

    def gen_first_jobs(self, curr_time:int, overhead_factor=0.) -> List[Job]:
        '''
        Parameters
        ----------
        curr_time: simulator's time (in ts)
        overhead_factor: coefficient multiplied by layer input size to determine overhead

        Returns
        -------
        jobs: Jobs corresponding to first layers of request (that are not dependent on each other/parallel)
        '''
        first_layers = [(layer_id, *self.model.layer_index[layer_id]) for layer_id in self.model.independent_layers]
        jobs = []
        for layer_id, input_size, vvps, children in first_layers:
            overhead_time = math.ceil(overhead_factor*input_size)
            jobs.append(Job(curr_time+overhead_time, self.req_id, layer_id, vvps, input_size, is_first_job=True))
        return jobs