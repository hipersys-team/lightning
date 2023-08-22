from sim_classes import Processor, Request, Job, Event, Task, JobEnd
from dnn_classes import ReadableModel
from collections import deque
from heapq import merge
from typing import List, Dict, Tuple, Set, Union
import math
from models import build_model
from utils import gen_jobs
import argparse
import time
import pickle

PROCESSORS = {
    'Brainwave': Processor('Brainwave', 96000, 0.25, {
        "AlexNet": 0,
        "ResNet-18": 0,
        "VGG-16": 0,
        "VGG-19": 0,
        "BERT": 0,
        "GPT-2": 0,
        "ChatGPT": 0,
        "DLRM": 0
    }),
    'A100': Processor('A100', 6912, 1.41, {
        "AlexNet": 581000,
        "ResNet-18": 615000,
        "VGG-16": 607000,
        "VGG-19": 596000,
        "BERT": 1176000,
        "GPT-2": 6605000,
        "ChatGPT": 6605000*116, # estimated based on model size
        "DLRM": 6605000*2
    }),
    'P4': Processor('P4', 2560, 1.114, {
        "AlexNet": 1600000,
        "ResNet-18": 1532000,
        "VGG-16": 1555000,
        "VGG-19": 1552000,
        "BERT": 3702000,
        "GPT-2": 12761000,
        "ChatGPT": 12761000*116, # estimated based on model size
        "DLRM": 12761000*2
    }),
    'DPU': Processor('DPU', 6912, 1.41, {
        "AlexNet": 0,
        "ResNet-18": 0,
        "VGG-16": 0,
        "VGG-19": 0,
        "BERT": 0,
        "GPT-2": 0,
        "ChatGPT": 0,
        "DLRM": 0
    })
}

class Simulator():
    def __init__(self, processor:Processor, pkl_num:int, last_req:int, batch_size=1, preemptive=False) -> None:
        self.processor = processor                                                                                  # selected processor for simulation
        self.last_req = last_req                                                                                    # last req to end simulation on
        self.dpl = processor.dpl                                                                                    # table of datapath latencies (in ts) before every request
        self.overhead_factor = processor.overhead_factor                                                            # latency factor between layers of request (proportional to input size of next layer)
        self.cores = [Core(i) for i in range(processor.num_cores)]                                                  # cores that the simulator will run on
        self.batch_size = batch_size                                                                                # number of similar requests that are batched in cases of congestion
        self.next_core = 0                                                                                          # core that follows the last scheduled core (round-robin style)
        self.queue:deque = deque([])                                                                                # holds all scheduled events
        self.time = 0                                                                                               # simulator's internal time (in ts)
        self.preemptive = preemptive                                                                                # whether simulation employs preemptive (or non-preemptive) scheduling strategy
        self.req_id = 0                                                                                             # ID that will be assigned to the next Request
        self.req_start_times:Dict[int,int] = {}                                                                     # maps req_ids to their request's scheduled start times
        self.req_end_times:Dict[int,int] = {}                                                                       # maps req_ids to latest end times of any VVPs associated with that Request
        self.req_progress:Dict[int,ReadableModel] = {}                                                              # req_ids to Request objects (to track progress)
        self.req_layers_left:Dict[int,int] = {}                                                                     # req_ids to integer representing number of layers left
        self.req_times:Dict[str, List[float]] = {}                                                                  # completion times of finished requests (at current time) for each model
        self.curr_req_count = 0                                                                                     # number of DNN requests currently being served
        self.reqs_over_time:List[Tuple[int,int]] = []                                                               # list of tuples (time, num_active_reqs)
        self.batches:Dict[int,List[int]] = {}                                                                       # maps req_ids of the first request in a batch to other list of up to `batch_size`-1 requests of same type
        self.congested_reqs_set:Set[int] = set()                                                                    # set of IDs of Requests that arrived and haven't been scheduled to any cores
        self.congested_reqs_queue:List[int] = []                                                                    # list of IDs of Requests that arrived and haven't been scheduled to any cores (in order of arrival)
        self.first_job_seen:Set[int] = set()                                                                        # set of IDs of requests that have been seen (started)
        self.pkl_filename = f"./job_stats/jobs_{processor.name}_{batch_size}_BS_{pkl_num}_{time.time()}.pkl"        # name of file about statistics of simulation's Jobs
        self.pkl_file = None                                                                                        # object for file about statistics of simulation's Jobs
        self.earliest_core_avail_t = 0                                                                              # time (in ts) of the next core availability (updated real-time in non-preemptive simulation)
        self.req_absolute_start_times = {}                                                                          # maps req_ids to their start times (in ns)
        self.outstanding_reqs = set([i for i in range(last_req)])                                                   # set of IDs for outstanding requests

    def schedule_request(self, model:ReadableModel, start_t:int) -> None:
        '''
        Parameters
        ----------
        model: object that represents request's layer dependency graph
        start_t: time (in ts of simulator) of request's arrival
        '''
        self._merge_into_queue([Request(start_t, model, self.req_id)])
        self.req_id += 1

    def _merge_into_queue(self, events:List[Event]) -> None:
        '''
        Parameters
        ----------
        events: list of events to merged into queue (by start_t and then req_id for tiebreaking)
        '''
        # added left-handedly to prioritize recents
        self.queue = deque(merge(events, self.queue, key=lambda event:(event.start_t, event.req_id)))

    def _handle_request(self, req:Request) -> None:
        '''
        Processes Request object (updating simulator state)

        Parameters
        ----------
        req: Request object corresponding to a DNN inference
        '''
        print(f'Request {req.req_id} started on {req.model.name}: {self.time} ts')
        
        self.req_start_times[req.req_id] = self.time
        self.req_progress[req.req_id] = req.model
        self.req_layers_left[req.req_id] = len(req.model.layer_index)
        self.req_absolute_start_times[req.req_id] = time.time()
        self.curr_req_count += 1
        self.reqs_over_time.append((self.time, self.curr_req_count))
        # first independent layers of DAG
        first_jobs = req.gen_first_jobs(self.time + self.dpl[req.model.name], self.overhead_factor)
        self._merge_into_queue(first_jobs)

    def _handle_job(self, job:Job) -> None:
        '''
        Processes Job object (updating simulator state) into partial Job or JobEnd

        Parameters
        ----------
        job: Job object corresponding to a DNN layer
        '''
        # PREEMPTIVE SCHEDULING (doesn't support batching)
        if self.preemptive:
            job_end = 0
            job_start = None
            idle_times = {}
            for _ in range(job.vvps):
                assigned_core_id = self.next_core % self.processor.num_cores
                core = self.cores[assigned_core_id]
                task_start = max(core.current_end_time,self.time)
                task_end, idle_time = core.schedule_vvp(job.task, self.time)
                if idle_time:
                    if idle_time in idle_times:
                        idle_times[idle_time].append(assigned_core_id)
                    else:
                        idle_times[idle_time] = [assigned_core_id]
                if not job_start:
                    job_start = task_start
                else:
                    job_start = min(job_start, task_start)
                job_end = max(job_end, task_end)
                self.next_core += 1
            num_cores_used = min(job.vvps, len(self.cores))
            self._merge_into_queue([JobEnd(job_end, job.req_id, job.layer_id)])
            self.pkl_file.write(f"{self.req_progress[job.req_id].name},{job.req_id},{job.layer_id},{job_start},{job_end},{num_cores_used}\n")
            
            if len(idle_times) > 0:
                for idle_time in idle_times:
                    core_ids = sorted(idle_times[idle_time])
                    intervals = []
                    start_interval = core_ids[0]
                    end_interval = core_ids[0]

                    for i in range(1, len(core_ids)):
                        if core_ids[i] == end_interval + 1:
                            end_interval = core_ids[i]
                        else:
                            intervals.append((start_interval, end_interval))
                            start_interval = core_ids[i]
                            end_interval = core_ids[i]
                    
                    intervals.append((start_interval, end_interval))
            
                    # self.pkl_file.write(f"{idle_time[0]},{idle_time[1]},{intervals}\n")
            
        # NON-PREEMPTIVE SCHEDULING (supports batching)
        else:
            idle_cores:Set[int] = set()
            new_earliest_core_avail_t = None
            if self.earliest_core_avail_t <= self.time:
                for core in self.cores:
                    if core.current_end_time <= self.time:
                        idle_cores.add(core.id)
                    else:
                        if new_earliest_core_avail_t != None:
                            new_earliest_core_avail_t = min(core.current_end_time, new_earliest_core_avail_t)
                        else:
                            new_earliest_core_avail_t = core.current_end_time

            job_end = None
            num_tasks = job.vvps
            is_first_job = False

            if job.is_first_job and job.req_id not in self.first_job_seen:
                if len(idle_cores) == 0:
                    if job.req_id not in self.congested_reqs_set:
                        self.congested_reqs_set.add(job.req_id)
                        self.congested_reqs_queue.append(job.req_id)
                        print("Adding to congested queue:", job.req_id)
                    is_first_job = True
                else:
                    self.first_job_seen.add(job.req_id)
                    print(f"{job.req_id} off congestion queue")
                    if job.req_id in self.congested_reqs_set and self.batch_size > 1:
                        new_congest_reqs_queue = []
                        curr_batch_size = 0
                        curr_req_type = self.req_progress[job.req_id].name
                        dependent_reqs = []
                        for req_id in self.congested_reqs_queue:
                            if curr_batch_size >= self.batch_size or self.req_progress[req_id].name != curr_req_type:
                                new_congest_reqs_queue.append(req_id)
                            else:
                                curr_batch_size += 1
                                if req_id != job.req_id:
                                    dependent_reqs.append(req_id)
                                # print(req_id)
                                self.congested_reqs_set.remove(req_id)
                        self.congested_reqs_queue = new_congest_reqs_queue
                        if len(dependent_reqs) > 0:
                            print(f"batching {job.req_id} with {dependent_reqs}")
                            self.batches[job.req_id] = dependent_reqs
                            dependent_reqs_set = set(dependent_reqs)
                            new_queue = deque([])
                            for event in self.queue:
                                if event.req_id not in dependent_reqs_set:
                                    new_queue.append(event)
                            self.queue = new_queue

            for core_id in idle_cores:
                if num_tasks == 0:
                    break
                core = self.cores[core_id]
                task_end = core.schedule_vvp(job.task, self.pkl_file, self.time)
                num_tasks -= 1
                if job_end == None:
                    job_end = task_end
                    if new_earliest_core_avail_t != None:
                        new_earliest_core_avail_t = min(job_end, new_earliest_core_avail_t)
                    else:
                        new_earliest_core_avail_t = job_end

            if num_tasks > 0:
                # print(f"partial layer {job.layer_id} of Req {job.req_id} with {num_tasks} task left: {self.time} ts")
                # Partial Job
                if new_earliest_core_avail_t:
                    self.earliest_core_avail_t = new_earliest_core_avail_t
                future_jobs = [Job(self.earliest_core_avail_t, job.req_id, job.layer_id, num_tasks, job.input_size, job.task, is_first_job)]
                while len(self.queue) > 0 and self.queue[0].req_id == job.req_id and isinstance(self.queue[0], Job) and self.queue[0].start_t == job.start_t:
                    unprocessable_job = self.queue.popleft()
                    future_jobs.append(Job(self.earliest_core_avail_t, unprocessable_job.req_id, unprocessable_job.layer_id, unprocessable_job.vvps, unprocessable_job.input_size, unprocessable_job.task, is_first_job))
                self._merge_into_queue(future_jobs)
            else:
                # print(f"finished layer {job.layer_id} of Req {job.req_id}: {self.time} ts")
                # JobEnd
                self._merge_into_queue([JobEnd(job_end, job.req_id, job.layer_id)])

    def _handle_jobend(self, jobend:JobEnd) -> None:
        '''
        Processes JobEnd object (updating simulator state)

        Parameters
        ----------
        jobend: JobEnd object corresponding to end time of a DNN layer
        '''
        req_id = jobend.req_id
        layer_id = jobend.layer_id
        vec_len, num_vvps, children = self.req_progress[req_id].layer_index[layer_id]
        self.req_layers_left[req_id] -= 1
        if len(children) > 0: # when there are still children layers
            next_layer_set = []
            for c_layer_id in children:
                self.req_progress[req_id].prereqs[c_layer_id].remove(layer_id)
                if len(self.req_progress[req_id].prereqs[c_layer_id]) == 0:
                    next_layer_set.append((c_layer_id, *self.req_progress[req_id].layer_index[c_layer_id]))
            if len(next_layer_set) > 0:
                next_jobs = gen_jobs(next_layer_set, self.time, req_id, self.overhead_factor)
                # print(f"Scheduled next jobs: {[job[0] for job in next_layer_set]} for {self.time} ts")
                self._merge_into_queue(next_jobs)
        else: # request done
            self.req_end_times[req_id] = self.time
            if self.req_layers_left[req_id] == 0:
                total_req_time = self.req_end_times[req_id] - self.req_start_times[req_id]
                new_req_times = [total_req_time / self.processor.freq] # ts -> ns
                if req_id in self.outstanding_reqs:
                    self.outstanding_reqs.remove(req_id)
                # self.pkl_file.write(f"REQ_TIME,{self.req_progress[jobend.req_id].name},{jobend.req_id},{time.time()-self.req_absolute_start_times[jobend.req_id]}\n")
                print(f"Request {req_id} done: {self.time} ts")
                print(f"Total req time for {req_id}: {total_req_time / self.processor.freq} ns")
                if req_id in self.batches:
                    for other_req_id in self.batches[req_id]:
                        other_total_req_time = self.req_end_times[req_id] - self.req_start_times[other_req_id]
                        new_req_times.append(other_total_req_time / self.processor.freq)
                        print(f"Request {other_req_id} done: {self.time} ts")
                        print(f"Total req time: {other_total_req_time / self.processor.freq} ns")

                if self.req_progress[req_id].name in self.req_times:
                    self.req_times[self.req_progress[req_id].name].extend(new_req_times)
                else:
                    self.req_times[self.req_progress[req_id].name] = new_req_times
                self.curr_req_count -= len(new_req_times)
                self.reqs_over_time.append((self.time, self.curr_req_count))
                # remove_from_layer_progress = True

    def simulate(self) -> Dict[str,float]:
        '''
        Performs simulation based on scheduled requests

        Returns
        -------
        average_request_times: dictionary of average lifetime (in ns) of a request in simulation for each DNN
        '''
        self.pkl_file = open(self.pkl_filename, "at")
        s_time = time.time()
        while self.queue and len(self.outstanding_reqs) > 0:
            event = self.queue.popleft()
            self.time = event.start_t
            if isinstance(event, Request):
                self._handle_request(event)
            elif isinstance(event, Job):
                self._handle_job(event)
            elif isinstance(event, JobEnd):
                self._handle_jobend(event)

        avg_req_times = {}
        for model in self.req_times:
            avg_req_times[model] = sum(self.req_times[model]) / len(self.req_times[model])
        print(f"Total runtime: {time.time()-s_time} secs")
        self.pkl_file.close()
        return avg_req_times
        
    def get_request_count_vs_time(self) -> List[Tuple[float,int]]:
        '''
        Returns
        -------
        request_count_vs_time: list of (time_stamp_in_ns, number_of_active_requests)
        '''
        request_count_vs_time = [(0.,0)]
        for time_ts, num_reqs in self.reqs_over_time:
            request_count_vs_time.append((time_ts / self.processor.freq, num_reqs))
        return request_count_vs_time


class Core():
    def __init__(self, core_id:int) -> None:
        self.id = core_id                                                                                           # unique identifier for core
        self.current_end_time = 0                                                                                   # time when core is no longer occupied with current task
    
    def schedule_vvp(self, task:Task, sim_time=0) -> Tuple[int,Union[None,Tuple[float,float]]]:
        '''
        Schedules a task to the core (as soon as available)

        Parameters
        ----------
        task: a VVP that wants to be processed on this core

        Returns
        -------
        task_end: details for end of just-scheduled task (unless there is already a task or event-driven simulation-mode, then None)
        '''
        # if there's a difference between sim_time > self.current_end_time, log difference
        if sim_time > self.current_end_time:
            idle_time = (self.current_end_time,sim_time)
        else:
            idle_time = None
        self.current_end_time = max(sim_time, self.current_end_time) + task.size
        return self.current_end_time, idle_time


def build_sim_for_mixed_arrivals(processor:Processor, network_speed:float, pkl_num:int, last_req:int, min_vec_len:int, batch_size=1, preemptive=False):
    '''
    Parameters
    ----------
    processor: see Simulator spec
    network_speed: network speed in Gbps
    pkl_num: request schedule pickle file identifier
    last_req: last req to end simulation on
    min_vec_len: minimum length of vector multiplication (for granularity)
    batch_size: maximum batch size for processor
    preemptive: whether simulator should use preemptive (or non-preemptive) scheduling strategy

    Returns
    -------
    simulator: Simulator object which is properly scheduled
    '''
    simulator = Simulator(processor, pkl_num, last_req, batch_size, preemptive)
    with open(f'sim_scheds/mixed_sched_{network_speed}_Gbps_{pkl_num}.pkl', 'rb') as file:
        schedule = pickle.load(file)
    for model_name, arrival_ns in schedule:
        model = build_model(model_name, min_vec_len)
        arrival_ts = math.ceil(arrival_ns * processor.freq) # ns => ts
        simulator.schedule_request(model, arrival_ts)
    print(f"Arrival times (in ts): {schedule}")
    return simulator

def ParseOpt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--processor', type=str, help='processor on which to simulate')
    parser.add_argument('--num_reqs', type=int, help="exact number of requests to simulate")
    parser.add_argument('--network_speed', type=float, help="network speed in Gbps")
    parser.add_argument('--gran', type=float, help="granularity (minimum vector length)")
    parser.add_argument('--preemptive', type=bool, help="preemptive or non-preemptive scheduling strategy")
    parser.add_argument('--batch_size', type=int, help="maximum batch size for processor")
    parser.add_argument('--pkl_num', type=int, help='request schedule pickle file identifier')
    parser.add_argument('--lightning_core_count', type=int, help='number of cores for lightning')
    parser.add_argument('--last_req', type=int, help="last req to end simulation on")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()

    return opt


def sim_mixed_arrivals_on_processor(processor_name:str, network_speed:float, pkl_num:int, last_req:int, min_vec_len=1, batch_size=1, preemptive=False) -> None:
    '''
    Simulates multiple DNN models on a processor and prints and returns the average request processing times

    Parameters
    ----------
    processor_name: name of processor
    network_speed: network speed in Gbps
    pkl_num: request schedule pickle file identifier
    last_req: last req to end simulation on
    min_vec_len: minimum length of vector multiplication (for granularity)
    batch_size: maximum batch size for processor
    preemptive: preemptive (or non-preemptive) scheduling strategy

    Returns
    -------
    average_req_times: dictionary of average lifetime (in ns) of a request in simulation for each DNN
    '''
    processor = PROCESSORS[processor_name]
    print(f"Running mixed arrivals on {processor.name}...")
    simulator = build_sim_for_mixed_arrivals(processor, network_speed, pkl_num, last_req, min_vec_len, batch_size, preemptive)
    print("Simulator scheduling complete...")
    average_req_times = simulator.simulate()
    print(f'Average request times (in ns): {average_req_times}')
    request_count_over_time = simulator.get_request_count_vs_time()
    print(f'Request count over time (in ns): {request_count_over_time}')
    return average_req_times

if __name__=="__main__":
    opt = ParseOpt()
    
    PROCESSORS['Lightning-1-200-100'] = Processor('Lightning-1-200-100', opt.lightning_core_count, 97, {
        "AlexNet": 115*8,
        "ResNet-18": 115*21,
        "VGG-16": 115*16,
        "VGG-19": 115*19,
        "BERT": 115*(1+7*24), # Encoder=4 for Self-Attention + 3 for Feed-Forward, 1 FC
        "GPT-2": 115*(2+7*48), # Encoder=4 for Self-Attention + 3 for Feed-Forward, 2 FC
        "ChatGPT": 115*24291,
        "DLRM": 115*8 # Embeddings=1, Bottom MLP=3, Top MLP=4
    })

    sim_mixed_arrivals_on_processor(opt.processor, opt.network_speed, opt.pkl_num, opt.last_req, opt.gran, opt.batch_size, opt.preemptive)