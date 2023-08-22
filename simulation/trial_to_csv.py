from typing import Tuple
import re
import ast
import argparse

def trial_parser(filename:str) -> Tuple[str,str]:
    '''
    Parses trial file into two strings: one for average request completion times (and runtime) and one for active request count over time

    Parameters
    ----------
    filename: name of file where trial stored

    Returns
    -------
    str1: average request completion times (and runtime)
    str2: active request count over time
    '''
    total_runtime = None
    avg_req_completion = None
    active_reqs_v_time = None
    finished = False
    with open(filename, "r") as trial:
        lines = trial.readlines()
        for line in lines:
            if line.startswith("Total runtime:"):
                finished = True
                pattern = r"\d+\.\d+"
                match = re.search(pattern, line)
                if match:
                    total_runtime = float(match.group())
            elif line.startswith("Average request times (in ns):"):
                avg_req_completion = ast.literal_eval(line[31:])
            elif line.startswith("Request count over time (in ns):"):
                active_reqs_v_time = ast.literal_eval(line[33:])
    if finished:
        str1 = f"Runtime,{total_runtime}\n"

        for model_name in avg_req_completion:
            str1 += f"{model_name},{avg_req_completion[model_name]}\n"
        str2 = '\n'.join(','.join(map(str, tpl)) for tpl in active_reqs_v_time)
        return str1, str2
    else:
        req_completions = {}
        with open(filename, "r") as trial:
            lines = trial.readlines()
            arrivals = []
            for line in lines:
                if line.startswith("Arrival times (in ts):"):
                    arrivals = ast.literal_eval(line[23:])
                elif line.startswith("Total req time for"):
                    tokens = line.split()
                    req_id = int(tokens[4][:-1])
                    rt = float(tokens[5])
                    model_name = arrivals[req_id-1][0]
                    if model_name in req_completions:
                        req_completions[model_name].append(rt)
                    else:
                        req_completions[model_name] = [rt]
        s = ""
        for model_name in req_completions:
            s += f"{model_name},{sum(req_completions[model_name])/len(req_completions[model_name])}\n"
        return s, ""

def ParseOpt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help="maximum batch size for processor")
    parser.add_argument('--lightning_core_count', type=int, help="number of cores for Lightning")
    parser.add_argument('--num_reqs', type=int, help="exact number of requests to simulate")
    parser.add_argument('--network_speed', type=str, help="network speed (in Gbps)")
    parser.add_argument('--pkl_num', type=int, help="request schedule pickle file identifier")
    parser.add_argument('--processor', type=str, help="name of processor")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()

    return opt

if __name__=="__main__":
    opt = ParseOpt()
    filename = f"trials/{opt.processor}_{opt.network_speed}_Gbps_l{opt.lightning_core_count}_cores_{opt.num_reqs}_reqs_{opt.batch_size}_BS_{opt.pkl_num}.txt"
    str1, str2 = trial_parser(filename)
    with open(f"results/runtimes/{opt.processor}_{opt.network_speed}_Gbps_l{opt.lightning_core_count}_cores_{opt.num_reqs}_reqs_{opt.batch_size}_BS_{opt.pkl_num}.csv", "w") as file:
        file.write(str1)
    with open(f"results/active_reqs/{opt.processor}_{opt.network_speed}_Gbps_l{opt.lightning_core_count}_cores_{opt.num_reqs}_reqs_{opt.batch_size}_BS_{opt.pkl_num}.csv", "w") as file:
        file.write(str2)