from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import argparse

def congestion_plot(lightning_over_t:List[Tuple[int,int]], \
                    a100_over_t:List[Tuple[int,int]], \
                    dpu_over_t:List[Tuple[int,int]], \
                    brainwave_over_t:List[Tuple[int,int]], \
                    out_filepath:str,
                    network_speed:float) -> None:
    lightning_x_1 = [t[0] for t in lightning_over_t]
    lightning_y_1 = [t[1] for t in lightning_over_t]
    a100_x_1 = [t[0] for t in a100_over_t]
    a100_y_1 = [t[1] for t in a100_over_t]
    a100x_x_1 = [t[0] for t in dpu_over_t]
    a100x_y_1 = [t[1] for t in dpu_over_t]
    brainwave_x_1 = [t[0] for t in brainwave_over_t]
    brainwave_y_1 = [t[1] for t in brainwave_over_t]

    fig, ax = plt.subplots()
    ax.plot(a100_x_1, a100_y_1, label="A100")
    ax.plot(a100x_x_1, a100x_y_1, label="A100X")
    ax.plot(brainwave_x_1, brainwave_y_1, label="Brainwave")
    ax.plot(lightning_x_1, lightning_y_1, label="Lightning")

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Active Requests')
    ax.set_title(f'Active Requests vs Time for {network_speed}Gbps')
    ax.legend()
    plt.xscale("log")

    plt.savefig(out_filepath)
    print(f"Output accessible in {out_filepath}")

def ParseOpt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help="maximum batch size for processor")
    parser.add_argument('--lightning_core_count', type=int, help="number of cores for Lightning")
    parser.add_argument('--num_reqs', type=int, help="exact number of requests to simulate")
    parser.add_argument('--network_speed', type=str, help="network speed (in Gbps)")
    parser.add_argument('--pkl_num', type=int, help="request schedule pickle file identifier")
    parser.add_argument('--preemptive', type=str, help='P if preemptive scheduling (NP otherwise)')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()

    return opt

if __name__=="__main__":
    opt = ParseOpt()

    lightning_over_t = []
    a100_over_t = []
    dpu_over_t = []
    brainwave_over_t = []
    network_speed= opt.network_speed
    with open(f"./results/active_reqs/lightning_{network_speed}_Gbps_l{opt.lightning_core_count}_cores_{opt.num_reqs}_reqs_{opt.batch_size}_BS_{opt.pkl_num}.csv", "r") as file:
        lines = file.readlines()
        for line in lines:
            line_as_list = line.split(',')
            lightning_over_t.append((float(line_as_list[0]),int(line_as_list[1])))
    with open(f"./results/active_reqs/a100_{network_speed}_Gbps_l{opt.lightning_core_count}_cores_{opt.num_reqs}_reqs_{opt.batch_size}_BS_{opt.pkl_num}.csv", "r") as file:
        lines = file.readlines()
        for line in lines:
            line_as_list = line.split(',')
            a100_over_t.append((float(line_as_list[0]),int(line_as_list[1])))
    with open(f"./results/active_reqs/dpu_{network_speed}_Gbps_l{opt.lightning_core_count}_cores_{opt.num_reqs}_reqs_{opt.batch_size}_BS_{opt.pkl_num}.csv", "r") as file:
        lines = file.readlines()
        for line in lines:
            line_as_list = line.split(',')
            dpu_over_t.append((float(line_as_list[0]),int(line_as_list[1])))
    with open(f"./results/active_reqs/brainwave_{network_speed}_Gbps_l{opt.lightning_core_count}_cores_{opt.num_reqs}_reqs_{opt.batch_size}_BS_{opt.pkl_num}.csv", "r") as file:
        lines = file.readlines()
        for line in lines:
            line_as_list = line.split(',')
            brainwave_over_t.append((float(line_as_list[0]),int(line_as_list[1])))
    out_filepath = f"final_{opt.preemptive}_request_count_vs_time_{network_speed}_Gbps_l{opt.lightning_core_count}_{opt.batch_size}_BS_{opt.num_reqs}_reqs_{opt.pkl_num}.png"
    congestion_plot(lightning_over_t, a100_over_t, dpu_over_t, brainwave_over_t, out_filepath, opt.network_speed)