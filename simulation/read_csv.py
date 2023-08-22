import argparse

def read_csv(file_prefix:str, file_suffix:str) -> str:
    '''
    Generates list of runtimes in appropriate display form
    '''
    time_dict = {}

    for proc in ["lightning", "a100", "dpu", "brainwave"]:
        with open(file_prefix + proc + file_suffix, "r") as file:
            lines = file.readlines()
            for line in lines:
                key_val = line.split(",")
                time_dict[proc + key_val[0]] = float(key_val[1])

    output = "\tLightning\tA100\tA100X\tBrainwave\tA100/Lightning\tA100X/Lightning\tBrainwave/Lightning\n"

    max_ratios = {}

    for model in ["AlexNet", "ResNet-18", "VGG-16", "VGG-19", "BERT", "GPT-2", "DLRM"]:
        output += f"{model}"
        for proc in ["lightning", "a100", "dpu", "brainwave"]:
            output += f"\t{time_dict[proc + model]}"
        lightning_time = time_dict["lightning" + model]
        for proc in ["a100", "dpu", "brainwave"]:
            ratio = time_dict[proc+model]/lightning_time
            output += f"\t{ratio}"
            if proc in max_ratios:
                max_ratios[proc] = max(max_ratios[proc], ratio)
            else:
                max_ratios[proc] = ratio
        output += "\n"

    output += "Max\t\t\t\t\t"
    for proc in ["a100", "dpu", "brainwave"]:
        output += f"\t{max_ratios[proc]}"

    return output

def ParseOpt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help="maximum batch size for processor")
    parser.add_argument('--lightning_core_count', type=int, help="number of cores for Lightning")
    parser.add_argument('--num_reqs', type=int, help="exact number of requests to simulate")
    parser.add_argument('--network_speed', type=float, help="network speed (in Gbps)")
    parser.add_argument('--pkl_num', type=int, help="request schedule pickle file identifier")
    parser.add_argument('--preemptive', type=str, help="whether there is a preemptive flag")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()

    return opt

if __name__=="__main__":
    opt = ParseOpt()
    prefix = f"results/runtimes/"
    new_str = ""
    for pkl_num in range(1,11):
        try:
            suffix = f"_{opt.network_speed}_Gbps_l{opt.lightning_core_count}_cores_{opt.num_reqs}_reqs_{opt.batch_size}_BS_{pkl_num}.csv"
            new_str += read_csv(prefix, suffix) + "\n"
            print(f"SUCCESS: {opt.preemptive}, {opt.network_speed}Gbps, {opt.lightning_core_count} cores, {opt.num_reqs} reqs, {opt.batch_size} BS, {pkl_num} pickle num")
        except:
            print(f"Result for {opt.preemptive}, {opt.network_speed}Gbps, {opt.lightning_core_count} cores, {opt.num_reqs} reqs, {opt.batch_size} BS, {pkl_num} pickle num failed...")
            continue
    out_filename = f"results/runtimes/P_" + "final" + f"_{opt.network_speed}_Gbps_l{opt.lightning_core_count}_cores_{opt.num_reqs}_reqs_{opt.batch_size}_BS.tsv"
    with open(out_filename, "w") as file:
        file.write(new_str)
    print(f"Output accessible at {out_filename}.")