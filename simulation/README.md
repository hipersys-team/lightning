# Lightning Simulation Code
The Lightning simulations compare the performance of large DNNs on Lightning against their performance on GPUs and AI accelerators. The code in this folder reproduces results reported in Section 9 of the Lightning paper.

## Overview
Event-driven simulation occurs in two phases:

1. Scheduling DNN requests. `build_sim_for_mixed_arrivals()` does this by scheduling a particular number of different inference requests (spaced based on their associated DNN input sizes using a Poisson distribution).

2. Simulating the schedule. `Simulator.simulate()` will act out the requests at their specified times and return the average completion times per model.

Supports comparisons between Lightning, NVIDIA's A100 and A100X, and Microsoft Brainwave.

## Folder structure
|  Source Files               |Description                                                                                                                    |
|  -----                      |  -----                                                                                                                          |
|  `orders/` |  Different mixed orders of DNN requests to be converted to schedules   |
|  `sim_scheds/` |   Schedules of DNN request traces for simulation (specific to a network speed)     |
|  `congestion_plot.py` |  Plots the active DNN requests over time of a finished simulation |
|  `csv_gen.sh` |   Converts the simulator's trial outputs to CSV format (in `results/`) for further analysis   |
|  `dnn_classes.py` |   Foundational class structures for representing deep neural network (DNN) architectures |
|  `final_gen.sh` |  Bash utility for batching process reading and processing of CSV files using the `read_csv.py` Python script |
|  `gen_mixed.py` |   Converts DNN request order into a network speed-specific schedule for simulation. |
|  `make_order.py` |   Generates and saves a random order of DNN requests  |
|  `models.py` |   Provides a way to generate and represent the layers for popular DNNs such as BERT-Large, GPT-2 (Extra-Large), LeNet-300-100, AlexNet, ResNet-18, VGG-16, VGG-19, and Meta's DLRM  |
|  `read_csv.py` |  Processes runtime data for DNNs executed on different processors and then stores the results in a TSV-formatted file   |
|  `README.md` |  This file, describing the requirements and instructions to reproduce the experiments.                |
|  `requirements.txt` |  List of all of the Python dependencies |
|  `run.sh` |   Conducts a series of simulations (via the `sim.py` script) on different types of processors |
|  `sched_gen.sh` |   Generates multiple DNN request traces for simulations |
|  `sim_classes.py` |   Useful data structures for simulation  |
|  `sim.py` |  Event-driven simulator code |
|  `trial_to_csv.py` |  Parses trial file and extracts information about the simulation, specifically the average request completion times, total runtime, and the active request count over time, and stores that in CSV format. |
|  `utils.py` |  Utility functions for simulator  |

## Usage

### 1. Install requirements
Verify that you have Python3 set up and then install necessary packages with `python3 -m pip install -r requirements.txt`.

### 2. Launch simulations
Run simulations in parallel using `bash run.sh` (default configuration: 10 unique DNN traces over a 60Mbps network). Note: `run.sh` launches 40 simulations in parallel.

If you'd like new schedules, execute `sched_gen.sh` before `run.sh`.

### 3. Convert logs to CSVs
Once the 40 simulations initiated by `run.sh` are complete, execute `bash csv_gen.sh` to parse the trial logs.

### 4. Generate average speedups by DNN model
Run `bash final_gen.sh` to generate a TSV-formatted file with the average runtimes of each DNN model over each processor. 

### 5. Other useful information
`congestion_plot.py` provide plots for active DNN request count over time for a single simulation. Be sure to read its ParseOpt functions to pass in the correct arguments. `job_stats/` includes the start and end times for each DNN layer of any past simulation.