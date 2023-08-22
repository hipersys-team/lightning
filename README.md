# Lightning: A Reconfigurable Photonic-Electronic SmartNIC for Fast and Energy-Efficient Inference

[![DOI:10.1145/3603269.3604821](http://img.shields.io/badge/DOI-10.1145/3603269.3604821-69B7DB.svg)](https://doi.org/10.1145/3603269.3604821)
[![DNN build](https://github.com/hipersys-team/lightning/actions/workflows/dnn_single_core.yml/badge.svg)](https://github.com/hipersys-team/lightning/actions/workflows/dnn_single_core.yml)

Welcome to the Lightning, a reconfigurable photonic-electronic neural network inference system integrated with the a 100 Gbps smartNIC.

## 1. Overview

We propose Lightning, the first reconfigurable photonic-electronic smartNIC to serve real-time deep neural network inference requests. Lightning uses a fast datapath to feed traffic from the NIC into the photonic domain without creating digital packet processing and data movement bottlenecks. To do so, Lightning leverages a novel reconfigurable count-action abstraction that keeps track of the required computation operations of each inference packet. Our count-action abstraction decouples the compute control plane from the data plane by counting the number of operations in each task and triggers the execution of the next task(s) without interrupting the dataflow. To the best of our knowledge, our prototype is the highest-frequency photonic computing system, capable of serving real-time inference queries at 4.055 GHz end-to-end.

For a full technical description on Lightning, please read our ACM SIGCOMM 2023 paper and demo:

> Z. Zhong, M. Yang, J. Lang, C. Williams, L. Kronman, A. Sludds, H. Esfahanizadeh, D. Englund, M. Ghobadi, "Lightning: A Reconfigurable Photonic-Electronic SmartNIC for Fast and Energy-Efficient Inference," ACM SIGCOMM, 2023. <https://doi.org/10.1145/3603269.3604821>

> Z. Zhong, M. Yang, J. Lang, D. Englund, M. Ghobadi, "Demo: First Demonstration of Real-Time Photonic-Electronic DNN Acceleration on SmartNICs," ACM SIGCOMM, 2023. <https://doi.org/10.1145/3603269.3610842>

For more details on Lightning, please visit our website: <https://lightning.mit.edu>

This is an active project, if you want to have a community discussion, please start a new discussion thread in the discussion tab, and we will get back to you as soon as possible.

For any questions, please contact Zhizhen Zhong at zhizhenz [at] mit.edu. We welcome all contributions and feedbacks.

## 2. Artifact Structure

This repo has submodules, please clone the repo using git recursive clone.
```
git clone --recursive 
```

### 2.1 Verilog RTL code on the Lightning datapath design and implementation

This part of artifact contains Lightning's RTL-based datapath design and implementation (Sections 4, 5, and 6 of the Lightning SIGCOMM paper). We also include an emulated photonic MAC core to build a cycle-accurate testbench using Verilator.

|  Source Files      |  Description                                                                                                             |
|  -----             |  -----                                                                                                                   |
|  `rtl/datapath/`   |  This folder contains the code of Lightning's datapath modules (packet I.O, memory controller, count-action logic, etc.) |
|  `rtl/emulate/`    |  This folder contains the code of emulated photonic multiplier modules                                                   |
|  `rtl/sram/`       |  This folder contains the code of SRAM modules                                                                           |
|  `rtl/tb/`         |  This folder contains the code of Verilator-based testbench modules                                                      |
|  `rtl/utils/`      |  This folder contains the code of customized AXI-related modules and third-party AXI libraries                           |
|  `rtl/Makefile`    |  This folder contains the Makefile for running the Verilator-based testbench                                             |
|  `rtl/README.md`   |  This README file explains the dependencies and steps to run the RTL cycle-accurate testbench                            |

### 2.2 FPGA firmware and library code for Lightning's Python API

This part of artifact contains the FPGA firmware and library code to enable Lightning's Python API (Section 6 and Appendix G of the Lightning SIGCOMM paper).

|  Source Files         |  Description                                                                              |
|  -----                |  -----                                                                                    |
|  `api/firmware/`      |  This folder contains the code of the FPGA firmwares to support the Python API            |
|  `api/lightning_lib/` |  This folder contains the code of Lightning Python API libraries                          |

### 2.3 Photonic noise emulation code

This part of artifact contains Lightning's photonic emulation (Section 7 of the Lightning SIGCOMM paper).

|  Source Files           |  Description                                                                                     |
|  -----                  |  -----                                                                                           |
|  `emulation/`           |  This folder contains the code for running photonic emulation on large deep neural networks      |
|  `emulation/README.md`  |  This file contains the dependencies and steps to run the emulation code                         |

### 2.4 Large-scale event-driven simulation code

This part of artifact contains Lightning's event-driven simulation study on seven real-world large DNN models (Section 9 of the Lightning SIGCOMM paper).
|  Source Files            |  Description                                                                                    |
|  -----                   |  -----                                                                                          |
|  `simulation/`           |  This folder contains the code for running event-driven simulations on DNN inference queries    |
|  `simulation/README.md`  |  This file contains the dependencies and steps to run the simulations code                      |

### 2.5 Developer kit manufacture files

This part of artifact contains manufacture files for the Lightning developer kit (Appendix G of the Lightning SIGCOMM paper).

|  Source Files                 |  Description                                                                                                             |
|  -----                        |  -----                                                                                                                   |
|  `kit/laser_cutting/`         |  This folder contains the design files for manufactoring the package of the developer kit using laser-cutting machines   |
|  `kit/3D_printing/`           |  This folder contains the design files for manufactoring the device support components using 3D printers                 |
|  `kit/lightning_devkit_v2.jpg`|  This file contains the shopping list to assembly the open-source developer kit                                          |

### 2.6 Data files

This part of artifact contains the data to reproduce our results.

|  Source Files              |  Description                                                                   |
|  -----                     |  -----                                                                         |
|  `data/saved_dataset/`     |  This folder stores the datasets for corresponding DNN models                  |
|  `data/saved_bitstreams/`  |  This folder stores the FPGA bitstream firmwares generated by Vivado           |
|  `data/saved_models/`      |  This folder stores the considered DNN models                                  |
|  `data/saved_results/`     |  This folder stores some FPGA ILA results                                      |

### 2.7 Automatic DNN accuracy checker

This part of artifact contains automatic github actions like the DNN accuracy checker through our Verilator-based testbench.

|  Source Files              |  Description                                                                   |
|  -----                     |  -----                                                                         |
|  `.github/workflows/`      |  This folder stores the automatic workflows like DNN accuracy checker          |

## 3. Major Dependencies

* Xilinx Vivado 2022.2
* Xilinx PetaLinux 2022.2
* Ubuntu 20.04.6 LTS (Focal Fossa)
* Verilator 4.038 2020-07-11 rev v4.036-114-g0cd4a57ad
* Python 3.10.8
* Detailed dependencies is described in the README.md of each folder

## 4. License

Lightning is MIT-licensed.
