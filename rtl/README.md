# Lightning Register Transfer Level (RTL) Code
The Lightning RTL code implements the fast datapath 
 The code in this folder reproduces results reported in Lightning paper Section 6.

## Reproduce the testbench results

### 1. Install [Verilator](https://verilator.org/guide/latest/install.html) and dependencies
Run ```python tb/create_venv.py``` to create the environments and dependencies<br>
Run ```sudo apt install -y verilator python3 python3-pip python3-venv``` to install verilator<br>
Run ```verilator --version``` to check if the install version is ```Verilator 4.038 2020-07-11 rev v4.036-114-g0cd4a57ad```. Other Verilator versions may or may not be compatible due to frequent Verilator codebase changes.

### 2. Build the verilator testbench
Run ```make build-sw-lenet-single-core``` <br>
If build is successful, the compiled Verilator testbench will appear in the ```obj_dir/``` folder.

### 3. Run the verilator testbench (e.g., LeNet-300-100 DNN)
Run ```make run-sw-lenet-single-core``` <br>
After running the testbench, results will appear in the ```tb/``` folder:
- ```tb_lenet_sim.vcd```: the generated waveform file containing all registers values over the simulation period.
- ```tb_lenet_sim_reg_values```: the register values at each cycle in .csv format recorded by the Verilator [Direct Programming Interface (DPI)
](https://verilator.org/guide/latest/connecting.html#direct-programming-interface-dpi).