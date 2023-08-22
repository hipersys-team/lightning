mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))
repo_base_dir := $(abspath $(mkfile_dir)/../../)

VERILATOR_PATH = $(realpath $(shell which verilator)/../..)
VERILATOR_INCLUDE_PATH = $(VERILATOR_PATH)/share/verilator/include

CURDIR := $(shell pwd)

#############################################
# stuff here is common across model_runner _and_ the simulated
# top level logics - most files should probably live in here
VERILATOR_COMMON=	datapath/analog_interfaces/calibration.v			\
					datapath/analog_interfaces/loss_compensator.v 		\
					datapath/analog_interfaces/preamble_detect.v 		\
					datapath/digital_compute/adder_tree.v				\
					datapath/digital_compute/averager_tree.v 			\
					datapath/digital_compute/exp.v 						\
					datapath/digital_compute/log.v 						\
					datapath/digital_compute/parallel_add_and_copy.v 	\
					datapath/digital_compute/parallel_exp.v 			\
					datapath/digital_compute/parallel_logsoftmax.v 		\
					datapath/digital_compute/parallel_max_and_copy.v 	\
					datapath/digital_compute/relu.v 					\
					datapath/neural_network/integration.v 				\
					datapath/neural_network/inter_layer_buffer.v 		\
					datapath/neural_network/layer_activation_switch.v 	\
					datapath/neural_network/normalization.v 			\
					datapath/neural_network/sparsity_detect.v 			\
					emulate/photonic_multiplier.v 				    	\
					tb/verilator_dpi.v									\
					utils/asym_ram_sdp_read_wider.v 					\
					utils/axis_adjustable_delay.v 						\
					utils/axis_adjustable_intra_cycle_delay.v			\
					utils/axis_delay.v 									\
					utils/axis_intra_cycle_delay.v 						\

VERILATOR_ON_LENET_SINGLE_CORE=		${VERILATOR_COMMON}				\
					datapath/lenet_single_core.v 					\
					tb/lenet_sim_single_core.v						\
					sram/lenet/input_activation_lenet.v				\
					sram/lenet/weight_matrix_absolute_lenet.v		\
					sram/lenet/weight_matrix_sign_lenet.v			\

VERILATOR_ON_IOT_SINGLE_CORE=		${VERILATOR_COMMON}				\
					datapath/iot_single_core.v						\
					tb/iot_sim_single_core.v						\
					sram/iot/input_activation_iot.v					\
					sram/iot/weight_matrix_absolute_iot.v			\
					sram/iot/weight_matrix_sign_iot.v				\

build-sim-lenet-single-core: lenet_sim.cpp
	verilator -DOBSOLETE=1 -DONESPIN=1 --error-limit 200 --trace --cc -y .. -y ../utils/verilog-axis/rtl -y ../utils/verilog-axi/rtl -y ../sram/lenet --top-module lenet_sim $(VERILATOR_ON_LENET_SINGLE_CORE) --exe lenet_sim.cpp
	make -j`nproc` -C obj_dir -f Vlenet_sim.mk Vlenet_sim

run-sim-lenet-single-core: build-sim-lenet-single-core
	./obj_dir/Vlenet_sim 0

build-sim-iot-single-core: iot_sim.cpp
	verilator -DOBSOLETE=1 -DONESPIN=1 --error-limit 200 --trace --cc -y .. -y ../utils/verilog-axis/rtl -y ../utils/verilog-axi/rtl -y ../sram/iot --top-module iot_sim $(VERILATOR_ON_IOT_SINGLE_CORE) --exe iot_sim.cpp
	make -j`nproc` -C obj_dir -f Viot_sim.mk Viot_sim

run-sim-iot-single-core: build-sim-iot-single-core
	./obj_dir/Viot_sim 0

#############################################
ACCURACY=		accuracy.py
CREATE_VENV=		create_venv.py

ACCURACY_DEPENDS=	numpy torch
ACCURACY_PARALLELISM=	20
ACCURACY_TRIALS=	200
ACCURACY_VENV=		env

.PHONY: accuracy
accuracy: $(ACCURACY_VENV)
	. ./$</bin/activate &&	\
	python3 $(ACCURACY) -n $(ACCURACY_PARALLELISM) -t $(ACCURACY_TRIALS) -v

$(ACCURACY_VENV):
	python3 ${CREATE_VENV} --name $@
	. ./$@/bin/activate && pip install $(ACCURACY_DEPENDS)

#############################################
# clean all sim related files
clean-sims:
	rm -rf obj_dir tb_fpga_sim.vcd $(ACCURACY_VENV)
