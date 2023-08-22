all:

# build the verilator for cycle-accurate simulatiom on the LeNet DNN
build-sw-lenet-single-core:
	$(MAKE) -C tb build-sim-lenet-single-core

# run the verilator for cycle-accurate simulatiom on the LeNet DNN
run-sw-lenet-single-core:
	$(MAKE) -C tb run-sim-lenet-single-core

# perform accuracy checking for all models
accuracy:
	$(MAKE) -C tb accuracy

# clean the verilator files
clean-sw:
	$(MAKE) -C tb clean-sims
