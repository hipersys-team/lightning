all:
	# petalinux-config --silentconfig --get-hw-description ../hardware/imp/top.xsa
	# petalinux-build -c pmu-firmware -c fsbl-firmware -c device-tree -x cleansstate
	# petalinux-build -c pmu-firmware -c fsbl-firmware -c device-tree
	petalinux-build -c pmu-firmware -x cleansstate
	petalinux-build -c pmu-firmware
	petalinux-build -c fsbl-firmware -x cleansstate
	petalinux-build -c fsbl-firmware
