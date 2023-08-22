NETWORK_SPEED=0.06
NUM_REQS=100
LIGHTNING_CORE_COUNT=576
LIGHTNING_BATCH_SIZE=1

mkdir trials
mkdir job_stats

for PKL_NUM in 1 2 3 4 5 6 7 8 9 10
do
    python3 sim.py --processor="Lightning-1-200-100" --gran=2048 --network_speed=$NETWORK_SPEED --lightning_core_count=$LIGHTNING_CORE_COUNT --batch_size=$LIGHTNING_BATCH_SIZE --preemptive=True --pkl_num=$PKL_NUM --last_req=100 > trials/lightning_$NETWORK_SPEED\_Gbps_l$LIGHTNING_CORE_COUNT\_cores_$NUM_REQS\_reqs_$LIGHTNING_BATCH_SIZE\_BS_$PKL_NUM.txt &
    sleep 2;
    python3 sim.py --processor="A100" --gran=2048 --network_speed=$NETWORK_SPEED --batch_size=1 --preemptive=True --pkl_num=$PKL_NUM --last_req=100 > trials/a100_$NETWORK_SPEED\_Gbps_l$LIGHTNING_CORE_COUNT\_cores_$NUM_REQS\_reqs_$LIGHTNING_BATCH_SIZE\_BS_$PKL_NUM.txt &
    sleep 2;
    python3 sim.py --processor="DPU" --gran=2048 --network_speed=$NETWORK_SPEED --batch_size=1 --preemptive=True --pkl_num=$PKL_NUM --last_req=100 > trials/dpu_$NETWORK_SPEED\_Gbps_l$LIGHTNING_CORE_COUNT\_cores_$NUM_REQS\_reqs_$LIGHTNING_BATCH_SIZE\_BS_$PKL_NUM.txt &
    sleep 2;
    python3 sim.py --processor="Brainwave" --gran=2048 --network_speed=$NETWORK_SPEED --batch_size=1 --preemptive=True --pkl_num=$PKL_NUM --last_req=100 > trials/brainwave_$NETWORK_SPEED\_Gbps_l$LIGHTNING_CORE_COUNT\_cores_$NUM_REQS\_reqs_$LIGHTNING_BATCH_SIZE\_BS_$PKL_NUM.txt &
    sleep 2;
done
