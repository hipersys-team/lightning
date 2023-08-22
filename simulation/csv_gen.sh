NETWORK_SPEED="0.06"
NUM_REQS=100
LIGHTNING_BATCH_SIZE=1
LIGHTNING_CORE_COUNT=576

mkdir results
mkdir results/runtimes
mkdir results/active_reqs

for PKL_NUM in 1 2 3 4 5 6 7 8 9 10
do
    for PROCESSOR in "lightning" "a100" "dpu" "brainwave"
    do
        python3 trial_to_csv.py --lightning_core_count=$LIGHTNING_CORE_COUNT --batch_size=$LIGHTNING_BATCH_SIZE --num_reqs=$NUM_REQS --network_speed=$NETWORK_SPEED --pkl_num=$PKL_NUM --processor=$PROCESSOR &
    done
done