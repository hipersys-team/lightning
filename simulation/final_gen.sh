CORES=576
BS=1
PREEMPTIVE="P"

for NS in 0.06
do
    python3 read_csv.py --num_reqs=100 --lightning_core_count=$CORES --network_speed=$NS --batch_size=$BS --preemptive=$PREEMPTIVE &
done