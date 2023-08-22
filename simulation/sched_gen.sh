NETWORK_SPEEDS=(0.01 0.02 0.03 0.04 0.05 0.06)

for PKL_NUM in 1 2 3 4 5 6 7 8 9 10
do
    python3 make_order.py --num_reqs=5000 --pkl_num=$PKL_NUM

    for NS in $NETWORK_SPEEDS
    do
        python3 gen_mixed.py --network_speed=$NS --pkl_num=$PKL_NUM
    done
done