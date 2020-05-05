# !/bin/bash

MPLBACKEND=agg python main.py --k 8 --l 2 --hidden_ch 32 --warmup_steps 10000 --init_lr 0.001 \
                              --batch_size 64 --epochs 100 --cuda True --log_step 100 \
                              --dataset mnist --data_dir ../../data \
