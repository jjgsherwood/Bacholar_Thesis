# !/bin/bash

MPLBACKEND=agg python main.py --k 32 --l 4 --hidden_ch 128 --warmup_steps 10000 --init_lr 0.001 \
                              --batch_size 64 --epochs 100 --cuda True --log_step 100 \
                              --dataset mnist --data_dir /storage/datasets/ \
