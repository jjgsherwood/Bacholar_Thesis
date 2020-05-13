# !/bin/bash

MPLBACKEND=agg python main.py --k 4 --l 2 --hidden_ch 16 --warmup_steps 10 --init_lr 0.001 --batch_size 2 --epochs 100 --cuda False --log_step 10 --output_dir ./liver_experiment --dataset liver --structure liver --data_dir ../data/HSI/liver_whole_sample2.npy
MPLBACKEND=agg python main.py --k 10 --l 3 --hidden_ch 16 --warmup_steps 500 --init_lr 0.001 --batch_size 32 --epochs 100 --cuda True --log_step 20 --output_dir ./liver_experiment2 --dataset liver --structure liver --data_dir ../data/HSI/Liver_map_150z25_60s_1TCPOBOP.npy
MPLBACKEND=agg python main.py --k 8 --l 3 --hidden_ch 16 --warmup_steps 1000 --init_lr 0.001 --batch_size 64 --epochs 100 --cuda True --log_step 10 --output_dir ./liver_experiment3 --dataset liver --structure liver --data_dir ../data/HSI/Liver_map_150z25_60s_1TCPOBOP.npy
MPLBACKEND=agg python main.py --k 8 --l 3 --hidden_ch 16 --warmup_steps 1000 --init_lr 0.001 --batch_size 64 --epochs 100 --cuda True --log_step 10 --output_dir ./liver_experiment4 --dataset liver --structure liver_real_nvp --data_dir ../data/HSI/Liver_map_150z25_60s_1TCPOBOP.npy
MPLBACKEND=agg python main.py --k 8 --l 2 --hidden_ch 64 --warmup_steps 10000 --init_lr 0.001 --batch_size 64 --epochs 100 --cuda True --log_step 100 --dataset mnist --structure mnist_real_nvp --data_dir ../data

# load model
MPLBACKEND=agg python load_model.py --output_dir ./liver_experiment3


# PCA

MPLBACKEND=agg python PCA_approach.py --n_dims 20 --output_dir ./PCA_output --data_dir ../data/HSI/Liver_map_150z25_60s_1TCPOBOP.npy
