#!/usr/bin/env bash
sleep 5

# Default
python main.py --max_trans 8000000 --n_workers 16 --train_every 20 --copy_every 5000 --start_eps .95 --eps_decay 2500000 --lr 0.001 --lr_decay 1 --l2alpha 0.01 --discount_factor 0.99 --n_hidden_nodes 64 --save_name more_its

killall -9 rcssserver
sleep 10

python main.py --max_trans 5000000 --n_workers 16 --train_every 50 --copy_every 5000 --start_eps .95 --eps_decay 2500000 --lr 0.0001 --lr_decay 1 --l2alpha 0.01 --discount_factor 0.99 --n_hidden_nodes 64 --save_name lower_lr

killall -9 rcssserver
sleep 10

python main.py --max_trans 8000000 --n_workers 16 --train_every 80 --copy_every 3500 --start_eps .95 --eps_decay 4000000 --lr 0.0025 --lr_decay 1 --l2alpha 0.03 --discount_factor 0.99 --n_hidden_nodes 128 --save_name more_hnodes

killall -9 rcssserver
sleep 10

python main.py --max_trans 8000000 --n_workers 12 --train_every 100 --copy_every 3500 --start_eps .95 --eps_decay 4000000 --lr 0.005 --lr_decay 1 --l2alpha 0.05 --discount_factor 0.99 --n_hidden_nodes 256 --save_name too_many_hnodes

killall -9 rcssserver
sleep 10

