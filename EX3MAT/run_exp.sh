#!/usr/bin/env bash
sleep 5

# Default
python main.py --max_trans 5000000 --n_workers 12 --train_every 50 --copy_every 1500 --start_eps .95 --eps_decay 2500000 --lr 0.001 --lr_decay 1 --l2alpha 0.01 --discount_factor 0.99 --n_hidden_nodes 64 --save_name base_model

killall -9 rcssserver
sleep 10

python main.py --max_trans 5000000 --n_workers 20 --train_every 150 --copy_every 3000 --start_eps .95 --eps_decay 2500000 --lr 0.001 --lr_decay 1 --l2alpha 0.01 --discount_factor 0.99 --n_hidden_nodes 64 --save_name more_workers

killall -9 rcssserver
sleep 10

python main.py --max_trans 2000000 --n_workers 12 --train_every 50 --copy_every 1500 --start_eps .95 --eps_decay 1200000 --lr 0.001 --lr_decay 1 --l2alpha 0.01 --discount_factor 0.99 --n_hidden_nodes 64 --save_name less_iters

killall -9 rcssserver
sleep 10

python main.py --max_trans 5000000 --n_workers 12 --train_every 50 --copy_every 1500 --start_eps .95 --eps_decay 2500000 --lr 0.01 --lr_decay 1 --l2alpha 0.01 --discount_factor 0.99 --n_hidden_nodes 64 --save_name start_lr_big

killall -9 rcssserver
sleep 10

python main.py --max_trans 5000000 --n_workers 12 --train_every 50 --copy_every 1500 --start_eps .95 --eps_decay 2500000 --lr 0.1 --lr_decay 1 --l2alpha 0.01 --discount_factor 0.99 --n_hidden_nodes 64 --save_name start_lr_bigger

killall -9 rcssserver
sleep 10

# Default
python main.py --max_trans 5000000 --n_workers 12 --train_every 50 --copy_every 1500 --start_eps .95 --eps_decay 2500000 --lr 0.001 --lr_decay 1 --l2alpha 0. --discount_factor 0.99 --n_hidden_nodes 64 --save_name no_weight_decay

killall -9 rcssserver
sleep 10

python main.py --max_trans 5000000 --n_workers 12 --train_every 50 --copy_every 1500 --start_eps .95 --eps_decay 2500000 --lr 0.001 --lr_decay 1 --l2alpha 0.01 --discount_factor 0.995 --n_hidden_nodes 64 --save_name higher_gamma

killall -9 rcssserver
sleep 10

python main.py --max_trans 5000000 --n_workers 7 --train_every 5 --copy_every 10000 --start_eps 1. --eps_decay 2000000 --lr 0.0001 --lr_decay 1 --l2alpha 0.0001 --discount_factor 0.99 --n_hidden_nodes 105 --save_name giovannis

killall -9 rcssserver
sleep 10
