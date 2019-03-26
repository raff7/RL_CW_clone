#!/usr/bin/env python3
# encoding utf-8
import torch.multiprocessing as mp
import multiprocessing
from Environment import HFOEnv
from Networks import ValueNetwork
from SharedAdam import SharedAdam
from Worker import train, hard_update
import numpy as np
from argparse import ArgumentParser
import time
import matplotlib.pyplot as plt
import sys
import os
import json

# Use this script to handle arguments and
# initialize important components of your experiment.
# These might include important parameters for your experiment, and initialization of
# your models, torch's multiprocessing methods, etc.
if __name__ == "__main__":
    # sys.stdout = open('/dev/null', 'w')

    default_path_name = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    parser = ArgumentParser()
    # Example on how to initialize global locks for processes
    # and counters.
    parser.add_argument('--max_trans', type=int, default=5e6)
    parser.add_argument('--n_workers', type=int, default=12)
    parser.add_argument('--train_every', type=int, default=50)
    parser.add_argument('--copy_every', type=int, default=1500)
    parser.add_argument('--save_every', type=int, default=1e5)
    parser.add_argument('--save_name', type=str, default=default_path_name)
    parser.add_argument('--start_eps', type=float, default=.95)
    parser.add_argument('--eps_decay', type=int, default=2e6)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--lr_decay', type=bool, default=True)
    parser.add_argument('--l2alpha', type=float, default=0.01)
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--n_hidden_nodes', type=int, default=64)
    parser.add_argument('--live_plot', type=bool, default=True)

    args = parser.parse_args()

    save_path = os.path.join('models', args.save_name)

    # Low level feature set has: 68
    # High level feature set has: 15
    state_dim = 15

    value_network = ValueNetwork(input_dim=state_dim, hidden_dim=args.n_hidden_nodes)
    target_value_network = ValueNetwork(input_dim=state_dim, hidden_dim=args.n_hidden_nodes)

    hard_update(target_value_network, value_network)

    # Create queues for exchanging data to plot
    reward_queue = mp.Queue()
    scored_queue = mp.Queue()
    count_queue = mp.Queue()
    main_terminated = multiprocessing.Event()

    counter = mp.Value('i', 0)
    lock = mp.Lock()
    optimizer = SharedAdam(value_network.parameters(), lr=args.lr, weight_decay=args.l2alpha)

    processes = []

    all_rewards = []
    mean_rew = -3
    all_scored = []
    mean_scored = .5
    all_counts = []
    mean_count = 150

    save_plots_every = 20000
    last_saved_plot = 0
    transition_count = 0

    ma_coeff = 0.999

    f, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    rew_lines = axs[0].plot([0], [0])[0]
    axs[0].set_title('Avg cumulative rewards (x1000)')
    scored_lines = axs[1].plot([0], [0])[0]
    axs[1].set_title('Score chances (x1000)')
    count_lines = axs[2].plot([0], [0])[0]
    axs[2].set_title('Avg episode length (x1000)')

    axs[3].set_title('Info')
    n_ite_txt = axs[3].text(0.1, 0.5, '# steps = 0')
    epsilon_txt = axs[3].text(0.1, 0.2, 'eps_value = {}'.format(args.start_eps))


    if args.live_plot:
        plt.ion()
        plt.show()


    last_counter_val = 0

    # Example code to initialize torch multiprocessing.
    for idx in range(0, args.n_workers):
        trainingArgs = (idx, args, value_network, target_value_network, optimizer, counter, reward_queue,
                        scored_queue, count_queue, main_terminated)
        p = mp.Process(target=train, args=trainingArgs)
        p.start()
        processes.append(p)


    last_updated = time.time()

    try:
        while True:
            time.sleep(0.00001)
            rew_q_size = reward_queue.qsize()

            coeff = ma_coeff #if len(all_rewards) > 100 else len(all_rewards)/100 * 0.099 + 0.9
            if rew_q_size > 0:
                new_rew = reward_queue.get()
                if new_rew is None:
                    break
                mean_rew = coeff * mean_rew + (1-coeff) * new_rew
                all_rewards.append(mean_rew)
            scored_q_size = scored_queue.qsize()
            if scored_q_size > 0:
                new_score = scored_queue.get()
                if new_score is None:
                    break
                mean_scored = coeff * mean_scored + (1 - coeff) * new_score
                all_scored.append(mean_scored)
            count_q_size = count_queue.qsize()
            if count_q_size > 0:
                new_count = count_queue.get()
                mean_count = coeff * mean_count + (1-coeff) * new_count
                all_counts.append(mean_count)


            if len(all_scored) > 50 and time.time() - last_updated > 2:
                transition_count = counter.value
                scored_data = all_scored[-len(all_scored)+20:]
                rew_data = all_rewards[-len(all_rewards)+20:]
                count_data = all_counts[-len(all_counts)+20:]
                xs = np.linspace(1/1000, transition_count/1000, max(len(scored_data), len(rew_data), len(count_data)))
                scored_lines.set_ydata(scored_data)
                scored_lines.set_xdata(xs[:len(scored_data)])
                rew_lines.set_ydata(rew_data)
                rew_lines.set_xdata(xs[:len(rew_data)])
                count_lines.set_ydata(count_data)
                count_lines.set_xdata(xs[:len(count_data)])
                [ax.relim() for ax in axs[:-1]]
                [ax.autoscale_view() for ax in axs[:-1]]

                n_ite_txt.set_text('# steps = {}'.format(transition_count))
                epsilon_txt.set_text('eps_value = {:.3f}'.format(args.start_eps * np.exp(-transition_count/args.eps_decay)))

                f.canvas.draw()
                f.canvas.flush_events()

                last_updated = time.time()

                if counter.value - last_saved_plot > save_plots_every:
                    f.tight_layout()
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    f.savefig(os.path.join(save_path, 'plots.png'))
                    last_saved_plot = counter.value

                    if not os.path.exists(os.path.join(save_path, 'cfg.txt')):
                        with open(os.path.join(save_path, 'cfg.txt'), 'w+') as file:
                            json.dump(vars(args), file)


            curr_counter_val = counter.value
            if curr_counter_val != last_counter_val:
                last_counter_val = curr_counter_val
    finally:
        main_terminated.is_set()
        for p in processes:
            p.join()

        os.system("killall -9 rcssserver")
        time.sleep(10)
        print('\n\n\n\n\n\n\nEnded!!\n\n\n\n\n\n\n')
        exit()
