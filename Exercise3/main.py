#!/usr/bin/env python3
# encoding utf-8


from Environment import HFOEnv
import multiprocessing as mp
import argparse
from Networks import ValueNetwork
from SharedAdam import SharedAdam
from Worker import *
from ctypes import c_bool
import os
import matplotlib.pyplot as plt
import time

# Use this script to handle arguments and 
# initialize important components of your experiment.
# These might include important parameters for your experiment, and initialization of
# your models, torch's multiprocessing methods, etc.
if __name__ == "__main__" :
    
    os.system("killall -9 rcssserver")
    parser = argparse.ArgumentParser()
    parser.add_argument('--numEpisodes', type=int, default=30000000)
    parser.add_argument('--numWorkers', type=int, default=6)
    parser.add_argument('--initEpsilon', type=int, default=0.6)
    parser.add_argument('--updateTarget', type=int, default=50)
    parser.add_argument('--trainIter', type=int, default=10)
    parser.add_argument('--lr', type=int, default=0.0001)
    parser.add_argument('--weightDecay', type=int, default=0.1)
    parser.add_argument('--discountFactor', type=int, default=0.95)
    
    done = mp.Value(c_bool, False)
    time_goal = mp.Queue()
    goals = mp.Queue()
    cum_rew = mp.Queue()
    all_time_goal= []
    all_goals = []
    all_cum_rew=[]
    avg_time_goal= 200
    avg_goals = 0.5
    avg_cum_rew=0
    avg_coef = 0.001
    last_time = time.time()
    f, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = ax.flatten()
    
    time_line = ax[0].plot([0],[0])[0]
    ax[0].set_title("Time")

    goal_line = ax[1].plot([0],[0])[0]
    ax[1].set_title("Goals")
    
    rew_line = ax[2].plot([0],[0])[0]
    ax[2].set_title("reward")
    
    text_params = ax[3].text(0.5,0.5,'TESTP')
    ax[3].set_title("shit")
    
    plt.ion()
    plt.show()


    value_network = ValueNetwork()
    target_value_network = ValueNetwork()
    hard_update(target_value_network, value_network)

    args=parser.parse_args()

    optimizer = SharedAdam(value_network.parameters(),lr=args.lr, weight_decay=args.weightDecay)
        
    counter = mp.Value('i', 0)
    lock = mp.Lock()
    
    processes =[]
    for idx in range(0, args.numWorkers):
        trainingArgs = (idx, args, value_network, target_value_network, optimizer, lock, counter, done,time_goal, goals, cum_rew)
        p = mp.Process(target=train, args=(trainingArgs))
        p.start()
        processes.append(p)
    
    while True:
        time.sleep(0.0001)
        
        if(time_goal.qsize()>0):
            new_time_goal = time_goal.get()
            avg_time_goal = (1-avg_coef)*(avg_time_goal) + avg_coef*new_time_goal
            all_time_goal.append(avg_time_goal)
            
        if(goals.qsize()>0):
            new_goals = goals.get()
            avg_goals = (1-avg_coef)*(avg_goals) + avg_coef*new_goals
            all_goals.append(avg_goals)
            
        if(cum_rew.qsize()>0):
            new_cum_rew = cum_rew.get()
            avg_cum_rew = (1-avg_coef)*(avg_cum_rew) + avg_coef*new_cum_rew
            all_cum_rew.append(avg_cum_rew)
        
        
        if(time.time()-last_time>2):
            time_line.set_ydata(all_time_goal)
            time_line.set_xdata(range(len(all_time_goal)))
            
            goal_line.set_ydata(all_goals)
            goal_line.set_xdata(range(len(all_goals)))
            
            
            rew_line.set_ydata(all_cum_rew)
            rew_line.set_xdata(range(len(all_cum_rew)))
            
            
            [axxx.relim() for axxx in ax[:-1]]
            [axxx.autoscale_view() for axxx in ax[:-1]]
            
            text_params.set_text('Counter {}'.format(counter.value))
            
            f.canvas.draw()
            f.canvas.flush_events()
            last_time = time.time()
        if(done.value):
            break
            
        
        
    for p in processes:
        p.join()




