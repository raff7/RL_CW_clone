#!/usr/bin/env python3
# encoding utf-8



from Environment import HFOEnv
import multiprocessing as mp
import argparse
from Networks import ValueNetwork
from SharedAdam import SharedAdam
from Worker import *
from ctypes import c_bool, c_double
import os
import matplotlib.pyplot as plt
import time

try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')

# Use this script to handle arguments and 
# initialize important components of your experiment.
# These might include important parameters for your experiment, and initialization of
# your models, torch's multiprocessing methods, etc.
if __name__ == "__main__" :
    
    os.system("killall -9 rcssserver")
    parser = argparse.ArgumentParser()
    parser.add_argument('--numEpisodes', type=int, default=30000000)
    parser.add_argument('--numWorkers', type=int, default=4)
    parser.add_argument('--initEpsilon', type=int, default=0.6)
    parser.add_argument('--updateTarget', type=int, default=500)
    parser.add_argument('--trainIter', type=int, default=100)
    parser.add_argument('--lr', type=int, default=0.0001)
    parser.add_argument('--weightDecay', type=int, default=0.1)
    parser.add_argument('--discountFactor', type=int, default=0.95)

    args=parser.parse_args()

    #PLOT
    done = mp.Value(c_bool, False)
    print_eps = mp.Value(c_double ,args.initEpsilon)
    time_goal = mp.Queue()
    goals = mp.Queue()
    cum_rew = mp.Queue()
    all_time_goal= []
    all_goals = []
    all_cum_rew=[]
    avg_time_goal= 200
    avg_goals = 0.5
    avg_cum_rew=0
    avg_coef = 0.004
    last_time = time.time()
    f, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = ax.flatten()
    
    time_line = ax[0].plot([0],[0])[0]
    ax[0].set_title("Time-steps between goals")

    goal_line = ax[1].plot([0],[0])[0]
    ax[1].set_title("Goal probability")
    
    rew_line = ax[2].plot([0],[0])[0]
    ax[2].set_title("Cumulative reward")
    
    text_params = ax[3].text(0.5,0.5,'TESTP')
    ax[3].set_title("Parameters")
    
    plt.ion()
    plt.show()

    #CREATE NETWORKS
    value_network = ValueNetwork()
    target_value_network = ValueNetwork()
    hard_update(target_value_network, value_network)


    optimizer = SharedAdam(value_network.parameters(),lr=args.lr, weight_decay=args.weightDecay)
        
    counter = mp.Value('i', 0)
    games_counter = mp.Value('i', 0)

    lock = mp.Lock()
    
    processes =[]
    #Start Training
    for idx in range(0, args.numWorkers):
        trainingArgs = (idx, args, value_network, target_value_network, optimizer, lock, counter, games_counter, done,time_goal, goals, cum_rew,print_eps)
        p = mp.Process(target=train, args=(trainingArgs))
        p.start()
        processes.append(p)
    
    while True:

        #Print update
        time.sleep(0.001)
        if not time_goal.empty():
            c_coef = avg_coef*2 if len(all_time_goal)>100 else 0.1*np.exp(-len(all_cum_rew)/50)
            new_time_goal = time_goal.get()
            avg_time_goal = (1-c_coef)*(avg_time_goal) + c_coef*new_time_goal
            all_time_goal.append(avg_time_goal)
        if not goals.empty():
            c_coef = avg_coef if len(all_cum_rew)>100 else 0.05*np.exp(-len(all_cum_rew)/50)
            new_goals = goals.get()
            avg_goals = (1-c_coef)*(avg_goals) + c_coef*new_goals
            all_goals.append(avg_goals)
        if(not cum_rew.empty()):
            c_coef = avg_coef*2 if len(all_cum_rew)>100 else 0.1*np.exp(-len(all_cum_rew)/50)
            new_cum_rew = cum_rew.get()
            avg_cum_rew = (1-c_coef)*(avg_cum_rew) + c_coef*new_cum_rew
            all_cum_rew.append(avg_cum_rew)

        if(time.time()-last_time>2):
            print("\n\n\n\n\n\n\nTIMEEEEEEEE\n-----------------------\n\n\n\n\n\n\n\n\n\n\n\n")
            time_line.set_ydata(all_time_goal)
            time_line.set_xdata(range(len(all_time_goal)))
            #time_line.set_xdata(np.linspace(0, counter.value,len(all_time_goal)))
            
            goal_line.set_ydata(all_goals)
            goal_line.set_xdata(range(len(all_goals)))
            #rew_line.set_xdata(np.linspace(0, counter.value,len(all_goals)))

            rew_line.set_ydata(all_cum_rew)
            rew_line.set_xdata(range(len(all_cum_rew)))
            #rew_line.set_xdata(np.linspace(0, counter.value,len(all_cum_rew)))
            #print('\n\nax {}\n\n'.format(ax))
            [axxx.relim() for axxx in ax[:-1]]
            [axxx.autoscale_view() for axxx in ax[:-1]]
            
            text_params.set_text('Game Counter: {}\nCounter: {}\nEpsilon: {}\nTotal goal percentage: {}'.format(games_counter.value,counter.value,print_eps.value, len(all_goals)/len(all_time_goal)))
            
            f.canvas.draw()
            f.canvas.flush_events()
            last_time = time.time()
            plt.show()
        if(done.value):
            break
            
        
        
    for p in processes:
        p.join()




