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
    parser.add_argument('--save', type=str, default="MODEL_3")
    parser.add_argument('--numEpisodes', type=int, default=10000000)
    parser.add_argument('--numWorkers', type=int, default=8)
    parser.add_argument('--initEpsilon', type=int, default=0.95)
    parser.add_argument('--updateTarget', type=int, default=10000)
    parser.add_argument('--trainIter', type=int, default=50)
    parser.add_argument('--lr', type=int, default=0.0005)
    parser.add_argument('--weightDecay', type=int, default=0.00001)#0.00001
    parser.add_argument('--discountFactor', type=int, default=0.99)

    args=parser.parse_args()
    path =  os.path.join('v', args.save)
    save_plots_every = 100000

    #print('\n\n\n\n\n\n\n\n\n\ncores {}\n\n\n\n\n\n\n\n'.format(mp.cpu_count()))
    #PLOT
    done = mp.Value(c_bool, False)
    print_eps = mp.Value(c_double ,args.initEpsilon)
    print_lr = mp.Value(c_double,args.lr)
    time_goal = mp.Queue()
    goals = mp.Queue()
    cum_rew = mp.Queue()
    all_time_goal= []
    all_goals = []
    all_cum_rew=[]
    avg_time_goal= 500
    avg_goals = 0.5
    avg_cum_rew=0
    avg_coef = 0.0005
    last_time = time.time()
    last_saved_plot = 0
    f, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = ax.flatten()
    
    time_line = ax[0].plot([0],[0])[0]
    ax[0].set_title("avg Time-steps to score a goal")

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
    iter_update = counter.value
    games_counter = mp.Value('i', 0)

    lock = mp.Lock()
    
    processes =[]
    #Start Training
    for idx in range(0, args.numWorkers):
        trainingArgs = (idx, args, value_network, target_value_network, optimizer, lock, counter, games_counter, done,time_goal, goals, cum_rew,print_eps, print_lr)
        p = mp.Process(target=train, args=(trainingArgs))
        p.start()
        processes.append(p)
    
    while True:

        #Print update
        time.sleep(0.001)
        if not time_goal.empty():
            #print("\nMain Agent CHECK")
            c_coef = avg_coef*2 if len(all_time_goal)>500 else 0.025*np.exp(-len(all_cum_rew)/200)
            new_time_goal = time_goal.get()
            avg_time_goal = (1-c_coef)*(avg_time_goal) + c_coef*new_time_goal
            all_time_goal.append(avg_time_goal)
        if not goals.empty():
            c_coef = avg_coef if len(all_cum_rew)>500 else 0.01*np.exp(-len(all_cum_rew)/200)
            new_goals = goals.get()
            avg_goals = (1-c_coef)*(avg_goals) + c_coef*new_goals
            all_goals.append(avg_goals)
        if(not cum_rew.empty()):
            c_coef = avg_coef*2 if len(all_cum_rew)>500 else 0.025*np.exp(-len(all_cum_rew)/200)
            new_cum_rew = cum_rew.get()
            avg_cum_rew = (1-c_coef)*(avg_cum_rew) + c_coef*new_cum_rew
            all_cum_rew.append(avg_cum_rew)

        if(time.time()-last_time>2):
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
            
            text_params.set_text('Game Counter: {}\nCounter: {}\nEpsilon: {}\nLearning Rate {}\niterations in update: {}\nTime left{}'.format(games_counter.value,counter.value,print_eps.value,print_lr.value, counter.value-iter_update,(args.numEpisodes-counter.value)/((counter.value+0.01-iter_update)/2)/3600))
            iter_update = counter.value
            f.canvas.draw()
            f.canvas.flush_events()
            last_time = time.time()
            plt.show()
        if(counter.value - last_saved_plot > save_plots_every):
            f.tight_layout()
            if not os.path.exists(path):
                os.makedirs(path)
            f.savefig(os.path.join(path, 'plot.png'))
            last_saved_plot = counter.value

        if(done.value):
            break
            
        
        
    for p in processes:
        print("\nKILLING {} IT AT {}\n".format(p,counter.value)*100)
        p.join()




