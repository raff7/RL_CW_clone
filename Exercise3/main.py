#!/usr/bin/env python3
# encoding utf-8

from Environment import HFOEnv
import multiprocessing as mp
import argparse
from Networks import ValueNetwork
from SharedAdam import SharedAdam
from Worker import *

# Use this script to handle arguments and 
# initialize important components of your experiment.
# These might include important parameters for your experiment,
# your models, torch's multiprocessing methods, etc.
if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--numEpisodes', type=int, default=30000000)
    parser.add_argument('--numWorkers', type=int, default=4)
    parser.add_argument('--initEpsilon', type=int, default=0.6)
    parser.add_argument('--updateTarget', type=int, default=50)
    parser.add_argument('--trainIter', type=int, default=10)
    parser.add_argument('--lr', type=int, default=0.0001)
    parser.add_argument('--weightDecay', type=int, default=0.1)
    parser.add_argument('--discountFactor', type=int, default=0.95)



    value_network = ValueNetwork()
    target_value_network = ValueNetwork()
    hard_update(target_value_network, value_network)

    args=parser.parse_args()

    optimizer = SharedAdam(value_network.parameters(),lr=args.lr, weight_decay=args.weightDecay)
        
    counter = mp.Value('i', 0)
    lock = mp.Lock()
    
    processes =[]
    for idx in range(0, args.numWorkers):
        trainingArgs = (idx, args, value_network, target_value_network, optimizer, lock, counter)
        p = mp.Process(target=train, args=(trainingArgs))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()



