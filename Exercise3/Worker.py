import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from Networks import ValueNetwork
from torch.autograd import Variable
from Environment import HFOEnv
import random
import matplotlib.pyplot as plt
import time
from hfo import GOAL, OUT_OF_BOUNDS,IN_GAME ,CAPTURED_BY_DEFENSE ,OUT_OF_TIME ,SERVER_DOWN





def train(idx, args, value_network, target_value_network, optimizer, lock, counter, games_counter, mp_done,time_goal, goals, cum_rew,print_eps,print_lr):
    new_lr = args.lr
    port = 2600+idx*20
    seed =idx*100
    hfoEnv = HFOEnv(numTeammates=0, numOpponents=1, port=port, seed=seed)
    hfoEnv.connectToServer()
    newObservation =hfoEnv.reset()
    episodeN=0
    steps_to_goal = 0
    cum_reward = 0
    while counter.value <args.numEpisodes:
        if(True):#idx!=0):
            steps_to_goal +=1
            epsilon,new_lr = updateParams(args, counter.value)
            print_eps.value, print_lr.value = epsilon,new_lr
            action, actionID = act(newObservation,value_network,args,hfoEnv,epsilon)
            newObservation, reward, done, status, info = hfoEnv.step(action)
            cum_reward += reward
            reward = torch.Tensor([reward])
            tar = computeTargets(reward, newObservation, args.discountFactor, done, target_value_network)
            pred = computePrediction(torch.Tensor(newObservation),actionID,value_network)
            loss = 0.5*(tar-pred)**2
            loss.backward()
        else:
            steps_to_goal +=1
            epsilon,new_lr = updateParams(args, counter.value)
            epsilon = 0
            print_eps.value, print_lr.value = epsilon,new_lr
            action, actionID = act(newObservation,value_network,args,hfoEnv,epsilon)
            newObservation, reward, done, status, info = hfoEnv.step(action)
            cum_reward += reward
            reward = torch.Tensor([reward])
            tar = computeTargets(reward, newObservation, args.discountFactor, done, target_value_network)
            pred = computePrediction(torch.Tensor(newObservation),actionID,value_network)
            loss = 0.5*(tar-pred)**2
        with lock:
            counter.value +=1
            if(counter.value % args.trainIter):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                optimizer.step()
                optimizer.zero_grad()
            if(counter.value % args.updateTarget ==0):
                hard_update(target_value_network,value_network)
            if(counter.value% 500000 ==0 ):
                saveModelNetwork(value_network,'model/saveModel.pt')
                
        if done:
            games_counter.value +=1
            if(True):#idx==0):
                goals.put_nowait(1.0 if status == GOAL else 0.0)
                if status != GOAL:
                    steps_to_goal = 500
                time_goal.put_nowait(steps_to_goal)
                steps_to_goal = 0
                cum_rew.put_nowait(cum_reward)
                cum_reward= 0
            episodeN+=1
            newObservation =hfoEnv.reset()
            

    with lock:
        mp_done.value = True

        
        

        
def act(state,value_network,args,hfoEnv,epsilon):
    if(random.random()<epsilon):
        actionID = random.randint(0,3)
        action = hfoEnv.possibleActions[actionID]
    else:
        qVals = value_network(torch.Tensor(state))
        actionID = np.argmax(qVals.detach().numpy())
        action = hfoEnv.possibleActions[actionID]
    return action, actionID
    
def updateParams(args,episodeN):
    eps= args.initEpsilon*(pow(np.e, (-episodeN / 5000000)))
    lr= args.lr *(args.numEpisodes-episodeN )/args.numEpisodes
    return eps, lr

def computeTargets(reward, nextObservation, discountFactor, done, targetNetwork):
    if(done):
        target = reward
    else:
        qVals = targetNetwork(torch.Tensor(nextObservation)).detach()
        target = reward +discountFactor*max(qVals)
    return target

        
    
    

def computePrediction(state, action, valueNetwork):
    qVals = valueNetwork(torch.Tensor(state))
    return qVals[action]
	
# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
	torch.save(model.state_dict(), strDirectory)
	
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)



