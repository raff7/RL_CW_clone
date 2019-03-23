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
from hfo import GOAL





def train(idx, args, value_network, target_value_network, optimizer, lock, counter, mp_done,time_goal, goals, cum_rew):
    port =5000+idx*10
    seed =0
    hfoEnv = HFOEnv(numTeammates=0, numOpponents=1, port=port, seed=seed)
    hfoEnv.connectToServer()
    newObservation =hfoEnv.reset()
    episodeN=0
    steps_episode = 0
    cum_reward = 0
    while counter.value <args.numEpisodes:
        steps_episode +=1
        action, actionID = act(newObservation,value_network,args,hfoEnv,episodeN)
        newObservation, reward, done, status, info = hfoEnv.step(action)
        cum_reward += reward
        reward = torch.Tensor([reward])
        tar = computeTargets(reward, newObservation, args.discountFactor, done, target_value_network)
        pred = computePrediction(torch.Tensor(newObservation),actionID,value_network)
        loss = 0.5*(pred-tar)**2
        loss.backward()
        with lock:
            counter.value +=1
            if(counter.value % args.trainIter):
                optimizer.step()
                optimizer.zero_grad()
            if(counter.value % args.updateTarget ==0):
                hard_update(target_value_network,value_network)
            if(counter.value% 1000000 ==0 ):
                saveModelNetwork(value_network,'model/saveModel.pt')
                
        if done:
            goals.put_nowait(1.0 if status == GOAL else 0.0)
            time_goal.put_nowait(steps_episode)
            cum_rew.put_nowait(cum_reward)
            episodeN+=1
            newObservation =hfoEnv.reset()
            steps_episode=0
            cum_reward= 0
    with lock:
        mp_done.value = True

        
        

        
def act(state,value_network,args,hfoEnv,episodeN):
    epsilon = updateEpsilon(args.initEpsilon,episodeN)
    if(random.random()<epsilon):
        actionID = random.randint(0,3)
        action = hfoEnv.possibleActions[actionID]
    else:
        qVals = value_network(torch.Tensor(state))
        actionID = np.argmax(qVals.detach().numpy())
        action = hfoEnv.possibleActions[actionID]
    return action, actionID
    
def updateEpsilon(initEps,episodeN):
    return initEps*(pow(np.e, (-episodeN / 1000000)))

def computeTargets(reward, nextObservation, discountFactor, done, targetNetwork):
    if(done):
        target = reward
    else:
        qVals = targetNetwork(torch.Tensor(nextObservation))
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



