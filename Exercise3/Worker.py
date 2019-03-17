import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from Networks import ValueNetwork
from torch.autograd import Variable
from Environment import HFOEnv
import random

def train(idx, args, value_network, target_value_network, optimizer, lock, counter):
    port =5000+idx*10
    seed =0
    hfoEnv = HFOEnv(numTeammates=0, numOpponents=1, port=port, seed=seed)
    hfoEnv.connectToServer()
    newObservation =hfoEnv.reset()
    episodeN=0
    local_counter=0
    old_count=0
    while counter.value <args.numEpisodes:
        action, actionID = act(newObservation,value_network,args,hfoEnv,episodeN)
        newObservation, reward, done, status, info = hfoEnv.step(action)
        reward = torch.Tensor([reward])
        tar = computeTargets(reward, newObservation, args.discountFactor, done, target_value_network)
        pred = computePrediction(torch.Tensor(newObservation),actionID,value_network)
        loss = 0.5*(pred-tar)**2
        loss.backward()
        
        with lock:
            counter.value +=1
            local_counter +=1
            if(counter.value % args.trainIter):
                optimizer.step()
                optimizer.zero_grad()
            if(counter.value % args.updateTarget ==0):
                hard_update(target_value_network,value_network)
            if(counter.value% 1000000 ==0 ):
                saveModelNetwork(value_network,'model/saveModel.pt')
                
        if done:
            if(episodeN % 1 ==0):
                print('\ntook {} timesteps, reward is {}'.format(local_counter-old_count,reward))
                old_count = local_counter
            episodeN+=1
            newObservation =hfoEnv.reset()

        
        

        
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



