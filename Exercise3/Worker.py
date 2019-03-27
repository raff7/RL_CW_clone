import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
from Networks import ValueNetwork
from torch.autograd import Variable
from Environment import HFOEnv
import random
import matplotlib.pyplot as plt
import time
from hfo import GOAL, OUT_OF_BOUNDS,IN_GAME ,CAPTURED_BY_DEFENSE ,OUT_OF_TIME ,SERVER_DOWN





def train(idx, args, value_network, target_value_network, optimizer, lock, counter, games_counter, mp_done,time_goal, goals, cum_rew,print_eps,print_lr):
    print("\n"*10)
    print("CALL LEARN AGENT ",idx)
    new_lr = args.lr
    port = 1258+idx*582#nt(random.random()*10000)+idx*20
    seed =idx*100

    print("try HFO init FOR {}, PORT {}, SEED {}".format(idx,port, seed))
    hfoEnv = HFOEnv(numTeammates=0, numOpponents=1, port=port, seed=seed)
    print("MADE HFO, try connect to server  FOR {}".format(idx))
    hfoEnv.connectToServer()
    print("CONNECTED {}, reset".format(idx))
    observation =hfoEnv.reset()
    print("HFO INIT FOR {}".format(idx))

    steps_to_goal = 0
    cum_reward = 0

    while counter.value < args.numEpisodes:
        print("AGENT {} CHECK {}".format(idx, counter.value))
        steps_to_goal +=1
        epsilon,new_lr = updateParams(args, counter.value)
        print_eps.value, print_lr.value = epsilon,new_lr
        #ACT
        action, actionID = act(observation,value_network,args,hfoEnv,epsilon)
        #STEP
        newObservation, reward, done, status, info = hfoEnv.step(action)
        cum_reward += reward
        #TRAIN
        tar = computeTargets(reward, newObservation, args.discountFactor, done, target_value_network)
        pred = computePrediction(observation,actionID,value_network)
        loss = 0.5*(pred-tar)**2
        loss.backward()
            

        with lock:
            counter.value +=1
            if(counter.value % args.trainIter ==0):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                optimizer.step()
                optimizer.zero_grad()

            if(counter.value % args.updateTarget ==0):
                hard_update(target_value_network,value_network)

            if(counter.value% 100000 ==0 ):
                folder = os.path.join('v', args.save)
                if(not os.path.exists(folder)):
                    os.makedirs(folder)
                saveModelNetwork(value_network,os.path.join(folder, 'Saving_step={}.pt'.format(counter.value)))

            if counter.value >= args.numEpisodes:
                mp_done.value = True

        observation = newObservation

        if done:
            games_counter.value +=1
            goals.put_nowait(1.0 if status == GOAL else 0.0)
            if status != GOAL:
                steps_to_goal = 500
            time_goal.put_nowait(steps_to_goal)
            steps_to_goal = 0
            cum_rew.put_nowait(cum_reward)
            cum_reward= 0
            observation =hfoEnv.reset()
            



        
        

        
def act(state,value_network,args,hfoEnv,epsilon):
    if(random.random()<epsilon):
        actionID = random.randint(0,3)
        action = hfoEnv.possibleActions[actionID]
    else:
        qVals = value_network(torch.Tensor(state))
        #print("\n"*5,"Possible actions\n",hfoEnv.possibleActions,"Qvals\n",qVals,"\n"*5)
        actionID = np.argmax(qVals.detach().numpy())
        action = hfoEnv.possibleActions[actionID]
    return action, actionID
    
def updateParams(args,episodeN):
    eps= args.initEpsilon*(pow(np.e, (-episodeN / 1500000)))
    lr= args.lr *(args.numEpisodes-episodeN )/args.numEpisodes
    return eps, lr

def computeTargets(reward, nextObservation, discountFactor, done, targetNetwork):
    reward = torch.Tensor([reward])
    if(done):
        target = reward
    else:
        qVals = targetNetwork(torch.Tensor(nextObservation)).detach()
        maxqval = torch.max(qVals)
        target = reward + discountFactor*maxqval
    return target

        
    
    

def computePrediction(state, action, valueNetwork):
    qVals = valueNetwork(torch.Tensor(state))#
    return qVals[action]
	
# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
	torch.save(model.state_dict(), strDirectory)
	
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)



