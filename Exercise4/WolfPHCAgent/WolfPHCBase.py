#!/usr/bin/env python3
# encoding utf-8
import sys
sys.path.append('../')
import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import numpy as np
import operator
        
class WolfPHCAgent(Agent):
    def __init__(self, learningRate, discountFactor, winDelta=0.01, loseDelta=0.1, initVals=0.0):
        super(WolfPHCAgent, self).__init__()
        self.qValues = {}
        self.pi = {}
        self.mean_pi ={}
        self.C = {}
        self.discountFactor = discountFactor
        self.winDelta = winDelta
        self.loseDelta = loseDelta
        self.learningRate = learningRate
        self.printing = False
    
    def greedyAction(self):
        max_k = max(self.qValues[self.state].items(), key=operator.itemgetter(1))[0]
        max_v = self.qValues[self.state][max_k]
        actions = [key for (key, value) in self.qValues[self.state].items() if value == max_v]
        return actions[random.randint(0, len(actions) - 1)]  # chose at random among actions with highest q_value
    
    def setExperience(self, state, action, reward, status, nextState):
        self.action = action
        self.R = reward
        self.nextState = nextState

    def learn(self):
        diff = self.learningRate * (self.R + self.discountFactor * self.qValues[self.nextState][self.greedyAction()] - self.qValues[self.state][self.A])
        self.qValues[self.state][self.action]+= diff
        return diff

    def act(self):
        return np.random.choice(list(self.pi.keys()),p=list(self.pi.values()))

    def calculateAveragePolicyUpdate(self):
        if (not self.state in self.mean_pi.keys()):
            self.mean_pi[self.state] = self.pi[self.state]
        else:
            for action in self.possibleActions:
                self.mean_pi[self.state][action] += (1/self.C[self.state] ) * (self.pi[self.state][action]-self.mean_pi[self.state][action])
        return self.mean_pi[self.state]
    def calculatePolicyUpdate(self):
        v1=0
        v2=0
        for action in self.possibleActions:
            v1 += self.pi[self.state][action]*self.qValues[self.state][action]
            v2 += self.mean_pi[self.state][action]*self.qValues[self.state][action]
        if(v1>v2):
            delta = self.winDelta
        else:
            delta = self.loseDelta
        gr_action = self.greedyAction()
        for action in self.possibleActions:
            if(action == gr_action):
                self.pi[self.state][action] += delta
            else:
                self.pi[self.state][action] -= delta/(len(self.possibleActions)-1)
        
        return self.pi[self.state]

    
    def toStateRepresentation(self, state):
        if(isinstance(state,str)):
            return state
        return(((state[0][0][0],state[0][0][1]),(state[0][1][0],state[0][1][1])))


    def setState(self, state):
        self.state = state
        if (not state in self.qValues.keys()):
            self.C[state] = 1#initialize C(S) and start it at 1 (as you just visited it)
            self.qValues[state] = dict.fromkeys(self.possibleActions, 0)
            self.pi[state] = dict.fromkeys(self.possibleActions,1/len(self.possibleActions))
        else:
            self.C[self.state] +=1
    def setLearningRate(self,lr):
        self.learningRate = lr
        
    def setWinDelta(self, winDelta):
        self.winDelta = winDelta
        
    def setLoseDelta(self, loseDelta):
        self.loseDelta = loseDelta
    
    def computeHyperparameters(self, numTakenActions, episodeNumber):
        return self.loseDelta, self.winDelta, self.learningRate 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--numOpponents', type=int, default=1)
    parser.add_argument('--numAgents', type=int, default=2)
    parser.add_argument('--numEpisodes', type=int, default=100000)

    args=parser.parse_args()

    numOpponents = args.numOpponents
    numAgents = args.numAgents
    MARLEnv = DiscreteMARLEnvironment(numOpponents = numOpponents, numAgents = numAgents)

    agents = []
    for i in range(args.numAgents):
        agent = WolfPHCAgent(learningRate = 0.2, discountFactor = 0.99, winDelta=0.01, loseDelta=0.1)
        agents.append(agent)

    numEpisodes = args.numEpisodes
    numTakenActions = 0
    for episode in range(numEpisodes):    
        status = ["IN_GAME","IN_GAME","IN_GAME"]
        observation = MARLEnv.reset()
        
        while status[0]=="IN_GAME":
            for agent in agents:
                loseDelta, winDelta, learningRate = agent.computeHyperparameters(numTakenActions, episode)
                agent.setLoseDelta(loseDelta)
                agent.setWinDelta(winDelta)
                agent.setLearningRate(learningRate)
            actions = []
            perAgentObs = []
            agentIdx = 0
            for agent in agents:
                obsCopy = deepcopy(observation[agentIdx])
                perAgentObs.append(obsCopy)
                agent.setState(agent.toStateRepresentation(obsCopy))
                actions.append(agent.act())
                agentIdx += 1
            nextObservation, reward, done, status = MARLEnv.step(actions)
            numTakenActions += 1

            agentIdx = 0
            for agent in agents:
                agent.setExperience(agent.toStateRepresentation(perAgentObs[agentIdx]), actions[agentIdx], reward[agentIdx], 
                    status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
                agent.learn()
                agent.calculateAveragePolicyUpdate()
                agent.calculatePolicyUpdate()
                agentIdx += 1
            
            observation = nextObservation
