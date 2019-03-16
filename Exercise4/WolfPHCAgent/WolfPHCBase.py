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
import time
        
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
        self.epsilon = 0
    
    def greedyAction(self,state):
        max_k = max(self.qValues[state].items(), key=operator.itemgetter(1))[0]
        max_v = self.qValues[state][max_k]
        actions = [key for (key, value) in self.qValues[state].items() if value == max_v]
        return actions[random.randint(0, len(actions) - 1)]  # chose at random among actions with highest q_value
    
    def setExperience(self, state, action, reward, status, nextState):
        self.state = state
        self.action = action
        self.R = reward
        self.nextState = nextState
        if (not nextState in self.qValues.keys()):
            self.qValues[nextState] = dict.fromkeys(self.possibleActions, 0)

    def learn(self):
        diff = self.learningRate * (self.R + self.discountFactor * self.qValues[self.nextState][self.greedyAction(self.nextState)] - self.qValues[self.state][self.action])
        self.qValues[self.state][self.action] += diff
        return diff

    def act(self):
        return np.random.choice(list(self.pi[self.state].keys()),p=list(self.pi[self.state].values()))


    def calculateAveragePolicyUpdate(self):
        if (not self.state in self.mean_pi.keys()):
            self.mean_pi[self.state] = self.pi[self.state]
        else:
            for action in self.possibleActions:
                self.mean_pi[self.state][action] += (1/self.C[self.state] ) * (self.pi[self.state][action]-self.mean_pi[self.state][action])
        return self.mean_pi[self.state]
    def calculatePolicyUpdate(self):
        #Calculate delta
        v1=0
        v2=0
        for action in self.possibleActions:
            v1 += self.pi[self.state][action]*self.qValues[self.state][action]
            v2 += self.mean_pi[self.state][action]*self.qValues[self.state][action]
        if(v1>v2):
            delta = self.winDelta
        else:
            delta = self.loseDelta
        #Update
        gr_action = self.greedyAction(self.state)
        p_mass = 0
        max_q_actions = []
        subopt_actions = []
        for action in self.possibleActions:
            if(self.qValues[self.state][action] == self.qValues[self.state][gr_action]):
                max_q_actions.append(action)

            else:
                subopt_actions.append(action)



        for action in subopt_actions:
            subtr = (delta / (len(subopt_actions)))
            p_mass += min(self.pi[self.state][action], subtr)
            self.pi[self.state][action] -= min(self.pi[self.state][action], subtr)

        for action in max_q_actions:
            self.pi[self.state][action] += p_mass/len(max_q_actions)


        return self.pi[self.state].values()

    def toStateRepresentation(self, state):
        if(isinstance(state,str)):
            return state
        return(((state[0][0][0],state[0][0][1]),(state[0][1][0],state[0][1][1])))


    def setState(self, state):
        self.state = state
        if (not state in self.qValues.keys()):
            self.qValues[state] = dict.fromkeys(self.possibleActions, 0)
        if(not state in self.pi.keys()):
            self.C[state] = 1#initialize C(S) and start it at 1 (as you just visited it)
            self.pi[state] = dict.fromkeys(self.possibleActions,1.0/len(self.possibleActions))
        else:
            self.C[self.state] +=1
    def setLearningRate(self,lr):
        self.learningRate = lr
        
    def setWinDelta(self, winDelta):
        self.winDelta = winDelta
        
    def setLoseDelta(self, loseDelta):
        self.loseDelta = loseDelta
    
    def computeHyperparameters(self, numTakenActions, episodeNumber):

        #ld = 0.009 * (pow(np.e, (-episodeNumber / 20000)))+0.001
        #wd = 0.0009 * (pow(np.e, (-episodeNumber / 20000)))+0.0001
        #ld = 0.25*(np.log(episodeNumber / 8000 + 1))
        #wd = 0.025*(np.log(episodeNumber / 8000 + 1))
        ld = self.loseDelta
        wd = self.winDelta
        #lr = 0.4 * (pow(np.e, (-episodeNumber / 35000)))
        lr = self.learningRate
        return ld, wd, lr

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--numOpponents', type=int, default=1)
    parser.add_argument('--numAgents', type=int, default=2)
    parser.add_argument('--numEpisodes', type=int, default=50000)

    args=parser.parse_args()

    numOpponents = args.numOpponents
    numAgents = args.numAgents
    MARLEnv = DiscreteMARLEnvironment(numOpponents = numOpponents, numAgents = numAgents)

    agents = []
    for i in range(args.numAgents):
        agent = WolfPHCAgent(learningRate = 0.3, discountFactor = 0.95, winDelta=0.001, loseDelta=0.01)#winDelta=0.00075, loseDelta=0.0075
        agents.append(agent)

    numEpisodes = args.numEpisodes
    numTakenActions = 0
    wins=0
    prew=0

    for episode in range(numEpisodes):
        if(episode%200==0):
            print("Epsilon {}".format(agent.epsilon))
            print("Episode {}/{}, tot win% {}, partial win %: {}".format(wins,episode,wins/max(1,episode) ,(wins-prew)/200))
            prew=wins

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
        if (reward[0] == 1):
            wins += 1
