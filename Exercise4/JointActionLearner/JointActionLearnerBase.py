#!/usr/bin/env python3
# encoding utf-8
import sys
sys.path.append('../')
import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import itertools
import argparse
import numpy as np
        
class JointQLearningAgent(Agent):
    def __init__(self, learningRate, discountFactor, epsilon, numTeammates, initVals=0.0):
        super(JointQLearningAgent, self).__init__()
        self.lr = learningRate
        self.df = discountFactor
        self.eps = epsilon
        self.numMates = numTeammates

        self.qValues = {}
        self.C = {}
        self.n = {}

    def setExperience(self, state, action, oppoActions, reward, status, nextState):
        self.state = state
        self.A = action
        self.oppA = oppoActions[0]
        self.R = reward
        self.nextState = nextState
        if (not nextState in self.qValues.keys()):
            self.qValues[nextState] = dict.fromkeys([(x,y) for x in self.possibleActions for y in self.possibleActions], 0)
        
    def learn(self):
        a = (self.A,self.oppA)
        V = self.getBestAction(self.nextState)[1]
        diff = self.lr*(self.R + self.df * V - self.qValues[self.state][a])
        self.qValues[self.state][a] += diff
        self.C[self.state][self.oppA] +=1
        self.n[self.state]+=1
        return diff

    def getBestAction(self,state):
        max_v = -99999999999
        if (not state in self.n.keys() or self.n[state] == 0):
            max_A = []
            for A in self.possibleActions:
                v = 0
                for oppA in self.possibleActions:
                    v += (1 / len(self.possibleActions)) * (self.qValues[state][(A, oppA)])
                if (v >= max_v):
                    max_v = v
                    max_A.append(A)
        else:
            max_A = []
            for A in self.possibleActions:
                v = 0
                for oppA in self.possibleActions:
                    v += (self.C[state][oppA] / self.n[state]) * (self.qValues[state][(A, oppA)])
                if (v >= max_v):
                    max_v = v
                    max_A.append(A)

        return np.random.choice(max_A), max_v#/len(self.possibleActions)

    def act(self):
        if (random.random() < self.eps):  # epsilon greedy probability of chosing random
            return self.possibleActions[random.randint(0, len(self.possibleActions)-1)]
        else:
            return self.getBestAction(self.state)[0]



    def setEpsilon(self, epsilon) :
        self.eps = epsilon

    def setLearningRate(self, learningRate) :
        self.lr = learningRate

    def setState(self, state):
        self.state = state
        if (not state in self.qValues.keys()):
            self.qValues[state] = dict.fromkeys([(x,y) for x in self.possibleActions for y in self.possibleActions],0)  # when first time in a state add it to qValue table
        if (not state in self.n.keys()):
            self.n[state] = 0
            self.C[state] = dict.fromkeys(self.possibleActions,0)
    def toStateRepresentation(self, state):
        if (isinstance(state, str)):
            return state
        return (((state[0][0][0], state[0][0][1]), (state[0][1][0], state[0][1][1])))
        
    def computeHyperparameters(self, numTakenActions, episodeNumber):
        eps =  0.8 * (pow(np.e, (-episodeNumber / 10000)))
        #lr = 0.5 * (pow(np.e, (-episodeNumber / 10000)))
        lr = 0.5*(50000-episodeNumber )/50000#0.5

        return lr, eps

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--numOpponents', type=int, default=1)
    parser.add_argument('--numAgents', type=int, default=2)
    parser.add_argument('--numEpisodes', type=int, default=50000)

    args=parser.parse_args()

    MARLEnv = DiscreteMARLEnvironment(numOpponents = args.numOpponents, numAgents = args.numAgents)
    agents = []
    numAgents = args.numAgents
    numEpisodes = args.numEpisodes
    for i in range(numAgents):
        agent = JointQLearningAgent(learningRate = 0.1, discountFactor = 0.95, epsilon = 1.0, numTeammates=args.numAgents-1)
        agents.append(agent)

    numEpisodes = numEpisodes
    numTakenActions = 0
    wins = 0
    prew = 0

    for episode in range(numEpisodes):
        if (episode % 200 == 0):
            print("\nEpsilon {}".format(agent.eps))
            print("LR {}".format(agent.lr))
            print("Episode {}/{}, tot win% {}, partial win %: {}".format(wins, episode, wins / max(1, episode),    (wins - prew) / 200))
            prew = wins
        status = ["IN_GAME","IN_GAME","IN_GAME"]
        observation = MARLEnv.reset()
            
        while status[0]=="IN_GAME":
            for agent in agents:
                learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
                agent.setEpsilon(epsilon)
                agent.setLearningRate(learningRate)
            actions = []
            stateCopies = []
            for agentIdx in range(args.numAgents):
                obsCopy = deepcopy(observation[agentIdx])
                stateCopies.append(obsCopy)
                agents[agentIdx].setState(agents[agentIdx].toStateRepresentation(obsCopy))
                actions.append(agents[agentIdx].act())

            nextObservation, reward, done, status = MARLEnv.step(actions)
            numTakenActions += 1

            for agentIdx in range(args.numAgents):
                oppoActions = actions.copy()
                del oppoActions[agentIdx]
                agents[agentIdx].setExperience(agents[agentIdx].toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx], oppoActions, 
                    reward[agentIdx], status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
                agents[agentIdx].learn()
                
            observation = nextObservation
        if (reward[0] == 1):
            wins += 1