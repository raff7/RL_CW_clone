#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import random
import operator
import numpy as np


class MonteCarloAgent(Agent):
    def __init__(self, discountFactor, epsilon, initVals=0.0):
        super(MonteCarloAgent, self).__init__()
        self.epsilon = epsilon
        self.discountFactor = discountFactor
        self.qValues = {}
        self.returns = {}
        self.printing=True


    def learn(self):
        if(self.printing):
            print("------------------LEARN STAGE------------------")
            print(self.path)
        G = 0
        visited = []
        returns = []
        for i in range(len(self.path)-1,-1,-1):
            step = self.path[i]
            if (self.printing):
                print()
                print("Step ",i)
                print(step)
            if (self.printing):
                print("G:{} =  discG[{}] + R[{}] ".format(self.discountFactor*G + step[2],self.discountFactor*G,step[2]))
            G = self.discountFactor*G + step[2]
            if(not (step[0],step[1]) in visited):
                if (self.printing):
                    print("step not visited")
                visited.append((step[0],step[1]))
                if(not (step[0],step[1]) in self.returns.keys()):
                    self.returns[(step[0],step[1])] = []
                self.returns[(step[0],step[1])].append(G)
                if(self.printing):
                    print("QValues ",self.qValues[step[0]][step[1]])
                if(not step[0] in self.qValues.keys()):
                    self.qValues[step[0]] = dict.fromkeys(self.possibleActions,0)
                self.qValues[step[0]][step[1]] = np.mean(self.returns[(step[0],step[1])])
                returns.append(self.qValues[step[0]][step[1]])
                if (self.printing):
                    print("updated QValue= ",self.qValues[step[0]][step[1]])
        print("\nReturns ",returns[::-1])
        return returns[::-1]

    def toStateRepresentation(self, state):
        return state[0]

    def setExperience(self, state, action, reward, status, nextState):
        self.path.append((state,action,reward))

    def setState(self, state):
        self.state = state
        if(not state in self.qValues.keys()):
            self.qValues[state]= dict.fromkeys(self.possibleActions,0)				#when first time in a state add it to qValue table


    def reset(self):
        self.path =  []

    def act(self):
        if(self.printing):
            print()
            print("1111111111111ACT1111111111111111")
            print("Chose among ",self.qValues[self.state])
            print("From state ",self.state)
        if(random.random() < self.epsilon #epsilon greedy probability of chosing random
                or len(self.qValues[self.state]) == 0):# or when no action was ever performed from this state (all values are 0_)
            action = self.possibleActions[random.randint(0, 4)]
            if(self.printing):
                print("epsilon explore")
        else:
            if(self.printing):
                print("greedy")
            action = self.getGreedy(self.state)

        if (not action in self.qValues[self.state].keys()):
            self.qValues[self.state][action] = 0  # when randomly chose an action we never explored initialize it to 0.
        if(self.printing):
            print("Chosen action: {}".format(action))
        return action

    def getGreedy(self, state):
        max_k = max(self.qValues[state].items(), key=operator.itemgetter(1))[0]
        max_v = self.qValues[state][max_k]
        actions = [key for (key, value) in self.qValues[state].items() if value == max_v]
        return actions[random.randint(0, len(actions) - 1)]  # chose at random among actions with highest q_value



    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def computeHyperparameters(self, numTakenActions, episodeNumber):
        ep = self.epsilon
        if (episodeNumber > 2000):
            ep = 0.03
        elif (episodeNumber > 700):
            ep = 0.1
        elif(episodeNumber>400):
            ep=0.2
        elif(episodeNumber>300):
            ep=0.3
        elif(episodeNumber>150):
            ep=0.5
        return ep


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--numOpponents', type=int, default=0)
    parser.add_argument('--numTeammates', type=int, default=0)
    parser.add_argument('--numEpisodes', type=int, default=500)

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--numOpponents', type=int, default=0)
    parser.add_argument('--numTeammates', type=int, default=0)
    parser.add_argument('--numEpisodes', type=int, default=500)

    args=parser.parse_args()

    #Init Connections to HFO Server
    hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
    hfoEnv.connectToServer()

    # Initialize a Monte-Carlo Agent
    agent = MonteCarloAgent(discountFactor = 0.9, epsilon = 1.0)
    numEpisodes = args.numEpisodes
    numTakenActions = 0
    # Run training Monte Carlo Method
    for episode in range(numEpisodes):
        agent.reset()
        observation = hfoEnv.reset()
        status = 0

        while status==0:
            epsilon = agent.computeHyperparameters(numTakenActions, episode)
            agent.setEpsilon(epsilon)
            obsCopy = observation.copy()
            agent.setState(agent.toStateRepresentation(obsCopy))
            action = agent.act()
            numTakenActions += 1
            nextObservation, reward, done, status = hfoEnv.step(action)
            agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
            observation = nextObservation

        agent.learn()
