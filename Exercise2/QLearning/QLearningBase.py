#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import operator
import random
import sys
import argparse


class QLearningAgent(Agent):
    def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
        super(QLearningAgent, self).__init__()
        self.qValues = {}
        self.discountFactor = discountFactor
        self.epsilon = epsilon
        self.learningRate=learningRate


    def learn(self):
        greedy_action = self.getGreedy(self.nextState)
        diff = self.learningRate*(self.R + self.discountFactor*self.qValues[self.nextState][greedy_action] - self.qValues[self.state][self.A])

        self.qValues[self.state][self.A] = self.qValues[self.state][self.A] + diff
        print()
        print("LEARN START")
        print("Return{}.".format(diff))
        print("Qvalues:")
        print(self.qValues)
        return diff

    def act(self):
        print("???????ACT????????")
        print(self.qValues[self.state])
        if(random.random() < self.epsilon #epsilon greedy probability of chosing random
                or len(self.qValues[self.state]) == 0):# or when no action was ever performed from this state (all values are 0_)
            action = self.possibleActions[random.randint(0, 4)]
            print("epsilon explore")
        else:
            print("greedy")
            action = self.getGreedy(self.state)

        if (not action in self.qValues[self.state].keys()):
            self.qValues[self.state][action] = 0  # when randomly chose an action we never explored initialize it to 0.
        print("Chosen action: {}".format(action))
        print("??????????????<>????????????")
        return action
    def toStateRepresentation(self, state):
        return state[0]

    def getGreedy(self,state):
        print("CALLED GETGREEDY")
        max_k = max(self.qValues[state].items(), key=operator.itemgetter(1))[0]
        max_v = self.qValues[state][max_k]
        actions = [key for (key,value) in self.qValues[state].items() if value ==max_v]
        print(max_v)
        print(self.qValues[state])
        print(actions)
        print(random.randint(0,len(actions)-1))
        print(random.randint(0,len(actions)-1))
        print(random.randint(0,len(actions)-1))
        print(random.randint(0,len(actions)-1))
        print(random.randint(0,len(actions)-1))
        print("Test done")
        print()
        return actions[random.randint(0,len(actions)-1)]#chose at random among actions with highest q_value
    def setState(self, state):
        self.state = state
        if(not state in self.qValues.keys()):
            self.qValues[state]= dict.fromkeys(self.possibleActions,0)				#when first time in a state add it to qValue table

    def setExperience(self, state, action, reward, status, nextState):
        self.R = reward
        self.A = action
        self.nextState = nextState
        print("=============")
        print("setExperience")
        print("=============")
        print("Current state: {}".format(state))
        print("Action: {}".format(action))
        print("Reward: {}".format(reward))
        print("Next state: {}".format(nextState))
        if(not nextState in self.qValues.keys()):
            self.qValues[nextState] = dict.fromkeys(self.possibleActions,0)


    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def reset(self):
        pass

    def computeHyperparameters(self, numTakenActions, episodeNumber):
        lr = self.learningRate
        ep = self.epsilon
        return lr, ep

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--numOpponents', type=int, default=0)
    parser.add_argument('--numTeammates', type=int, default=0)
    parser.add_argument('--numEpisodes', type=int, default=500)

    args=parser.parse_args()

    # Initialize connection with the HFO server
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--numOpponents', type=int, default=0)
    parser.add_argument('--numTeammates', type=int, default=0)
    parser.add_argument('--numEpisodes', type=int, default=500)

    args = parser.parse_args()

    hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
    hfoEnv.connectToServer()

    # Initialize a Q-Learning Agent
    agent = QLearningAgent(learningRate = 0.1, discountFactor = 0.99, epsilon = 0.1)

    numEpisodes = args.numEpisodes

    # Run training using Q-Learning
    numTakenActions = 0

    for episode in range(numEpisodes):
        print()
        print("episode ",episode)
        status = 0
        observation = hfoEnv.reset()

        while status==0:
            learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
            agent.setEpsilon(epsilon)
            agent.setLearningRate(learningRate)

            obsCopy = observation.copy()
            agent.setState(agent.toStateRepresentation(obsCopy))
            action = agent.act()
            numTakenActions += 1

            nextObservation, reward, done, status = hfoEnv.step(action)
            agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
            update = agent.learn()

            observation = nextObservation

