#!/usr/bin/env python3
# encoding utf-8

import sys
sys.path.append('../')
from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import numpy as np
import operator
import argparse
import random
import matplotlib.pyplot as plt
import time


class SARSAAgent(Agent):
    def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
          super(SARSAAgent, self).__init__()
          self.qValues = {}
          self.discountFactor = discountFactor
          self.epsilon =epsilon
          self.printing=False
          self.learningRate=learningRate

    def learn(self):
        if(self.nextA is None):#Terminal state, Qvalue 0
            diff = self.learningRate * (self.R -self.qValues[self.state][self.A])
            if (self.printing):
                print()
                print("LEARN START")
                print("Return {} = a[{}]*(R[{}]+expV_next[0] - Qval[{}] =".format(diff, self.learningRate, self.R,self.qValues[self.state][self.A]))
                print("Qvalues:")
                print("previous state{}:".format(self.state))
                print("Did action: ",self.A)
                print(self.qValues[self.state])

            self.qValues[self.state][self.A] = self.qValues[self.state][self.A] + diff
            if (self.printing):
                print("updated previous state{}:".format(self.state))
                print(self.qValues[self.state])
        else:
            diff = self.learningRate * (self.R + self.discountFactor * self.qValues[self.nextState][self.nextA] - self.qValues[self.state][self.A])
            if (self.printing):
                print()
                print("LEARN START")
                print("Return {} = a[{}]*(R[{}]+expV_next[{}] - Qval[{}] =".format(diff, self.learningRate, self.R,self.discountFactor * self.qValues[self.nextState][self.nextA], self.qValues[self.state][self.A]))
                print("Qvalues:")
                print("previous state{}:".format(self.state))
                print("Did action: ",self.A)
                print(self.qValues[self.state])

            self.qValues[self.state][self.A] = self.qValues[self.state][self.A] + diff
            if(self.printing):
                print("updated previous state{}:".format(self.state))
                print(self.qValues[self.state])
        return diff
    def act(self):
        if (self.printing):
            print("1111111111111ACT1111111111111111")
            print("Chose among ", self.qValues[self.actState])
        action = self.policy(self.actState)
        if (self.printing):
            print("From state ",self.actState)
            print("Chosen action: {}".format(action))
        return action
    def policy(self,state):
        if (random.random() < self.epsilon or len(self.qValues[state]) == 0):#epsilon greedy policy, chose random with probability epsilon, or when no action was ever performed from this state (all values are 0_)
            if (self.printing):
                print("Epsilon Explore")
            action = self.possibleActions[random.randint(0, 4)]
        else:
            if (self.printing):
                print("GREEDY")
            action = self.getGreedy(state)
        return action
        if (not action in self.qValues[state].keys()):
            self.qValues[state][action] = 0  # when randomly chose an action we never explored initialize it to 0.
         #print('ACTION',action)
        return action

    def getGreedy(self,state):
        max_k = max(self.qValues[state].items(), key=operator.itemgetter(1))[0]
        max_v = self.qValues[state][max_k]
        actions = [key for (key, value) in self.qValues[state].items() if value == max_v]
        return actions[random.randint(0, len(actions) - 1)]  # chose at random among actions with highest q_value

    def setState(self, state):
        self.actState = state
        if (not state in self.qValues.keys()):
            self.qValues[state] = dict.fromkeys(self.possibleActions,0)  # when first time in a state add it to qValue table

    def setExperience(self, state, action, reward, status, nextState):
        self.state = self.nextState
        self.nextState = state
        self.A = self.nextA
        self.R = self.nextR
        self.nextR = reward
        self.nextA = action
        if (not nextState in self.qValues.keys()):
            self.qValues[nextState] = dict.fromkeys(self.possibleActions, 0)

    def computeHyperparameters(self, numTakenActions, episodeNumber):
        #running: lr = 0.3
        lr = 0.3 * (5000 - episodeNumber) / 5000
        ep = 0.7 * (pow(np.e, (-episodeNumber / 600)))

        return lr, ep
    def toStateRepresentation(self, state):
        return state[0]

    def reset(self):
        self.nextA = None
        self.nextState = None
        self.nextR = None
        self.A = None
        self.state = None
        self.R = None

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--numOpponents', type=int, default=0)
    parser.add_argument('--numTeammates', type=int, default=0)
    parser.add_argument('--numEpisodes', type=int, default=5000)

    args=parser.parse_args()

    numEpisodes = args.numEpisodes
    # Initialize connection to the HFO environment using HFOAttackingPlayer
    hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
    hfoEnv.connectToServer()

    # Initialize a SARSA Agent
    agent = SARSAAgent(0.3, 0.9, 0.6)#Best so far 0.3 0.9 0.6(600)

    # Run training using SARSA
    numTakenActions = 0
    for episode in range(numEpisodes):
        if (agent.printing):
            print()
            print()
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("\n\nepisode ", episode)

        agent.reset()
        status = 0

        observation = hfoEnv.reset()
        nextObservation = None
        epsStart = True

        while status==0:
            if(agent.printing):
                print("\nEVENT:")
                print("==============================")

            learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
            agent.setEpsilon(epsilon)
            agent.setLearningRate(learningRate)

            obsCopy = observation.copy()
            agent.setState(agent.toStateRepresentation(obsCopy))
            action = agent.act()
            numTakenActions += 1
            nextObservation, reward, done, status = hfoEnv.step(action)
            agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
            if (agent.printing):
                print("end in state ",agent.toStateRepresentation(nextObservation))
                print("Ger reward: ",reward)
            if not epsStart :
            	agent.learn()
            else:
                epsStart = False

            observation = nextObservation
        if (agent.printing):
            print()
            print("EPISODE OVER, DO ONE MORE TRAIN:")
            print("6666666666666666666666666666666666")
        agent.setExperience(agent.toStateRepresentation(nextObservation), None, None, None, None)
        agent.learn()


        if (agent.printing and episode % 50 == 0):
            st = (0, 2)
            ac = 'S'
            print("Epsilon ",agent.epsilon)
            print('\n\nGreedy policy for episode {}:'.format(episode))
            cout = 0
            while ac != 'G' and st in agent.qValues.keys() and cout < 15:
                cout += 1
                ac = agent.getGreedy(st)
                print("From state {} do action {}".format(st, ac))
                if (ac == 'DRIBBLE_UP'):
                    st = (st[0], st[1] - 1)
                elif (ac == 'DRIBBLE_DOWN'):
                    st = (st[0], st[1] + 1)
                elif (ac == 'DRIBBLE_RIGHT'):
                    st = (st[0] + 1, st[1])
                elif (ac == 'DRIBBLE_LEFT'):
                    st = (st[0] - 1, st[1])
                elif (ac == 'KICK'):
                    ac = 'G'
                else:
                    print('ERROR')
            print()
