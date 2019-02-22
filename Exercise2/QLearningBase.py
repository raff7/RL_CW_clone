#!/usr/bin/env python3
# encoding utf-8
import numpy as np
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
        self.initEpsilon = epsilon
        self.epsilon = epsilon
        self.learningRate=learningRate
        self.printing=False


    def learn(self):
        greedy_action = self.getGreedy(self.nextState)
        diff = self.learningRate*(self.R + self.discountFactor*self.qValues[self.nextState][greedy_action] - self.qValues[self.state][self.A])
        if(self.printing):
            print()
            print("222222222222222222222222222222222222222")
            print("LEARN START")
            print("Return {} = a[{}]*(R[{}]+expV_next[{}] - Qval[{}] =".format(diff,self.learningRate,self.R,self.discountFactor*self.qValues[self.nextState][greedy_action],self.qValues[self.state][self.A]))
            print("Qvalues:")
            print("current state{}:".format(self.state))
            print(self.qValues[self.state])
        self.qValues[self.state][self.A] = self.qValues[self.state][self.A] + diff
        if(self.printing):
            print("updated state{}:".format(self.state))
            print(self.qValues[self.state])
            print("<<<<<<<<<<<<<<<<<2 END>>>>>>>>>>>>>>>>>>>")

        return diff

    def act(self):
        if(self.printing):
            print()
            print("1111111111111ACT1111111111111111")
            print("Chose among ",self.qValues[self.state])
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
    def toStateRepresentation(self, state):
        return state[0]

    def getGreedy(self,state):
        max_k = max(self.qValues[state].items(), key=operator.itemgetter(1))[0]
        max_v = self.qValues[state][max_k]
        actions = [key for (key,value) in self.qValues[state].items() if value ==max_v]
        return actions[random.randint(0,len(actions)-1)]#chose at random among actions with highest q_value
    def setState(self, state):
        self.state = state
        if(not state in self.qValues.keys()):
            self.qValues[state]= dict.fromkeys(self.possibleActions,0)				#when first time in a state add it to qValue table

    def setExperience(self, state, action, reward, status, nextState):
        self.R = reward
        self.A = action
        self.nextState = nextState
        if(self.printing):
            print()
            print()
            print("Current state: {}".format(state))
            print("Action: {}".format(action))
            print("Reward: {}".format(reward))
            print("Next state: {}".format(nextState))
        if(not nextState in self.qValues.keys()):
            self.qValues[nextState] = dict.fromkeys(self.possibleActions,0)
        if(self.printing):
            print("<<<<<<<<<<<<<<<<<1 END>>>>>>>>>>>>>>>>>>>")

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def reset(self):
        pass

    def computeHyperparameters(self, numTakenActions, episodeNumber):
        lr = self.learningRate
        ep = self.initEpsilon*(pow(np.e,(-episodeNumber/600)))

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
    agent = QLearningAgent(learningRate = 0.4, discountFactor = 0.9, epsilon = 0.6)

    numEpisodes = args.numEpisodes

    # Run training using Q-Learning
    numTakenActions = 0

    for episode in range(numEpisodes):
        if(agent.printing):
            print()
            print()
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("episode ",episode)
        status = 0
        observation = hfoEnv.reset()

        while status==0:
            if(agent.printing):
                print()
                print("WHILE LOOP")
                print("----------------------------------------------")
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
        if(episode%25==0):
            st = (0,2)
            ac = 'S'
            print('\n\nGreedy policy for episode {}:'.format(episode))
            cout = 0
            while ac != 'G' and st in agent.qValues.keys() and cout < 15:
                cout +=1
                ac = agent.getGreedy(st)
                print("From state {} do action {}".format(st,ac))
                if(ac == 'DRIBBLE_UP'):
                    st= (st[0],st[1]-1)
                elif(ac == 'DRIBBLE_DOWN'):
                    st=(st[0],st[1]+1)
                elif(ac == 'DRIBBLE_RIGHT'):
                    st= (st[0]+1,st[1])
                elif(ac == 'DRIBBLE_LEFT'):
                    st=(st[0]-1, st[1])
                elif(ac == 'KICK'):
                    ac = 'G'
                else:
                    print('ERROR')
            print()
