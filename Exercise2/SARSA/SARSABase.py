#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import numpy as np
import argparse
import random

class SARSAAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
          super(SARSAAgent, self).__init__()
          self.qValues = {}
          self.discountFactor = discountFactor
          self.epsilon = epsilon
          self.learningRate=learningRate
	def learn(self):
		if(self.A is None):#Terminal state, Qvalue 0
			diff = self.learningRate * (self.preR -self.qValues[self.preS][self.preA])
			print()
			print("LEARN START")
			print("Return {} = a[{}]*(R[{}]+expV_next[0] - Qval[{}] =".format(diff, self.learningRate, self.preR,self.qValues[self.preS][self.preA]))
			print("Qvalues:")
			print("previous state{}:".format(self.preS))
			print(self.qValues[self.preS])

			self.qValues[self.preS][self.preA] = self.qValues[self.preS][self.preA] + diff

			print("updated previous state{}:".format(self.preS))
			print(self.qValues[self.preS])
		else:
			diff = self.learningRate * (self.preR + self.discountFactor * self.qValues[self.state][self.A] - self.qValues[self.preS][self.preA])

			print()
			print("LEARN START")
			print("Return {} = a[{}]*(R[{}]+expV_next[{}] - Qval[{}] =".format(diff, self.learningRate, self.preR,self.discountFactor * self.qValues[self.state][self.A], self.qValues[self.preS][self.preA]))
			print("Qvalues:")
			print("previous state{}:".format(self.preS))
			print(self.qValues[self.preS])

			self.qValues[self.preS][self.preA] = self.qValues[self.preS][self.preA] + diff

			print("updated previous state{}:".format(self.preS))
			print(self.qValues[self.preS])
		return diff
	def act(self):
		print("1111111111111ACT1111111111111111")
		print("Chose among ", self.qValues[self.state])
		action = self.policy(self.state)
		print("From state ",self.state)
		print("Chosen action: {}".format(action))
		return action
	def policy(self,state):
		if (random.random() < self.epsilon or len(self.qValues[state]) == 0):#epsilon greedy policy, chose random with probability epsilon, or when no action was ever performed from this state (all values are 0_)
			print("Epsilon Explore")
			action = self.possibleActions[random.randint(0, 4)]
		else:
			print("GREEDY")
			action = max(self.qValues[state])

		if (not action in self.qValues[state].keys()):
			self.qValues[state][action] = 0  # when randomly chose an action we never explored initialize it to 0.
         #print('ACTION',action)
		return action

	def setState(self, state):
		self.state = state
		if (not state in self.qValues.keys()):
			self.qValues[state] = dict.fromkeys(self.possibleActions,0)  # when first time in a state add it to qValue table

	def setExperience(self, state, action, reward, status, nextState):
		self.preA=self.A
		self.preR =self.R
		self.preS= self.state
		self.R = reward
		self.A = action
		self.nextState = nextState
		self.state = state
		if (not nextState in self.qValues.keys()):
			self.qValues[nextState] = dict.fromkeys(self.possibleActions, 0)

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		lr = self.learningRate
		ep = 0.75 * (pow(np.e, (-episodeNumber / 700)))

		return lr, ep
		return lr, ep
	def toStateRepresentation(self, state):
		return state[0]

	def reset(self):
		self.preA = None
		self.preS = None
		self.preR = None
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
	parser.add_argument('--numEpisodes', type=int, default=500)

	args=parser.parse_args()

	numEpisodes = args.numEpisodes
	# Initialize connection to the HFO environment using HFOAttackingPlayer
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a SARSA Agent
	agent = SARSAAgent(0.15, 0.9, 0.8)

	# Run training using SARSA
	numTakenActions = 0 
	for episode in range(numEpisodes):
		print()
		print()
		print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
		print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
		print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
		print("episode ", episode)

		agent.reset()
		status = 0

		observation = hfoEnv.reset()
		nextObservation = None
		epsStart = True

		while status==0:
			print("EVENT:")
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
			print("end in state ",agent.nextState)
			if not epsStart :
				agent.learn()
			else:
				epsStart = False

			observation = nextObservation

		print()
		print("EPISODE OVER, DO ONE MORE TRAIN:")
		print("6666666666666666666666666666666666")
		agent.setExperience(agent.toStateRepresentation(nextObservation), None, None, None, None)
		agent.learn()
