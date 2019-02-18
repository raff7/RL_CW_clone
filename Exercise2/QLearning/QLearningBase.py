#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
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
		diff = self.learningRate*(self.R + self.discountFactor*self.qValues[self.nextState][max(self.qValues[self.nextState])] - self.qValues[self.state][self.A])
		self.qValues[self.state][self.A] = self.qValues[self.state][self.A] + diff
		return diff

	def act(self):
		if (random.random() < self.epsilon or len(self.qValues[self.state]) == 0):#epsilon greedy policy, chose random with probability epsilon, or when no action was ever performed from this state (all values are 0_)
			action = self.possibleActions[random.randint(0, 4)]
		else:
			action = max(self.qValues[self.state])

		if (not action in self.qValues[self.state].keys()):
			self.qValues[self.state][action] = 0  # when randomly chose an action we never explored initialize it to 0.

		return action
	def toStateRepresentation(self, state):
		return state[0]

	def setState(self, state):
		self.state = state
		if(not state in self.qValues.keys()):
			self.qValues[state]={}				#when first time in a state add it to qValue table

	def setExperience(self, state, action, reward, status, nextState):
		self.R = reward
		self.A = action
		self.nextState = nextState
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
	
