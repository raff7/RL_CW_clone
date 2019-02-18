#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
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
		if(self.nextState is None):
			diff = self.learningRate * (self.R + self.discountFactor * 0 -self.qValues[self.state][self.A])
			self.qValues[self.state][self.A] = self.qValues[self.state][self.A] + diff
		else:
			diff = self.learningRate * (self.R + self.discountFactor * self.qValues[self.state][self.policy(self.nextState)] -self.qValues[self.state][self.A])
			self.qValues[self.state][self.A] = self.qValues[self.state][self.A] + diff
		return diff
	def act(self):
		self.policy(self.state)
	def policy(self,state):
		if (random.random() < self.epsilon or len(self.qValues[state]) == 0):#epsilon greedy policy, chose random with probability epsilon, or when no action was ever performed from this state (all values are 0_)
			action = self.possibleActions[random.randint(0, 4)]
		else:
			action = max(self.qValues[state])

		if (not action in self.qValues[state].keys()):
			self.qValues[state][action] = 0  # when randomly chose an action we never explored initialize it to 0.
          print('ACTION',action)
		return action

	def setState(self, state):
		self.state = state
		if(not state in self.qValues.keys()):
			self.qValues[state]={}				#when first time in a state add it to qValue table

	def setExperience(self, state, action, reward, status, nextState):
		self.R = reward
		self.A = action
		self.nextState = nextState

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		return self.learningRate, self.epsilon

	def toStateRepresentation(self, state):
		return state[0]

	def reset(self):
		pass

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
	agent = SARSAAgent(0.1, 0.99, 0.05)

	# Run training using SARSA
	numTakenActions = 0 
	for episode in range(numEpisodes):	
		agent.reset()
		status = 0

		observation = hfoEnv.reset()
		nextObservation = None
		epsStart = True

		while status==0:
			learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			agent.setLearningRate(learningRate)
			
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1

			nextObservation, reward, done, status = hfoEnv.step(action)
			print(obsCopy, action, reward, nextObservation)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			
			if not epsStart :
				agent.learn()
			else:
				epsStart = False
			
			observation = nextObservation

		agent.setExperience(agent.toStateRepresentation(nextObservation), None, None, None, None)
		agent.learn()

	
