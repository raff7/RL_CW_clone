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
		if(self.state is None):#Terminal state, Qvalue 0
			print("REWARD: ",self.R)#TODO fix this mess.. pre-not pre
			print("QVALUE FOR ACTION: ",self.qValues[self.preS][self.preA])
			diff = self.learningRate * (self.preR -self.qValues[self.preS][self.preA])#TODO CHECK IF I INVERTED NEST STATE AND STATE
			self.qValues[self.preS][self.preA] = self.qValues[self.preS][self.preA] + diff
		else:
<<<<<<< HEAD
=======
			print("pre A",self.preA)
			print("pre s",self.preS)
			print("pre r",self.preR)
			print("a",self.A)
			print(" s",self.state)
			print(" r",self.R)

>>>>>>> 8a8bef896326d80a48f992da52dbcd0262b9bab3
			diff = self.learningRate * (self.preR + self.discountFactor * self.qValues[self.state][self.A] -self.qValues[self.preS][self.preA])
			self.qValues[self.preS][self.preA] = self.qValues[self.preS][self.preA] + diff
		return diff
	def act(self):
		return self.policy(self.state)

	def policy(self,state):
		if (random.random() < self.epsilon or len(self.qValues[state]) == 0):#epsilon greedy policy, chose random with probability epsilon, or when no action was ever performed from this state (all values are 0_)
			action = self.possibleActions[random.randint(0, 4)]
		else:
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
		ep = self.epsilon
		if (episodeNumber > 2000):
			ep = 0.03
		elif (episodeNumber > 700):
			ep = 0.1
		elif (episodeNumber > 400):
			ep = 0.2
		elif (episodeNumber > 300):
			ep = 0.3
		elif (episodeNumber > 150):
			ep = 0.5
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
	agent = SARSAAgent(0.25, 0.9, 0.8)

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
			print("ACTION: ",action)
			nextObservation, reward, done, status = hfoEnv.step(action)
			print(obsCopy, action, reward, nextObservation)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))

			if not epsStart :
				agent.learn()
			else:
				epsStart = False

			observation = nextObservation
			print("==============================")
			print("\nPLAY")
			print("\nState {}".format(agent.state))
			print("QValues {}".format(agent.qValues))
			print("Action: {}".format(agent.A))
			print("Reward: {}".format(agent.R))
			print("Next State: {}".format(agent.nextState))


		agent.setExperience(agent.toStateRepresentation(nextObservation), None, None, None, None)
		agent.learn()
