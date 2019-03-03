import sys
sys.path.append('../')

#!/usr/bin/env python3
# encoding utf-8
import random
import numpy as np
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import operator
import argparse
import time
import matplotlib.pyplot as plt


def plot_greedy_policy(q_vals, f, ax, iteration, agentID):
    if f == ax == None:
        plt.ion()
        f, ax = plt.subplots(1, 1, figsize=(6, 4))
        plt.show()
        f.canvas.set_window_title(agentID)

    possible_actions = ['MOVE_UP', 'MOVE_DOWN', 'MOVE_LEFT', 'MOVE_RIGHT', 'KICK']
    ax.clear()
    ax.set_title('Iteration {}'.format(iteration))
    ax.set_ylim([0, 6])
    ax.set_xlim([0, 5])
    ax.invert_yaxis()

    for y in range(6):
        for x in range(5):
            if (not (x, y) in q_vals.keys()):
                continue

            max_k = max(q_vals[(x, y)].items(), key=operator.itemgetter(1))[0]
            max_v = q_vals[(x, y)][max_k]
            actions = [key for (key, value) in q_vals[(x, y)].items() if value == max_v]

            for action in actions:
                if action == 'MOVE_UP':
                    ax.annotate('', (x + 0.5, y), (x + 0.5, y + 0.5), arrowprops={'width': 0.1})
                if action == 'MOVE_DOWN':
                    ax.annotate('', (x + 0.5, y + 1), (x + 0.5, y + 0.5), arrowprops={'width': 0.1})
                if action == 'MOVE_RIGHT':
                    ax.annotate('', (x + 1, y + 0.5), (x + 0.5, y + 0.5), arrowprops={'width': 0.1})
                if action == 'MOVE_LEFT':
                    ax.annotate('', (x, y + 0.5), (x + 0.5, y + 0.5), arrowprops={'width': 0.1})
                if action == 'KICK':
                    ax.text(x + 0.5, y + 0.5, action, horizontalalignment='center', verticalalignment='center', )

    f.canvas.draw()
    f.canvas.flush_events()
    # f.canvas.set_window_title('Window Title')
    time.sleep(0.001)

    return f, ax
        
class IndependentQLearningAgent(Agent):
    def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
        super(IndependentQLearningAgent, self).__init__()
        self.qValues = {}
        self.discountFactor = discountFactor
        self.epsilon = epsilon
        self.learningRate = learningRate
        self.printing = False


    def setExperience(self, state, action, reward, status, nextState):
        self.R = reward
        self.A = action
        self.nextState = nextState
        if (self.printing):
            print()
            print()
            print("Current state: {}".format(state))
            print("Action: {}".format(action))
            print("Reward: {}".format(reward))
            print("Next state: {}".format(nextState))
        if (not nextState in self.qValues.keys()):
            self.qValues[nextState] = dict.fromkeys(self.possibleActions, 0)
        if (self.printing):
            print("<<<<<<<<<<<<<<<<<1 END>>>>>>>>>>>>>>>>>>>")
    
    def learn(self):
        greedy_action = self.getGreedy(self.nextState)
        diff = self.learningRate * (self.R + self.discountFactor * self.qValues[self.nextState][greedy_action] -
                                    self.qValues[self.state][self.A])
        if (self.printing):
            print()
            print("222222222222222222222222222222222222222")
            print("LEARN START")
            print("Return {} = a[{}]*(R[{}]+expV_next[{}] - Qval[{}] =".format(diff, self.learningRate, self.R,
                                                                               self.discountFactor *
                                                                               self.qValues[self.nextState][
                                                                                   greedy_action],
                                                                               self.qValues[self.state][self.A]))
            print("Qvalues:")
            print("current state{}:".format(self.state))
            print(self.qValues[self.state])
        self.qValues[self.state][self.A] = self.qValues[self.state][self.A] + diff
        if (self.printing):
            print("updated state{}:".format(self.state))
            print(self.qValues[self.state])
            print("<<<<<<<<<<<<<<<<<2 END>>>>>>>>>>>>>>>>>>>")

        return diff

    def act(self):
        if (self.printing):
            print()
            print("1111111111111ACT1111111111111111")
            print("Chose among ", self.qValues[self.state])
        if (random.random() < self.epsilon  # epsilon greedy probability of chosing random
                or len(self.qValues[
                           self.state]) == 0):  # or when no action was ever performed from this state (all values are 0_)
            action = self.possibleActions[random.randint(0, 4)]
            if (self.printing):
                print("epsilon explore")
        else:
            if (self.printing):
                print("greedy")
            action = self.getGreedy(self.state)

        if (not action in self.qValues[self.state].keys()):
            self.qValues[self.state][action] = 0  # when randomly chose an action we never explored initialize it to 0.
        if (self.printing):
            print("Chosen action: {}".format(action))
        return action

    def getGreedy(self, state):
        max_k = max(self.qValues[state].items(), key=operator.itemgetter(1))[0]
        max_v = self.qValues[state][max_k]
        actions = [key for (key, value) in self.qValues[state].items() if value == max_v]
        return actions[random.randint(0, len(actions) - 1)]  # chose at random among actions with highest q_value

    def toStateRepresentation(self, state):
        if(isinstance(state,str)):
            return state
        return(((state[0][0][0],state[0][0][1]),(state[0][1][0],state[0][1][1])))

    def setState(self, state):
        self.state = state
        if (not state in self.qValues.keys()):
            self.qValues[state] = dict.fromkeys(self.possibleActions, 0)  # when first time in a state add it to qValue table

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon
        
    def computeHyperparameters(self, numTakenActions, episodeNumber):
        lr = 0.3
        ep = 0.8 * (pow(np.e, (-episodeNumber / 6000)))

        return lr, ep

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--numOpponents', type=int, default=1)
    parser.add_argument('--numAgents', type=int, default=2)
    parser.add_argument('--numEpisodes', type=int, default=50000)

    args=parser.parse_args()

    MARLEnv = DiscreteMARLEnvironment(numOpponents = args.numOpponents, numAgents = args.numAgents)
    agents = []
    for i in range(args.numAgents):
        agent = IndependentQLearningAgent(learningRate = 0.1, discountFactor = 0.9, epsilon = 1.0)
        agents.append(agent)

    numEpisodes = 50000
    numTakenActions = 0

    f1, ax1 = plot_greedy_policy(agents[0].qValues, None, None, 0,'agent 1')
    f2, ax2 = plot_greedy_policy(agents[1].qValues, None, None, 0,'agent 2')
    score = 0
    moving_average = np.zeros(500)
    for episode in range(numEpisodes):
        if(agent.printing):
            print()
            print()
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print("Episode {}".format(episode))
        status = ["IN_GAME","IN_GAME","IN_GAME"]
        observation = MARLEnv.reset()
        totalReward = 0.0
        timeSteps = 0

        while status[0]=="IN_GAME":
            if(agent.printing):
                i=0
            for agent in agents:
                learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
                agent.setEpsilon(epsilon)
                agent.setLearningRate(learningRate)
            actions = []
            stateCopies = []
            for agentIdx in range(args.numAgents):
                obsCopy = deepcopy(observation[agentIdx])
                stateCopies.append(obsCopy)
                agents[agentIdx].setState(agent.toStateRepresentation(obsCopy))
                actions.append(agents[agentIdx].act())
            numTakenActions += 1
            nextObservation, reward, done, status = MARLEnv.step(actions)

            for agentIdx in range(args.numAgents):
                if(agent.printing):
                    print()
                    i+=1
                    print("AGENT LEARN {}".format(agentIdx))
                agents[agentIdx].setExperience(agent.toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx], reward[agentIdx],status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
                agents[agentIdx].learn()

            observation = nextObservation
            if(reward[0] > 0 or reward[1]>0):
                score +=1
#            for i in range(0,len(moving_average)-1):
#                moving_average[i] = moving_average[i+1]
#            moving_average[0]= reward[0]
        if (episode % 100 == 0):
            print("Score {}/{} {}".format(score,episode,score/max(episode,1)))
#            print("Moving average {}".format(moving_average.mean()))
            f1, ax1 = plot_greedy_policy(agents[0].qValues, f1, ax1, episode,'Agent 1')
            f2, ax2 = plot_greedy_policy(agents[1].qValues, f2, ax2, episode, 'Agent 2')