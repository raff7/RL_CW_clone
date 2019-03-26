import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Networks import ValueNetwork
from torch.autograd import Variable
from Environment import HFOEnv
import random
import numpy as np
import os
from hfo import GOAL


def train(idx, args, value_network, target_value_network, optimizer, counter, rew_queue, scored_queue, count_queue,
        main_terminated):
    port = 4000 + idx * 20
    seed = idx * 10
    random.seed(seed)
    hfoEnv = HFOEnv(numTeammates=0, numOpponents=1, port=port, seed=seed)
    hfoEnv.connectToServer()

    observation = hfoEnv.reset()

    # This runs a random agent
    episodeNumber = 0
    cum_reward = 0
    steps_in_episode = 0
    while not main_terminated.is_set():
        eps = calculate_eps(counter, args)
        action_id = act_epsgreedy(observation, value_network, hfoEnv, eps)
        act = hfoEnv.possibleActions[action_id]
        newObservation, reward, done, status, info = hfoEnv.step(act)

        # Train network
        target = computeTargets(reward, newObservation, args.discount_factor, done, target_value_network)
        state = torch.Tensor(observation)
        prediction = computePrediction(state, action_id, value_network)
        loss = 0.5 * (prediction - target) ** 2
        loss.backward()

        with counter.get_lock():
            counter.value += 1
            if counter.value % args.train_every == 0:
                if args.lr_decay:
                    curr_lr = linear_sched(args.lr, counter.value, args.max_trans)
                    change_lr(optimizer, curr_lr)
                optimizer.step()
                optimizer.zero_grad()

            if counter.value % args.copy_every == 0:
                hard_update(target_value_network, value_network)

            if counter.value % args.save_every == 0:
                main_folder = os.path.join('models', args.save_name)
                if not os.path.exists(main_folder):
                    os.makedirs(main_folder)
                saveModelNetwork(value_network, os.path.join(main_folder, 'checkpoint-steps={}.pt'
                                                             .format(counter.value)))

            if counter.value >= args.max_trans:
                rew_queue.put_nowait(None)
                scored_queue.put_nowait(None)
                break

        cum_reward += reward
        observation = newObservation
        steps_in_episode += 1

        if done:
            episodeNumber += 1
            observation = hfoEnv.reset()

            rew_queue.put_nowait(cum_reward)
            scored = 1 if status == 1 else 0
            scored_queue.put_nowait(scored)

            steps_in_episode = 500 if status != GOAL else steps_in_episode
            count_queue.put_nowait(steps_in_episode)

            steps_in_episode = 0
            cum_reward = 0



def computeTargets(reward, nextObservation, discountFactor, done, targetNetwork):
    reward = torch.Tensor([reward])

    if done:
        return reward
    else:
        next_state = torch.Tensor(nextObservation)
        q_vals = targetNetwork(next_state).detach()
        max_qval = torch.max(q_vals)

        return reward + discountFactor * max_qval

def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def linear_sched(init_val, counter, max_iters):
    return init_val * (max_iters - counter)/max_iters

def calculate_eps(counter, args):
    eps = args.start_eps * np.exp(-counter.value/args.eps_decay)

    return eps

def act_epsgreedy(state, valueNetwork, env, epsilon=0.0):
    if random.random() < epsilon:
        return random.choice(range(len(env.possibleActions)))
    else:
        state = torch.Tensor(state)
        q_vals = valueNetwork(state)

        id = np.argmax(q_vals.detach().numpy())

        return id


def computePrediction(state, action_id, valueNetwork):
    state = torch.Tensor(state)

    q_vals = valueNetwork(state)

    return q_vals[action_id]


# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
    torch.save(model.state_dict(), strDirectory)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
