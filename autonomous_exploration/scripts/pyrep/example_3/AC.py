#!/usr/bin/env python

import rospy
import argparse
import numpy as np
import random
from itertools import count
from collections import namedtuple
import matplotlib
import matplotlib.pyplot as plt
from os.path import dirname, join, abspath

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from espeleo_env import Environment


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=0, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


SCENE_FILE = join(dirname(abspath(__file__)),
                  'Espeleo_office_map.ttt')
env = Environment(SCENE_FILE, POS_MIN = [7.0, 0.0, 0.0],POS_MAX = [7.0, 0.0, 0.0])
rospy.sleep(5)

torch.manual_seed(args.seed)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(30, 128)

        # actor's layer
        self.action_head = nn.Linear(128, 5)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return action_prob, state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-4)
eps = np.finfo(np.float32).eps.item()


def select_action(state, eps):
    state = torch.from_numpy(state).float()
    model.eval()
    probs, state_value = model(state)


    # action = np.argmax(probs.cpu().data.numpy())
    # model.saved_actions.append(SavedAction(torch.log(probs[action]), state_value))

    

    # and sample an action using the distribution
    if(random.random() > eps):
        action = np.argmax(probs.cpu().data.numpy())
        model.saved_actions.append(SavedAction(torch.log(probs[action]), state_value))
        print("\33[92m Categorical Sample \33[0m")
    else:
        action = random.choice(np.arange(5))
        model.saved_actions.append(SavedAction(torch.log(probs[action]), state_value))
        # print("\33[41m Aleatório \33[0m")

    # m = Categorical(probs)
    # action = m.sample()

    # save to action buffer
    # model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # return action.item()
    return action


def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss 
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]



def plot_durations2(d):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(d)

    plt.savefig('EXP_Espeleo.png')


episode_durations = []
scores = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    scores_t = torch.tensor(scores, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    # plt.ylabel('Duration')
    # plt.plot(durations_t.numpy())
    plt.ylabel('Score')
    plt.plot(scores_t.numpy())

    # Take 100 episode averages and plot them too
    # if len(durations_t) >= 100:
    #   means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #   means = torch.cat((torch.zeros(99), means))
    #   plt.plot(means.numpy())
    if len(durations_t) >= 10:
        means = scores_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())



rospy.sleep(5)
episode_length = int(50)
# def main():
running_reward = 10
eps = 1.0

# run inifinitely many episodes
for i_episode in count(1):
    print("Episode :=", i_episode)

    # reset environment and episode reward
    state = env.reset()
    # print(state)
    # print(state.shape)
    ep_reward = 0

    if(i_episode >= 500):
        episode_length = int(150)

    # for each episode, only run 9999 steps so that we don't 
    # infinite loop while learning
    for t in range(1, episode_length):

        # select action from policy
        action = select_action(state,eps)
        print("Action :=", action)

        # take the action
        reward, state, done = env.step(action)
        # print(state)


        model.rewards.append(reward)
        ep_reward += reward
        if done:
            break

    eps = max(0.01, 0.99*eps)

    episode_durations.append(t+1)
    scores.append(ep_reward)
    plot_durations()

    # update cumulative reward
    running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

    # perform backprop
    finish_episode()

    # log results
    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
              i_episode, ep_reward, running_reward))

    # check if we have "solved" the cart pole problem
    # if running_reward > env.spec.reward_threshold:
    if(i_episode > 5000):
        print("Solved! Running reward is now {} and "
              "the last episode runs to {} time steps!".format(running_reward, t))
        break


print('Completed episodes')
env.shutdown()
plot_durations2(scores)


# if __name__ == '__main__':
    # main()
    