#!/usr/bin/env python

import rospy

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from os.path import dirname, join, abspath
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from espeleo_env import Environment


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100
TARGET_UPDATE = 30

SCENE_FILE = join(dirname(abspath(__file__)),
				  'Espeleo_office_map.ttt')
env = Environment(SCENE_FILE, POS_MIN = [7.0, 0.0, 0.0],POS_MAX = [7.0, 0.0, 0.0])
rospy.sleep(5)


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


Transition = namedtuple('Transition',
						('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

	def __init__(self, capacity):
		self.memory = deque([],maxlen=capacity)

	def push(self, *args):
		"""Save a transition"""
		self.memory.append(Transition(*args))

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)



class DQN(nn.Module):
	def __init__(self, input_shape, n_actions):
		super(DQN, self).__init__()

		self.dense = nn.Sequential(
			nn.Linear(input_shape, 512),
			nn.ReLU(),
			nn.Linear(512,512),
			nn.ReLU(),
			nn.Linear(512, n_actions)
		)


		# self.conv = nn.Sequential(
		# 	nn.Conv2d(input_shape[0],32,kernel_size=8,stride=4),
		# 	nn.BatchNorm2d(32),
		# 	nn.ReLU(),
		# 	nn.Conv2d(32,64,kernel_size=4,stride=2),
		# 	nn.BatchNorm2d(64),
		# 	nn.ReLU(),
		# 	nn.Conv2d(64,64,kernel_size=3,stride=1),
		# 	nn.BatchNorm2d(64),
		# 	nn.ReLU()
		# )

		# cnw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(w,3,1), 4, 2), 8, 4)
		# cnh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(h,3,1), 4, 2), 8, 4)
		# conv_size = cnw * cnh * 32

		# self.fc = nn.Sequential(
		# 	nn.Linear(conv_size, 512),
		# 	nn.ReLU(),
		# 	nn.Linear(512,n_actions)
		# )


	def conv2d_size_out(self, size, kernel_size, stride):
		return (size - (kernel_size - 1) - 1) // stride  + 1


	def forward(self, x):
		return self.dense(x.float())

		# out = self.conv(x).view(x.size()[0],-1)
		# return self.fc(out)



def select_action(state):
	global steps_done
	sample = random.random()
	eps_threshold = EPS_END + (EPS_START - EPS_END) * \
		math.exp(-1. * steps_done / EPS_DECAY)
	steps_done += 1
	if sample > eps_threshold:
		with torch.no_grad():
			# t.max(1) will return largest column value of each row.
			# second column on max result is index of where max element was
			# found, so we pick action with the larger expected reward.
			return policy_net(state).max(1)[1].view(1, 1)
	else:
		return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)




n_actions = 5
state_dim = 11
num_episodes = 2000
num_steps = 200

policy_net = DQN(state_dim, n_actions).to(device)
target_net = DQN(state_dim, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

steps_done = 0

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)



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
	# 	means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
	# 	means = torch.cat((torch.zeros(99), means))
	# 	plt.plot(means.numpy())
	if len(durations_t) >= 10:
		means = scores_t.unfold(0, 10, 1).mean(1).view(-1)
		means = torch.cat((torch.zeros(9), means))
		plt.plot(means.numpy())

	plt.pause(0.001)  # pause a bit so that plots are updated
	if is_ipython:
		display.clear_output(wait=True)
		display.display(plt.gcf())



def optimize_model():
	if len(memory) < BATCH_SIZE:
		return
	transitions = memory.sample(BATCH_SIZE)
	# Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
	# detailed explanation). This converts batch-array of Transitions
	# to Transition of batch-arrays.
	batch = Transition(*zip(*transitions))

	# Compute a mask of non-final states and concatenate the batch elements
	# (a final state would've been the one after which simulation ended)
	non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
										  batch.next_state)), device=device, dtype=torch.bool)
	non_final_next_states = torch.cat([s for s in batch.next_state
												if s is not None])
	state_batch = torch.cat(batch.state)
	action_batch = torch.cat(batch.action)
	reward_batch = torch.cat(batch.reward)

	# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
	# columns of actions taken. These are the actions which would've been taken
	# for each batch state according to policy_net
	state_action_values = policy_net(state_batch).gather(1, action_batch)

	# Compute V(s_{t+1}) for all next states.
	# Expected values of actions for non_final_next_states are computed based
	# on the "older" target_net; selecting their best reward with max(1)[0].
	# This is merged based on the mask, such that we'll have either the expected
	# state value or 0 in case the state was final.
	next_state_values = torch.zeros(BATCH_SIZE, device=device)
	next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
	# Compute the expected Q values
	expected_state_action_values = (next_state_values * GAMMA) + reward_batch

	# Compute Huber loss
	criterion = nn.SmoothL1Loss()
	loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

	# Optimize the model
	optimizer.zero_grad()
	loss.backward()
	for param in policy_net.parameters():
		param.grad.data.clamp_(-1, 1)
	optimizer.step()



rospy.sleep(10)

############ TRAINING
for i_episode in range(num_episodes):
	score = 0
	state = env.reset()
	# state = torch.tensor(state, device=device)
	state = torch.from_numpy(state).float().unsqueeze(0).to(device)
	print("Episode :=", i_episode)
	for t in count():
		print(t)
		action = select_action(state)
		action_cp = action.to('cpu').data.numpy().item()
		# print("action :=", test.data.numpy().item())

		reward, next_state, done = env.step(action_cp)
		score += reward

		reward = torch.tensor([reward], device=device)
		next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
		# next_state = torch.tensor(next_state, device=device)


		memory.push(state, action, next_state, reward)
#
		state = next_state

		# # Perform one step of the optimization (on the policy network)
		optimize_model()


		if done or t>= num_steps:
			episode_durations.append(t + 1)
			# plot_durations()
			scores.append(score)
			plot_durations()
			break


	if i_episode % TARGET_UPDATE == 0:
		target_net.load_state_dict(policy_net.state_dict())





print('Completed episodes')
env.shutdown()


np.savetxt(join(dirname(abspath(__file__)),'scores.txt'), scores, delimiter=',')
fig2 = plt.figure()
plt.plot(scores)
plt.ylabel('score')
plt.xlabel('episodes')
plt.title('Training score')