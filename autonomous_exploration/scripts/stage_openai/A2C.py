#!/usr/bin/env python

import rospy
from config import Config
from ros_stage_env import StageEnvironment
import rospkg

import torch  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import numpy as np
from collections import deque
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns



# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


# hyperparameters
hidden_size = 256
learning_rate = 3e-4

# Constants
GAMMA = 0.99
num_steps = 5
max_episodes = 200


def conv_block(input_size, output_size):
	block = nn.Sequential(
		nn.Conv2d(input_size, output_size, kernel_size=3,stride=1,padding=1), nn.BatchNorm2d(output_size), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
		nn.Conv2d(output_size, output_size, kernel_size=3,stride=1,padding=1), nn.BatchNorm2d(output_size), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
	)

	return block
 


class ConcatNetwork(nn.Module):
	def __init__(self):
		super(ConcatNetwork, self).__init__()
		# test convolution
		self.conv1 = conv_block(1, 4)
		self.ln1 = nn.Linear(4 * 16 * 16, 32)


	def forward(self, x):

		x[1] = x[1].view(x[1].size(0), -1)

		# conv image
		x[2] = self.conv1(x[2])
		x[2] = x[2].view(x[2].size(0), -1)
		x[2] = self.ln1(x[2])


		x = torch.cat((x[0], x[1], x[2]), dim=1)

		return x


class ActorCritic(nn.Module):
	def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
		super(ActorCritic, self).__init__()

		self.cc_network = ConcatNetwork()

		self.num_actions = num_actions
		self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
		self.critic_linear2 = nn.Linear(hidden_size, 1)

		self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
		self.actor_linear2 = nn.Linear(hidden_size, num_actions)
	
	def forward(self, state):
		state_v = []
		for k in state:
			state_v.append(Variable(torch.from_numpy(k).float().unsqueeze(0)))

		state = self.cc_network(state_v)

		# state = Variable(torch.from_numpy(state).float().unsqueeze(0))
		value = F.relu(self.critic_linear1(state))
		value = self.critic_linear2(value)
		
		policy_dist = F.relu(self.actor_linear1(state))
		policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

		return value, policy_dist



episode_durations = []
scores = []

def plot_():
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
	if len(durations_t) >= 5:
		means = scores_t.unfold(0, 5, 1).mean(1).view(-1)
		means = torch.cat((torch.zeros(4), means))
		plt.plot(means.numpy())

	plt.pause(0.001)  # pause a bit so that plots are updated
	if is_ipython:
		display.clear_output(wait=True)
		display.display(plt.gcf())





#Make environment
args = Config().parse()
env = StageEnvironment(args, False)

while(env.observation_space.shape[0] == 0):
	rospy.sleep(1)


def a2c():
	flag_first = True
	changed_pose = []
	num_inputs = 43
	num_outputs = env.action_space
	
	actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size)
	ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

	all_lengths = []
	average_lengths = []
	all_rewards = []
	entropy_term = 0
	# scores = []

	for episode in range(max_episodes):
		if flag_first:
			state,n_pose = env.reset()
			changed_pose = list(n_pose)
			flag_first = False
			print("Reset")
			print(changed_pose)
		else:
			# state, _, _ = env.step(new_action) 
			# changed_pose = list(bk_state[0])
			print("Reset pose")
			print(changed_pose)
			state,_ = env.reset_pose(changed_pose)
			rospy.sleep(5)

		log_probs = []
		values = []
		rewards = []
		score = 0
		print("Episode: %d" % episode)

		# print(type(state))
		# print(state)

		# state = env.reset()
		for steps in range(num_steps):
			value, policy_dist = actor_critic.forward(state)
			value = value.detach().numpy()[0,0]
			dist = policy_dist.detach().numpy() 

			action = np.random.choice(num_outputs, p=np.squeeze(dist))
			log_prob = torch.log(policy_dist.squeeze(0)[action])
			entropy = -np.sum(np.mean(dist) * np.log(dist))
			#### STEP
			print("Action :=",action," =--= Step :=", steps)
			new_state, reward, done = env.step(action)
			# print("Done :=", done)

			score += reward

			rewards.append(reward)
			values.append(value)
			log_probs.append(log_prob)
			entropy_term += entropy
			state = new_state
			
			if done or steps == num_steps-1:
				Qval, _ = actor_critic.forward(new_state)
				Qval = Qval.detach().numpy()[0,0]
				all_rewards.append(np.sum(rewards))
				all_lengths.append(steps)
				average_lengths.append(np.mean(all_lengths[-10:]))
				if episode % 10 == 0:					
					sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode, np.sum(rewards), steps, average_lengths[-1]))
				break
		
		scores.append(score)
		print("Score = %d" % score)
		plot_()

		# compute Q values
		Qvals = np.zeros_like(values)
		for t in reversed(range(len(rewards))):
			Qval = rewards[t] + GAMMA * Qval
			Qvals[t] = Qval
  
		#update actor critic
		values = torch.FloatTensor(values)
		Qvals = torch.FloatTensor(Qvals)
		log_probs = torch.stack(log_probs)
		
		advantage = Qvals - values
		actor_loss = (-log_probs * advantage).mean()
		critic_loss = 0.5 * advantage.pow(2).mean()
		ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

		ac_optimizer.zero_grad()
		ac_loss.backward()
		ac_optimizer.step()



if __name__ == "__main__":
	a2c()  
	np.savetxt(join(dirname(abspath(__file__)),'scores_A2C.txt'), scores, delimiter=',')
	fig2 = plt.figure()
	plt.plot(scores)
	plt.ylabel('score')
	plt.xlabel('episodes')
	plt.title('Training score')
	plt.show()