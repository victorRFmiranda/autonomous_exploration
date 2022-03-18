# -*- coding: utf-8 -*-
import torch, gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
from collections import deque
# import matplotlib
# from sklearn.linear_model import LinearRegression
from queue import Queue
import cv2


from environment.environment import Environment
from environment.environment_node_data import Mode
import action_mapper


# set up matplotlib
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
# 	from IPython import display

 
def conv2d_size_out(size, kernel_size = 5, stride = 2):
			return (size - (kernel_size - 1) - 1) // stride  + 1

#Using a neural network to learn our policy parameters
class PolicyNetwork(nn.Module):
	
	#Takes in observations and outputs actions
	def __init__(self, observation_space, action_space, map_size):
		# observation_space -> quantidade de estados
		# action_space -> quantidade de acoes
		
		super(PolicyNetwork, self).__init__()

		self.conv_pos = nn.Sequential(
						nn.Conv1d(in_channels=3,out_channels=2,kernel_size=1,stride=2),
						nn.BatchNorm1d(2),
						nn.ReLU()
						)
		pos_input = 6

		self.conv_frontier = nn.Sequential(
						nn.Conv2d(4,8,kernel_size=2,stride=2),
						nn.BatchNorm2d(8),
						nn.ReLU()
						)
		cw = conv2d_size_out(4,kernel_size=2)
		ch = conv2d_size_out(2,kernel_size=2)
		frontier_input = cw*ch*8


		self.conv2d = nn.Sequential(
				nn.Conv2d(map_size[0],16,kernel_size=1,stride=2),
				nn.BatchNorm2d(16),
				nn.ReLU(),
				nn.Conv2d(16,32,kernel_size=3,stride=2),
				nn.BatchNorm2d(32),
				nn.ReLU(),
				nn.Conv2d(32,32,kernel_size=1,stride=2),
				nn.BatchNorm2d(32),
				nn.ReLU(),
				)
		convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(map_size[0],kernel_size=1),kernel_size=3),kernel_size=1)  
		convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(map_size[1],kernel_size=1),kernel_size=3),kernel_size=1)
		map_input = 800 #???


		cv1_inputsize = map_input + frontier_input + pos_input
							# MAPA + Frontiers + robot_pose


		# self.conv = nn.Sequential(
		# 		nn.Conv1d(in_channels=cv1_inputsize,out_channels=16,kernel_size=1,stride=2),
		# 		nn.BatchNorm1d(16),
		# 		nn.ReLU(),
		# 		nn.Conv1d(in_channels=16,out_channels=32,kernel_size=1,stride=2),
		# 		nn.BatchNorm1d(32),
		# 		nn.ReLU(),
		# 		nn.Conv1d(in_channels=32,out_channels=32,kernel_size=1,stride=2),
		# 		nn.BatchNorm1d(32),
		# 		nn.ReLU(),
		# 		)
		# linear_input_size = 32
		self.dense = nn.Sequential(
					nn.Linear(cv1_inputsize,512),
					nn.ReLU(),
					nn.Linear(512,512),
					nn.ReLU(),
					nn.Linear(512,action_space)
					)

		
	
	#forward pass
	def forward(self, observ):
	
		x = list([])

		x0 = self.conv_pos(observ[0])
		x0 = x0.view(x0.size(0), -1)
		
		x1 = self.conv_frontier(observ[1])
		x1 = x1.view(x1.size(0), -1)		

		x2 = self.conv2d(observ[2])
		x2 = x2.view(x2.size(0), -1)


		x = torch.cat((x0, x1, x2), dim=1)
		

		# x = self.conv(x)
		actions = self.dense(x)
		
		#get softmax for a probability distribution
		action_probs = F.softmax(actions, dim=1)
		
		return action_probs


#Using a neural network to learn state value
class StateValueNetwork(nn.Module):
	
	#Takes in state
	def __init__(self, observation_space):
		super(StateValueNetwork, self).__init__()
		# # observation_space -> quantidade de estados

		self.conv_pos = nn.Sequential(
						nn.Conv1d(in_channels=3,out_channels=2,kernel_size=1,stride=2),
						nn.BatchNorm1d(2),
						nn.ReLU()
						)
		pos_input = 6

		self.conv_frontier = nn.Sequential(
						nn.Conv2d(4,8,kernel_size=2,stride=2),
						nn.BatchNorm2d(8),
						nn.ReLU()
						)
		cw = conv2d_size_out(4,kernel_size=2)
		ch = conv2d_size_out(2,kernel_size=2)
		frontier_input = cw*ch*8


		self.conv2d = nn.Sequential(
				nn.Conv2d(map_size[0],16,kernel_size=1,stride=2),
				nn.BatchNorm2d(16),
				nn.ReLU(),
				nn.Conv2d(16,32,kernel_size=3,stride=2),
				nn.BatchNorm2d(32),
				nn.ReLU(),
				nn.Conv2d(32,32,kernel_size=1,stride=2),
				nn.BatchNorm2d(32),
				nn.ReLU(),
				)
		convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(map_size[0],kernel_size=1),kernel_size=3),kernel_size=1)  
		convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(map_size[1],kernel_size=1),kernel_size=3),kernel_size=1)
		map_input = 800 #???


		cv1_inputsize = map_input + frontier_input + pos_input
							# MAPA + Frontiers + robot_pose


		# self.conv = nn.Sequential(
		# 		nn.Conv1d(in_channels=cv1_inputsize,out_channels=16,kernel_size=1,stride=2),
		# 		nn.BatchNorm1d(16),
		# 		nn.ReLU(),
		# 		nn.Conv1d(in_channels=16,out_channels=32,kernel_size=1,stride=2),
		# 		nn.BatchNorm1d(32),
		# 		nn.ReLU(),
		# 		nn.Conv1d(in_channels=32,out_channels=32,kernel_size=1,stride=2),
		# 		nn.BatchNorm1d(32),
		# 		nn.ReLU(),
		# 		)
		# linear_input_size = 32
		self.dense = nn.Sequential(
					nn.Linear(cv1_inputsize,512),
					nn.ReLU(),
					nn.Linear(512,512),
					nn.ReLU(),
					nn.Linear(512,1)
					)
		
	def forward(self, observ):

		x = list([])

		x0 = self.conv_pos(observ[0])
		x0 = x0.view(x0.size(0), -1)
		
		x1 = self.conv_frontier(observ[1])
		x1 = x1.view(x1.size(0), -1)		

		x2 = self.conv2d(observ[2])
		x2 = x2.view(x2.size(0), -1)
		

		x = torch.cat((x0, x1, x2), dim=1)
		

		# x = self.conv(x)
		state_value = self.dense(x)
		
		return state_value


def select_action(network, state):
	''' Selects an action given current state
	Args:
	- network (Torch NN): network to process state - PolicyNetwork
	- state (Array): Array of action space in an environment - State
	
	Return:
	- (int): action that is selected
	- (float): log probability of selecting that action given state and network
	'''

	
	# s0 = np.stack(state[0], axis=1)
	# s1 = np.stack(state[1],axis=0)
	# s1 = np.transpose(s1, [1, -1, 0])
	# s2 = np.stack(state[2],axis=0)
	# s2 = np.transpose(s2, [1, -1, 0])
	# state = np.asarray([s0,s1,s2],dtype=object)

	
	#convert state to float tensor, add 1 dimension, allocate tensor on device	
	state_v = []
	# for k in state:
	# 	state_v.append(torch.from_numpy(k).float().unsqueeze(0).to(DEVICE))


	state_v = concat_states(state)


	
	#use network to predict action probabilities
	network.eval()
	action_probs = network(state_v)
	network.train()
	# state = state.detach()
	
	#sample an action using the probability distribution
	m = Categorical(action_probs)
	action = m.sample()

	lp = m.log_prob(action)
	
	#return action
	return action.item(), lp



def process_rewards(rewards, state_vals, decay):
	''' Processes rewards and statevals into lambda returns
	Args:
	- rewards (Array): Array of rewards with index as timestep
	- state_vals (Array): Array of state values with index as timestep
	- decay (Float): lambda constant, decay rate of weights
	
	Return:
	G (Array): array of lambda returns with index as timestep
	'''

	#length of episode
	episode_length = len(rewards)
	
	#get weights for all N-steps for each timestep - (1-lambda(n))lambda(n)^(n-1)
	episode_weights = get_weights(episode_length, decay)
	episode_weights = [np.array(x) for x in episode_weights]

	#get returns for all N-steps for each timestep - G(n) = R*lambda^(n), esse lambda e diferente do de cima
	episode_returns = get_Nstep_returns(rewards, state_vals)
	episode_returns = [np.array(x) for x in episode_returns]

	#multiple returns by weights and sum up all weighted returns for each timestep
	#G is lambda returns with index as timestep. The sum of all N-step weights at each timestep should be 1.
	G = [sum(weights * nsteps) for weights, nsteps in zip(episode_weights, episode_returns)]

	#whitening rewards to prevent gradient explosion and decrease variance
	G = torch.tensor(G).float().to(DEVICE)
	G = (G - G.mean())/G.std()

	return G


def get_Nstep_returns(rewards, state_vals):
	''' Get N-step returns for each timestep
	Args:
	- rewards (Array): Array of rewards with index as timestep
	- state_vals (Array): Array of state values with index as timestep
	
	Return:
	episode_returns (Array): array of N-step returns with index as timestep
	'''
	
	#episode length
	episode_length = len(rewards)

	#store episode returns
	episode_returns = []

	#iterate through each timestep
	for t in range(episode_length):
		
		#store nstep returns for a timestep
		nstep_returns = []

		#iterate from timestep to end of episode
		for i in range(t, episode_length):
			
			#discounted cumulative reward for N-steps
			nstep_return = 0
			
			#iterate from timestep to i, calculate (i - timestep)-step return
			for j in range(t, i+1):

				#if on Nth step and its not the terminal state, use bootstrapped return
				if j == i and j != episode_length - 1:
					nstep_return += state_vals[j].item() * DISCOUNT_FACTOR ** (i-t)
					
				#else use discounted reward
				else:
					nstep_return += rewards[j] * DISCOUNT_FACTOR ** (i-t)
			
			#append nstep return
			nstep_returns.append(nstep_return)
			
		#append nstep returns
		episode_returns.append(nstep_returns)
	
	return episode_returns



def get_weights(episode_length, decay):
	'''Get weights for all N-steps in each timestep
	Args:
	- episode_length (int): length of episode
	- decay (float): lambda constant, decay rate of weights
	
	Returns:
	- episode_weights (Array): weights for all N-steps at each timestep with index as timestep
	'''

	#store weights for each timestep
	episode_weights = []

	#iterate through each timestep in episode
	for t in range(episode_length):
	#weights for different Ns for current timestep
		weights = []

		#iterate from current timestep until end
		for i in range(t, episode_length):

			#if at terminal state
			if i == episode_length - 1:
				#append weight. Note that we are doing ep_len - t instead of ep_len - t - 1 since the loop is 0-indexed
				weights.append(decay**(episode_length - t - 1))
			else:
				#append weight
				weights.append((1-decay) * decay**(i-t))

		#append weights
		episode_weights.append(weights)
	
	return episode_weights


def train_policy(deltas, log_probs, optimizer):
	''' Update policy parameters
	Args:
	- deltas (Array): difference between predicted stateval and actual stateval (Gt - process_rewards)
	- log_probs (Array): trajectory of log probabilities of action taken (ln pi(At|St))
	- optimizer (Pytorch optimizer): optimizer to update policy network parameters - Rede criada para a politica
	'''
	
	#store updates
	policy_loss = []


	#calculate loss to be backpropagated
	for d, lp in zip(deltas, log_probs):
		#add negative sign since we are performing gradient ascent
		policy_loss.append(-d * lp)
	
	#Backpropagation
	optimizer.zero_grad()
	sum(policy_loss).backward()
	optimizer.step()



def train_value(G, state_vals, optimizer):
	''' Update state-value network parameters
	Args:
	- G (Array): trajectory of cumulative discounted rewards 
	- V state_vals (Array): trajectory of predicted state-value at each step
	- optimizer (Pytorch optimizer): optimizer to update state-value network parameters - Rede criada para o state-value
	'''
	
	#calculate MSE loss
	val_loss = F.mse_loss(state_vals, G)
		
	#Backpropagate
	optimizer.zero_grad()
	val_loss.backward()
	optimizer.step()





episode_durations = []
scores = []


def update_frame(q, frame):
	if q.full():
		q.get()
	q.put(frame)


def concat_states(state):
	s0 = np.stack(state[0], axis=1)
	s1 = np.stack(state[1],axis=0)
	s1 = np.transpose(s1, [1, -1, 0])
	s2 = np.stack(state[2],axis=0)
	s2 = np.transpose(s2, [1, -1, 0])
	state = np.asarray([s0,s1,s2],dtype=object)

	state_v = []
	for k in state:
		state_v.append(torch.from_numpy(k).float().unsqueeze(0).to(DEVICE))

	return state_v




###################################################################
if __name__ == "__main__":
	env = Environment("./environment/world/ufmg_2")
	env.set_mode(Mode.ALL_RANDOM, False)
	env.use_ditance_angle_to_end(True)
	env.set_observation_rotation_size(128)
	env.use_observation_rotation_size(True)
	env.set_cluster_size(1)

	observation, _, flag_colide, _ = env.reset_2()

	s_map,_ = env._get_map()
	map_size = s_map.shape


	state_size = env.observation_size()
	action_size = 4

	DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

	actor = PolicyNetwork(state_size, action_size, map_size).to(DEVICE)
	critic = StateValueNetwork(state_size).to(DEVICE)

	actor_optimizer = optim.Adam(actor.parameters(), lr=3e-6)
	critic_optimizer = optim.Adam(critic.parameters(), lr=3e-6)

	recent_scores = deque(maxlen=10000)

	MAX_EPISODES = 100000
	MAX_STEPS = 500
	vector_len = 5
	CRITIC_LAMBDA = 0.9
	DISCOUNT_FACTOR = 0.9

	queue_state = Queue(maxsize = vector_len)
	flag_colide = False 

	ep = 0
	# for ep in range(MAX_EPISODES):
	while (ep < MAX_EPISODES):

		queue_state.queue.clear()
		observation, _, flag_colide, _ = env.reset_2()


		while(not queue_state.full()):
			update_frame(queue_state, observation)
			observation, _, flag_colide, _ = env.step_2(-1)

		if(flag_colide):
			flag_colide = False
			continue

		state = np.array(queue_state.queue)
		state = np.transpose(state, [1, 0])  # move channels

		# 

 
 
		trajectory = []
		score = 0

		print("Episode: %d" % ep)
		
		for step in range(MAX_STEPS):

			env.visualize()
			print("Robot pose := ", [env._env.get_robot_pose_x(),env._env.get_robot_pose_y()])

			mapa, _ = env._get_map()  ## NEW STATE
			# print(mapa)
			# cv2.imshow('Mapa',mapa)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
			frontier, _ = env._get_frontier()
			print("frontier := ", frontier)
			input("Wait")

			# state = np.random.rand(1211,5)
			action, lp = select_action(actor, state)
			print("Action :=", action)
			# linear, angular = action_mapper.map_action(action)

			#execute action
			next_observation, reward, done, _ = env.step_2(action)
			update_frame(queue_state, next_observation)
			next_state = np.array(queue_state.queue)
			next_state = np.transpose(next_state, [1, 0])

			#track episode score
			score += reward

			#store into trajectory
			trajectory.append([state, action, reward, lp])

			state = next_state 

			

			if done and step > 1:
				print("Episode %d - Score = %f" % (ep,score))
				break


		# cv2.imshow('Mapa',mapa)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		scores.append(score)
		recent_scores.append(score)
		episode_durations.append(step+1)

		
		#get items from trajectory
		states = [step[0] for step in trajectory]
		actions = [step[1] for step in trajectory]
		rewards = [step[2] for step in trajectory]
		lps = [step[3] for step in trajectory]

		#calculate state values
		state_vals = []
		for state in states:
			state = concat_states(state)
			# state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
			critic.eval()
			state_vals.append(critic(state))
			critic.train()
		
		#get lambda returns for each timestep
		#we use this lambda returns for critic and actor so CRITIC_LAMBDA is used for both
		G = process_rewards(rewards, state_vals, CRITIC_LAMBDA)

		state_vals = torch.stack(state_vals).squeeze()

		
		train_value(G, state_vals, critic_optimizer)
			
		#calculate TD lambda for actor
		deltas = [gt - val for gt, val in zip(G, state_vals)]
		deltas = torch.tensor(deltas).to(DEVICE)
		
		train_policy(deltas, lps, actor_optimizer)

		ep += 1

		
		# input("AAAAAAAAAAAAAa")




print("Finish")
torch.save(actor.state_dict(), "./Models/actor.pt")
torch.save(critic.state_dict(), "./Models/critic.pt")


