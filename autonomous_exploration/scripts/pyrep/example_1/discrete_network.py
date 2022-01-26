#!/usr/bin/env python


import torch, gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
DISCOUNT_FACTOR = 0.9



#Using a neural network to learn our policy parameters
class PolicyNetwork(nn.Module):
	
	#Takes in observations and outputs actions
	def __init__(self, observation_space, action_space):
		# observation_space -> quantidade de estados
		# action_space -> quantidade de acoes
		
		super(PolicyNetwork, self).__init__()
		# Resumo, entra observation_space entradas e saem action_space saidas, com 512 neuronios na camada escondida.
		# Camada de entrada da rede de acordo com o observation_space
		# 512 neuronios na camada escondida
		self.input_layer = nn.Linear(observation_space, 512)
		# hidden Layer
		self.h_layer = nn.Linear(512,512)
		# Liga a camada escondida com a saida de tamanho definido pelo action_space
		self.output_layer = nn.Linear(512, action_space)

		
	
	#forward pass
	def forward(self, x):

		#input states
		x = self.input_layer(x)
		#relu activation
		x = F.relu(x)

		# hidden layer
		x2 = F.relu(self.h_layer(x))

		
		#actions
		actions = self.output_layer(x2)
		
		#get softmax for a probability distribution
		action_probs = F.softmax(actions, dim=1)
		
		return action_probs


#Using a neural network to learn state value
class StateValueNetwork(nn.Module):
	
	#Takes in state
	def __init__(self, observation_space):
		super(StateValueNetwork, self).__init__()
		# observation_space -> quantidade de estados
		# 512 neuronios na camada escondida
		
		self.input_layer = nn.Linear(observation_space, 512)
		# hidden Layer
		self.h_layer = nn.Linear(512,512)
		#
		self.output_layer = nn.Linear(512, 1)
		
	def forward(self, x):
		#input layer
		x = self.input_layer(x)
		
		#activiation relu
		x = F.relu(x)

		# hidden layer
		x2 = F.relu(self.h_layer(x))
		
		#get state value
		state_value = self.output_layer(x2)
		
		return state_value


# def select_action(network, state, ccnetwork):
def select_action(network, state):
	''' Selects an action given current state
	Args:
	- network (Torch NN): network to process state - PolicyNetwork
	- state (Array): Array of action space in an environment - State
	
	Return:
	- (int): action that is selected
	- (float): log probability of selecting that action given state and network
	'''
	
	#convert state to float tensor, add 1 dimension, allocate tensor on device
	# state_v = []
	# for k in state:
		# state_v.append(torch.from_numpy(k).float().unsqueeze(0).to(DEVICE))

	# state = ccnetwork(state_v)
	state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

	
	#use network to predict action probabilities
	action_probs = network(state)
	state = state.detach()
	
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

	# print(deltas)
	# print(log_probs)
	
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