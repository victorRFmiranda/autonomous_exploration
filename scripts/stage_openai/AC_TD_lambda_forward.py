#!/usr/bin/env python

import rospy
from config import Config
from ros_stage_env import StageEnvironment

import torch, gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gym
import numpy as np
from collections import deque




def conv_block(input_size, output_size):
	block = nn.Sequential(
		nn.Conv2d(input_size, output_size, (3, 3)), nn.ReLU(), nn.BatchNorm2d(output_size), nn.MaxPool2d((2, 2)),
	)

	return block

 


class ConcatNetwork(nn.Module):
	def __init__(self):
		super(ConcatNetwork, self).__init__()
		# test convolution
		self.conv1 = conv_block(3, 16)
		self.conv2 = conv_block(16, 32)
		# self.conv3 = conv_block(32, 64)
		#self.ln1 = nn.Linear(64 * 58 * 58, 16)
		self.ln1 = nn.Linear(32 * 14 * 14, 16)
		self.relu = nn.ReLU()
		self.batchnorm = nn.BatchNorm1d(16)
		self.dropout = nn.Dropout2d(0.5)
		self.ln2 = nn.Linear(16, 1)

		self.p1 = nn.Linear(3, 2)
		self.p2 = nn.Linear(2, 2)
		self.p3 = nn.Linear(2, 1)

		self.f1 = nn.Linear(2, 4)
		self.f2 = nn.Linear(4, 4)
		self.f3 = nn.Linear(16, 1)

		self.last = nn.Linear(3,3)

	def forward(self, x):
		# mlp pose
		x[0] = self.p1(x[0])
		x[0] = self.relu(x[0])
		x[0] = self.p2(x[0])
		x[0] = self.relu(x[0])
		x[0] = self.p3(x[0])
		x[0] = self.relu(x[0])
		# mlp frontiers
		x[1] = self.f1(x[1])
		x[1] = self.relu(x[1])
		x[1] = self.f2(x[1])
		x[1] = self.relu(x[1])
		x[1] = x[1].reshape(x[1].shape[0], -1)
		x[1] = self.f3(x[1])
		x[1] = self.relu(x[1])
		# conv image
		x[2] = self.conv1(x[2])
		x[2] = self.conv2(x[2])
		# x[2] = self.conv3(x[2])
		x[2] = x[2].reshape(x[2].shape[0], -1)
		x[2] = self.ln1(x[2])
		x[2] = self.relu(x[2])
		x[2] = self.dropout(x[2])
		x[2] = self.ln2(x[2])
		x[2] = self.relu(x[2])


		x = torch.cat((x[0], x[1], x[2]), dim=1)


		x = self.last(x)

		return x







#Using a neural network to learn our policy parameters
class PolicyNetwork(nn.Module):
	
	#Takes in observations and outputs actions
	def __init__(self, observation_space, action_space):
		# observation_space -> quantidade de estados
		# action_space -> quantidade de acoes
		
		super(PolicyNetwork, self).__init__()
		# Resumo, entra observation_space entradas e saem action_space saidas, com 128 neuronios na camada escondida.
		# Camada de entrada da rede de acordo com o observation_space
		# 128 neuronios na camada escondida
		self.input_layer = nn.Linear(observation_space, 128)
		# Liga a camada escondida com a saida de tamanho definido pelo action_space
		self.output_layer = nn.Linear(128, action_space)

		
	
	#forward pass
	def forward(self, x):

		#input states
		x = self.input_layer(x)
		
		#relu activation
		x = F.relu(x)
		
		#actions
		actions = self.output_layer(x)
		
		#get softmax for a probability distribution
		action_probs = F.softmax(actions, dim=1)
		
		return action_probs


#Using a neural network to learn state value
class StateValueNetwork(nn.Module):
	
	#Takes in state
	def __init__(self, observation_space):
		super(StateValueNetwork, self).__init__()
		# observation_space -> quantidade de estados
		# 128 neuronios na camada escondida
		
		self.input_layer = nn.Linear(observation_space, 128)
		self.output_layer = nn.Linear(128, 1)
		
	def forward(self, x):
		#input layer
		x = self.input_layer(x)
		
		#activiation relu
		x = F.relu(x)
		
		#get state value
		state_value = self.output_layer(x)
		
		return state_value


def select_action(network, state, ccnetwork):
	''' Selects an action given current state
	Args:
	- network (Torch NN): network to process state - PolicyNetwork
	- state (Array): Array of action space in an environment - State
	
	Return:
	- (int): action that is selected
	- (float): log probability of selecting that action given state and network
	'''
	
	#convert state to float tensor, add 1 dimension, allocate tensor on device
	state_v = []
	for k in state:
		state_v.append(torch.from_numpy(k).float().unsqueeze(0).to(DEVICE))

	state = ccnetwork(state_v)

	
	#use network to predict action probabilities
	action_probs = network(state)
	state = state.detach()
	
	#sample an action using the probability distribution
	m = Categorical(action_probs)
	action = m.sample()
	
	#return action
	return action.item(), m.log_prob(action)



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



####################################################
######Preprocessing Data For Policy#################


class MLP(nn.Module):
  '''
	Multilayer Perceptron.
  '''
  def __init__(self):
	super().__init__()
	self.layers = nn.Sequential(
	  nn.Flatten(),
	  nn.Linear(32 * 32 * 3, 64),
	  nn.ReLU(),
	  nn.Linear(64, 32),
	  nn.ReLU(),
	  nn.Linear(32, 10)
	)


  def forward(self, x):
	'''Forward pass'''
	return self.layers(x)

def create_mlp(dim, regress=False):
	# define our MLP network
	model = Sequential()
	model.add(Dense(4, input_dim=dim, activation="relu"))
	model.add(Dense(2, activation="relu"))
	# check to see if the regression node should be added
	if regress:
		model.add(Dense(1, activation="linear"))
	# return our model
	return model



def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)
	chanDim = -1
	# define the model input
	inputs = Input(shape=inputShape)
	# loop over the number of filters
	for (i, f) in enumerate(filters):
		# if this is the first CONV layer then set the input
		# appropriately
		if i == 0:
			x = inputs
		# CONV => RELU => BN => POOL
		x = Conv2D(f, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		
	# flatten the volume, then FC => RELU => BN => DROPOUT
	x = Flatten()(x)
	x = Dense(16)(x)
	x = Activation("relu")(x)
	x = BatchNormalization(axis=chanDim)(x)
	x = Dropout(0.5)(x)
	# apply another FC layer, this one to match the number of nodes
	# coming out of the MLP
	x = Dense(4)(x)
	x = Activation("relu")(x)
	# check to see if the regression node should be added
	if regress:
		x = Dense(1, activation="linear")(x)
	# construct the CNN
	model = Model(inputs, x)
	# return the CNN
	return model

####################################################
####################################################





########### MAIN ##############
args = Config().parse()

#discount factor for future utilities
DISCOUNT_FACTOR = args.DISCOUNT_FACTOR
#number of episodes to run
NUM_EPISODES = args.NUM_EPISODES
#Max number episodes to train
MAX_EPISODES = args.TOTAL_EPISODES
#max steps per episode
MAX_STEPS = args.MAX_STEPS
#device to run model on 
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

if torch.cuda.is_available():
	gc.collect()
	torch.cuda.empty_cache()
#score agent needs for environment to be solved
SOLVED_SCORE = args.SOLVED_SCORE
#decay constant for critic
CRITIC_LAMBDA = args.CRITIC_LAMBDA


#Make environment
env = StageEnvironment(args)

while(env.observation_space.shape[0] == 0):
	rospy.sleep(1)

print("Number of states = %d" % env.observation_space.shape[0])
print("Number of possible actions = %d" % env.action_space)

#Init network
policy_network = PolicyNetwork(env.observation_space.shape[0], env.action_space).to(DEVICE)
stateval_network = StateValueNetwork(env.observation_space.shape[0]).to(DEVICE)
concat_network = ConcatNetwork().to(DEVICE)

#Init optimizer
policy_optimizer = optim.Adam(policy_network.parameters(), lr=1e-2)
stateval_optimizer = optim.Adam(stateval_network.parameters(), lr=1e-2)



#track scores
scores = []

#recent 100 scores
recent_scores = deque(maxlen=100)
flag_change = False
flag_first = True
changed_pose = []


rate = rospy.Rate(10)

#iterate through episodes
episode = 0
episode_main = 0
while not rospy.is_shutdown() or (episode_main <= TOTAL_EPISODES):
	if flag_first == False:
		action, lp = select_action(policy_network, state, concat_network)
		new_state, reward, done = env.step(action,flag_change) 

		changed_pose = new_state[0]
		flag_change = False

	while not rospy.is_shutdown() or (episode <= NUM_EPISODES):
	# for episode in range(NUM_EPISODES):
		print("Episode: %d" % episode)
		
		#reset environment, initiable variables
		if(flag_first):
			state = env.reset()		# reseta o ambiente e retorna o estado atual
		else:
			state = env.reset_pose(changed_pose)

		trajectory = []
		score = 0


		# clear cuda memory
		if torch.cuda.is_available():
			gc.collect()
			torch.cuda.empty_cache()
		

		#generate episode
		for step in range(MAX_STEPS):

			print("Step: %d" % step)
			#env.render()
			

			#select action
			action, lp = select_action(policy_network, state, concat_network)
		
			
			#execute action
			# think in my case: if the actions is correct reward = 1 and done = false, else reward = 1 and done = true
			new_state, reward, done = env.step(action,flag_change)  

			
			#track episode score
			score += reward

			
			#store into trajectory
			trajectory.append([state, action, reward, lp])
			
			#end episode - in failure case
			# if done or score < -5	:
			if done and step > 0:
				print("Done\n")
				break
			
			#move into new state
			state = new_state
		
		#append score
		scores.append(score)
		recent_scores.append(score)

		print("Score = %d" % np.array(recent_scores).mean())
		# print("Action = ", action)

		#early stopping if we meet solved score goal
		if np.array(recent_scores).mean() >= SOLVED_SCORE:
			if(step > 100):
				flag_change = True
			break

		if (np.array(recent_scores).mean() < -400):
			#Init network
			policy_network = PolicyNetwork(env.observation_space.shape[0], env.action_space).to(DEVICE)
			stateval_network = StateValueNetwork(env.observation_space.shape[0]).to(DEVICE)
			concat_network = ConcatNetwork().to(DEVICE)

			#Init optimizer
			policy_optimizer = optim.Adam(policy_network.parameters(), lr=1e-2)
			stateval_optimizer = optim.Adam(stateval_network.parameters(), lr=1e-2)

			#track scores
			scores = []

			#recent 100 scores
			recent_scores = deque(maxlen=100)
			#iterate through episodes
			episode = 0
			episode_main = 0

		
		#get items from trajectory
		states = [step[0] for step in trajectory]
		actions = [step[1] for step in trajectory]
		rewards = [step[2] for step in trajectory]
		lps = [step[3] for step in trajectory]

		#calculate state values
		state_vals = []
		for state in states:
			state_v = []
			for k in state:
				state_v.append(torch.from_numpy(k).float().unsqueeze(0).to(DEVICE))

			state = concat_network(state_v)
			# state = state_v
			# state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
			# state.required_grad = True
			state_vals.append(stateval_network(state))
		
		#get lambda returns for each timestep
		#we use this lambda returns for critic and actor so CRITIC_LAMBDA is used for both
		G = process_rewards(rewards, state_vals, CRITIC_LAMBDA)

		state_vals = torch.stack(state_vals).squeeze()

		
		train_value(G, state_vals, stateval_optimizer)
			
		#calculate TD lambda for actor
		deltas = [gt - val for gt, val in zip(G, state_vals)]
		deltas = torch.tensor(deltas).to(DEVICE)
		
		train_policy(deltas, lps, policy_optimizer)

		episode += 1

		rate.sleep()

	flag_first = False
	episode_main += 1
	

torch.save(policy_network, '/home/victor/ROS/catkin_ws/src/autonomous_exploration/scripts/stage_openai/model/actor.pkl')
torch.save(stateval_network, '/home/victor/ROS/catkin_ws/src/autonomous_exploration/scripts/stage_openai/model/critic.pkl')




import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import numpy as np

sns.set()

plt.plot(scores)
plt.ylabel('score')
plt.xlabel('episodes')
plt.title('Training score with Forward-view TD')

reg = LinearRegression().fit(np.arange(len(scores)).reshape(-1, 1), np.array(scores).reshape(-1, 1))
y_pred = reg.predict(np.arange(len(scores)).reshape(-1, 1))
plt.plot(y_pred)
plt.show()

