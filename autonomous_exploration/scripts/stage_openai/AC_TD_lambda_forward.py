#!/usr/bin/env python
import rospy
from config import Config
from ros_stage_env import StageEnvironment
import rospkg

import torch, gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

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
 

def conv2d_size_out(size, kernel_size = 5, stride = 2):
			return (size - (kernel_size - 1) - 1) // stride  + 1

class ConcatNetwork(nn.Module):
	def __init__(self):
		super(ConcatNetwork, self).__init__()

		self.conv = nn.Sequential(
				nn.Conv2d(1,16,kernel_size=5,stride=2),
				nn.BatchNorm2d(16),
				nn.ReLU(),
				nn.Conv2d(16,32,kernel_size=5,stride=2),
				nn.BatchNorm2d(32),
				nn.ReLU(),
				nn.Conv2d(32,32,kernel_size=5,stride=2),
				nn.BatchNorm2d(32),
				nn.ReLU(),
				)
		convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(96)))  # image 96x96
		convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(96)))
		linear_input_size = convw * convh * 32
		self.dense = nn.Linear(linear_input_size,10)


	def forward(self, x):

		x[1] = x[1].view(x[1].size(0), -1)

		# conv image
		x[2] = self.conv(x[2])
		x[2] = self.dense(x[2].view(x[2].size(0), -1))


		x = torch.cat((x[0], x[1], x[2]), dim=1)

		# CHANGE HERE
		# x = torch.cat((x[0], x[1]), dim=1)

		return x






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
		x3 = F.relu(self.h_layer(x2))

		
		#actions
		actions = self.output_layer(x3)
		
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
		x3 = F.relu(self.h_layer(x2))
		
		#get state value
		state_value = self.output_layer(x3)
		
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

	state = np.asarray(state)
	
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
	if len(durations_t) >= 30:
		means = scores_t.unfold(0, 30, 1).mean(1).view(-1)
		means = torch.cat((torch.zeros(29), means))
		plt.plot(means.numpy())

	plt.pause(0.001)  # pause a bit so that plots are updated
	if is_ipython:
		display.clear_output(wait=True)
		display.display(plt.gcf())




########### MAIN ##############
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('autonomous_exploration')
file_path = pkg_path + "/scripts/stage_openai/model/"
LOAD_NETWORK = False

args = Config().parse()

#discount factor for future utilities
DISCOUNT_FACTOR = args.DISCOUNT_FACTOR
#number of episodes to run
NUM_EPISODES = args.NUM_EPISODES
#Max number episodes to train
MAX_EPISODES = args.TOTAL_EPISODES
#max steps per episode
MAX_STEPS = args.MAX_STEPS
# number of states (ANN inputs)
NUM_STATES = args.num_states
#device to run model on 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
# 
if torch.cuda.is_available():
	gc.collect()
	torch.cuda.empty_cache()
#score agent needs for environment to be solved
SOLVED_SCORE = args.SOLVED_SCORE
#decay constant for critic
CRITIC_LAMBDA = args.CRITIC_LAMBDA


#Make environment
env = StageEnvironment(args, False)

while(env.observation_space.shape[0] == 0):
	rospy.sleep(1)

print("Number of states = %d" % NUM_STATES)
print("Number of possible actions = %d" % env.action_space)

#Init network
if(LOAD_NETWORK == True):
	print("Loading Network")
	policy_network = torch.load(file_path+"actor.pkl")
	stateval_network = torch.load(file_path+"critic.pkl")
else:
	print("Creating Network")
	policy_network = PolicyNetwork(NUM_STATES, env.action_space).to(DEVICE)
	stateval_network = StateValueNetwork(NUM_STATES).to(DEVICE)

concat_network = ConcatNetwork().to(DEVICE)

#Init optimizer
policy_optimizer = optim.Adam(policy_network.parameters(), lr=1e-6)
stateval_optimizer = optim.Adam(stateval_network.parameters(), lr=1e-6)



#track scores
# scores = []

#recent 100 scores
recent_scores = deque(maxlen=10000)
flag_change = False
flag_first = True
changed_pose = []


rate = rospy.Rate(10)

#iterate through episodes
episode = 0
episode_main = 0


# state = env.reset()
while (episode <= MAX_EPISODES) and not rospy.is_shutdown():

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


	print("Episode: %d" % episode)


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
		
		if (type(state[1]) is list):
			while(len(state[1])==0):
				rospy.sleep(2)
				print("Wainting Frontiers")
			# print("State :=", state)
			state[1] = np.asarray(state[1])
			# print("State after :=", state)
			# print("State Type :=", type(state[1]))
			# input("\33[41m ERRO STATE IS LIST -- Press Enter to continue...\33[0m")
		
			#select action
		action, lp = select_action(policy_network, state, concat_network)
		print("action :=", action)
	
		
		#execute action
		new_state, reward, done = env.step(action) 

		# print("reward = ", reward)

		
		#track episode score
		score += reward

		
		#store into trajectory
		trajectory.append([state, action, reward, lp])

		#
		state = new_state 
		
		#end episode - in failure case
		# if done or score < -5	:
		if (done and step > 0) or (score < -5 and step > 0):
			print("Done\n")
			break
		
		#move into new state
		# if(step == MAX_STEPS - 1):
		# 	print("FIM!!!\n\n")
		# 	state = new_state
		# 	bk_state = state
		# else:
		# 	state = env.reset_pose(changed_pose)

	
	#append score
	scores.append(score)
	recent_scores.append(score)
	episode_durations.append(step+1)
	plot_()

	print("Score = %d" % np.array(recent_scores).mean())
	# print("Action = ", action)

	#early stopping if we meet solved score goal
	if np.array(recent_scores).mean() >= SOLVED_SCORE:
		break

	
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

	if(episode%20==0):
		torch.save(policy_network, file_path+"actor.pkl")
		torch.save(stateval_network, file_path+"critic.pkl")

		np.savetxt(file_path+"scores.txt", scores, delimiter=',')

		# sns.set()
		# plt.ion()
		# fig = plt.figure()
		# plt.plot(scores)
		# plt.ylabel('score')
		# plt.xlabel('episodes')
		# plt.title('Training score with Forward-view TD')
		# fig.savefig(file_path+"/figures/partial.png", dpi=fig.dpi)
		# plt.draw()
		# plt.show()
		# plt.pause(0.1)
		# plt.close('all')


	rate.sleep()

	

torch.save(policy_network, file_path+"actor.pkl")
torch.save(stateval_network, file_path+"critic.pkl")



reg = LinearRegression().fit(np.arange(len(scores)).reshape(-1, 1), np.array(scores).reshape(-1, 1))
y_pred = reg.predict(np.arange(len(scores)).reshape(-1, 1))


## Export TXT
np.savetxt(file_path+"scores.txt", scores, delimiter=',')
np.savetxt(file_path+"predict.txt", y_pred, delimiter=',')

## PLOT


sns.set()
fig2 = plt.figure()
plt.plot(scores)
plt.ylabel('score')
plt.xlabel('episodes')
plt.title('Training score with Forward-view TD')

plt.plot(y_pred)
plt.show()

