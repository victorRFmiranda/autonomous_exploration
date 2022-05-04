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
from collections import deque
import cv2


from environment.environment import Environment
from environment.environment_node_data import Mode
import action_mapper


# set up matplotlib
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
# 	from IPython import display

class ReplayBuffer():
	def __init__(self, max_size, input_dims):
		input_shape = input_dims[0]*input_dims[1]
		self.mem_size = max_size
		self.mem_cntr = 0
		# self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
		# self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
		self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
		self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)

		self.log_prob_memory = np.zeros(self.mem_size, dtype=np.float32)
		self.reward_memory = np.zeros(self.mem_size,dtype=np.float32)
		self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

	def store_transition(self, state, log_prob, reward, state_, done):
		index = self.mem_cntr % self.mem_size
		self.state_memory[index] = state
		self.log_prob_memory[index] = log_prob
		self.reward_memory[index] = reward
		self.new_state_memory[index] = state_
		self.terminal_memory[index] = done

		self.mem_cntr += 1

	def sample_buffer(self, batch_size):
		max_mem = min(self.mem_cntr, self.mem_size)
		batch = np.random.choice(max_mem, batch_size, replace=False)
		states = self.state_memory[batch]
		probs = self.log_prob_memory[batch]
		rewards = self.reward_memory[batch]
		states_ = self.new_state_memory[batch]
		dones = self.terminal_memory[batch]

		return states, probs, rewards, states_, dones



class ActorCritic(nn.Module):
	def __init__(self, lr, input_dims, n_actions):
		super(ActorCritic, self).__init__()
		self.input_dims = input_dims[0]*input_dims[1]
		self.lr = lr
		self.n_actions = n_actions
		self.hidden_dim = 512


		# ACTOR
		self.dense_actor= nn.Sequential(
					nn.Linear(self.input_dims,self.hidden_dim),
					nn.ReLU(),
					nn.Linear(self.hidden_dim, self.hidden_dim),
					nn.ReLU(),
					nn.Linear(self.hidden_dim, self.hidden_dim),
					nn.ReLU(),
					nn.Linear(self.hidden_dim,self.n_actions)
					)

		self.dense_value= nn.Sequential(
					nn.Linear(self.input_dims,self.hidden_dim),
					nn.ReLU(),
					nn.Linear(self.hidden_dim, self.hidden_dim),
					nn.ReLU(),
					nn.Linear(self.hidden_dim, self.hidden_dim),
					nn.ReLU(),
					nn.Linear(self.hidden_dim,1)
					)

		# self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
		self.optimizer = optim.RAdam(self.parameters(), lr=self.lr)
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, state):
		# print(state.shape)
		# Convolution step
		# x = self.conv(state)
		# input("Wait")

		# actor
		actions = self.dense_actor(state)
		# actions = self.dense_actor(x.view(x.size(0), -1))
		probs = F.softmax(actions, dim=1)

		# critic
		# state_value = self.dense_value(x.view(x.size(0), -1))
		state_value = self.dense_value(state)

		return (probs, state_value)


class Agent():
	def __init__(self, lr, input_dims, n_actions, gamma= 0.99, batch_size=32, mem_size=1000000):
		self.gamma = gamma
		self.batch_size = batch_size
		self.memory = ReplayBuffer(mem_size, input_dims)
		self.actor_critic = ActorCritic(lr, input_dims, n_actions=n_actions)

	def store_transition(self, state, prob, reward, state_, done):
		self.memory.store_transition(state, prob, reward, state_, done)


	def choose_action(self, observation):
		self.actor_critic.eval()
		state = torch.tensor([observation]).to(self.actor_critic.device, dtype=torch.float)
		probabilities, _ = self.actor_critic.forward(state)
		action_probs = torch.distributions.Categorical(probabilities)
		action = action_probs.sample()
		log_probs = action_probs.log_prob(action)

		return action.item(), log_probs


	def learn(self):
		if self.memory.mem_cntr < self.batch_size:
			return
		self.actor_critic.optimizer.zero_grad()

		state, prob, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
		states = torch.tensor(state).to(self.actor_critic.device)
		probs = torch.tensor(prob).to(self.actor_critic.device)
		rewards = torch.tensor(reward).to(self.actor_critic.device)
		dones = torch.tensor(done).to(self.actor_critic.device)
		states_ = torch.tensor(new_state).to(self.actor_critic.device)

		_, critic_value = self.actor_critic.forward(states)
		_, critic_value_ = self.actor_critic.forward(states_)


		critic_value_[dones] = 0.0

		delta = rewards + self.gamma*critic_value_

		actor_loss = -torch.mean(probs*(delta - critic_value))
		critic_loss = F.mse_loss(delta, critic_value)

		(actor_loss + critic_loss).backward()
		self.actor_critic.optimizer.step()

	def save_model(self):
		torch.save(self.actor_critic, "./Models/AC_model.pkl")



episode_durations = []
scores = []


def update_frame(q, frame):
	if q.full():
		q.get()
	q.put(frame)


###################################################################
if __name__ == "__main__":
	env = Environment("./environment/world/room")
	env.set_mode(Mode.ALL_RANDOM, False)
	env.use_ditance_angle_to_end(True)
	env.set_observation_rotation_size(128)
	env.use_observation_rotation_size(True)
	env.set_cluster_size(1)

	MAX_EPISODES = 30000
	MAX_STEPS = 500
	vector_len = 2

	_, _, _, _ = env.reset()

	# torch.randn(20, 16, 5)	
	# print(torch.randn(5, 16, 32).shape)
	# input("WAIT")

	state_size = env.observation_size() -2
	action_size = action_mapper.ACTION_SIZE
		

	agent = Agent(input_dims = [state_size,vector_len], n_actions = action_size, lr= 3e-5, gamma = 0.99)

	recent_scores = deque(maxlen=10000)

	

	queue_state = Queue(maxsize = vector_len)
	flag_colide = False 

	

	ep = 0
	while (ep < MAX_EPISODES):

		queue_state.queue.clear()
		observation, _, flag_colide, _ = env.reset()

		while(not queue_state.full()):
			update_frame(queue_state, observation)
			observation, _, flag_colide, _ = env.step(0.0,0.0,20)

		if(flag_colide):
			flag_colide = False
			continue

		state = np.array(queue_state.queue)
		# state = np.expand_dims(state, axis=0)
		# state = np.transpose(state, [1, 2, 0])  # move channels
		state = np.transpose(state, [1, 0])  # move channels
		state = np.reshape(state, -1)


		trajectory = []
		score = 0

		
		for step in range(MAX_STEPS):
			
			action, prob = agent.choose_action(state)


			linear, angular = action_mapper.map_action(action)

			#execute action
			next_observation, reward, done, _ = env.step(linear, angular, 20)
			update_frame(queue_state, next_observation)
			next_state = np.array(queue_state.queue)
			# next_state = np.expand_dims(next_state, axis=0)
			# next_state = np.transpose(next_state, [1, 2, 0])
			next_state = np.transpose(next_state, [1, 0])  # move channels
			next_state = np.reshape(next_state, -1)

			#track episode score
			score += reward

			agent.store_transition(state, prob, reward, next_state, int(done))
			agent.learn()

			state = next_state 

			env.visualize()


			if done and step > 1:
				break

		scores.append(score)
		avg_score = np.mean(scores[-100:])
		
		print("Episode ", ep, "score %.1f" % score, "Avg score %.2f" % avg_score)

		ep += 1



agent.save_model()
print("Finish")


