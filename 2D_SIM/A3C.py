import torch
import torch.nn as nn
from utils import v_wrap, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import os
from queue import Queue
import numpy as np
# import multiprocessing
from threading import Thread

from environment.environment import Environment
from environment.environment_node_data import Mode
import action_mapper



# os.environ["OMP_NUM_THREADS"] = "10"



class Net(nn.Module):
	def __init__(self, s_dim, a_dim):
		super(Net, self).__init__()


		self.conv = nn.Sequential(
				nn.Conv1d(in_channels=s_dim,out_channels=16,kernel_size=1,stride=5),
				nn.BatchNorm1d(16),
				nn.ReLU(),
				nn.Conv1d(in_channels=16,out_channels=32,kernel_size=1,stride=3),
				nn.BatchNorm1d(32),
				nn.ReLU()
				)
		linear_input_size = 32
		self.dense_A = nn.Sequential(
					nn.Linear(linear_input_size,512),
					nn.ReLU(),
					nn.Linear(512,512),
					nn.ReLU(),
					nn.Linear(512,a_dim)
					)

		self.dense_V = nn.Sequential(
					nn.Linear(linear_input_size,512),
					nn.ReLU(),
					nn.Linear(512,512),
					nn.ReLU(),
					nn.Linear(512,1)
					)

		self.distribution = torch.distributions.Categorical

	def forward(self, x):
		x_a = self.conv(x)
		logits = self.dense_A(x_a.view(x_a.size(0), -1))
		
		x_v = self.conv(x)
		values = self.dense_V(x_v.view(x_v.size(0), -1))

		return logits, values

	def choose_action(self, s):
		self.eval()
		logits, _ = self.forward(s)
		prob = F.softmax(logits, dim=1).data
		m = self.distribution(prob)
		return m.sample().numpy()[0]


	def loss_func(self, s, a, v_t):
		self.train()
		logits, values = self.forward(s)
		td = v_t - values
		c_loss = td.pow(2)

		
		probs = F.softmax(logits, dim=1)
		m = self.distribution(probs)

		# print(values)
		# print(td.detach().squeeze())
		# print(m.log_prob(a))
		# input("AAAAAA")

		exp_v = m.log_prob(a) * td.detach().squeeze()
		a_loss = -exp_v
		total_loss = (c_loss + a_loss).mean()
		return total_loss


def update_frame(q, frame):
	if q.full():
		q.get()
	q.put(frame)

class Worker(Thread):
	def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, agent_id, name):
		super(Worker, self).__init__()
		self.name = 'w%i_%02i' % (agent_id,name)
		self.id = name
		self.mapas = mapas
		self.agent_id = agent_id
		# print(name)
		self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
		self.gnet, self.opt = gnet, opt
		self.lnet = Net(N_S, N_A)		   # local network
		# self.env = gym.make('CartPole-v0').unwrapped
		self.env = Environment(self.mapas[self.id % len(self.mapas)])
		self.env.set_mode(Mode.ALL_RANDOM, False)
		self.env.use_ditance_angle_to_end(True)
		self.env.set_observation_rotation_size(128)
		self.env.use_observation_rotation_size(True)
		self.env.set_cluster_size(1)

		self.queue_state = Queue(maxsize = 5)

	def run(self):
		total_step = 1
		flag_colide = False
		while self.g_ep.value < MAX_EP:
			self.queue_state.queue.clear()
			# s = self.env.reset()
			observation, _, flag_colide, _ = self.env.reset()


			while(not self.queue_state.full()):
				update_frame(self.queue_state, observation)
				observation, _, flag_colide, _ = env.step(0.0,0.0,20)

			if(flag_colide):
				flag_colide = False
				continue

			state = np.array(self.queue_state.queue)
			state = np.transpose(state, [1, 0])  # move channels

			buffer_s, buffer_a, buffer_r = [], [], []
			ep_r = 0.
			step = 0
			while step < MAX_STEP:
				if (self.agent_id == 0) and (int(self.id / len(self.mapas)) == 0):
					self.env.visualize()


				
				action = self.lnet.choose_action(v_wrap(state[None, :]))
				linear, angular = action_mapper.map_action(action)
				next_observation, r, done, _ = self.env.step(linear, angular, 20)
				update_frame(self.queue_state, next_observation)
				next_state = np.array(self.queue_state.queue)
				next_state = np.transpose(next_state, [1, 0])
				# if done: r = -1
				ep_r += r
				buffer_a.append(action)
				buffer_s.append(state)
				buffer_r.append(r)

				if (total_step % UPDATE_GLOBAL_ITER == 0 or done) and len(buffer_s) > 1:  # update global and assign to local net
					# sync
					push_and_pull(self.opt, self.lnet, self.gnet, done, next_state, buffer_s, buffer_a, buffer_r, GAMMA)
					buffer_s, buffer_a, buffer_r = [], [], []

					if done:  # done and print information
						record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
						break
				state = next_state
				total_step += 1
				step += 1
		self.res_queue.put(None)


class Agent(mp.Process):
	def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, a_id):
		super(Agent, self).__init__()
		self.gnet = gnet
		self.opt = opt
		self.global_ep = global_ep
		self.global_ep_r = global_ep_r
		self.res_queue = res_queue
		self.id = a_id

		self.number = 8

	def run(self):
		workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, self.id, i) for i in range(self.number)]
		[w.start() for w in workers]


if __name__ == "__main__":

	# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

	np.random.seed(100)

	UPDATE_GLOBAL_ITER = 5
	GAMMA = 0.9
	MAX_EP = 100000
	MAX_STEP = 300

	# mapas = ["./environment/world/room", "./environment/world/four_rooms" ,"./environment/world/square"]
	mapas = ["./environment/world/room"]


	env = Environment("./environment/world/square")
	env.set_mode(Mode.ALL_RANDOM, False)
	env.use_ditance_angle_to_end(True)
	env.set_observation_rotation_size(128)
	env.use_observation_rotation_size(True)
	env.set_cluster_size(1)
	env.reset()

	N_S = env.observation_size() - 2
	N_A = action_mapper.ACTION_SIZE

	# mp.set_start_method('spawn')

	gnet = Net(N_S, N_A)	# global network
	gnet.share_memory()		 # share the global parameters in multiprocessing
	opt = SharedAdam(gnet.parameters(), lr=1e-6, betas=(0.92, 0.999))	  # global optimizer
	global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

	# parallel training
	# workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
	# workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
	workers = [Agent(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(3)]
	[w.start() for w in workers]
	res = []					# record episode reward to plot
	while True:
		r = res_queue.get()
		if r is not None:
			res.append(r)
		else:
			break
	[w.join() for w in workers]

	torch.save(gnet, "./Models/A3C_Net.pkl")

	import matplotlib.pyplot as plt
	plt.plot(res)
	plt.ylabel('Moving average ep reward')
	plt.xlabel('Step')
	plt.show()


	