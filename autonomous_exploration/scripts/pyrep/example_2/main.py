#!/usr/bin/env python

from __future__ import division
# import gym
import numpy as np
import torch
from torch.autograd import Variable
from torch.distributions import Categorical
import os
import psutil
import gc
from os.path import dirname, join, abspath
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import train
import buffer

# ROS
import rospy

from espeleo_env import Environment
 

# Constants
SCENE_FILE = join(dirname(abspath(__file__)),
                  'Espeleo_office_map.ttt')

# env = gym.make('BipedalWalker-v3')
# env = gym.make('Pendulum-v1')
env = Environment(SCENE_FILE, POS_MIN = [7.0, 0.0, 0.0],POS_MAX = [7.0, 0.0, 0.0])
rospy.sleep(5)

MAX_EPISODES = 500
MAX_STEPS = 200
MAX_BUFFER = 1000
MAX_TOTAL_REWARD = 300
S_DIM = 11
A_DIM = 1
A_MAX = 4

print(' State Dimensions :- ', S_DIM)
print(' Action Dimensions :- ', A_DIM)
# print(' Action Max :- ', A_MAX)

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)

scores = []

for _ep in range(MAX_EPISODES):
	observation = env.reset()
	score = 0
	# print(type(observation))
	print('EPISODE :- ', _ep)
	for r in range(MAX_STEPS):
		
		state = np.float32(observation)
		# state = observation

		action = trainer.get_exploration_action(state)
		# m = Categorical(action)
		# c_action = m.sample().item()
		# print(test.item())
		# action = action.data.numpy() 
		print('STEP :- ', r)
		# print('Action :-', c_action)
		# print('Action2 :-',action)

		# new_observation, reward, done, info = env.step(action)
		reward, new_observation, done = env.step(action)

		score += reward

		if done:
			new_state = None
		else:
			new_state = np.float32(new_observation)
			# push this exp in ram
			ram.add(state, action, reward, new_state)

		observation = new_observation

		# perform optimization
		trainer.optimize()
		if done:
			break


	scores.append(score)
	print("\33[92m Score =:", score, "\33[0m")
	# check memory consumption and clear memory
	gc.collect()
	# process = psutil.Process(os.getpid())
	# print(process.memory_info().rss)

	if _ep%100 == 0:
		path = dirname(abspath(__file__)) + "/Models/"
		trainer.save_models(path,_ep)


print('Completed episodes')
env.shutdown()



reg = LinearRegression().fit(np.arange(len(scores)).reshape(-1, 1), np.array(scores).reshape(-1, 1))
y_pred = reg.predict(np.arange(len(scores)).reshape(-1, 1))


## Export TXT
np.savetxt(join(dirname(abspath(__file__)),'scores.txt'), scores, delimiter=',')
np.savetxt(join(dirname(abspath(__file__)),'predict.txt'), y_pred, delimiter=',')

## Plot
sns.set()
fig2 = plt.figure()
plt.plot(scores)
plt.ylabel('score')
plt.xlabel('episodes')
plt.title('Training score')

plt.plot(y_pred)
plt.show()
