#!/usr/bin/env python

# ROS
import rospy
import rospkg
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import Pose, PoseStamped, Twist, Polygon, Point32, Point, TransformStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import Bool

# System or python
from os.path import dirname, join, abspath
import numpy as np
from collections import deque
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



# Network
from pyrep import PyRep
from pyrep.objects.shape import Shape
import torch, gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from discrete_network import PolicyNetwork, StateValueNetwork, select_action, process_rewards, get_Nstep_returns, get_weights, train_policy, train_value
from sklearn.linear_model import LinearRegression

from espeleo_env import Environment


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
	from IPython import display

 
# Constants
SCENE_FILE = join(dirname(abspath(__file__)),
                  'Espeleo_office_map.ttt')
MAX_STEPS = 200
MAX_EPISODES = 4000
# POS_MIN, POS_MAX = [5.0, -2.0, 0.0], [9.0, 2.0, 0.0]
POS_MIN, POS_MAX = [7.0, 0.0, 0.0], [7.0, 0.0, 0.0]
DISCOUNT_FACTOR = 0.9
NUM_STATES = 4
NUM_ACTIONS = 5
CRITIC_LAMBDA = 0.9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"


# # Create Environment Class
# class Environment(object):
	
# 	def __init__(self):
# 		# ROS STUFFS
# 		rospy.init_node("PyRep", anonymous=True)
# 		self.pub_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
# 		self.set_pose = rospy.Publisher("/robot_newpose",Pose,queue_size=1)
# 		rospy.Subscriber("/pose_gt", Odometry, self.callback_pose)
# 		rospy.Subscriber("/odom", Odometry, self.callback_odom)
# 		rospy.Subscriber("/coppel_collision", Bool, self.callback_collision)


# 		# Create PyRep Object
# 		self.pr = PyRep()
# 		# Launch the application with a scene file
# 		# headless: TRUE - NO Graphics / FALSE - Open Coppelia
# 		self.pr.launch(SCENE_FILE, headless=False)

# 		# Start sim
# 		self.pr.start()  # Start the simulation


# 		# create target
# 		self.target = Shape('target')

# 		# variables
# 		self.initial_pose = [0,0,0,0,0,0,1]
# 		self.robot_pose = np.asarray([0.0,0.0,0.0])
# 		self.velocities = np.asarray([0.0,0.0])
# 		self.distancy_before = 0.0
# 		self.collision = False

# 		# Starting sim!!
# 		self.pr.step()
# 		self.pr.step()
# 		self.pr.step()


# 	def _get_state(self):
# 		return np.concatenate([self.robot_pose,
# 								self.velocities,
# 								self.target.get_position()])


# 	def reset(self):
# 		pos = list(np.random.uniform(POS_MIN, POS_MAX))
# 		self.target.set_position(pos)
# 		r_newPose = Pose()
# 		r_newPose.position.x = self.initial_pose[0]
# 		r_newPose.position.y = self.initial_pose[1]
# 		r_newPose.position.z = self.initial_pose[2]
# 		r_newPose.orientation.x = self.initial_pose[3]
# 		r_newPose.orientation.y = self.initial_pose[4]
# 		r_newPose.orientation.z = self.initial_pose[5]
# 		r_newPose.orientation.w = self.initial_pose[6]
# 		self.set_pose.publish(r_newPose)

# 		self.distancy_before = np.sqrt( (pos[0] - self.initial_pose[0])**2 + (pos[1] - self.initial_pose[0])**2 )

# 		return self._get_state()

# 	def detect_action(self, action):
# 		if(action == 0):
# 			V = 0.7
# 			W = 0.0
# 		elif(action == 1):
# 			V = 0.5
# 			W = 1.5
# 		elif(action == 2):
# 			V = 0.5
# 			W = -1.5
# 		elif(action == 3):
# 			V = 0.0
# 			W = 1.5
# 		else:
# 			V = 0.0
# 			W = -1.5

# 		return V,W

# 	def step(self, action):
# 		V, W = self.detect_action(action)
# 		vel_msg = Twist()
# 		vel_msg.linear.x = V
# 		vel_msg.angular.z = W
# 		self.pub_vel.publish(vel_msg)
# 		for i in range(2):
# 			self.pr.step()

# 		# Compute reward
# 		tx, ty, tz = self.target.get_position()
# 		D = np.sqrt( (self.robot_pose[0] - tx)**2 + (self.robot_pose[1] - ty)**2 )

# 		# Compute Reward
# 		if(D < self.distancy_before):
# 			reward = 1
# 		else:
# 			reward = -1

# 		self.distancy_before = D
# 		# reward = -D
# 		# reward = 1/(D+0.00001)

# 		if (self.collision):
# 			done = True
# 			reward = -100
# 		elif(D <= 0.5):
# 			done = True
# 			reward = 100
# 		else:
# 			done = False

# 		return reward, self._get_state(), done



# 	def shutdown(self):
# 		self.pr.stop()
# 		self.pr.shutdown()


# 	def callback_pose(self,data):
# 		angles = euler_from_quaternion([data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w])
# 		self.robot_pose = np.asarray([data.pose.pose.position.x,data.pose.pose.position.y,data.pose.pose.position.z,angles[2]])

# 	def callback_odom(self,data):
# 		self.velocities = np.asarray([data.twist.twist.linear.x,data.twist.twist.angular.z])

# 	def callback_collision(self, data):
# 		self.collision = data.data




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
	if len(durations_t) >= 50:
		means = scores_t.unfold(0, 50, 1).mean(1).view(-1)
		means = torch.cat((torch.zeros(49), means))
		plt.plot(means.numpy())

	plt.pause(0.001)  # pause a bit so that plots are updated
	if is_ipython:
		display.clear_output(wait=True)
		display.display(plt.gcf())


SCENE_FILE = join(dirname(abspath(__file__)),
                  'Espeleo_office_map.ttt')
env = Environment(SCENE_FILE, POS_MIN = [7.0, 0.0, 0.0],POS_MAX = [7.0, 0.0, 0.0])

# env = Environment()
rospy.sleep(2)

# Create Nets
policy_network = PolicyNetwork(NUM_STATES, NUM_ACTIONS).to(DEVICE)
stateval_network = StateValueNetwork(NUM_STATES).to(DEVICE)
# concat_network = ConcatNetwork().to(DEVICE)

# Create Optmizers
policy_optimizer = optim.Adam(policy_network.parameters(), lr=1e-6)
stateval_optimizer = optim.Adam(stateval_network.parameters(), lr=1e-6)

#track scores
# scores = []
recent_scores = deque(maxlen=10000)
# recent_scores = deque(maxlen=32)

episode = 0


while (episode <= MAX_EPISODES) and not rospy.is_shutdown():
	print('Starting episode %d' % episode)
	state = env.reset()
	score = 0
	trajectory = []
	# clear cuda memory
	if torch.cuda.is_available():
		gc.collect()
		torch.cuda.empty_cache()

	k = 0
	while (k <= MAX_STEPS) and not rospy.is_shutdown():
	# for k in range(MAX_STEPS):
		print('Step %d' % k)

		action, lp = select_action(policy_network, state)
		print("Action :=",action)
		reward, next_state, done = env.step(action)
		print("Reward = %f" % reward)
		# print("State :=", next_state)

		score += reward

		#store into trajectory
		trajectory.append([state, action, reward, lp])

		#
		state = next_state 

		if done:
			break

		k += 1


	scores.append(score)
	recent_scores.append(score)
	episode_durations.append(k)
	plot_()

	print("Score = %d" % score)

	#get items from trajectory
	states = [step[0] for step in trajectory]
	actions = [step[1] for step in trajectory]
	rewards = [step[2] for step in trajectory]
	lps = [step[3] for step in trajectory]


	#calculate state values
	state_vals = []
	for state in states:
		state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
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

		# action = list(np.random.uniform(0, 1.0, size=(2,)))
		# reward, next_state = env.step(action)
		# state = next_state


# Save network
path = dirname(abspath(__file__)) 
torch.save(policy_network.state_dict(), path  + '_actor.pt')
torch.save(stateval_network.state_dict(), path  + '_critic.pt')
print('Models saved successfully')


print('Done!')
env.shutdown()


reg = LinearRegression().fit(np.arange(len(scores)).reshape(-1, 1), np.array(scores).reshape(-1, 1))
y_pred = reg.predict(np.arange(len(scores)).reshape(-1, 1))


## Export TXT
np.savetxt(join(dirname(abspath(__file__)),'scores.txt'), scores, delimiter=',')
np.savetxt(join(dirname(abspath(__file__)),'predict.txt'), y_pred, delimiter=',')