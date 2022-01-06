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
import matplotlib.pyplot as plt


# Network
from pyrep import PyRep
from pyrep.objects.shape import Shape
import torch, gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from network import conv_block, ConcatNetwork, PolicyNetwork, StateValueNetwork, select_action, process_rewards, get_Nstep_returns, get_weights, train_policy, train_value
from sklearn.linear_model import LinearRegression


# Constants
SCENE_FILE = join(dirname(abspath(__file__)),
                  'Espeleo_office_map.ttt')
MAX_STEPS = 500
MAX_EPISODES = 1000
# POS_MIN, POS_MAX = [5.0, -2.0, 0.0], [9.0, 2.0, 0.0]
POS_MIN, POS_MAX = [7.0, 0.0, 0.0], [7.0, 0.0, 0.0]
DISCOUNT_FACTOR = 0.99
NUM_STATES = 9
NUM_ACTIONS = 2
CRITIC_LAMBDA = 0.9
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"


# Create Environment Class
class Environment(object):
	
	def __init__(self):
		# ROS STUFFS
		rospy.init_node("PyRep", anonymous=True)
		self.pub_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
		self.set_pose = rospy.Publisher("/robot_newpose",Pose,queue_size=1)
		rospy.Subscriber("/pose_gt", Pose, self.callback_pose)
		rospy.Subscriber("/odom", Odometry, self.callback_odom)
		rospy.Subscriber("/coppel_collision", Bool, self.callback_collision)


		# Create PyRep Object
		self.pr = PyRep()
		# Launch the application with a scene file
		# headless: TRUE - NO Graphics / FALSE - Open Coppelia
		self.pr.launch(SCENE_FILE, headless=False)

		# Start sim
		self.pr.start()  # Start the simulation


		# create target
		self.target = Shape('target')

		# variables
		self.initial_pose = [0,0,0,0,0,0,1]
		self.robot_pose = np.asarray([0.0,0.0,0.0])
		self.velocities = np.asarray([0.0,0.0])
		self.collision = False

		# Starting sim!!
		self.pr.step()
		self.pr.step()
		self.pr.step()


	def _get_state(self):
		return np.concatenate([self.robot_pose,
								self.velocities,
								self.target.get_position()])


	def reset(self):
		pos = list(np.random.uniform(POS_MIN, POS_MAX))
		self.target.set_position(pos)
		r_newPose = Pose()
		r_newPose.position.x = self.initial_pose[0]
		r_newPose.position.y = self.initial_pose[1]
		r_newPose.position.z = self.initial_pose[2]
		r_newPose.orientation.x = self.initial_pose[3]
		r_newPose.orientation.y = self.initial_pose[4]
		r_newPose.orientation.z = self.initial_pose[5]
		r_newPose.orientation.w = self.initial_pose[6]
		self.set_pose.publish(r_newPose)
		return self._get_state()


	def step(self, action):
		vel_msg = Twist()
		vel_msg.linear.x = action[0]
		vel_msg.angular.z = action[1]
		self.pub_vel.publish(vel_msg)
		for i in range(2):
			self.pr.step()

		# Compute reward
		tx, ty, tz = self.target.get_position()
		D = np.sqrt( (self.robot_pose[0] - tx)**2 + (self.robot_pose[1] - ty)**2 )
		reward = -D
		# reward = 1/(D+0.00001)

		if (self.collision):
			done = True
			reward = -300
		elif(D <= 0.5):
			done = True
			reward = 300
		else:
			done = False

		return reward, self._get_state(), done




	def shutdown(self):
		self.pr.stop()
		self.pr.shutdown()


	def callback_pose(self,data):
		angles = euler_from_quaternion([data.orientation.x,data.orientation.y,data.orientation.z,data.orientation.w])
		self.robot_pose = np.asarray([data.position.x,data.position.y,data.position.z,angles[2]])

	def callback_odom(self,data):
		self.velocities = np.asarray([data.twist.twist.linear.x,data.twist.twist.angular.z])

	def callback_collision(self, data):
		self.collision = data.data






env = Environment()
rospy.sleep(2)

# Create Nets
policy_network = PolicyNetwork(NUM_STATES, NUM_ACTIONS).to(DEVICE)
stateval_network = StateValueNetwork(NUM_STATES).to(DEVICE)
# concat_network = ConcatNetwork().to(DEVICE)

# Create Optmizers
policy_optimizer = optim.Adam(policy_network.parameters(), lr=1e-2)
stateval_optimizer = optim.Adam(stateval_network.parameters(), lr=1e-2)

#track scores
scores = []
recent_scores = deque(maxlen=1000000)
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
		# print(action)
		reward, next_state, done = env.step(action)
		print("Reward = %f" % reward)

		score += reward

		#store into trajectory
		trajectory.append([state, action, reward, lp])

		if done:
			break

		k += 1


	scores.append(score)
	recent_scores.append(score)

	print("Score = %d" % np.array(recent_scores).mean())

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
torch.save(policy_network, join(dirname(abspath(__file__)),
                  'actor.pkl'))
torch.save(stateval_network, join(dirname(abspath(__file__)),
                  'critic.pkl'))


reg = LinearRegression().fit(np.arange(len(scores)).reshape(-1, 1), np.array(scores).reshape(-1, 1))
y_pred = reg.predict(np.arange(len(scores)).reshape(-1, 1))


## Export TXT
np.savetxt(join(dirname(abspath(__file__)),'scores.txt'), scores, delimiter=',')
np.savetxt(join(dirname(abspath(__file__)),'predict.txt'), y_pred, delimiter=',')

## Plot
# sns.set()
fig2 = plt.figure()
plt.plot(scores)
plt.ylabel('score')
plt.xlabel('episodes')
plt.title('Training score with Forward-view TD')

plt.plot(y_pred)
plt.show()


print('Done!')
env.shutdown()