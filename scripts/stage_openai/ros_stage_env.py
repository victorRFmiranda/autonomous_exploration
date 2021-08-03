#!/usr/bin/env python

import rospy
import roslaunch
import numpy as np

import gym
import time
import copy
import os

from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import Pose, PoseStamped, Twist
from nav_msgs.msg import Odometry

from math import pi, atan2, tan, cos, sin, sqrt, hypot, floor, ceil, log


class StageEnvironment(gym.Env):
	def __init__(self,args):
		self.action_space = args.num_actions		# num frontiers (fixed)
		self.max_actions = args.max_action_count    # num actions for epoch (how many times check all frontiers)
		self.num_initstates = args.n_init 			# num start positions before change the map
		self.maps = args.maps_gt					# vector with name of stage maps for training
		self.map_count = 0

		self.init_pose = [-20.0, -20.0, 0.0]		# x, y, theta  -- Came from a parameters ?
		self.robot_pose = [0.0, 0.0, 0.0]			# x, y, theta

		rospy.init_node("Stage_environment", anonymous=True)
		rospy.Subscriber("/base_pose_ground_truth", Odometry, self.callback_pose)
		self.pub_pose = rospy.Publisher("/cmd_pose", Pose, queue_size=1)
		self.pub_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1)


	# change the map for training
	def reset_map(self):
		node = "/stageros"
		os.system("rosnode kill "+ node)
		time.sleep(1)
		os.system("gnome-terminal -x roslaunch autonomous_exploration test_stage.launch map:="+self.maps[self.map_count])
		time.sleep(1)
		print("map changed")
		self.map_count += 1

	# restart stage (robot back to the init position) - change the robot pose in training code
	def reset_pose(self, data):
		msg_pos = Pose()
		q = quaternion_from_euler(0,0,data[2])
		msg_pos.orientation.x = q[0]
		msg_pos.orientation.y = q[1]
		msg_pos.orientation.z = q[2]
		msg_pos.orientation.w = q[3]
		msg_pos.position.x = data[0]
		msg_pos.position.y = data[1]
		msg_pos.position.z = 0.0
		self.pub_pose.publish(msg_pos)
		print("Pose reseted\n")

		# Reset Gmapping
		node = "/GMAP"
		os.system("rosnode kill "+ node)
		time.sleep(1)
		os.system("gnome-terminal -x roslaunch autonomous_exploration gmapping.launch xmin:=-25.0 ymin:=-25.0 xmax:=25.0 ymax:=25.0 delta:=0.1 odom_fram:=world")
		time.sleep(1)
		print("gmapping reseted")

	def reset(self):
		rospy.wait_for_service('reset_positions')


	# move robot to the selected frontier (action)
	def step(self, frontier):
		D = _dist(self.robot_pose,frontier)
		msg = Twist()
		while(D>0.2):
			D = _dist(self.robot_pose,frontier)
			v,w = _PotControl(self.robot_pose,frontier)
			msg.linear.x = v
			msg.angular.z = w
			self.pub_vel.publish(msg)


	# compute the reward for this action
	def compute_reward(self):
		print("reward")

	def callback_pose(self, data):
		self.robot_pose[0] = data.pose.pose.position.x
		self.robot_pose[1] = data.pose.pose.position.y
		q = [data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w]
		angles = euler_from_quaternion(q)
		self.robot_pose[2] = angles[2]


def _PotControl(robot_pose,goal):
	d = 0.2
	k = 1
	#pot field
	Ux = k * (goal[0] - robot_pose[0])
	Uy = k * (goal[1] - robot_pose[1])
	#feedback_linearization
	vx = cos(robot_pose[2]) * Ux + sin(robot_pose[2]) * Uy
	w = -(sin(robot_pose[2]) * Ux)/ d + (cos(robot_pose[2]) * Uy) / d
	return vx,w


def _dist(p1,p2): 
	return ((p1[0]-p2[0])**2 +(p1[1]-p2[1])**2)**(0.5)


from config import Config
args = Config().parse()
# set argumments
# args.num_actions = 1
# args.max_action_count = 2
# args.n_init = 4
# args.maps_gt = 5
env = StageEnvironment(args)

cont = 0
rate = rospy.Rate(1)
goal = [-20,-15]
init_pose = env.init_pose
# os.system("gnome-terminal -x roslaunch autonomous_exploration test_stage.launch map:="+args.maps_gt[0])
env.reset_map()
time.sleep(2)
while not rospy.is_shutdown() and cont < args.n_init:
	print("Epoch: %d" % cont)
	for times in range(args.max_action_count):
		print("Event: %d" % times)
		env.reset_pose(init_pose)
		rospy.sleep(1)
		env.step(goal)

	# if(cont == 1):	

	env.reset_map()			
	init_pose = copy.deepcopy(env.robot_pose)
	cont += 1
	goal[1] += 5
	#print(init_pose)
	rate.sleep()

