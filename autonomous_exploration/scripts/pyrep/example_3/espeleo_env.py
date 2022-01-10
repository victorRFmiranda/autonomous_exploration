#!/usr/bin/env python

# ROS
import rospy
import rospkg
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import Pose, PoseStamped, Twist, Polygon, Point32, Point, TransformStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Bool

# System or python
from os.path import dirname, join, abspath
import numpy as np
from collections import deque
import time
from math import atan2, pi


# Network
from pyrep import PyRep
from pyrep.objects.shape import Shape


 
# Constants
# SCENE_FILE = join(dirname(abspath(__file__)),
                  # 'Espeleo_office_map.ttt')

# POS_MIN, POS_MAX = [5.0, -2.0, 0.0], [9.0, 2.0, 0.0]
# POS_MIN, POS_MAX = [7.0, 0.0, 0.0], [7.0, 0.0, 0.0]


# Create Environment Class
class Environment(object):
	
	def __init__(self, SCENE_FILE, POS_MIN = [7.0, 0.0, 0.0], POS_MAX= [7.0, 0.0, 0.0]):
		# ROS STUFFS
		rospy.init_node("PyRep", anonymous=True)
		self.pub_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
		self.set_pose = rospy.Publisher("/robot_newpose",Pose,queue_size=1)
		rospy.Subscriber("/pose_gt", Odometry, self.callback_pose)
		rospy.Subscriber('/scan', LaserScan, self.callback_laser)
		rospy.Subscriber("/coppel_collision", Bool, self.callback_collision)
		# rospy.Subscriber("/odom", Odometry, self.callback_odom)


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
		self.pos_min = POS_MIN
		self.pos_max = POS_MAX
		self.distancy_before = 0.0
		self.laser = []
		self.l_range_max = 0.0
		self.l_range_min = 0.0
		self.collision = False
		self.closest_direction = 0.0
		self.closest_point = -100.0

		self.memory_len = 5
		self.memory_pose  = deque(maxlen=self.memory_len)
		self.memory_velocities = deque(maxlen=self.memory_len)
		self.memory_target_pos = deque(maxlen=self.memory_len)
		self.memory_lidar = deque(maxlen=self.memory_len)
		self.memory_distancy  = deque(maxlen=self.memory_len)

		# Starting sim!!
		self.pr.step()
		self.pr.step()
		self.pr.step()


	def initiate_memory(self):
		for _ in range(4):
			self.memory_distancy.append([0.0,0.0])
			self.memory_velocities.append([0.0,0.0])
			self.memory_lidar.append([0.0,0.0])



	def _get_state(self):
		tx,ty,tz = self.target.get_position()
		D = np.sqrt( (self.robot_pose[0] - tx)**2 + (self.robot_pose[1] - ty)**2 )

		angle = np.arctan2((ty-self.robot_pose[1]),(tx-self.robot_pose[0]))


		self.memory_distancy.append([D,angle])
		# self.memory_pose.append(self.robot_pose)
		self.memory_velocities.append(self.velocities)
		# self.memory_target_pos.append(self.target.get_position())
		self.memory_lidar.append([self.closest_point,self.closest_direction])

		# print("Distance :=", self.memory_distancy)
		# s = np.hstack(self.memory_distancy)
		# print(s)
		# print(s.shape)
		# print("vel :=", self.memory_velocities)
		# print("lidar :=", self.memory_lidar)

		# aaaaaaa = np.concatenate( self.memory_lidar )

		return np.hstack( (np.hstack(self.memory_distancy), np.hstack(self.memory_velocities), np.hstack(self.memory_lidar)) )




		# return np.concatenate([self.robot_pose, 
		# 						self.velocities,
		# 						self.target.get_position(),
		# 						[self.closest_point,self.closest_direction]])


	def aleatory_pose(self):
		vec_x = [[0,5],[6,9],[6,8]]
		vec_y = [[-1,1],[2,5],[-5,-2]]
		aux = np.random.randint(3)

		x = np.random.uniform(vec_x[aux][0],vec_x[aux][1])
		y = np.random.uniform(vec_y[aux][0],vec_y[aux][1])

		q = quaternion_from_euler(0, 0, np.random.uniform(-pi,pi))

		return x, y, q

	def reset(self):
		self.initiate_memory()

		pos = list(np.random.uniform(self.pos_min, self.pos_max))
		self.target.set_position(pos)

		x,y, q = self.aleatory_pose()
		r_newPose = Pose()
		r_newPose.position.x = x
		r_newPose.position.y = y
		r_newPose.position.z = 0.0
		r_newPose.orientation.x = q[0]
		r_newPose.orientation.y = q[1]
		r_newPose.orientation.z = q[2]
		r_newPose.orientation.w = q[3]
		self.set_pose.publish(r_newPose)


		self.distancy_before = np.sqrt( (pos[0] - self.initial_pose[0])**2 + (pos[1] - self.initial_pose[0])**2 )

		return self._get_state()

	def detect_action(self, action):
		if(action == 0):
			V = 0.7
			W = 0.0
		elif(action == 1):
			V = 0.4
			W = 1.0
		elif(action == 2):
			V = 0.4
			W = -1.0
		elif(action == 3):
			V = 0.0
			W = 1.5
		# elif(action == 4):
		else:
			V = 0.0
			W = -1.5
		# elif(action == 5):
		# 	V = -0.7
		# 	W = 0.0
		# elif(action == 6):
		# 	V = -0.5
		# 	W = 1.5
		# else:
		# 	V = -0.5
		# 	W = -1.5

		return V,W

	def step(self, action):
		# action = round(np.clip(action, -4, 3)[0])
		# print(action)
		V, W = self.detect_action(action)
		
		self.velocities = np.asarray([V,W])

		vel_msg = Twist()
		vel_msg.linear.x = V
		vel_msg.angular.z = W
		self.pub_vel.publish(vel_msg)
		for i in range(2):
			self.pr.step()
			time.sleep(0.01)

		# Compute Distancy to target
		tx, ty, tz = self.target.get_position()
		D = np.sqrt( (self.robot_pose[0] - tx)**2 + (self.robot_pose[1] - ty)**2 )

		# Compute Reward
		if(D < self.distancy_before):
			reward = 1
		else:
			reward = -1

		self.distancy_before = D

		# reward = -D
		# reward = 1/(D+0.00001)


		if (self.collision):
			done = True
			reward = -200
		elif(D <= 0.5):
			done = True
			reward = 200
		else:
			done = False

		return reward, self._get_state(), done



	def shutdown(self):
		self.pr.stop()
		self.pr.shutdown()


	def callback_pose(self,data):
		angles = euler_from_quaternion([data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w])
		# self.robot_pose = np.asarray([data.pose.pose.position.x,data.pose.pose.position.y,data.pose.pose.position.z,angles[2]])
		self.robot_pose = [data.pose.pose.position.x,data.pose.pose.position.y,data.pose.pose.position.z,angles[2]]
		# self.velocities = np.asarray([data.twist.twist.linear.x,data.twist.twist.angular.z])

	def callback_odom(self,data):
		self.velocities = np.asarray([data.twist.twist.linear.x,data.twist.twist.angular.z])


	def callback_laser(self, data):
		self.laser = data.ranges					 # Distancias detectadas
		self.closest_direction = np.argmin(self.laser)
		self.closest_point = self.laser[self.closest_direction]
		# print("Direction :=", self.closest_direction)
		# print("Distancy :=", self.closest_point)

		# print("Lidar Size := ", len(self.laser))
		self.l_range_max = data.range_max		  # range max do lidar
		self.l_range_min = data.range_min		  # range min do lidar

	def callback_collision(self, data):
		self.collision = data.data





# env = Environment()
# rospy.sleep(2)

# print('Done!')
# env.shutdown()