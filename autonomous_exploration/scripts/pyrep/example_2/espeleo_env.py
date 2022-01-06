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
		self.pr.launch(SCENE_FILE, headless=True)

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

		# Starting sim!!
		self.pr.step()
		self.pr.step()
		self.pr.step()

		while (self.closest_point == -100.0):
			rospy.sleep(1)
			print("Waiting lidar")


	def _get_state(self):
		return np.concatenate([self.robot_pose,
								self.velocities,
								self.target.get_position(),
								[self.closest_point,self.closest_direction]])


	def reset(self):
		pos = list(np.random.uniform(self.pos_min, self.pos_max))
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


		self.distancy_before = np.sqrt( (pos[0] - self.initial_pose[0])**2 + (pos[1] - self.initial_pose[0])**2 )

		return self._get_state()

	def detect_action(self, action):
		if(action == 0):
			V = 0.7
			W = 0.0
		elif(action == 1):
			V = 0.5
			W = 1.5
		elif(action == 2):
			V = 0.5
			W = -1.5
		elif(action == 3):
			V = 0.0
			W = 1.5
		elif(action == -1):
		# else:
			V = 0.0
			W = -1.5
		elif(action == -2):
			V = -0.7
			W = 0.0
		elif(action == -3):
			V = -0.5
			W = 1.5
		else:
			V = -0.5
			W = -1.5

		return V,W

	def step(self, action):
		action = round(np.clip(action, -4, 3)[0])
		# print(action)
		V, W = self.detect_action(action)
		# V = action[0]
		# W = action[1]
		vel_msg = Twist()
		vel_msg.linear.x = V
		vel_msg.angular.z = W
		self.pub_vel.publish(vel_msg)
		for i in range(2):
			self.pr.step()

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
			reward = -10
		elif(D <= 0.5):
			done = True
			reward = 10
		else:
			done = False

		return reward, self._get_state(), done



	def shutdown(self):
		self.pr.stop()
		self.pr.shutdown()


	def callback_pose(self,data):
		angles = euler_from_quaternion([data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w])
		self.robot_pose = np.asarray([data.pose.pose.position.x,data.pose.pose.position.y,data.pose.pose.position.z,angles[2]])
		self.velocities = np.asarray([data.twist.twist.linear.x,data.twist.twist.angular.z])

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