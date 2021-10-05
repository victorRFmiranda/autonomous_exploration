#!/usr/bin/env python


#--------Include modules---------------
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PointStamped, Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from getfrontier import getfrontier
from tf.transformations import euler_from_quaternion
import numpy as np
from math import pi, atan2, tan, cos, sin, sqrt, hypot, floor, ceil, log, atan
from autonomous_exploration.msg import frontier

#--------Sklearn modules----------------------
from sklearn.cluster import DBSCAN, KMeans

#--------OpenCv------------------------------
import cv2





# global variables
mapData=OccupancyGrid()
robot_states = [0.0, 0.0, 0.0]
origem_map = [0.0,0.0]
width = 0
height = 0
resol = 0




########################################
'''			Callbacks			 '''
########################################
def callback_pose(data):
	global robot_states

	robot_states[0] = data.pose.pose.position.x  # robot pos x
	robot_states[1] = data.pose.pose.position.y  # robot pos y

	x_q = data.pose.pose.orientation.x
	y_q = data.pose.pose.orientation.y
	z_q = data.pose.pose.orientation.z
	w_q = data.pose.pose.orientation.w
	euler = euler_from_quaternion([x_q, y_q, z_q, w_q])

	robot_states[2] = euler[2]  # robot orientation
			
	return

def callback_map(msg):
	global size, origem_map, width, height, resol, mapData
	mapData = msg
	resol = msg.info.resolution
	width = msg.info.width
	height = msg.info.height
	origem_map = [msg.info.origin.position.x,msg.info.origin.position.y]
	size = [origem_map[0]+(msg.info.width * resol),origem_map[1]+(msg.info.height * resol)]



########################################
'''			Compute Frontier		 '''
########################################
def compute_frontiers(width,height,resol,origem_map,mapData, robot_states):
	front_vect = []
	for i in range(width):
		for j in range(height):
			if(mapData.data[j*width + i] == 0):
				if(i>0 and i < width-1 and j>0 and j < height-1):
					s = np.array([mapData.data[j*width + i+1], mapData.data[j*width + i-1], mapData.data[(j-1)*width + i], mapData.data[(j+1)*width + i]
							, mapData.data[(j-1)*width + i+1], mapData.data[(j+1)*width + i+1], mapData.data[(j-1)*width + i-1], mapData.data[(j+1)*width + i-1]])

					s1 = np.array([mapData.data[(j+2)*width + i-2],mapData.data[(j+2)*width + i-1],mapData.data[(j+2)*width + i],mapData.data[(j+2)*width + i+1],mapData.data[(j+2)*width + i+2],
								mapData.data[(j+1)*width + i-2],mapData.data[(j+1)*width + i-1],mapData.data[(j+1)*width + i],mapData.data[(j+1)*width + i+1],mapData.data[(j+1)*width + i+2],
								mapData.data[(j)*width + i-2],mapData.data[(j)*width + i-1],mapData.data[(j)*width + i],mapData.data[(j)*width + i+1],mapData.data[(j)*width + i+2],
								mapData.data[(j-1)*width + i-2],mapData.data[(j-1)*width + i-1],mapData.data[(j-1)*width + i],mapData.data[(j-1)*width + i+1],mapData.data[(j-1)*width + i+2],
								mapData.data[(j-2)*width + i-2],mapData.data[(j-2)*width + i-1],mapData.data[(j-2)*width + i],mapData.data[(j-2)*width + i+1],mapData.data[(j-2)*width + i-2]]
						)
					if( (len(np.where(s==-1)[0]) >= 3) and (len(np.where(s1==100)[0])<=0)):
						x = origem_map[0] + (i * resol + resol/2.0)
						y = origem_map[1] + (j * resol + resol/2.0)
						theta_s = atan2( (y - robot_states[1]), (x - robot_states[0]))
						theta_s = np.mod(theta_s, 2*pi)
						theta_r = np.mod(robot_states[2], 2*pi)
						theta = theta_s - theta_r + pi*0
						if theta > 2*pi:
							theta -= 2*pi

						front_vect.append([x,y])
	return front_vect



def create_image(mapa, centers):
	matrix = np.asarray(mapa.data)
	h = mapa.info.height
	w = mapa.info.width
	r = mapa.info.resolution
	om = [mapa.info.origin.position.x, mapa.info.origin.position.y]

	# for k in range(len(centers)):
	# 	i_w = np.around( (centers[k][0] - om[0] - r/2.0)/r ).astype(int)
	# 	j_h = np.around( (centers[k][1] - om[1] - r/2.0)/r ).astype(int)
	# 	# print(matrix[j_h*w + i_w])
	# 	matrix[j_h*w + i_w] = -100
	# 	matrix[(j_h+1)*w + i_w] = -100
	# 	matrix[(j_h-1)*w + i_w] = -100
	# 	matrix[j_h*w + i_w + 1] = -100
	# 	matrix[j_h*w + i_w - 1] = -100
	# 	matrix[(j_h+1)*w + i_w + 1] = -100
	# 	matrix[(j_h+1)*w + i_w - 1] = -100
	# 	matrix[(j_h-1)*w + i_w + 1] = -100
	# 	matrix[(j_h-1)*w + i_w - 1] = -100

	image = np.zeros((h,w,3)).astype(np.uint8)
	for i in range(0,h):
		for j in range(0,w):
			if(matrix[i*w+j] == 100):
				image[i,j] = [0,0,0]
			elif(matrix[i*w+j] == -1):
				image[i,j] = [192,192,192]
			elif(matrix[i*w+j] == 0):
				image[i,j] = [255,255,255]
			elif(matrix[i*w+j] == -100):
				image[i,j] = [255,0,0]

	

	# image = cv2.flip(image, 0);
	return image



def run():
	## ROS STUFFS
	rospy.init_node("lidar_frontier", anonymous=True)

	# Subscribers
	rospy.Subscriber('/odom', Odometry, callback_pose)
	rospy.Subscriber('/map', OccupancyGrid, callback_map)

	# Publishers
	#targetspub = rospy.Publisher('/frontier_points', PointStamped, queue_size=10)
	pub = rospy.Publisher('/frontier_markers', MarkerArray, queue_size=1)
	pub2 = rospy.Publisher('/cluster_markers', MarkerArray, queue_size=1)
	pub_map = rospy.Publisher("/map_image", Image, queue_size=1)
	pub_frontiers = rospy.Publisher("/frontier_points", frontier, queue_size=1)

	# routine frequency
	rate = rospy.Rate(10)

	while len(mapData.data)<1:
		print("Waiting Map")
		rate.sleep()
		pass

	#exploration_goal=PointStamped()
	

	while not rospy.is_shutdown():
		frontiers = compute_frontiers(width,height,resol, origem_map, mapData, robot_states)
		# print(len(frontiers))
		ft_array = frontier()
		if(len(frontiers) > 2):

			pointArray=MarkerArray()
			for i in range(len(frontiers)):
				ft = Point()
				points=Marker()
				#Set the frame ID and timestamp.  See the TF tutorials for information on these.
				points.header.frame_id=mapData.header.frame_id
				points.header.stamp=rospy.Time.now()

				# points.ns= "markers"
				points.id = i

				points.type = Marker.SPHERE
				points.scale.x = 0.1
				points.scale.y = 0.1
				points.scale.z = 0.1
				#Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
				points.action = Marker.ADD;

				points.pose.orientation.w = 1.0;
				points.scale.x=points.scale.y=0.1;
				points.color.r = 255.0/255.0
				points.color.g = 0.0/255.0
				points.color.b = 0.0/255.0
				points.color.a=1;
				points.lifetime == rospy.Duration();


				x=frontiers[i]

				points.pose.position.x = x[0]
				points.pose.position.y = x[1]
				points.pose.position.z = 0
				

				#exploration_goal.header.frame_id= mapData.header.frame_id
				#exploration_goal.header.stamp=rospy.Time(0)
				#exploration_goal.point.x=x[0]
				#exploration_goal.point.y=x[1]
				#exploration_goal.point.z=0	

				#targetspub.publish(exploration_goal)
				ft.x = x[0]
				ft.y = x[1]
				ft.z = 0.0
				ft_array.frontiers.append(ft)

				# points.points=[exploration_goal.point]

				pointArray.markers.append(points)
				
			pub.publish(pointArray) 




			# clustering = DBSCAN(eps=0.5, min_samples=10).fit(frontiers)
			# num_clusters = int(round(len(frontiers)*0.2))
			num_clusters = 4
			# print(num_clusters)
			kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(frontiers)

			clusterArray= MarkerArray()
			for k in range(num_clusters):
				p = Marker()
				ft = Point()
				#Set the frame ID and timestamp.  See the TF tutorials for information on these.
				p.header.frame_id=mapData.header.frame_id
				p.header.stamp=rospy.Time.now()

				# points.ns= "markers"
				p.id = k

				p.type = Marker.SPHERE
				p.scale.x = 0.2
				p.scale.y = 0.2
				p.scale.z = 0.2
				#Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
				p.action = Marker.ADD;

				p.pose.orientation.w = 1.0;
				p.scale.x=p.scale.y=0.1;
				p.color.r = 0.0/255.0
				p.color.g = 0.0/255.0
				p.color.b = 255.0/255.0
				p.color.a=1;
				p.lifetime == rospy.Duration();

				pos = kmeans.cluster_centers_[k]
				p.pose.position.x = pos[0]
				p.pose.position.y = pos[1]
				p.pose.position.z = 0

				clusterArray.markers.append(p)

				ft.x = pos[0]
				ft.y = pos[1]
				ft.z = 0.0
				ft_array.clusters.append(ft)

			pub2.publish(clusterArray) 

			pub_frontiers.publish(ft_array)

			image = create_image(mapData, kmeans.cluster_centers_)
			# cv2.imshow('image',image)
			# cv2.waitKey(0)

			bridge = CvBridge()
			map_image = bridge.cv2_to_imgmsg(image, encoding="bgr8")
			pub_map.publish(map_image)



		rate.sleep()



########################################
'''			Start Code			'''
########################################
if __name__ == '__main__':
	run()
