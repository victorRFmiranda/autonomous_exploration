#!/usr/bin/env python


#--------Include modules---------------
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PointStamped
from getfrontier import getfrontier
from tf.transformations import euler_from_quaternion
import numpy as np

#-----------------------------------------------------
# Subscribers' callbacks------------------------------
mapData=OccupancyGrid()

robot_states = np.zeros((2,3))
goal_point = np.zeros((2,2))
origem_map = [0.0,0.0]
width = 0
height = 0
resol = 0


def mapCallBack(data):
	global mapData, width,height,resol,origem_map
	resol = data.info.resolution
	width = data.info.width
	height = data.info.height
	origem_map = [data.info.origin.position.x,data.info.origin.position.y]
	mapData=data
	

	

# Node----------------------------------------------
def node():
		global mapData, robot_states, width,height,resol,origem_map
		exploration_goal=PointStamped()
		goal = PointStamped()
		rospy.init_node('FrontierDetector', anonymous=False)
		# map_topic= rospy.get_param('~map_topic','/robot_1/map')
		map_topic = '/map'
		rospy.Subscriber(map_topic, OccupancyGrid, mapCallBack)

		targetspub = rospy.Publisher('/frontier_points', PointStamped, queue_size=10)
		# pub = rospy.Publisher('shapes', Marker, queue_size=10)
		pub = rospy.Publisher('/frontier_markers', MarkerArray, queue_size=1)

		# wait until map is received, when a map is received, mapData.header.seq will not be < 1
		while mapData.header.seq<1 or len(mapData.data)<1:
			pass
		   	
		rate = rospy.Rate(20)			
		
#-------------------------------OpenCV frontier detection------------------------------------------
		while not rospy.is_shutdown():
			frontiers=getfrontier(mapData)
			pointArray=MarkerArray()
			for i in range(len(frontiers)):
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

				exploration_goal.header.frame_id= mapData.header.frame_id
				exploration_goal.header.stamp=rospy.Time(0)
				exploration_goal.point.x=x[0]
				exploration_goal.point.y=x[1]
				exploration_goal.point.z=0	

				targetspub.publish(exploration_goal)
				# points.points=[exploration_goal.point]

				pointArray.markers.append(points)
				


			pub.publish(pointArray) 
			rate.sleep()
		  	
		

	  	#rate.sleep()



#_____________________________________________________________________________

if __name__ == '__main__':
	try:
		node()
	except rospy.ROSInterruptException:
		pass
 
 
 
 
