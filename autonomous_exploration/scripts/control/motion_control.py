#!/usr/bin/env python
import rospy
import rospkg

# ros-msgs
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TwistStamped, Twist

import sys



###############################################################
'''           Convert Vel to Differential Wheels           '''
###############################################################
# def differential_drive(v,w,r = 0.116855, L = 1.2):
def differential_drive(v,w,r = 0.0975, L = 0.2075):
	# default is Coppelia Summit XL params
	vel_right = ((2 * v) + (w * L)) / (2 * r)
	vel_left = ((2 * v) - (w * L)) / (2 * r)

	wr = JointState()
	wr.header.stamp = rospy.Time.now()
	wr.velocity = [vel_right]
	wl = JointState()
	wl.header.stamp = rospy.Time.now()
	wl.velocity = [vel_left]

	return wr, wl




########################################
'''            Callbacks             '''
########################################
def callback_vel(msg):
    global cmd_v, cmd_w, ID

    cmd_v = msg.linear.x
    cmd_w = msg.angular.z

    # if(msg.header.frame_id == str(ID)):
        # cmd_v = msg.twist.linear.x
        # cmd_w = msg.twist.angular.z





########################################
'''           Main Function          '''
########################################
def run():
    global cmd_v, cmd_w, ID


	## ROS STUFFS
    rospy.init_node("pioneer_control_"+str(ID), anonymous=True)
    pub_wheel1_vel = rospy.Publisher("/joint1_"+str(ID), JointState, queue_size=1)
    pub_wheel2_vel = rospy.Publisher("/joint2_"+str(ID), JointState, queue_size=1)
    # pub_wheel3_vel = rospy.Publisher("/joint3", JointState, queue_size=1)
    # pub_wheel4_vel = rospy.Publisher("/joint4", JointState, queue_size=1)


    rospy.Subscriber('/cmd_vel_'+str(ID), Twist, callback_vel)

    rate = rospy.Rate(10)

    # define velocities
    cmd_v = 0.0
    cmd_w = 0.0
    wr = JointState()
    wl = JointState()

    stop = JointState()
    stop.velocity = [0.0]

    while not rospy.is_shutdown():
        if(cmd_v == 0 and cmd_w == 0):
            wr.velocity = [0.0]
            wl.velocity = [0.0]
        else:
    	   # Computes wheel velocitise
    	   wr, wl = differential_drive(cmd_v,cmd_w)

        wr.header.stamp = rospy.get_rostime()
        wr.header.frame_id = str(ID)
        wl.header.stamp = rospy.get_rostime()
        wl.header.frame_id = str(ID)

    	pub_wheel1_vel.publish(wl)
    	pub_wheel2_vel.publish(wr)
    	# pub_wheel3_vel.publish(wr)
    	# pub_wheel4_vel.publish(wr)

        

    	rate.sleep()



# Vehicle ID
ID = 0

########################################
'''            Main Routine          '''
########################################
if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print("Informe o ID do robo como argumento!")
    else:
        ID = int(sys.argv[1])
        try:
            run()
        except rospy.ROSInterruptException:
            pass
