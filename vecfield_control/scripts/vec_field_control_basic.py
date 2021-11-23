#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Basic node to perform navigation control using topics (publishers subscribers)
    Authors:
        Adriano M. C. Rezende, <adrianomcr18@gmail.com>
        Hector Azpurua <hector.azpurua@itv.org>
	Victor R. F. Miranda <victormrfm@ufmg.br>
"""

import rospy

from geometry_msgs.msg import Twist
from std_msgs.msg import Int32
import vecfield_control.msg

from vector_field_control import rviz_helper, vec_field_node


class VecFieldNodeBasic(vec_field_node.VecFieldNode):
    """Basic implementation of the vec field controller node, using only publishers and subscribers"""

    def __init__(self):
        """Extends the VecFieldNode constructor to add a trajectory callback
        """
        super(VecFieldNodeBasic, self).__init__()
        rospy.loginfo("Initializing VecFieldNodeBasic...")

        rospy.Subscriber("/traj_points", vecfield_control.msg.Path,
                         super(VecFieldNodeBasic, self).callback_trajectory)

    def run(self):
        """Execute the controller loop
        """
        vel = Twist()
        rate = rospy.Rate(self.freq)
        rate_slow = rospy.Rate(self.freq_slow)
        reached_msg = Int32()

        prev_percentage = 0

        while not rospy.is_shutdown():
            #reached_msg.data = 0
            if not self.vec_field_obj.is_ready():
                # if the control algorithm is not ready to perform the path
                # then command the robot to stop, only once
                if vel.linear.x != 0.0 or vel.angular.z != 0.0:
                    vel.linear.x = 0.0
                    vel.angular.z = 0.0
                    self.pub_cmd_vel.publish(vel)

                rate_slow.sleep()

                # Publishing if reached the endpoint
                reached_msg.data = 0
                self.pub_reachend.publish(reached_msg)
                continue

            linear_vel_x, angular_vel_z, Vx_ref, Vy_ref, reached_endpoint, reached_percentage = \
                self.vec_field_obj.run_one_cycle()

            if int(reached_percentage) != int(prev_percentage) and (int(reached_percentage) % 5) == 0:
                prev_percentage = reached_percentage
                rospy.loginfo("goal reached?:%s, (%d%%)", reached_endpoint, reached_percentage)


            # Publishing if reached the endpoint
            reached_msg.data = int(reached_endpoint)
            self.pub_reachend.publish(reached_msg)

 

            vel.linear.x = linear_vel_x
            vel.angular.z = angular_vel_z

            self.pub_cmd_vel.publish(vel)
            rviz_helper.send_marker_to_rviz(Vx_ref, Vy_ref, self.pos, self.pub_rviz_ref)

            rate.sleep()


if __name__ == '__main__':
    vec_control_basic = VecFieldNodeBasic()
    vec_control_basic.run()

