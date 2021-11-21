#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Navigation control using Action Server
    Authors:
        Adriano M. C. Rezende, <adrianomcr18@gmail.com>
        Hector Azpurua <hector.azpurua@itv.org>
	Victor R. F. Miranda <victormrfm@ufmg.br>
"""

import actionlib
import rospy

import vecfield_control.msg
from geometry_msgs.msg import Twist
from vector_field_control import rviz_helper, vec_field_node


class VecFieldNodeAction(vec_field_node.VecFieldNode):
    """
    Action server interface for the navigation control
    """

    def __init__(self):
        """Extends the VecFieldNode constructor to add action server binds
        """
        super(VecFieldNodeAction, self).__init__()
        rospy.loginfo("Initializing VecFieldNodeAction...")

        # action server
        self.action_feedback = vecfield_control.msg.NavigatePathFeedback()
        self.action_result = vecfield_control.msg.NavigatePathResult()
        self.action_srv = actionlib.SimpleActionServer("espeleo_control_action",
                                                       vecfield_control.msg.NavigatePathAction,
                                                       execute_cb=self.execute_action_cb,
                                                       auto_start=False)
        self.action_srv.start()

    def execute_action_cb(self, goal):
        """Callback to obtain the trajectory to be followed by the robot
        :param goal: NavigatePath action ROS message
        """
        data = goal.path

        traj_points = []
        for k in range(len(data.path.points)):
            p = data.path.points[k]
            traj_points.append((p.x, p.y))

        rospy.loginfo("New path received (%d points)", len(traj_points))
        rospy.loginfo("Init path action")

        self.vec_field_obj.set_trajectory(traj_points, data.insert_n_points, data.filter_path_n_average,
                                          data.closed_path_flag)

        run_result = self.run()
        self.action_result.goal_reached = run_result
        self.action_srv.set_succeeded(self.action_result)

    def run(self, timeout=rospy.Duration(360)):
        """Execute the controller loop
        :return:
        """
        vel = Twist()
        rate = rospy.Rate(self.freq)
        rate_slow = rospy.Rate(self.freq_slow)
        timeout_time = rospy.get_rostime() + timeout

        while not rospy.is_shutdown():
            if timeout != rospy.Duration(0.0) and rospy.get_rostime() >= timeout_time:
                rospy.logerr('Timeout executing the path (%s)', timeout)
                break

            if self.action_srv.is_active() and self.action_srv.is_preempt_requested():
                rospy.loginfo('Action Preempted! Canceling this path execution...')
                self.vec_field_obj.reset()
                self.action_result.goal_reached = False
                self.action_srv.set_preempted(result=self.action_result)
                #self.action_srv.set_preempted()
                break

            if not self.vec_field_obj.is_ready():
                # if the control algorithm is not ready to perform the path
                # then command the robot to stop, only once
                if vel.linear.x != 0.0 or vel.angular.z != 0.0:
                    vel.linear.x = 0.0
                    vel.angular.z = 0.0
                    self.pub_cmd_vel.publish(vel)

                rate_slow.sleep()
                continue

            linear_vel_x, angular_vel_z, Vx_ref, Vy_ref, reached_endpoint, reached_percentage = \
                self.vec_field_obj.run_one_cycle()

            #rospy.loginfo("reached_endpoint:%s, reached_percentage:%d", reached_endpoint, reached_percentage)

            vel.linear.x = linear_vel_x
            vel.angular.z = angular_vel_z

            self.pub_cmd_vel.publish(vel)
            rviz_helper.send_marker_to_rviz(Vx_ref, Vy_ref, self.pos, self.pub_rviz_ref)

            # action server messages
            self.action_feedback.percent_complete = reached_percentage
            self.action_srv.publish_feedback(self.action_feedback)

            if reached_endpoint:
                return True

            rate.sleep()

        return False


if __name__ == '__main__':
    vec_control_action = VecFieldNodeAction()
    rospy.spin()
