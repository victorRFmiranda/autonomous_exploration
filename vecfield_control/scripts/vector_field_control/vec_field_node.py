#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Navigation control using Action Server
    Authors:
        Adriano M. C. Rezende, <adrianomcr18@gmail.com>
        Hector Azpurua <hector.azpurua@itv.org>
"""

import rospy
from geometry_msgs.msg import Twist, Pose, Point
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray
from math import cos, sin
from std_msgs.msg import Int32

import rviz_helper
import vec_field_controller


class VecFieldNode(object):
    """
    Navigation control using Action Server
    """

    def __init__(self):
        self.freq = 20.0  # Frequency of field computation in Hz
        self.freq_slow = 1.0

        self.pos = [0, 0, 0]  # Robot position and orientation
        self.rpy = [0, 0, 0]

        self.is_forward_motion = True
        self.flag_follow_obstacle = False

        # names and type of topics
        self.pose_topic_name = None
        self.pose_topic_type = None
        self.cmd_vel_topic_name = None
        self.obstacle_point_topic_name = None

        # potential field variables
        self.epsilon = 0.0
        self.switch_dist = 0.0

        # obtain the parameters
        self.v_r = 0.0
        self.k_f = 0.0
        self.d_feedback = 0.0

        # publishers
        self.pub_cmd_vel = None
        self.pub_rviz_ref = None
        self.pub_rviz_curve = None

        self.init_node()

        self.vec_field_obj = vec_field_controller.VecFieldController(self.v_r, self.k_f, self.d_feedback, self.epsilon,
                                                                     self.switch_dist, self.is_forward_motion,
                                                                     self.flag_follow_obstacle)

    def init_node(self):
        """Initialize ROS related variables, parameters and callbacks
        :return:
        """
        rospy.init_node("vec_field_node")

        # parameters (description in yaml file)
        self.v_r = float(rospy.get_param("/espeleo_control/vector_field/v_r", 1.0))
        self.k_f = float(rospy.get_param("/espeleo_control/vector_field/k_f", 5.0))
        self.is_forward_motion = rospy.get_param("/espeleo_control/vector_field/is_forward_motion", True)
        self.d_feedback = float(rospy.get_param("/espeleo_control/feedback_linearization/d_feedback", 0.2))

        self.pose_topic_name = rospy.get_param("/espeleo_control/robot_pose/pose_topic_name", "tf")
        self.pose_topic_type = rospy.get_param("/espeleo_control/robot_pose/pose_topic_type", "TFMessage")
        self.cmd_vel_topic_name = rospy.get_param("/espeleo_control/robot_cmd/cmd_vel_topic_name", "cmd_vel")

        self.flag_follow_obstacle = rospy.get_param("/espeleo_control/obstacle_avoidance/flag_follow_obstacle", False)
        self.epsilon = rospy.get_param("/espeleo_control/obstacle_avoidance/epsilon", 0.5)
        self.switch_dist = rospy.get_param("/espeleo_control/obstacle_avoidance/switch_dist", 1.0)
        self.obstacle_point_topic_name = rospy.get_param("/espeleo_control/obstacle_avoidance/obstacle_point_topic_name",
                                                         "/closest_obstacle_point")

        # publishers
        self.pub_end = rospy.Publisher("/reached_endpoint", Int32, queue_size=1)
        self.pub_cmd_vel = rospy.Publisher(self.cmd_vel_topic_name, Twist, queue_size=1)
        self.pub_rviz_ref = rospy.Publisher("/visualization_ref_vel", Marker, queue_size=1)
        self.pub_rviz_curve = rospy.Publisher("/visualization_trajectory", MarkerArray, queue_size=1)

        # subscribers
        rospy.Subscriber(self.obstacle_point_topic_name, Point, self.obstacle_point_cb)
        # rospy.Subscriber("/point_cloud_converter/clost_point", Point, self.obstacle_point_cb)

        if self.pose_topic_type == "TFMessage":
            rospy.Subscriber(self.pose_topic_name, TFMessage, self.tf_cb)
        elif self.pose_topic_type == "Pose":
            rospy.Subscriber(self.pose_topic_name, Pose, self.pose_cb)
        elif self.pose_topic_type == "Odometry":
            rospy.Subscriber(self.pose_topic_name, Odometry, self.odometry_cb)
        else:
            raise AssertionError("Invalid value for pose_topic_type:%s".format(self.pose_topic_type))

        # timers
        rospy.Timer(rospy.Duration(2), self.publish_trajectory_cb)

        rospy.loginfo("Vector field control configured:")
        rospy.loginfo("v_r: %s, kf:%s, d:%s",
                      self.v_r, self.k_f, self.d_feedback)
        rospy.loginfo("is_forward_motion:%s",
                      self.is_forward_motion)
        rospy.loginfo("pose_topic_name:%s, pose_topic_type:%s, cmd_vel_topic_name:%s",
                      self.pose_topic_name, self.pose_topic_type, self.cmd_vel_topic_name)
        rospy.loginfo("flag_follow_obstacle:%s",
                      self.flag_follow_obstacle)
        rospy.loginfo("obstacle_point_topic_name:%s", self.obstacle_point_topic_name)
        rospy.loginfo("flag_follow_obstacle:%s, epsilon:%s, switch_dist:%s",
                      self.flag_follow_obstacle, self.epsilon, self.switch_dist)

    def publish_trajectory_cb(self, event):
        """Publish the curve being followed at an interval
        """
        traj = self.vec_field_obj.get_traj()
        if traj and len(traj) > 0:
            rviz_helper.send_marker_array_to_rviz(traj, self.pub_rviz_curve)

    def tf_cb(self, data, frame_id="os1_imu_odom"):
        """Callback function to get the pose of the robot via a TF message
        :param frame_id: frame id to publish the marker
        :param data: tf ROS message
        """
        for T in data.transforms:
            if T.child_frame_id == frame_id:
                pos = (T.transform.translation.x, T.transform.translation.y, T.transform.translation.z)

                x_q = T.transform.rotation.x
                y_q = T.transform.rotation.y
                z_q = T.transform.rotation.z
                w_q = T.transform.rotation.w
                rpy = euler_from_quaternion([x_q, y_q, z_q, w_q])

                self.vec_field_obj.set_pos(pos, rpy)
                self.pos = pos
                self.rpy = rpy
                break

    def pose_cb(self, data):
        """Callback to get the pose of the robot
        :param data: pose ROS message
        """
        pos = (data.position.x, data.position.y, data.position.z)

        x_q = data.orientation.x
        y_q = data.orientation.y
        z_q = data.orientation.z
        w_q = data.orientation.w
        rpy = euler_from_quaternion([x_q, y_q, z_q, w_q])

        self.vec_field_obj.set_pos(pos, rpy)
        self.pos = pos
        self.rpy = rpy

    def odometry_cb(self, data):
        """Callback to get the pose from odometry data
        :param data: odometry ROS message
        """
        pos = (data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z)

        x_q = data.pose.pose.orientation.x
        y_q = data.pose.pose.orientation.y
        z_q = data.pose.pose.orientation.z
        w_q = data.pose.pose.orientation.w
        rpy = euler_from_quaternion([x_q, y_q, z_q, w_q])

        #Consider the position of the control point, instead of the robot's center
        pos = (pos[0] + self.d_feedback*cos(rpy[2]), pos[1] + self.d_feedback*sin(rpy[2]), pos[2])


        self.vec_field_obj.set_pos(pos, rpy)
        self.pos = pos
        self.rpy = rpy

    def obstacle_point_cb(self, data):
        """Callback to get the closest point obtained with the lidar
        used for obstacle avoidance
        :param data: point message
        """
        self.vec_field_obj.set_obstacle_point((data.x, data.y))

    def callback_trajectory(self, data):
        """Callback to obtain the trajectory to be followed by the robot
        :param data: trajectory ROS message
        """

        traj_points = []
        for k in range(len(data.path.points)):
            p = data.path.points[k]
            traj_points.append((p.x, p.y))

        rospy.loginfo("New path received (%d points) is closed?:%s", len(traj_points), data.closed_path_flag)

        self.vec_field_obj.set_trajectory(traj_points, data.insert_n_points, data.filter_path_n_average,
                                          data.closed_path_flag)
