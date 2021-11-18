#!/usr/bin/env python

import rospy
import math
import tf.transformations
from visualization_msgs.msg import Marker, MarkerArray


def send_marker_array_to_rviz(traj, pub_marker_array):
    """Function to send a array of markers, representing the curve, to rviz
    :param traj: trajectory list of points
    :param pub_rviz: ROS publisher object
    :return:
    """

    if not pub_marker_array:
        raise AssertionError("pub_marker_array is not valid:%s".format(pub_marker_array))

    points_marker = MarkerArray()
    for i in range(len(traj)):
        marker = Marker()
        marker.header.frame_id = "/os1_init"
        marker.header.stamp = rospy.Time.now()
        marker.id = i
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        # Size of sphere
        marker.scale.x = 0.06
        marker.scale.y = 0.06
        marker.scale.z = 0.06
        # Color and transparency
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        # Pose
        marker.pose.orientation.w = 1.0

        px, py = traj[i]

        marker.pose.position.x = px
        marker.pose.position.y = py
        marker.pose.position.z = 0.1

        # Append marker to array
        points_marker.markers.append(marker)

    # Publish marker array
    pub_marker_array.publish(points_marker)#


def send_marker_to_rviz(Vx, Vy, pos, pub_marker, frame_id="/initial_base"):
    """Function to send a markers, representing the value of the field
    :param Vx:
    :param Vy:
    :param pub_rviz: ROS publisher object
    :return:
    """
    marker = Marker()

    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.id = 0
    marker.type = marker.ARROW
    marker.action = marker.ADD
    marker.scale.x = 1.5 * (Vy ** 2 + Vx ** 2) ** (0.5)
    marker.scale.y = 0.08
    marker.scale.z = 0.08
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 0.0
    marker.color.b = 0.0

    # Position of the marker
    marker.pose.position.x = pos[0]
    marker.pose.position.y = pos[1]
    marker.pose.position.z = pos[2]

    # Orientation of the marker
    quaternio = tf.transformations.quaternion_from_euler(0, 0, math.atan2(Vy, Vx))
    marker.pose.orientation.x = quaternio[0]
    marker.pose.orientation.y = quaternio[1]
    marker.pose.orientation.z = quaternio[2]
    marker.pose.orientation.w = quaternio[3]

    # Publish marker
    pub_marker.publish(marker)
