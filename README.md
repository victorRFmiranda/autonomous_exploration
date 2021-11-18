# Deep Reinforcement Learning for Autnomous exploration

DRL training algorithm for selection of the best exploration frontier in under discover environment.


## Installation

### ROS Dependencies
```
$ sudo apt install ros-"your ros distro"-gmapping
$ sudo apt install ros-"your ros distro"-map-server
```

### Python Dependencies
```
Numpy
Pytorch
```


### Building Package

Install this package and the modified version of ROS Stage simulator on your catking workspace:

```
$ cd ~/catkin_ws/src/
$ git clone https://github.com/victorRFmiranda/autonomous_exploration.git
$ git clone https://github.com/victorRFmiranda/stage_ros.git
$ cd ..
$ catkin_make ## or catkin build
```

## Main CODE

The main code is located in:
`/scripts/stage_openai/AC_TD_lambda_forward.py`

Other auxiliary codes are:

Path Planner: `/scripts/stage_openai/a_star.py`

DRL Configuration File: `/scripts/stage_openai/config.py`

Training Environment: `/scripts/stage_openai/ros_stage_env.py`

Frontier Detection Algorithm: `/scripts/frontier/frontier_lidar.py`

## How to run

In order to start the training, run the following codes in two different terminal windows:

First:
```
$ roscore
```

Second:
```
$ rosrun autonomous_exploration AC_TD_lambda_forward.py
```
