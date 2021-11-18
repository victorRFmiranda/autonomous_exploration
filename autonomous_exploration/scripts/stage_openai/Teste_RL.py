#!/usr/bin/env python

import rospy
import numpy as np
import random
from ros_stage_env import StageEnvironment
from config import Config


max_epochs = 5001

args = Config().parse()
env = StageEnvironment(args)
q_table = np.zeros([10, 10, 4])

print(q_table)