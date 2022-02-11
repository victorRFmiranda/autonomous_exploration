import numpy as np
import math
import cv2


class Map:
    def __init__(self, size, resolution):
        self.ang_step = 0.00436332312998582394230922692122153178360717972135431364024297860042752278650862360920560392408627370553076123126
        self.angle_min = -2.3561944902
        self.angle_max = 2.3561944902
        self.range_min = 0.06
        self.range_max = 20
        self.angles = np.arange(self.angle_min,self.angle_max,self.ang_step)
        self.resolution = resolution
        self.size = size  # rows x cols
        # self.size = tuple(s/self.resolution for s in size)
        self.data = np.multiply(0.5, np.ones(self.size))
        self.L0 = np.log(np.divide(self.data, np.subtract(1, self.data)))
        self.L = self.L0

        self.map_increse = 0



    def inverse_scanner(self, pose, meas_r):
        alpha = 1 # Width of an obstacle (distance about measurement to fill in).
        m = np.zeros(self.size)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                # Find range and bearing relative to the input state (x, y, theta).
                r = math.sqrt((i - pose[0])**2 + (j - pose[1])**2)
                phi = (math.atan2(j - pose[1], i - pose[0]) - pose[2] + math.pi) % (2 * math.pi) - math.pi
                
                # Find the range measurement associated with the relative bearing.
                k = np.argmin(np.abs(np.subtract(phi, self.angles)))
                
                # If the range is greater than the maximum sensor range, or behind our range
                # measurement, or is outside of the field of view of the sensor, then no
                # new information is available.
                if (r > min(self.range_max, meas_r[k] + alpha / 2.0)) or (abs(phi - self.angles[k]) > self.ang_step / 2.0):
                    m[i, j] = 0.5
                
                # If the range measurement lied within this cell, it is likely to be an object.
                elif (meas_r[k] < self.range_max) and (abs(r - meas_r[k]) < alpha / 2.0):
                    m[i, j] = 0.7
                
                # If the cell is in front of the range measurement, it is likely to be empty.
                elif r < meas_r[k]:
                    m[i, j] = 0.3
                    
        return m


    def update_map(self, lidar_meas, r_pose):

        invmod = self.inverse_scanner(r_pose, lidar_meas)

        self.L = np.log(np.divide(invmod, np.subtract(1, invmod))) + self.L - self.L0

        p = np.exp(self.L)
        self.data = p / (1+p)

        return self.create_img()



    def create_img(self):
        self.map_increse = 0

        w,h = self.data.shape

        # image = np.zeros((w,h,3)).astype(np.uint8)
        image = np.zeros((w,h)).astype(np.uint8)
        for i in range(w):
            for j in range(h):
                if(self.data[i,j] == 0.5):
                    # image[i,j] = [192,192,192]
                    image[i,j] = 205
                elif(self.data[i,j] >= 0.6):
                    self.map_increse += 1
                    # image[i,j] = [0,0,0]
                    image[i,j] = 0
                elif(self.data[i,j] <= 0.4):
                    self.map_increse += 1
                    # image[i,j] = [255,255,255]
                    image[i,j] = 255

        # image = image.transpose()
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return image, self.map_increse