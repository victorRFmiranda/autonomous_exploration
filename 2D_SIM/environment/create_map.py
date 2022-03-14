import bresenham
from math import sin, cos, pi,tan, atan2,log
import math
from itertools import groupby
from operator import itemgetter
import numpy as np
import cv2


class localmap:
	def __init__(self, height, width, resolution,morigin):
		self.height=height
		self.width=width
		self.resolution=resolution
		self.punknown=-1.0
		self.localmap=[self.punknown]*int(self.width/self.resolution)*int(self.height/self.resolution)
		self.logodds=[0.0]*int(self.width/self.resolution)*int(self.height/self.resolution)
		self.origin=int(math.ceil(morigin[0]/resolution))+int(math.ceil(width/resolution)*math.ceil(morigin[1]/resolution))
		self.pfree=log(0.3/0.7)
		self.pocc=log(0.9/0.1)
		self.prior=log(0.5/0.5)
		self.max_logodd=100.0
		self.max_logodd_belief=10.0
		self.max_scan_range=1.0
		self.map_origin=morigin

		self.map_increse = 0



	def updatemap(self,scandata,pose):

		angle_min = -2.3561944902
		angle_max = 2.3561944902
		angle_increment = 0.00436332312998582394230922692122153178360717972135431364024297860042752278650862360920560392408627370553076123126
		range_min = 0.06
		range_max = 20

		robot_origin=int(pose[0])+int(math.ceil(self.width/self.resolution)*pose[1])
		centreray=len(scandata)/2+1
		for i in range(len(scandata)):
			if not math.isnan(scandata[i]):
				beta=(i-centreray)*angle_increment
				px=int(float(scandata[i])*cos(beta-pose[2])/self.resolution)
				py=int(float(scandata[i])*sin(beta-pose[2])/self.resolution)

				# l = bresenham.bresenham([0,0],[px,py])
				l = list(bresenham.bresenham(0,0,px,py))
				# print(l)
				# for j in range(len(l.path)):
				for j in range(len(l)):					
					# lpx=self.map_origin[0]+pose[0]+l.path[j][0]*self.resolution
					# lpy=self.map_origin[1]+pose[1]+l.path[j][1]*self.resolution
					lpx=self.map_origin[0]+pose[0]+l[j][0]*self.resolution
					lpy=self.map_origin[1]+pose[1]+l[j][1]*self.resolution

					if (0<=lpx<self.width and 0<=lpy<self.height):
						# index=self.origin+int(l.path[j][0]+math.ceil(self.width/self.resolution)*l.path[j][1])
						index=self.origin+int(l[j][0]+math.ceil(self.width/self.resolution)*l[j][1])
						if scandata[i]<self.max_scan_range*range_max:
							# if(j<len(l.path)-1):self.logodds[index]+=self.pfree
							if(j<len(l)-1):self.logodds[index]+=self.pfree
							else:self.logodds[index]+=self.pocc
						else:self.logodds[index]+=self.pfree						
						if self.logodds[index]>self.max_logodd:self.logodds[index]=self.max_logodd
						elif self.logodds[index]<-self.max_logodd:self.logodds[index]=-self.max_logodd
						if self.logodds[index]>self.max_logodd_belief:self.localmap[index]=100
						else:self.localmap[index]=0 
						self.localmap[self.origin]=100.0

		# return np.asarray(self.localmap)

		return self.create_img()




	def create_img(self):
		self.map_increse = 0

		w = int(self.width/self.resolution)
		h = int(self.height/self.resolution)

		# image = np.zeros((w,h,3)).astype(np.uint8)
		image = np.zeros((w,h)).astype(np.uint8)
		for i in range(w):
			for j in range(h):
				if(self.localmap[i*w+j] == -1.0):
					image[i,j] = 205
				elif(self.localmap[i*w+j] == 100.0):
					self.map_increse += 1
					image[i,j] = 0
				elif(self.localmap[i*w+j] == 0.0):
					self.map_increse += 1
					image[i,j] = 255

		# image = image.transpose()
		image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

		return image, self.map_increse