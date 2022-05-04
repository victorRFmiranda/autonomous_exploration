#--------Include modules---------------
from copy import copy

import numpy as np
import cv2

#-----------------------------------------------------

def getfrontier(mapData):
	w, h = mapData.shape
	resolution=1.0
	Xstartx=0.0
	Xstarty=0.0
	 
	img = mapData
	
	o=cv2.inRange(img,0,1)
	print(o)
	edges = cv2.Canny(img,0,255)
	contours, hierarchy = cv2.findContours(o,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(o, contours, -1, (255,255,255), 5)
	o=cv2.bitwise_not(o) 
	res = cv2.bitwise_and(o,edges)
	#------------------------------

	frontier=copy(res)
	contours, hierarchy = cv2.findContours(frontier,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(frontier, contours, -1, (255,255,255), 2)
	# cv2.drawContours(frontier, contours, -1, (255,255,255), 4)

	contours, hierarchy = cv2.findContours(frontier,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	all_pts=[]
	if len(contours)>0:
		upto=len(contours)-1
		i=0
		maxx=0
		maxind=0
		
		for i in range(0,len(contours)):
				cnt = contours[i]
				M = cv2.moments(cnt)
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])
				xr=cx*resolution+Xstartx
				yr=cy*resolution+Xstarty
				pt=[np.array([xr,yr])]
				if len(all_pts)>0:
					all_pts=np.vstack([all_pts,pt])
				else:
							
					all_pts=pt
	
	return all_pts


