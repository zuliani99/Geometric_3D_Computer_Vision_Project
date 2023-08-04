#from utils import find_distance_between_points, find_middle_point, compute_index_and_cc_coords
from polygon import Polygon

from typing import Dict, List, Tuple, Union
import cv2 as cv
import numpy as np


# Set the needed parameters to find the refined corners
winSize_sub = (5, 5)
zeroZone_sub = (-1, -1)
criteria_sub = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)


def check_mask(approx_cnt, mask):
	to_return = np.zeros((0,2), dtype=np.int32)
	for cnt in approx_cnt:
		x, y = cnt[0]
		if mask[y, x] == 0: 
			to_return = np.vstack((to_return, np.array([x, y])))
	return to_return


def find_interesting_points(imgray, mask):
	_, thresh = cv.threshold(imgray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
	
	# Consider only the board exluding all the object area that could be included erroneously
	mask_thresh = np.zeros((1080, 1920), dtype=np.uint8)
	mask_thresh[:, 1050:1600] = thresh[:, 1050:1600]

	# Finding the contourns
	contours, _ = cv.findContours(mask_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # [[[X Y]] [[X Y]] ... [[X Y]]]

	points_of_interests = np.zeros( (0,2), dtype=np.float32 )
  
	# Searching through every region selected to find the required polygon.
	for cnt in contours:
		
		# Shortlisting the regions based on there area.
		if cv.contourArea(cnt) > 1625: #1685, 1650 ok 1625
				
			approx_cnt = cv.approxPolyDP(cnt, 0.02 * cv.arcLength(cnt, True), True) # [[[X Y]] [[X Y]] ... [[X Y]]]
				
			# Checking if the number of sides of the selected region is 5.
			if (len(approx_cnt)) == 5:
				
				if(mask.shape[0] > 0):
					confermed_cnt = check_mask(approx_cnt, mask)
						
					if(confermed_cnt.shape[0] > 0):
						ref_approx_cnt = cv.cornerSubPix(imgray, np.float32(confermed_cnt), winSize_sub, zeroZone_sub, criteria_sub)
						points_of_interests = np.vstack((points_of_interests, np.squeeze(ref_approx_cnt)))
				else:
					ref_approx_cnt = cv.cornerSubPix(imgray, np.float32(approx_cnt), winSize_sub, zeroZone_sub, criteria_sub)
					points_of_interests = np.vstack((points_of_interests, np.squeeze(ref_approx_cnt)))
	return points_of_interests