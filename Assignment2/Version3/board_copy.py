#from utils import find_distance_between_points, find_middle_point, compute_index_and_cc_coords
from types import NoneType
from polygon import Polygon

from typing import Dict, List, Tuple, Union
import cv2 as cv
import numpy as np


# Set the needed parameters to find the refined corners
winSize_sub = (5, 5)
zeroZone_sub = (-1, -1)
criteria_sub = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)


def check_mask(approx_cnt, mask, lose_tracked_poly_vertices):
	to_return = np.zeros((0,3), dtype=np.int32)
	for cnt in approx_cnt:
		x, y = cnt[0]
		if mask[y, x] == 0: 
			nearest_index = 0
			if type(lose_tracked_poly_vertices) != NoneType:
				distances = np.linalg.norm(lose_tracked_poly_vertices - np.array([x, y]), axis=-1)
				nearest_index = np.unravel_index(np.argmin(distances), distances.shape)[0]
			to_return = np.vstack((to_return, np.array([x, y, nearest_index]))) # x and y coordinates and index of the polygon where it must be added
	return to_return


def find_interesting_points(imgray, mask, tracked_poly_vertices=None, lose_tracked_poly_vertices=None):
	_, thresh = cv.threshold(imgray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
	
	# Consider only the board exluding all the object area that could be included erroneously
	mask_thresh = np.zeros((1080, 1920), dtype=np.uint8)
	mask_thresh[:, 1050:1600] = thresh[:, 1050:1600]

	# Finding the contourns
	contours, _ = cv.findContours(mask_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # [[[X Y]] [[X Y]] ... [[X Y]]]

	new_points_of_interests = np.zeros((0,5,2), dtype=np.float32) # this array will store only the new vertices polygon found
  
	# Searching through every region selected to find the required polygon.
	for cnt in contours:
		
		# Shortlisting the regions based on there area.
		if cv.contourArea(cnt) > 1720:
				
			approx_cnt = cv.approxPolyDP(cnt, 0.02 * cv.arcLength(cnt, True), True) # [[[X Y]] [[X Y]] ... [[X Y]]]
				
			# Checking if the number of sides of the selected region is 5.
			if (len(approx_cnt)) == 5:
     
				confermed_cnt = check_mask(approx_cnt, mask, lose_tracked_poly_vertices)

				if(confermed_cnt.shape[0] > 0):
					ref_approx_cnt = cv.cornerSubPix(imgray, np.float32(confermed_cnt[:, :2]), winSize_sub, zeroZone_sub, criteria_sub)
					if(confermed_cnt.shape[0] == 5):
						new_points_of_interests = np.vstack((new_points_of_interests, np.expand_dims(ref_approx_cnt, axis=0)))
					else:
						for x, y, idx_poly in confermed_cnt:
							first_zero_pos = np.where(tracked_poly_vertices[idx_poly] == np.zeros((1,2)[0], dtype=np.float32))[0][:-1]
							tracked_poly_vertices[idx_poly][first_zero_pos] = np.array([x, y], dtype=np.float32)
	return tracked_poly_vertices, new_points_of_interests


