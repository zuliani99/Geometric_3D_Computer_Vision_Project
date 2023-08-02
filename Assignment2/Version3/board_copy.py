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


def check_mask(approx_cnt, mask, tracked_features):
	index_to_correct = None
	to_return = np.zeros((0,2), dtype=np.int32)
	for cnt in approx_cnt:
		x, y = cnt[0]
		if mask[y, x] == 0: 
			to_return = np.vstack((to_return, np.array([x, y])))
		elif index_to_correct is None:
			distances = np.linalg.norm(tracked_features - np.array([x, y]), axis=-1)
			index_to_correct = np.unravel_index(np.argmin(distances), distances.shape)[0]
			#index_to_correct = np.where((tracked_features[:, :, 0] == x) & (tracked_features[:, :, 1] == y))
			#print(index_to_correct)
	return index_to_correct, to_return


def find_interesting_points(imgray, mask, tracked_features = None):
	if(type(tracked_features) != NoneType):
		# get the empty polygon index
		empty_space = np.squeeze(np.where(np.all(tracked_features == np.zeros((5,2),  dtype=np.float32), axis=(1, 2))))
		#print(empty_space)
		#return
  
	_, thresh = cv.threshold(imgray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
	
	# Consider only the board exluding all the object area that could be included erroneously
	mask_thresh = np.zeros((1080, 1920), dtype=np.uint8)
	mask_thresh[:, 1050:1600] = thresh[:, 1050:1600]

	# Finding the contourns
	contours, _ = cv.findContours(mask_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # [[[X Y]] [[X Y]] ... [[X Y]]]

	points_of_interests = np.zeros( (0,5,2), dtype=np.float32 )
  
	# Searching through every region selected to find the required polygon.
	for cnt in contours:
		#print(cnt.shape)
		# Shortlisting the regions based on there area.
		if cv.contourArea(cnt) > 1720:
				
			approx_cnt = cv.approxPolyDP(cnt, 0.02 * cv.arcLength(cnt, True), True) # [[[X Y]] [[X Y]] ... [[X Y]]]
				
			# Checking if the number of sides of the selected region is 5.
			if (len(approx_cnt)) == 5:
				index_to_correct, confermed_cnt = check_mask(approx_cnt, mask, tracked_features)
    
				#confermed_cnt = np.expand_dims(check_mask(approx_cnt, mask),axis=0)
				if(confermed_cnt.shape[0] > 0):
					ref_approx_cnt = cv.cornerSubPix(imgray, np.float32(confermed_cnt), winSize_sub, zeroZone_sub, criteria_sub)
					#print(ref_approx_cnt.shape[0])
					if(ref_approx_cnt.shape[0] == 5):
						# new polygon found
						#print('found new polygon')
						if(type(tracked_features) != NoneType):
							tracked_features[empty_space] = ref_approx_cnt
						else:
							# I have to insert in the [0,0] of traked_fgeatures in the correct indwex 
							points_of_interests = np.vstack((points_of_interests, np.expand_dims(ref_approx_cnt, axis=0)))
							#print(points_of_interests.shape)
					else:
						#print('correcting points')
						#if tracked_features != None: pass
      					# find the correcmt 
						#print(index_to_correct, tracked_features[index_to_correct])
						zero_pos = np.where(np.all(tracked_features[index_to_correct] == np.array([0.,0.], dtype=np.float32), axis=1))
						#print('zero_pos', zero_pos)
						#print(zero_pos, ref_approx_cnt.shape[0])
						for pos, coords in zip (zero_pos[0], ref_approx_cnt):
							#print(pos, coords)
							tracked_features[index_to_correct][pos] = coords
	#print('exit')
	return tracked_features if(type(tracked_features) != NoneType) else points_of_interests