import cv2 as cv
import numpy as np


# Set the needed parameters to find the refined corners
winSize_sub = (5, 5)
zeroZone_sub = (-1, -1)
criteria_sub = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)


def sort_vertices_clockwise(vertices, x_centroid=None, y_centroid=None):
    new_centroid = np.mean(vertices, axis=0)
    if x_centroid is None and y_centroid is None:
        centroid = new_centroid
    elif x_centroid is None:
        centroid = [new_centroid[0], y_centroid]
    elif y_centroid is None:
        centroid = [x_centroid, new_centroid[1]]
    else:
        centroid = [x_centroid, y_centroid]
        
    angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    return centroid, vertices[sorted_indices]



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
	mask_thresh[:, 1130:1600] = thresh[:, 1130:1600]

	# Finding the contourns
	contours, _ = cv.findContours(mask_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # [[[X Y]] [[X Y]] ... [[X Y]]]

	points_of_interests = np.zeros( (0,2), dtype=np.float32 )
  
	# Searching through every region selected to find the required polygon.
	for cnt in contours:
		# Shortlisting the regions based on there area.
		#sorted_vertex = cnt
		_, sorted_vertex = sort_vertices_clockwise(np.squeeze(cnt, axis=1))
		if cv.contourArea(sorted_vertex) > 1625.0: #1685, 1650 ok 1625
														
			approx_cnt = cv.approxPolyDP(cnt, 0.015 * cv.arcLength(cnt, True), True) # [[[X Y]] [[X Y]] ... [[X Y]]]
				
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