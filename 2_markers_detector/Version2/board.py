from utils import find_distance_between_points, find_middle_point, compute_index_and_cc_coords, sort_vertices_clockwise
from polygon import Polygon

from typing import Dict, List, Tuple, Union
import cv2 as cv
import numpy as np


# Set the needed parameters to find the refined corners
winSize = (5, 5)
zeroZone = (-1, -1)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)



class Board:
    
	def __init__(self, n_polygons: int) -> None:
		self.polygon_list: List[Polygon] = [Polygon() for _ in range(n_polygons)]
    
    
    
	def draw_red_polygon(self, image: np.ndarray[int, np.uint8]) ->  np.ndarray[int, np.uint8]:
		'''
		PURPOSE: draw the red polygon, the cross in point A and the line crossing the polygon by length
		ARGUMENTS: 
			- image (np.ndarray[int, np.uint8]): image to edit
		RETURN:
			- (np.ndarray[int, np.uint8]): resuting image
		'''	
        
		for poly in self.polygon_list:
			if poly.cover == False:
				cv.drawContours(image, [poly.vertex_coords], 0, (0, 0, 255), 2, cv.LINE_AA)

				cv.drawMarker(image, poly.point_A, (0,255,0), cv.MARKER_CROSS, 20, 1, cv.LINE_AA)

				cv.line(image, poly.point_A, np.int32(poly.middle_point), (0, 255, 255), 1, cv.LINE_AA) 
		return image
            
       
            
	def draw_green_cross_and_blu_rectangle(self, image: np.ndarray[int, np.uint8]) ->  np.ndarray[int, np.uint8]:
		'''
		PURPOSE: draw a green cross and a blu rectangle in each circe centre 
		ARGUMENTS: 
			- image (np.ndarray[int, np.uint8]): image to edit
		RETURN:
			- (np.ndarray[int, np.uint8]): resuting image
		'''	
        
		for poly in self.polygon_list:
			if poly.cover == False:
				for idx, coords in enumerate(reversed(poly.circles_ctr_coords), start=1): 
					start = np.int32(coords[1])
					end = np.int32(coords[0])
					cv.drawMarker(image, (start, end), (0,255,0), cv.MARKER_CROSS, 10 * idx, 1, cv.LINE_AA)
					cv.drawMarker(image, (start, end), (255,0,0), cv.MARKER_SQUARE, 10, 1, cv.LINE_AA)

		return image		
  
  
  
	def draw_index(self, image: np.ndarray[int, np.uint8]) ->  np.ndarray[int, np.uint8]:
		'''
		PURPOSE: draw the polygon index
		ARGUMENTS: 
			- image (np.ndarray[int, np.uint8]): image to edit
		RETURN:
			- (np.ndarray[int, np.uint8]): resuting image
		'''	
     		
		for index, poly in enumerate(self.polygon_list):
			if poly.cover == False:
				cv.putText(image, str(index), poly.point_A, cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 8, cv.LINE_AA)
				cv.putText(image, str(index), poly.point_A, cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA)
		return image
  
  
  
	def draw_stuff(self, image: np.ndarray[int, np.uint8]) ->  np.ndarray[int, np.uint8]:
		'''
		PURPOSE: apply all the drawing function
		ARGUMENTS: 
			- image (np.ndarray[int, np.uint8]): image to edit
		RETURN:
			- (np.ndarray[int, np.uint8]): resuting image
		'''	
  
		return self.draw_index(self.draw_green_cross_and_blu_rectangle(self.draw_red_polygon(image)))



	def covered_polygon(self, polygons: np.ndarray[int, np.int32]) -> None:
		'''
		PURPOSE: apply all the drawing function
		ARGUMENTS: 
			- polygons (np.ndarray[int, np.int32]): array of index that express the covered polygons
		RETURN: None
		'''	
     
		for id_poly in polygons: self.polygon_list[id_poly].cover = True



	def find_markers(self, image: np.ndarray[int, np.uint8], frame_cnt: int, marker_reference: Dict[int, Tuple[int, int, int]]) \
			-> List[Dict[str, Union[int, int, np.float64, np.float64, int, int, int]]]:
		'''
		PURPOSE: computing the polygon index and the circles centre coordinates of a polygon
		ARGUMENTS: 
			- image (np.ndarray[int, np.uint8]): video frame
			- frame_cnt (int): index frmae 
			- marker_reference (Dict[int, Tuple[int, int, int]])): dictionary of the marker reference coordinates
		RETURN:
			- dict_stats_to_return (List[Dict[int, int, np.float64, np.float64, int, int, int]]): lis of dictionary containing the information to save in the .csv file
		'''	
		
		imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
		_, thresh = cv.threshold(imgray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
	
		# Consider only the board exluding all the object area that could be included erroneously
		mask_thresh = np.zeros((1080, 1920), dtype=np.uint8)
		mask_thresh[:, 1050:1600] = thresh[:, 1050:1600]

		# Finding the contourns
		contours, _ = cv.findContours(mask_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # [[[X Y]] [[X Y]] ... [[X Y]]]
	
		dict_stats_to_return = []
	
		# np.array of ones in which at the end of the computation will store only the covered polygons
		covered_polys = np.ones((1, 24))[0]

		# Searching through every region selected to find the required polygon.
		for cnt in contours:
   
			# Shortlisting the regions based on there area.
			if cv.contourArea(cnt) > 1650.0:  # previously was 1720
				
				approx_cnt = cv.approxPolyDP(cnt, 0.02 * cv.arcLength(cnt, True), True) # [[[X Y]] [[X Y]] ... [[X Y]]]
				
				# Checking if the number of sides of the selected region is 5.
				if (len(approx_cnt) == 5): 
					
					# Obtain the external point distance between the approximated board centre (1350,570) and each approximated polygon vertex
					external_points_dict = dict(enumerate(
						list(map(lambda x: find_distance_between_points(x[0], np.array([1350,570])), approx_cnt))
					))

					# Obtain the id of the two farthest point from the board centre
					id_external_points = sorted(external_points_dict.items(), key=lambda x:x[1])[-2:]

					# Obtain the point between the two farthest point
					middle_point = find_middle_point(
						approx_cnt[id_external_points[0][0]][0],
						approx_cnt[id_external_points[1][0]][0]
					)
					
					# Compute the convex hull of the contour
					hull = cv.convexHull(cnt, returnPoints=False)
					# The Convex Hull of a shape or a group of points is a tight fitting convex boundary around the points or the shape

					# Calculate the difference between the contour and its convex hull
					defects = cv.convexityDefects(cnt, hull)

					# Check for concave corners
					# The shape of defects is in the following form: [x,1,4] with x = vertices in the contourn
					for i in range(defects.shape[0]):
						
						# Each elements has: start_index, end_index, farthest_pt_index, fixpt_depth; but we are interested only in the last two
						_, _, f, d = defects[i, 0] 

						# In case the distance is grater than 1000 we are in a concave corner	 
						if d > 1000: 
							A = cnt[f][0]
		
							# Calculate the refined corner locations of the point A
							A_refined = cv.cornerSubPix(imgray, np.float32([A]), winSize, zeroZone, criteria)[0]

							# Compute the polygon index and all circles centre coordinates
							index, circles_ctr_coords = compute_index_and_cc_coords(A, middle_point, thresh) 
				
							#image = draw_stuff(image, A, approx_cnt, middle_point, index, circles_ctr_coords)
							self.polygon_list[index].update_info(False, circles_ctr_coords, approx_cnt, A, middle_point)
							covered_polys[index] = 0

							# Get the X, Y and Z marker reference 2D coordinates for the polygon with given index
							X, Y, Z = marker_reference[index] 

							# Append the information
							dict_stats_to_return.append({'frame': frame_cnt, 'mark_id': index, 'Px': A_refined[0], 'Py': A_refined[1], 'X': X, 'Y': Y, 'Z': Z})

							break
			
			# Set the cover cover attributo to true on all cover polygons
			self.covered_polygon(np.where(covered_polys == 1)[0])
   
		return dict_stats_to_return