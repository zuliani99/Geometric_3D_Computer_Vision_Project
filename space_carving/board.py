import cv2 as cv
import numpy as np

from utils import compute_index_and_cc_coords, find_distance_between_points, find_middle_point, sort_vertices_clockwise
from polygon import Polygon

from typing import List, Tuple, Dict



# Set the needed parameters to find the refined corners
winSize_sub = (3, 3)
zeroZone_sub = (-1, -1)
criteria_sub = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)


# Set the Lucas Kanade parameters
criteria_lk = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 100, 0.01)
maxlevel_lk = 4


# Board class to manage the board information

class Board:
    
	def __init__(self, n_polygons: int) -> None:
		self.__polygon_list: List[Polygon] = [Polygon() for _ in range(n_polygons)]
		self.__tracked_features = np.zeros((0,2), dtype=np.float32)
		self.__centroid = np.array([1259, 520], dtype=np.int32)



	def draw_red_polygon(self, image: np.ndarray[int, np.uint8]) -> np.ndarray[int, np.uint8]:
		'''
		PURPOSE: draw the red polygon, the cross in point A and the line crossing the polygon by length
		ARGUMENTS: 
			- image (np.ndarray[int, np.uint8]): image to edit
		RETURN:
			- (np.ndarray[int, np.uint8]): resuting image
		'''	
  
		for poly in self.__polygon_list:
			if poly.cover == False:
				cv.drawContours(image, [np.int32(poly.vertex_coords)], 0, (0, 0, 255), 1, cv.LINE_AA)

				cv.drawMarker(image, np.int32(poly.point_A), (0,255,0), cv.MARKER_CROSS, 20, 1, cv.LINE_AA)

				cv.line(image, np.int32(poly.point_A), np.int32(poly.middle_point), (0, 255, 255), 1, cv.LINE_AA) 
				
				for x, y in poly.vertex_coords:
					cv.circle(image, (int(x), int(y)), 4, poly.color, -1)
					cv.line(image, (int(x), int(y)), self.__centroid, poly.color, 1, cv.LINE_AA)
     
		return image
            
       
            
	def draw_green_cross_and_blu_rectangle(self, image: np.ndarray[int, np.uint8]) -> np.ndarray[int, np.uint8]:
		'''
		PURPOSE: draw a green cross and a blu rectangle in each circe centre 
		ARGUMENTS: 
			- image (np.ndarray[int, np.uint8]): image to edit
		RETURN:
			- (np.ndarray[int, np.uint8]): resuting image
		'''	
        
		for poly in self.__polygon_list:
			if poly.cover == False:
				for idx, coords in enumerate(reversed(poly.circles_ctr_coords), start=1): 
					start = np.int32(coords[1])
					end = np.int32(coords[0])
					cv.drawMarker(image, (start, end), (0,255,0), cv.MARKER_CROSS, 10 * idx, 1, cv.LINE_AA)
					cv.drawMarker(image, (start, end), (255,0,0), cv.MARKER_SQUARE, 10, 1, cv.LINE_AA)
					
		return image
  
  
  
	def draw_index(self, image: np.ndarray[int, np.uint8]) -> np.ndarray[int, np.uint8]:
		'''
		PURPOSE: draw the polygon index
		ARGUMENTS: 
			- image (np.ndarray[int, np.uint8]): image to edit
		RETURN:
			- (np.ndarray[int, np.uint8]): resuting image
		'''	
     		
		for index, poly in enumerate(self.__polygon_list):
			if poly.cover == False:
				cv.putText(image, str(index), np.int32(poly.point_A), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 8, cv.LINE_AA)
				cv.putText(image, str(index), np.int32(poly.point_A), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA)
		return image
  
  

	def draw_stuff(self, image: np.ndarray[int, np.uint8]) -> np.ndarray[int, np.uint8]:
		'''
		PURPOSE: apply all the drawing function
		ARGUMENTS: 
			- image (np.ndarray[int, np.uint8]): image to edit
		RETURN:
			- (np.ndarray[int, np.uint8]): resuting image
		'''	
  
		return self.draw_index(self.draw_green_cross_and_blu_rectangle(self.draw_red_polygon(image)))

  


	def find_interesting_points(self, thresh: np.ndarray[int, np.uint8], imgray: np.ndarray[int, np.uint8]) -> None:
		'''
		PURPOSE: find good features to track
		ARGUMENTS: 
			- thresh (np.ndarray[int, np.uint8]): threshold resut
			- imgray (np.ndarray[int, np.uint8]): gray image
		RETURN: None
		'''	

		# Reset the tracked features
		self.__tracked_features = np.zeros((0,2), dtype=np.float32)
		
		# Consider only the board exluding the area that could introduce some noise in the process
		mask_thresh = np.zeros_like(thresh, dtype=np.uint8)
		mask_thresh[:, 1140:1570] = thresh[:, 1140:1570]

		# Finding the contourns
		contours, _ = cv.findContours(mask_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # [[[X Y]] [[X Y]] ... [[X Y]]]
	
		# Searching through every region selected to find the required polygon
		for cnt in contours:
			
			# Shortlisting the regions based on there area sorted clockwise
			if(cv.contourArea(sort_vertices_clockwise(np.squeeze(cnt, axis=1)))) > 1550.0:

				approx_cnt = cv.approxPolyDP(cnt, 0.015 * cv.arcLength(cnt, True), True) # [[[X Y]] [[X Y]] ... [[X Y]]]
				
				# Checking if the number of sides of the selected region is 5
				if (len(approx_cnt)) == 5:
					ref_approx_cnt = cv.cornerSubPix(imgray, np.float32(approx_cnt), winSize_sub, zeroZone_sub, criteria_sub)
					self.__tracked_features = np.vstack((self.__tracked_features, np.squeeze(ref_approx_cnt)))




	def get_clockwise_vertices_initial(self) -> np.ndarray[int, np.float32]:
		'''
		PURPOSE: reshape the obtained features, sort them in clockwise order and remove the last polygon by area
		ARGUMENTS: None
		RETURN:
  			- (np.ndarray[int, np.float32]): sorted vertices polygon
		'''	
     
		self.__tracked_features = sort_vertices_clockwise(self.__tracked_features, self.__centroid)
		
		self.__tracked_features = self.__tracked_features[:int(self.__tracked_features.shape[0] // 5) * 5, :]
			
		reshaped_clockwise = np.reshape(self.__tracked_features, (int(self.__tracked_features.shape[0] // 5), 5, 2))
  
		# I have to sort clockwise the last polygon in order to compute correctly the contourArea
		if(cv.contourArea(sort_vertices_clockwise(reshaped_clockwise[-1,:,:])) <= 1550.0):
			reshaped_clockwise = reshaped_clockwise[:reshaped_clockwise.shape[0] - 1, :, :]
		
		return np.array([sort_vertices_clockwise(poly) for poly in reshaped_clockwise])




	def apply_LK_OF(self, prev_frameg: np.ndarray[int, np.uint8], frameg: np.ndarray[int, np.uint8], winsize_lk: Tuple[int, int]) -> None: 
		'''
		PURPOSE: apply Lucas Kanade Optical Flow to predict the position of the features based on its algorithm parameters
		ARGUMENTS: 
			- prev_frameg (np.ndarray[int, np.uint8]): previous gray frame
			- frameg (np.ndarray[int, np.uint8]): actual gray frame
			- winsize_lk (Tuple[int, int]): window size
		RETURN: None
		'''	
     
		# Forward Optical Flow
		p1, st, _ = cv.calcOpticalFlowPyrLK(prev_frameg, frameg, self.__tracked_features, None, winSize=winsize_lk, maxLevel=maxlevel_lk, criteria=criteria_lk)#, flags=cv.OPTFLOW_LK_GET_MIN_EIGENVALS, minEigThreshold=0.01)
		
		fb_good = p1[np.where(st == 1)[0]]
  
		self.__tracked_features = fb_good
					



	def covered_polygon(self, polygons: np.ndarray[int, np.int32]) -> None:
		'''
		PURPOSE: set the polygon attribute cover to True for the polygons that are behind the glass
		ARGUMENTS: 
			- polygons (np.ndarray[int, np.int32]): array of index that express the covered polygons
		RETURN: None
		'''	
     
		for id_poly in polygons: self.__polygon_list[id_poly].cover = True
  
  
  
  
	def compute_markers(self, thresh: np.ndarray[int, np.uint8], reshaped_clockwise: np.ndarray[int, np.float32], \
	     		marker_reference: Dict[int, Tuple[int, int, int]]) -> np.ndarray[int, np.float32]:
		'''
		PURPOSE: compute the markers informations for later use
		ARGUMENTS:
			- thresh (np.ndarray[int, np.uint8]): threshold image
			- reshaped_clockwise (np.ndarray[int, np.float32]): reshaped feature in clockwise order
			- marker_reference (Dict[int, Tuple[int, int, int]])): dictionary of the markers reference coordinates
		RETURN:
			- pixel_info (np.ndarray[int, np.float32]): marker informations array
		'''	
	    
		# np.array where to store the markers informations
		pixel_info = np.zeros((0,6), dtype=np.float32)

		# np.array of ones in which at the end of the computation will store only the covered polygons
		covered_polys = np.ones((1, 24))[0]
		
		# Iterate through the reshaped tracked features in clockwise order
		for poly in reshaped_clockwise:
			# Obtain the external point distance between the approximated board centroid and each approximated polygon vertex
			external_points_dict = dict(enumerate(
				list(map(lambda x: find_distance_between_points(x, self.__centroid), poly))
			))

			# Obtain the id of the two farthest point from the board centre
			id_external_points = sorted(external_points_dict.items(), key=lambda x:x[1])[-2:]

			# Obtain the point between the two farthest point
			middle_point = find_middle_point(poly[id_external_points[0][0]], poly[id_external_points[1][0]])
	
	 		# Compute the convex hull of the contour
			hull = np.squeeze(cv.convexHull(poly, returnPoints=False))
   			# The Convex Hull of a shape or a group of points is a tight fitting convex boundary around the points or the shape
	
			# Get the coordinate of the point A by getting the missing index
			A = np.squeeze(poly[np.squeeze(np.setdiff1d(np.arange(5), hull))])


			if(len(A.shape) == 1):
				index, circles_ctr_coords = compute_index_and_cc_coords(A, middle_point, thresh) 

				if(index < 24):
					self.__polygon_list[index].update_info(False, circles_ctr_coords, poly, A, middle_point)
					covered_polys[index] = 0

					# Get the X, Y and Z marker reference 2D coordinates for the polygon with given index
					X, Y, Z = marker_reference[index] 

					# Updating the array stackng the polygon index, the position of the A point in the image and in the marker reference coordinates 
					pixel_info = np.vstack((pixel_info, np.array([index, A[0], A[1], X, Y, Z], dtype=np.float32)))
		
  		# Set the cover cover attribute to True on all covered polygons
		self.covered_polygon(np.where(covered_polys == 1)[0])		

		return pixel_info




	def draw_origin(self, img: np.ndarray[int, np.uint8], imgpts: np.ndarray[int, np.int32]) -> np.ndarray[int, np.uint8]:
		'''
		PURPOSE: draw the origin with the axis
		ARGUMENTS:
			- img (np.ndarray[int, np.uint8]): image to modify
			- imgpts (np.ndarray[int, np.int32]): image points
		RETURN:
			- (np.ndarray[int, np.uint8]): modified frame
		'''	

		cv.arrowedLine(img, self.__centroid, tuple(imgpts[0].ravel()), (255,0,0), 4, cv.LINE_AA)
		cv.putText(img, 'Y', tuple(imgpts[0].ravel()), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv.LINE_AA)
	
		cv.arrowedLine(img, self.__centroid, tuple(imgpts[1].ravel()), (0,255,0), 4, cv.LINE_AA)
		cv.putText(img, 'X', tuple(imgpts[1].ravel()), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv.LINE_AA)
	
		cv.arrowedLine(img, self.__centroid, tuple(imgpts[2].ravel()), (0,0,255), 4, cv.LINE_AA)
		cv.putText(img, 'Z', tuple(imgpts[2].ravel()), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv.LINE_AA)

		return img

	