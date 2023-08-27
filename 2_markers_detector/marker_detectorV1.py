import numpy as np
import cv2 as cv
import time
import copy
import csv
import math
from typing import Dict, List, Tuple, Union


# List of object file name that we have to process
objs = ['obj01.mp4', 'obj02.mp4', 'obj03.mp4', 'obj04.mp4']

# Set the needed parameters to find the refined corners
winSize = (5, 5)
zeroZone = (-1, -1)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)



def set_marker_reference_coords() -> Dict[int, Tuple[int, int, int]]:
	'''
	PURPOSE: set the 2D marker reference coordinates in the 
	ARGUMENTS: None
	RETURN:
		- marker_reference_coords (Dict[int, Tuple[int, int, int]])
			key (int): id of the polygon
			value (Tuple[int, int, int]): X, Y, Z coordinates 
	'''
 
	marker_reference_coords = {}
    
	ox, oy = (0, 0)
	Ax, Ay = (70, 0)
 
	for id in range(24): # We have 24 polygons
		# Rotate the point (70, 0) counterclockwise by a -15° around the origin
		angle = id * math.radians(-15)
  
		marker_reference_coords[id] = (
      		ox + math.cos(angle) * (Ax - ox) - math.sin(angle) * (Ay - oy), # X coordinate
        	oy + math.sin(angle) * (Ax - ox) + math.cos(angle) * (Ay - oy), # Y coordinate
         	0	# Z coordinate
        )
	
	return marker_reference_coords
  
  

def save_stats(obj_id: str, dict_stats: List[Dict[int, Tuple[int, int, int]]]) -> None:
	'''
	PURPOSE: save as .csv file the list dictionaries
	ARGUMENTS: 
		- obj_id (str): name of the object that we are studing
		- dict_stats (List[Dict[int, Tuple[int, int, int]]]): list of dictionaries to save
	RETURN: None
	'''

	fields = ['frame', 'mark_id', 'Px', 'Py', 'X', 'Y', 'Z'] 
	
	# Name of csv file 
	filename = f'../output_part2/{obj_id}/{obj_id}_marker.csv'
		
	# Writing to csv file 
	with open(filename, 'w') as csvfile: 
		# Creating a csv dict writer object 
		writer = csv.DictWriter(csvfile, fieldnames = fields) 
			
		# Writing headers (field names) 
		writer.writeheader() 
			
		# Writing data rows 
		writer.writerows(dict_stats) 
	


def find_middle_point(Ep1: np.ndarray[int, np.int32], Ep2: np.ndarray[int, np.int32]) -> np.ndarray[int, np.float32]:
	'''
	PURPOSE: find the middle point between two points
	ARGUMENTS: 
		- Ep1 (np.ndarray[int, np.int32]): X and Y coordinates of the first extrame point
		- Ep2 (np.ndarray[int, np.int32]): X and Y coordinates of the second extreme point
	RETURN:
		- (np.ndarray[np.float32]): X and Y coordiantes of the middle point
	'''

	return np.array([(Ep1[0] + Ep2[0]) / 2, (Ep1[1] + Ep2[1]) / 2])



def find_distance_between_points(p1: np.ndarray[int, np.int32], p2: np.ndarray[int, np.int32]) -> np.float64:
	'''
	PURPOSE: find the euclidean distance between two points
	ARGUMENTS: 
		- p1 (np.ndarray[int, np.int32]): X and Y coordinates of the first point
		- p2 (np.ndarray[int, np.int32]): X and Y coordinates of the second point
	RETURN:
		- (np.float64): float distance between the two points
	'''
 
	return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5



def find_circles_centre_coords(dist_A_Cir_Ctr: List[int, np.float64], dist_A_Ext_Mid: np.float64,
                               	middle_point: np.ndarray[int, np.float64], A: np.ndarray[int, np.int32]) -> List[Tuple[np.float64, np.float64]]:
	'''
	PURPOSE: find the coordinates of the 5 circles centre
	ARGUMENTS: 
		- dist_A_Cir_Ctr (List[np.float64]): list of distance between the point A and each circle center
		- dist_A_Ext_Mid (np.float64): distance between the point A and the point between the two extreme points of a polygon
		- middle_point (np.ndarray[int, np.float64]): X and Y cordinates of the middle point between the two extreme points
		- A (np.ndarray[int, np.int32]): X and Y cordinates of the A point of a polygon
	RETURN:
		- circles_ctr_coords (List[Tuple[np.float64, np.float64]]): list of coordinates of each circle centre
	'''
	
	# Rememer that the y increase from the top to the bottom of the image
 
	circles_ctr_coords = []

	dx = A[0] - middle_point[0] # Difference the between X coordinates of point A and and the middle point between the two extreme points
	dy = A[1] - middle_point[1] # Difference the between Y coordinates of point A and and the middle point between the two extreme points
 
	for dist in dist_A_Cir_Ctr:
		# Find the rateo between the distace from A and the circle centre and the distance between A
  		# and the middle point between the two extreme points 
		rateo = dist / dist_A_Ext_Mid
 
		new_point_x = middle_point[0] + (rateo * dx) # Compute the X coordinates of a centre circle point
		new_point_y = middle_point[1] + (rateo * dy) # Compute the Y coordinates of a centre circle point

		circles_ctr_coords.append((new_point_y, new_point_x)) # Add coordinates to the list
	return circles_ctr_coords



def draw_stuff(image: np.ndarray[int, np.uint8], A: np.ndarray[int, np.int32], approx_cnt: np.ndarray[int, np.int32],
               middle_point: np.ndarray[int, np.float64], index: int, circles_ctr_coords: List[Tuple[np.float64, np.float64]]) -> np.ndarray[int, np.uint8]:
	'''
	PURPOSE: Draw contours, lines, circles an index of a specific polygon
	ARGUMENTS: 
		- image (np.ndarray[int, np.uint8]): original frame to modify
		- A (np.ndarray[int, np.int32]): X and Y cordinates of the A point of a polygon
		- approx_cnt (np.ndarray[int, np.int32]): array of coordinates of each approximate vertex position
		- middle_point (np.ndarray[int, np.float64]): X and Y cordinates of the middle point between the two extreme points
		- index (int): index of a polygon
		- circles_ctr_coords (List[Tuple[np.float64, np.float64]]): list of coordinates of each circle centre
	RETURN:
		- image (np.ndarray[int, np.uint8]): modified image
	'''	

 	# Drwaing the contourn of the polygon
	cv.drawContours(image, [approx_cnt], 0, (0, 0, 255), 2, cv.LINE_AA) 
    
    # Drwaing a cross in the A Point
	cv.line(image, (A[0], A[1] - 10), (A[0], A[1] + 10), (0,255,0), 1, cv.LINE_AA)
	cv.line(image, (A[0] - 10, A[1]), (A[0] + 10, A[1]), (0,255,0), 1, cv.LINE_AA)

	# Drawing a line from the A point up to the middle point between the two extreme point of a polygon
	cv.line(image, A, np.int32(middle_point), (0, 255, 255), 1, cv.LINE_AA) 
 
	# Drawing a cross and a rectangle in the centre of each circle
	for idx, coords in enumerate(reversed(circles_ctr_coords), start=1): 
		start = np.int32(coords[1])
		end = np.int32(coords[0])
  
		cv.line(image, (start, end - (5 * idx)), (start, end + (5 * idx)), (0,255,0), 1, cv.LINE_AA)
		cv.line(image, (start - (5 * idx), end), (start + (5 * idx), end), (0,255,0), 1, cv.LINE_AA)
		cv.rectangle(image, (start - 5, end - 5), (start + 5, end + 5), (255,0,0), 1, cv.LINE_AA)
	
	# Adding the text with the polygon index
	cv.putText(image, str(index), A, cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 8, cv.LINE_AA)
	cv.putText(image, str(index), A, cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA)

	return image



def compute_index_and_cc_coords(A: np.ndarray[int, np.int32], middle_point: np.ndarray[int, np.float64],
                                thresh: np.ndarray[int, np.uint8]) -> Tuple[int, List[Tuple[np.float64, np.float64]]]:
	'''
	PURPOSE: computing the polygon index and the circles centre coordinates of a polygon
	ARGUMENTS: 
		- A (np.ndarray[int, np.int32]): X and Y cordinates of the A point of a polygon
		- middle_point (np.ndarray[int, np.float64]): X and Y cordinates of the middle point between the two extreme points
		- thresh (np.ndarray[int, np.uint8]): thresholded image
	RETURN:
		- index (int): index of the polygon
		- circles_ctr_coords (List[Tuple[np.float64, np.float64]]): list of coordinates of each circle centre
	'''	

	# Computing the distance between the point A and the point between the two extreme points of a polygon
	dist_A_Ext_Mid = find_distance_between_points(A, np.int32(middle_point))
	  
	# Computing the distance from the point A and each circle centre
	dist_A_Cir_Ctr = [(dist_A_Ext_Mid * ((i * 4.5) + 5) / 28) for i in range(5)] 
	  
	# Obtaining the coordinates of each chircle centre
	circles_ctr_coords = find_circles_centre_coords(dist_A_Cir_Ctr, dist_A_Ext_Mid, middle_point, A) 

	# Obtain the reversed list of bit describing the polygon index
	bit_index = [1 if thresh[np.int32(coords[0]), np.int32(coords[1])] == 0 else 0 for coords in circles_ctr_coords] 

	# Obtain the index
	index = int("".join(str(x) for x in bit_index), 2) 
 
	return index, circles_ctr_coords



def find_markers(image: np.ndarray[int, np.uint8], frame_cnt: int, marker_reference: Dict[int, Tuple[int, int, int]]) \
    -> Tuple[np.ndarray[int, np.uint8], List[Dict[str, Union[int, int, np.float64, np.float64, int, int, int]]]]:
	'''
	PURPOSE: computing the polygon index and the circles centre coordinates of a polygon
	ARGUMENTS: 
		- image (np.ndarray[int, np.uint8]): video frame
		- frame_cnt (int): index frmae 
		- marker_reference (Dict[int, Tuple[int, int, int]])): dictionary of the marker reference coordinates
	RETURN:
		- image (np.ndarray[int, np.uint8]): modified image
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

	# Searching through every region selected to find the required polygon.
	for cnt in contours:
	   
		# Shortlisting the regions based on there area.
		if cv.contourArea(cnt) > 1720:
			  
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
			
						image = draw_stuff(image, A, approx_cnt, middle_point, index, circles_ctr_coords)

						# Get the X, Y and Z marker reference 2D coordinates for the polygon with given index
						X, Y, Z = marker_reference[index] 

						# Append the information
						dict_stats_to_return.append({'frame': frame_cnt, 'mark_id': index, 'Px': A_refined[0], 'Py': A_refined[1], 'X': X, 'Y': Y, 'Z': Z})

						break

	return image, dict_stats_to_return



def main() -> None:
	'''
	PURPOSE: main function
	ARGUMENTS: None
	RETURN: None
	'''	
	
 	# Set the marker reference coordinates for the 24 polygonls
	marker_reference = set_marker_reference_coords()
	
	# Iterate for each object
	for obj in objs:
		
		print(f'Marker Detector for {obj}...')
		input_video = cv.VideoCapture(f"../data/{obj}")

		# Get video properties
		frame_width = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
		frame_height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))
		fps = input_video.get(cv.CAP_PROP_FPS)

		actual_fps = 0
		obj_id = obj.split('.')[0]
  
		dict_stats = [] # Initialize the list of dictionary that we will save as .csv file

		# Create output video writer
		output_video = cv.VideoWriter(f"../output_part2/{obj_id}/{obj_id}_mask.mp4", cv.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
  
		# Until the video is open
		while input_video.isOpened():
			start = time.time() # Start the timer to compute the actual FPS 
			
			# Extract a frame
			ret, frame = input_video.read()

			if not ret:	break

			# Obtain the edited frame and the dictionary of statistics
			edited_frame, dict_stats_to_extend = find_markers(frame, actual_fps, marker_reference)

			frame_with_fps = copy.deepcopy(edited_frame) 
   
			end = time.time()
			fps = 1 / (end-start) # Compute the FPS

			# Output the frame with the FPS
			cv.putText(frame_with_fps, f"{fps:.2f} FPS", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
			cv.imshow(f'Marker Detector of {obj}', frame_with_fps)
   
			# Save the frame without the FPS count
			output_video.write(edited_frame)

			# Extends our list with the obtained dictionary
			dict_stats.extend(dict_stats_to_extend)

			if cv.waitKey(1) == ord('q'):
				break

			actual_fps += 1

		print(' DONE')

		print('Saving data...')
		save_stats(obj_id, dict_stats)
		print(' DONE\n')

		# Release the input and output streams
		input_video.release()
		output_video.release()
		cv.destroyAllWindows()



if __name__ == "__main__":
	main()