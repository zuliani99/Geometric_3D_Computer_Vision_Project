import math
from types import NoneType
import cv2 as cv
import numpy as np
import random

from typing import Dict, List, Tuple



def set_marker_reference_coords() -> Dict[int, Tuple[int, int, int]]:
	'''
	PURPOSE: set the 2D marker reference coordinates
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
			0.0	# Z coordinate
		)
	
	return marker_reference_coords




def find_extreme_middle_point(Ep1: np.ndarray[int, np.int32], Ep2: np.ndarray[int, np.int32]) -> np.ndarray[int, np.float32]:
	'''
	PURPOSE: find the middle point between two points
	ARGUMENTS: 
		- Ep1 (np.ndarray[int, np.int32]): X and Y coordinates of the first extreme point
		- Ep2 (np.ndarray[int, np.int32]): X and Y coordinates of the second extreme point
	RETURN:
		- (np.ndarray[int, np.float32]): X and Y coordiantes of the middle point
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




def find_circles_centre_coords(dist_A_Cir_Ctr: List[np.float64], dist_A_Ext_Mid: np.float64,
								   extreme_middle_point: np.ndarray[int, np.float64], A: np.ndarray[int, np.int32]) -> List[Tuple[np.float64, np.float64]]:
	'''
	PURPOSE: find the coordinates of the 5 circles centres
	ARGUMENTS: 
		- dist_A_Cir_Ctr (List[np.float64]): list of distance between the point A and each circle center
		- dist_A_Ext_Mid (np.float64): distance between the point A and the point between the two extreme points of a polygon
		- extreme_middle_point (np.ndarray[int, np.float64]): X and Y cordinates of the middle point between the two extreme points
		- A (np.ndarray[int, np.int32]): X and Y cordinates of the A point of a polygon
	RETURN:
		- circles_ctr_coords (List[Tuple[np.float64, np.float64]]): list of coordinates of each circle centre
	'''
	
	# Remember that the y increase from the top of the image to the bottom
 
	# x = x_middlepoint + ((dist_A_centre / dist_A_middlepoint) * (x_A - x_middlepoint))
	# y = y_middlepoint + ((dist_A_centre / dist_A_middlepoint) * (y_A - y_middlepoint))
 
	circles_ctr_coords = []

	dx = A[0] - extreme_middle_point[0] # Difference the between X coordinates of point A and the middle point between the two extreme points
	dy = A[1] - extreme_middle_point[1] # Difference the between Y coordinates of point A and the middle point between the two extreme points
 
	for dist in dist_A_Cir_Ctr:
		# Find the rateo between the distance from A and the circle centre and the distance between A
		# and the middle point between the two extreme points 
		rateo = dist / dist_A_Ext_Mid
 
		new_point_x = extreme_middle_point[0] + (rateo * dx) # Compute the X coordinates of a centre circle point
		new_point_y = extreme_middle_point[1] + (rateo * dy) # Compute the Y coordinates of a centre circle point

		circles_ctr_coords.append((new_point_y, new_point_x)) # Add coordinates to the list
	return circles_ctr_coords




def compute_index_and_cc_coords(A: np.ndarray[int, np.int32], extreme_middle_point: np.ndarray[int, np.float64],
								thresh: np.ndarray[int, np.uint8]) -> Tuple[int, List[Tuple[np.float64, np.float64]]]:
	'''
	PURPOSE: computing the polygon indexes and the circles centre coordinates
	ARGUMENTS: 
		- A (np.ndarray[int, np.int32]): X and Y cordinates of the A point of a polygon
		- extreme_middle_point (np.ndarray[int, np.float64]): X and Y cordinates of the middle point between the two extreme points
		- thresh (np.ndarray[int, np.uint8]): thresholded image
	RETURN:
		- index (int): index of the polygon
		- circles_ctr_coords (List[Tuple[np.float64, np.float64]]): list of coordinates of each circle centre
	'''	

	# Computing the distance from the point A and the point between the two extreme points of a polygon
	dist_A_Ext_Mid = find_distance_between_points(A, np.int32(extreme_middle_point))
	  
	# Computing the 5 proportions to obtain the distance from the point A and each circle centre
	# The consecutive distances within each polygon, considering the point A and the middle point as the two extremes and each circle centre as intermediate step are:
	# 5 - 4.5 - 4.5 - 4.5 - 4.5 - 5		and the sum is equal to 28
	dist_A_Cir_Ctr = [(dist_A_Ext_Mid * ((i * 4.5) + 5) / 28) for i in range(5)] 
	  
	# Obtaining the coordinates of each circles centre
	circles_ctr_coords = find_circles_centre_coords(dist_A_Cir_Ctr, dist_A_Ext_Mid, extreme_middle_point, A) 

	# Obtain the list of bits forming the polygon index by checking the circles centre color in the threshold frame
	bit_index = [1 if thresh[np.int32(coords[0]), np.int32(coords[1])] == 0 else 0 for coords in circles_ctr_coords] 

	# Obtain the index
	index = int("".join(str(x) for x in bit_index), 2) 
 
	return index, circles_ctr_coords




def sort_vertices_clockwise(vertices: np.ndarray[int, np.float32], centroid: NoneType | np.ndarray[int, np.int32] = None) \
		-> np.ndarray[int, np.float32] :
	'''
	PURPOSE: sort vertices clockwise respect a centroid 
	ARGUMENTS: 
		- vertices (np.ndarray[int, np.float32]): numpy array of vertices to sort
		- centroid (NoneType | np.ndarray[np.int32]): X and Y cordinates of the centroid
	RETURN:
		- (np.ndarray[int, np.float32]) sorted vertices
	'''	
	
	if centroid is None:
		centroid = np.mean(vertices, axis=0)     
	   
	angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
	sorted_indices = np.argsort(angles)

	return vertices[sorted_indices]




def random_bgr_color() -> Tuple[int, int, int]:
	'''
	PURPOSE: random BGR color
	ARGUMENTS: None
	RETURN:
		- (Tuple[int, int, int]): BGR color
	'''	
 
	blue = random.randint(0, 255)
	green = random.randint(0, 255)
	red = random.randint(0, 255)
	return (blue, green, red)




def resize_for_laptop(using_laptop: bool, frame: np.ndarray[int, np.uint8]) -> np.ndarray[int, np.uint8]:
	'''
	PURPOSE: resize the image if using_laptop is True
	ARGUMENTS:
		- using_laptop (bool)
		- frame (np.ndarray[int, np.uint8]): frame to resize
	RETURN:
		- (np.ndarray[int, np.uint8]): resized frmae
	'''	
				
	if using_laptop:
		frame = cv.resize(frame, (1080, 600), interpolation=cv.INTER_AREA)
	return frame




def write_ply_file(obj_id: str, voxels_cube_coords: np.ndarray[int, np.float32], voxels_cube_faces: np.ndarray[int, np.float32]) -> None:
	'''
	PURPOSE: write the .ply file that compose the object mesh
	ARGUMENTS:
		- obj_id (str)
		- voxels_cube_coords (np.ndarray[int, np.float32]): array of voxels cube coordinates
		- voxels_cube_faces (np.ndarray[int, np.float32]): array of voxels faces
	RETURN: None
	'''	

    # Create the header
	header = f"""ply
	format ascii 1.0
	element vertex {voxels_cube_coords.shape[0]}
	property float x
	property float y
	property float z
 	element face {voxels_cube_faces.shape[0]}
	property list uchar int vertex_index
	end_header
	"""

	# Write header and vertex data to a file
	with open(f'../output_project/{obj_id}/3d_{obj_id}.ply', 'w') as f:
		f.write(header)
		np.savetxt(f, voxels_cube_coords, fmt='%.4f', newline='\n')
  
	# Write face data to the same file
	with open(f'../output_project/{obj_id}/3d_{obj_id}.ply', 'a') as f:
		np.savetxt(f, voxels_cube_faces.astype(int), fmt='%i', newline='\n')
