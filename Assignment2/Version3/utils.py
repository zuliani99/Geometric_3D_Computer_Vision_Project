import csv
import math
from types import NoneType
import cv2 as cv
import numpy as np
import random

from typing import Dict, List, Tuple


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
    filename = f'../../output_part2/{obj_id}/{obj_id}_marker.csv'
        
    # Writing to csv file 
    with open(filename, 'w') as csvfile: 
        # Creating a csv dict writer object 
        writer = csv.DictWriter(csvfile, fieldnames = fields) 
            
        # Writing headers (field names) 
        writer.writeheader() 
            
        # Writing data rows 
        writer.writerows(dict_stats) 
    


def find_middle_point(Ep1: np.ndarray[np.int32], Ep2: np.ndarray[np.int32]) -> np.ndarray[np.float32]:
    '''
    PURPOSE: find the middle point between two points
    ARGUMENTS: 
        - Ep1 (np.ndarray[np.int32]): X and Y coordinates of the first extrame point
        - Ep2 (np.ndarray[np.int32]): X and Y coordinates of the second extreme point
    RETURN:
        - (np.ndarray[np.float32]): X and Y coordiantes of the middle point
    '''

    return np.array([(Ep1[0] + Ep2[0]) / 2, (Ep1[1] + Ep2[1]) / 2])



def find_distance_between_points(p1: np.ndarray[np.int32], p2: np.ndarray[np.int32]) -> np.float64:
    '''
    PURPOSE: find the middle point between two points
    ARGUMENTS: 
        - p1 (np.ndarray[np.int32]): X and Y coordinates of the first point
        - p2 (np.ndarray[np.int32]): X and Y coordinates of the second point
    RETURN:
        - (np.float64): float distance between the two points
    '''

    return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5



def find_circles_centre_coords(dist_A_Cir_Ctr: List[np.float64], dist_A_Ext_Mid: np.float64,
                                   middle_point: np.ndarray[np.float64], A: np.ndarray[np.int32]) -> List[Tuple[np.float64, np.float64]]:
    '''
    PURPOSE: find the coordinates of the 5 circles centre
    ARGUMENTS: 
        - dist_A_Cir_Ctr (List[np.float64]): list of distance between the point A and each circle center
        - dist_A_Ext_Mid (np.float64): distance between the point A and the point between the two extreme points of a polygon
        - middle_point (np.ndarray[np.float64]): X and Y cordinates of the middle point between the two extreme points
        - A (np.ndarray[np.int32]): X and Y cordinates of the A point of a polygon
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




def compute_index_and_cc_coords(A: np.ndarray[np.int32], middle_point: np.ndarray[np.float64],
                                thresh: np.ndarray[np.ndarray[np.ndarray[np.uint8]]]) -> Tuple[int, List[Tuple[np.float64, np.float64]]]:
    '''
    PURPOSE: computing the polygon index and the circles centre coordinates of a polygon
    ARGUMENTS: 
        - A (np.ndarray[np.int32]): X and Y cordinates of the A point of a polygon
        - middle_point (np.ndarray[np.float64]): X and Y cordinates of the middle point between the two extreme points
        - thresh (np.ndarray[np.ndarray[np.ndarray[np.uint8]]]): thresholded image
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




def sort_vertices_clockwise(vertices: np.ndarray[np.float32] | np.ndarray[np.ndarray[np.ndarray[np.float32]]], centroid: NoneType | np.ndarray[np.int32] = None) \
        -> np.ndarray[np.float32] | np.ndarray[np.ndarray[np.ndarray[np.float32]]]:
    '''
    PURPOSE: sort vertices clockwise respect a centroid 
    ARGUMENTS: 
        - vertices (np.ndarray[np.float32] | np.ndarray[np.ndarray[np.ndarray[np.float32]]]): numpy array of vertices to sort
        - centroid (NoneType | np.ndarray[np.int32]): X and Y cordinates of the centroid
    RETURN:
        - (np.ndarray[np.float32] | np.ndarray[np.ndarray[np.ndarray[np.float32]]]) sorted vertices
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



def resize_for_laptop(using_laptop: bool, frame: np.ndarray[np.ndarray[np.ndarray[np.uint8]]]) \
    -> np.ndarray[np.ndarray[np.ndarray[np.uint8]]]:
    '''
    PURPOSE: resize the image if using_laptop is True
    ARGUMENTS:
        - using_laptop (bool)
        - frame (np.ndarray[np.ndarray[np.ndarray[np.uint8]]]): frame to resze
    RETURN:
        - (np.ndarray[np.ndarray[np.ndarray[np.uint8]]]): resized frmae
    '''	
                
    if using_laptop:
        frame = cv.resize(frame, (1080, 600), interpolation=cv.INTER_AREA)
    return frame



def check_mask(approx_cnt: np.ndarray[np.ndarray[np.ndarray[np.uint8]]], mask: np.ndarray[np.uint8]) \
        -> np.ndarray[np.ndarray[np.uint8]]:
    '''
    PURPOSE: resize the image if using_laptop is True
    ARGUMENTS:
        - approx_cnt (np.ndarray[np.ndarray[np.ndarray[np.uint8]]])
        - mask (np.ndarray[np.uint8]): frame to resze
    RETURN:
        - (np.ndarray[np.ndarray[np.ndarray[np.uint8]]]): resized frmae
    '''	

    to_return = np.zeros((0,2), dtype=np.int32)
    for cnt in approx_cnt:
        x, y = cnt[0]
        
        # If the extracted point has coordinates in teh mask filled by black we accept the feature
        if mask[y, x] == 0: to_return = np.vstack((to_return, np.array([x, y])))
    return to_return


'''def are_lines_parallel(points1, points2):
    # Calculate the slopes of the lines using NumPy broadcasting
    #slopes1 = (points1[1, 1] - points1[0, 1]) / (points1[1, 0] - points1[0, 0])
    #slopes2 = (points2[1, 1] - points2[0, 1]) / (points2[1, 0] - points2[0, 0])

    #angle1 = np.arctan2(points1[1, 1] - points1[0, 1], points1[1, 0] - points1[0, 0]) * 180.0  / np.pi
    #angle2 = np.arctan2(points2[1, 1] - points2[0, 1], points2[1, 0] - points2[0, 0]) * 180.0  / np.pi

    # Check if the slopes are approximately equal
    #print(np.abs(angle1 - angle2) )
    #return np.all(np.abs(angle1 - angle2) > 0.3)
    #return np.isclose(angle1, angle2, atol=0.5)

    x_coords_1, y_coords_1 = zip(*points1)
    x_coords_2, y_coords_2 = zip(*points2)
    
    A_1 = np.vstack([x_coords_1, np.ones(len(x_coords_1))]).T
    m_1, _ = np.linalg.lstsq(A_1, y_coords_1)[0]
    
    A_2 = np.vstack([x_coords_2, np.ones(len(x_coords_2))]).T
    m_2, _ = np.linalg.lstsq(A_2, y_coords_2)[0]
    
    print(m_1, m_2)
    
    return np.isclose(m_1, m_2, atol=10)
    


def are_lines_near_parallel_to_y_axis(points1, points2, x_threshold=0.4):
    # Check if the x-coordinates are approximately equal for the corresponding y-coordinates
    y1, y2 = points1[:, 1], points2[:, 1]
    x1, x2 = points1[:, 0], points2[:, 0]

    return np.all(np.abs(x1 - x2) < x_threshold * np.abs(y1 - y2))'''