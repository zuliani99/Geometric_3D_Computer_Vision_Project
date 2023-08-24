import cv2 as cv
import numpy as np


def resize_for_laptop(using_laptop: bool, frame: np.ndarray[int, np.uint8]) -> np.ndarray[int, np.uint8]:
	'''
	PURPOSE: resize the image if using_laptop is True
	ARGUMENTS:
		- using_laptop (bool)
		- frame (np.ndarray[int, np.uint8]): frame to resze
	RETURN:
		- (np.ndarray[int, np.uint8]): resized frmae
	'''	
				
	if using_laptop:
		frame = cv.resize(frame, (1080, 600), interpolation=cv.INTER_AREA)
	return frame



def draw_origin(img: np.ndarray[int, np.uint8], centroid: np.ndarray[int, np.uint32], imgpts: np.ndarray[int, np.int32]) -> np.ndarray[int, np.uint8]:
	'''
	PURPOSE: draw the origin with the axis
	ARGUMENTS:
		- img (np.ndarray[int, np.uint8]): image to modify
 		- centroid (np.ndarray[int, np.uint32]): centroid coordinates
		- imgpts (np.ndarray[int, np.int32]): image points
	RETURN:
		- (np.ndarray[int, np.uint8]): modified frame
	'''	

	cv.arrowedLine(img, centroid, tuple(imgpts[0].ravel()), (255,0,0), 4, cv.LINE_AA)
	cv.putText(img, 'Y', tuple(imgpts[0].ravel()), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv.LINE_AA)
	
	cv.arrowedLine(img, centroid, tuple(imgpts[1].ravel()), (0,255,0), 4, cv.LINE_AA)
	cv.putText(img, 'X', tuple(imgpts[1].ravel()), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv.LINE_AA)
	
	cv.arrowedLine(img, centroid, tuple(imgpts[2].ravel()), (0,0,255), 4, cv.LINE_AA)
	cv.putText(img, 'Z', tuple(imgpts[2].ravel()), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv.LINE_AA)

	return img



def draw_cube(img: np.ndarray[int, np.uint8], imgpts: np.ndarray[int, np.int32]) -> np.ndarray[int, np.uint8]:
	'''
	PURPOSE: draw a red cube that inglobe the object
	ARGUMENTS:
		- img (np.ndarray[int, np.uint8]): image to modify
		- imgpts (np.ndarray[int, np.int32]): image points
	RETURN:
		- (np.ndarray[int, np.uint8]): modified frame
	'''	
	
	imgpts = np.int32(imgpts).reshape(-1,2)

	# Draw ground floor
	cv.drawContours(img, [imgpts[:4]], -1, (0,0,255), 3, cv.LINE_AA)

	# Draw pillars
	for i,j in zip(range(4), range(4,8)):
		cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (0,0,255), 3, cv.LINE_AA)

	# Draw top layer
	cv.drawContours(img, [imgpts[4:]], -1, (0,0,255), 3)

	return img