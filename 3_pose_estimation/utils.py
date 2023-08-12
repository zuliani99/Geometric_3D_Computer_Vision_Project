import cv2 as cv
import numpy as np


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




def draw_origin(img, corner, imgpts):
 
	cv.arrowedLine(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 4, cv.LINE_AA)
	cv.putText(img, 'Y', tuple(imgpts[0].ravel()), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv.LINE_AA)
	
	cv.arrowedLine(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 4, cv.LINE_AA)
	cv.putText(img, 'X', tuple(imgpts[1].ravel()), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv.LINE_AA)
	
	cv.arrowedLine(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 4, cv.LINE_AA)
	cv.putText(img, 'Z', tuple(imgpts[2].ravel()), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv.LINE_AA)

	return img



def draw_cube(img, imgpts):

	imgpts = np.int32(imgpts).reshape(-1,2)

	# draw ground floor in green
	cv.drawContours(img, [imgpts[:4]], -1, (0,0,255), 3, cv.LINE_AA)

	# draw pillars in blue color
	for i,j in zip(range(4), range(4,8)):
		cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (0,0,255), 3, cv.LINE_AA)

	# draw top layer in red color
	cv.drawContours(img, [imgpts[4:]], -1, (0,0,255), 3)

	return img
