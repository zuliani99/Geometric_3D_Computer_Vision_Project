import cv2 as cv
import numpy as np



hyperparameters = {
	'obj01.mp4': {
		'clipLimit': 8,
     	'first': (cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5)), 10),
      	'second': (cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3)), 12),
		'additional_mask_space': (300, 900, 270, 900),
       	'correction': (np.array([105,65,5]), np.array([140,255,255]))
    },
	'obj02.mp4': {
		'clipLimit': 8,
    	'first': (cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3)), 2),
    	'second': (cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3)), 3),
     	'correction': (np.array([105,65,5]), np.array([140,255,255]))
    },
	'obj03.mp4': {
		'clipLimit': 6,
    	'first': (cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5)), 12),
     	'second': (cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3)), 10),
		'additional_mask_space': (100, 1000, 500, 1000),
    	'correction': (np.array([105,55,0]), np.array([120,255,255]))
    },
	'obj04.mp4': {
		'clipLimit': 3,
    	'first': (cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3)), 4),
     	'second': (cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (2,2)), 3),
      	'correction': (np.array([105,55,0]), np.array([120,255,255]))
    }
}


def change_contrast(img: np.ndarray[int, np.uint8], clipLimit: int) -> np.ndarray[int, np.uint8]:
	'''
	PURPOSE: change the contrast of the image 
	ARGUMENTS:
		- img (np.ndarray[int, np.uint8]): image where apply the contrast changing
		- clipLimit
	RETURN:
		- (np.ndarray[int, np.uint8]) updated image
	'''
    
	lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)
	l_channel, a, b = cv.split(lab)

	# Applying CLAHE to L-channel
	l_clahe = cv.createCLAHE(clipLimit = clipLimit, tileGridSize = (20,20))
	l_channel = l_clahe.apply(l_channel)

	# Merge the CLAHE enhanced L-channel with the a and b channel
	limg = cv.merge((l_channel,a,b))

	# Converting image from LAB Color model to BGR color spcae
	return cv.cvtColor(limg, cv.COLOR_LAB2RGB)
    


def apply_segmentation(obj: str, frame: np.ndarray[int, np.uint8]) -> np.ndarray[int, np.uint8]:
	'''
	PURPOSE: apply the segmentation with all the color conversions and morphological operations
	ARGUMENTS:
		- obj (str): object name string
		- frame (np.ndarray[int, np.uint8]): image video frame
	RETURN:
		- morph_op_2 (np.ndarray[int, np.uint8]): final mask
	'''
 
    # Convert the imahe into RGB format
	rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

	# Change the image contast and confvert it into HSV format
	enhanced = cv.cvtColor(change_contrast(rgb, hyperparameters[obj]['clipLimit']), cv.COLOR_RGB2HSV)

	# Define the colored mask obtained from the image
	color_mask = cv.bitwise_not(cv.inRange(enhanced, hyperparameters[obj]['correction'][0], hyperparameters[obj]['correction'][1]))

	# Add a constant rectangle to mask the left part of the image
	rectangular_mask = np.full(rgb.shape[:2], 0, np.uint8)
	rectangular_mask[:,1180:rgb.shape[1]] = 255

	mask = cv.bitwise_or(cv.ellipse(color_mask,(1380,540),(700,300),89,0,180,255,-1), rectangular_mask)

	# Parse useful hyperparameters
	morph_op_1 = None
	morph1, kernel1, iter1 = hyperparameters[obj]['first']
	morph2, kernel2, iter2 = hyperparameters[obj]['second']

	# First Morphological Operation
	if 'additional_mask_space' in hyperparameters[obj]:

		# Add an additional masking ellipse tot ake into account the board
		y1, y2, x1, x2 = hyperparameters[obj]['additional_mask_space']
		additional_mask = np.full(frame.shape[:2], 0, np.uint8)
		additional_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

		morph_op_1 = cv.bitwise_or(
			cv.morphologyEx(additional_mask, morph1, kernel1, iterations = iter1),
			mask
		)		
	else:
		# In case we do not have to worry about the mask we just compute the specific morphological operation
		morph_op_1 = cv.morphologyEx(mask, morph1, kernel1, iterations = iter1)

	return cv.morphologyEx(morph_op_1, morph2, kernel2, iterations = iter2)