
import cv2 as cv
import numpy as np
from typing import Dict, Tuple


# Hyperparameters definitions
hyperparameters = {
	'obj01.mp4': {
		'clipLimit': 8,
     	'first': (cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (4,4)), 11),
      	'second': (cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3)), 9), #10?
		'additional_mask_space': (300, 900, 270, 900),
       	'correction': (np.array([105,65,5]), np.array([140,255,255]))
    },
	'obj02.mp4': {
		'clipLimit': 8,
    	'first': (cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (2,2)), 3),
    	'second': (cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3)), 5),
     	'correction': (np.array([105,65,5]), np.array([140,255,255]))
    },
	'obj03.mp4': {
		'clipLimit': 6,
    	'first': (cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5)), 11),
     	'second': (cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3)), 10),
		'additional_mask_space': (100, 1000, 500, 1000),
    	'correction': (np.array([105,55,0]), np.array([120,255,255]))
    },
	'obj04.mp4': {
		'clipLimit': 3,
    	'first': (cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3)), 10),
     	'second': (cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (2,2)), 6),
      	'correction': (np.array([105,55,0]), np.array([120,255,255]))
    }
}

def change_contrast(img: np.ndarray, clipLimit: int):
	'''
	PURPOSE: change the contrast of the image 
	ARGUMENTS:
		- img (np.ndarray): image where apply the contrast changing
		- clipLimit
	RETURN:
		- (np.ndarray) updated image
	'''
    
	lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)
	l_channel, a, b = cv.split(lab)

	# Applying CLAHE to L-channel
	l_clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=(20,20))
	l_channel = l_clahe.apply(l_channel)

	# Merge the CLAHE enhanced L-channel with the a and b channel
	limg = cv.merge((l_channel,a,b))

	# Converting image from LAB Color model to BGR color spcae
	return cv.cvtColor(limg, cv.COLOR_LAB2RGB)


def apply_foreground_background(mask: np.ndarray, img: np.ndarray) -> np.ndarray:
    '''
	PURPOSE: apply the foreground and background segmentation
	ARGUMENTS:
		- mask (np.ndarray)
		- img (np.ndarray): image where apply the segmentation 
	RETURN:
		- segmented (np.ndarray): black and white segmented image
	'''
    
    segmented = img
    
    foreground = np.where(mask==255)
    background = np.where(mask==0)
    
    segmented[foreground[0], foreground[1], :] = [255, 255, 255]
    segmented[background[0], background[1], :] = [0, 0, 0]
    
    return segmented
    


def apply_segmentation(obj: str, frame: np.ndarray):
	'''
	PURPOSE: apply the segmentation with all the color conversions and morphological operations
	ARGUMENTS:
		- obj (str): object name string
		- frame (np.ndarray): image video frame
	RETURN:
		- morph_op_2 (np.ndarray): final mask
		- apply_foreground_background return
	'''
    # Convert the imahe into RGB format
	rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
 
	# Change the image contast and confvert it into HSV format
	enhanced = cv.cvtColor(change_contrast(rgb, hyperparameters[obj]['clipLimit']), cv.COLOR_RGB2HSV)
	
	# Define the colored mask obtained from the image
	color_mask = cv.bitwise_not(cv.inRange(enhanced, hyperparameters[obj]['correction'][0], hyperparameters[obj]['correction'][1]))

	# Add a constant rectangle to mask the left part of the image
	rectangular_mask = np.full(rgb.shape[:2], 0, np.uint8)
	rectangular_mask[:,1210:rgb.shape[1]] = 255
 
	mask = cv.bitwise_or(cv.ellipse(color_mask,(1370,540),(670,250),89,0,180,255,-1), rectangular_mask)
	
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
		# In case we do not have to worry about the mask just compute the specific morphological operation
		morph_op_1 = cv.morphologyEx(mask, morph1, kernel1, iterations = iter1)
 
	# Second Morphological Operation
	morph_op_2 = cv.morphologyEx(morph_op_1, morph2, kernel2, iterations = iter2)

	return morph_op_2, apply_foreground_background(morph_op_2, rgb)



if __name__ == "__main__":
	
	for obj in list(hyperparameters.keys()):
		print(f'Segmentation of {obj} video')		
  
		input_video = cv.VideoCapture(f"../data/{obj}")

		# Get video properties
		frame_width = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
		frame_height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))
		fps = input_video.get(cv.CAP_PROP_FPS)
		tot_fps = int(input_video.get(cv.CAP_PROP_FRAME_COUNT))

		# Create output video writer
		output_video = cv.VideoWriter(f"../output_part1/{obj.split('.')[0]}_mask.mp4", cv.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
  
		while True:
      
			# Extract a frame
			ret, frame = input_video.read()

			if not ret:	break

			# Apply the segmentation
			resulting_mask, segmented_frame = apply_segmentation(obj, frame)

			# Draw the contours of the mask in the original frame
			contours_obj, _ = cv.findContours(resulting_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
			cv.drawContours(frame, contours_obj, -1, (0,255,0), 3)
   
			output_video.write(segmented_frame)

			# Display the segmented frame and the contourns
			cv.imshow(f"Segmented Video of {obj}", segmented_frame)
			cv.imshow(f"Countourns Segmentation of {obj}", frame)
   
			if cv.waitKey(1) == ord('q'):
				break
			
			
		input_video.release()
		output_video.release()
		cv.destroyAllWindows()
