
import cv2 as cv
import numpy as np
from tqdm import tqdm

kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))

hyperparameters = {
	'obj01.mp4': {
     	'closing':(kernel2, 5), 'opening': (kernel2, 4)
    },
	'obj02.mp4': {
    	'closing': (kernel1, 4), 'opening': (kernel2, 3)
    },
	'obj03.mp4': {
    	'closing': (kernel1, 4), 'opening': (kernel2, 4)
    },
	'obj04.mp4': {
    	'closing': (kernel1, 4), 'opening': (kernel2, 8)
    }
}


def change_contrast(img):
	lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)
	l_channel, a, b = cv.split(lab)

	# Applying CLAHE to L-channel
	l_clahe = cv.createCLAHE(clipLimit=8, tileGridSize=(13,13))
	l_channel = l_clahe.apply(l_channel)

	# merge the CLAHE enhanced L-channel with the a and b channel
	limg = cv.merge((l_channel,a,b))

	# Converting image from LAB Color model to BGR color spcae
	return cv.cvtColor(limg, cv.COLOR_LAB2RGB)



def apply_foreground_background(mask, img):
    segmented = img
    
    foreground = np.where(mask==255)
    background = np.where(mask==0)
    
    segmented[foreground[0], foreground[1], :] = [255, 255, 255]
    segmented[background[0], background[1], :] = [0, 0, 0]
    
    return segmented
    


def apply_segmentation(frame):
	rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
	enhanced = cv.cvtColor(change_contrast(rgb), cv.COLOR_RGB2HSV)
	color_mask = cv.bitwise_not(cv.inRange(enhanced, np.array([105,70,0]), np.array([165,255,255])))
 
	rectangular_mask = np.full(rgb.shape[:2], 0, np.uint8)
	rectangular_mask[:,1210:rgb.shape[1]] = 255

	mask = cv.bitwise_or(color_mask, rectangular_mask)
		
	# CLOSING
	closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, hyperparameters[obj]['closing'][0], iterations = hyperparameters[obj]['closing'][1])
 
	# OPENING
	opening = cv.morphologyEx(closing, cv.MORPH_OPEN, hyperparameters[obj]['opening'][0], iterations=hyperparameters[obj]['opening'][1])

	return apply_foreground_background(opening, rgb)



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
		output_video = cv.VideoWriter(f"./output/{obj.split('.')[0]}_mask.mp4", cv.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
  
		while True:
			ret, frame = input_video.read()

			if not ret:	break
		
			segmented_frame = apply_segmentation(frame)


			output_video.write(segmented_frame)

			cv.imshow(f"Segmented Video of {obj}",segmented_frame)
			if cv.waitKey(1) == ord('q'):
				break
			
			
		input_video.release()
		output_video.release()
		cv.destroyAllWindows()
