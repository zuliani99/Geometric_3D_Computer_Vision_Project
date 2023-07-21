import numpy as np
import cv2 as cv

def print_fps():
    pass

def find_markers():
    pass

def draw_polygons(image):
	
	imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	_, thresh = cv.threshold(imgray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
	mask_thresh = np.zeros((1080, 1920), dtype=np.uint8)
	
	# Detecting shapes in image by selecting region with same colors or intensity.

	mask_thresh[:, 1050:1600] = thresh[:, 1050:1600]
 
	# thresh[:, 1050:1600] -> consider only the board exluding all the object area that could be included erroneously
	contours, _ = cv.findContours(mask_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
 	
	# Searching through every region selected to find the required polygon.
	for cnt in contours:
		area = cv.contourArea(cnt)

		# Shortlisting the regions based on there area.
		if area > 1730:
			approx = cv.approxPolyDP(cnt, 0.015 * cv.arcLength(cnt, True), True)
	
			# Checking if the number of sides of the selected region is 5.
			if(len(approx) == 5): 
				# scaling
				cv.drawContours(image, [approx], 0, (0, 0, 255), 3)
    
	return image


if __name__ == "__main__":
	
		obj = 'obj01.mp4'
		paused = True
  
		input_video = cv.VideoCapture(f"../data/{obj}")

		while input_video.isOpened():
      
			# Extract a frame
			ret, frame = input_video.read()

			if not ret:	break
   
			polygon_frame = draw_polygons(frame)
   
			cv.imshow('Marker Detector', polygon_frame)

			#if cv.waitKey(1) == ord('q'):
				#break

			key = cv.waitKey(0 if paused else 10)
			if key==32:
				paused = not paused
			
		input_video.release()
		cv.destroyAllWindows()


