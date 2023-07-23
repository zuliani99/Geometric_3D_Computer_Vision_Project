import numpy as np
import cv2 as cv
import time
import copy
import math

objs = ['obj01.mp4']#, 'obj02.mp4', 'obj03.mp4', 'obj04.mp4']

origin = (0,0)
A = (70,0)
angle = math.radians(-15)
marker_reference_doords = {}

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def get_marker_reference_coords():
    for id in range(23):
        marker_reference_doords[id] = rotate(origin, A, id * angle)


def find_markers():
    pass

def save_stats():
    pass

def draw_red_polygons(image, actual_fps):
	
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
		if area > 1700: # 1730
			approx = cv.approxPolyDP(cnt, 0.02 * cv.arcLength(cnt, True), True) # 0.015
	
			# Checking if the number of sides of the selected region is 5.
			if(len(approx) == 5): 
    
				cv.drawContours(image, [approx], 0, (0, 0, 255), 2)
    
				hull = cv.convexHull(cnt, returnPoints=False)

				# Find the convexity defects in order to find the concave vertex
				defects = cv.convexityDefects(cnt, hull)

				# Check for concave corners
				if defects is not None:
					for i in range(defects.shape[0]):
						_, _, f, d = defects[i, 0]
						far = tuple(cnt[f][0])

						if d > 1000: # capire perche' funziona
							cv.line(image, (far[0], far[1] - 10), (far[0], far[1] + 10), (0,255,0), 1)
							cv.line(image, (far[0] - 10, far[1]), (far[0] + 10, far[1]), (0,255,0), 1)
    
	return image


if __name__ == "__main__":
    
	get_marker_reference_coords()
	
	for obj in objs:
		print(f'Marker Detector for {obj}')
		input_video = cv.VideoCapture(f"../data/{obj}")

		# Get video properties
		frame_width = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
		frame_height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))
		fps = input_video.get(cv.CAP_PROP_FPS)
		tot_fps = int(input_video.get(cv.CAP_PROP_FRAME_COUNT))

		actual_fps = 0

		# Create output video writer
		output_video = cv.VideoWriter(f"../output_part2/{obj.split('.')[0]}_mask.mp4", cv.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
  
		while input_video.isOpened():
			start = time.time()
			
			# Extract a frame
			ret, frame = input_video.read()

			if not ret:	break
   
			polygon_frame = draw_red_polygons(frame, actual_fps)

			frame_with_fps = copy.deepcopy(polygon_frame)
   
			end = time.time()
   
			fps = 1 / (end-start)
			cv.putText(frame_with_fps, f"{fps:.2f} FPS", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

			cv.imshow(f'Marker Detector of {obj}', frame_with_fps)
   
			# Save frame without the FPS count in it
			output_video.write(polygon_frame)

			if cv.waitKey(1) == ord('q'):
				break

			actual_fps += 1

			
		input_video.release()
		output_video.release()
		cv.destroyAllWindows()


