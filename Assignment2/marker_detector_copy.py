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


# Set the needed parameters to find the refined corners
winSize = (5, 5)
zeroZone = (-1, -1)
criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 40, 0.001)



# Rotate a point counterclockwise by a given angle around a given origin.
def rotate(origin, point, angle):
	
	ox, oy = origin
	px, py = point

	qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
	qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
	return qx, qy, 0


def get_marker_reference_coords():
	for id in range(24): # We have 24 markers
		marker_reference_doords[id] = rotate(origin, A, id * angle)
  

def save_stats():
	pass


def find_middle_pint(p1, p2):
	return (p1[0] + p2[0])/2, (p1[1] + p2[1])/2


def distanceCalculate(p1, p2):
	return ((p2[0][0] - p1[0][0]) ** 2 + (p2[0][1] - p1[0][1]) ** 2) ** 0.5


def draw_red_polygons(image, actual_fps):
	
	imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	_, thresh = cv.threshold(imgray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
	mask_thresh = np.zeros((1080, 1920), dtype=np.uint8)

	# Detecting shapes in image by selecting region with same colors or intensity.

	# Consider only the board exluding all the object area that could be included erroneously
	mask_thresh[:, 1050:1600] = thresh[:, 1050:1600]

	contours, _ = cv.findContours(mask_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


	#criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)


	# Searching through every region selected to find the required polygon.
	for cnt in contours:
	   
		# Shortlisting the regions based on there area.
		if cv.contourArea(cnt) > 1725: # 1730
			  
			approx_cnt = cv.approxPolyDP(cnt, 0.02 * cv.arcLength(cnt, True), True) # 0.015

			#corners = cv.cornerSubPix(imgray,np.float32(approx_cnt),(5,5),(-1,-1),criteria)

			# Checking if the number of sides of the selected region is 5.
			if (len(approx_cnt) == 5): 
			       
				external_points_list = list(map(lambda x: distanceCalculate(x, np.array([[1350,570]])), approx_cnt))
				external_points_dict = dict(enumerate(external_points_list))

				id_external_points = sorted(external_points_dict.items(), key=lambda x:x[1])[-2:]

				# Calculate the refined corner locations
    

				hull = cv.convexHull(cnt, returnPoints=False)

				external_point_coords = [approx_cnt[id_external_points[0][0]][0], approx_cnt[id_external_points[1][0]][0]]
				middle_point = find_middle_pint(external_point_coords[0], external_point_coords[1])

				cv.circle(image, np.int32(external_point_coords[0]), 4, (255,0,255), -1)
				cv.circle(image, np.int32(external_point_coords[1]), 4, (255,255,0), -1)

				# Find the convexity defects in order to find the concave vertex
				defects = cv.convexityDefects(cnt, hull)

				# Check for concave corners
				for i in range(defects.shape[0]):
					_, _, f, d = defects[i, 0]
					A = cnt[f][0]
					#print(A)

					if d > 1000: # capire perche' funziona
						cv.line(image, (A[0], A[1] - 10), (A[0], A[1] + 10), (0,255,0), 1)
						cv.line(image, (A[0] - 10, A[1]), (A[0] + 10, A[1]), (0,255,0), 1)

						cv.line(image, A, np.int32(middle_point), (0, 255, 255), 1)
      
      
						#print(A, np.int32(middle_point))
						distance_A_middle = distanceCalculate([A], [np.int32(middle_point)])
			
						bit_distance_from_A = [(distance_A_middle*((i*4.5) + 5) / 32.5) for i in range(5)]
			
						circe_centers_coords = []
			
						for dist in bit_distance_from_A:
							rateo = dist / distance_A_middle
							dx = A[0] - middle_point[0]
							dy = A[1] - middle_point[1]
							circe_centers_coords.append((A[0] + (rateo * dx), (A[1] + (rateo * dy))))

							cv.circle(image, [np.int32(A[0] + (rateo * dx)), np.int32(A[1] + (rateo * dy))], 2, (0,0,255), 2)

						#bit_conversion = [1 if thresh[np.int32(coords[1]), np.int32(coords[0])] == 0 else 0 for coords in circe_centers_coords]
						bit_conversion = []
						for coords in circe_centers_coords:
							#print(coords, thresh[np.int32(coords[1]), np.int32(coords[0])])
							if thresh[np.int32(coords[1]), np.int32(coords[0])] == 0:
								bit_conversion.append(1)
							else:
								bit_conversion.append(0)
			
			
						index = int("".join(str(x) for x in bit_conversion), 2)
			
						cv.putText(image, str(index), A, cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv.LINE_AA)
    
    
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


