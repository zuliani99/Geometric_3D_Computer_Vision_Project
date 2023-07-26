import numpy as np
import cv2 as cv
import time
import copy
import csv
import math

objs = ['obj01.mp4', 'obj02.mp4', 'obj03.mp4', 'obj04.mp4']

origin = (0,0)
A = (70,0)
angle = math.radians(-15)
marker_reference_coords = {}


# Set the needed parameters to find the refined corners
winSize = (5, 5)
zeroZone = (-1, -1)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)



# Rotate a point counterclockwise by a given angle around a given origin.
def rotate(origin, point, angle):
	
	ox, oy = origin
	px, py = point

	qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
	qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
	return qx, qy, 0


def get_marker_reference_coords():
	for id in range(24): # We have 24 markers
		marker_reference_coords[id] = rotate(origin, A, id * angle)
  

def save_stats(obj_id, dict_stats):
	fields = ['frame', 'mark_id', 'Px', 'Py', 'X', 'Y', 'Z'] 
    
	# name of csv file 
	filename = f'../output_part2/{obj_id}/{obj_id}_marker.csv'
		
	# writing to csv file 
	with open(filename, 'w') as csvfile: 
		# creating a csv dict writer object 
		writer = csv.DictWriter(csvfile, fieldnames = fields) 
			
		# writing headers (field names) 
		writer.writeheader() 
			
		# writing data rows 
		writer.writerows(dict_stats) 
	


def find_middle_pint(p1, p2):
	return (p1[0] + p2[0])/2, (p1[1] + p2[1])/2


def distanceCalculate(p1, p2):
	return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5


def find_markers(image, frame_cnt):
	
	imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	_, thresh = cv.threshold(imgray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
	mask_thresh = np.zeros((1080, 1920), dtype=np.uint8)

	# Detecting shapes in image by selecting region with same colors or intensity
	# Consider only the board exluding all the object area that could be included erroneously
	mask_thresh[:, 1050:1600] = thresh[:, 1050:1600]

	contours, _ = cv.findContours(mask_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
 
	dict_stats_to_return = []

	# Searching through every region selected to find the required polygon.
	for cnt in contours:
	   
		# Shortlisting the regions based on there area.
		if cv.contourArea(cnt) > 1720: # 1730
			  
			approx_cnt = cv.approxPolyDP(cnt, 0.02 * cv.arcLength(cnt, True), True) # 0.015
			
   			
			# Checking if the number of sides of the selected region is 5.
			if (len(approx_cnt) == 5): 

				external_points_dict = dict(enumerate(
					list(map(lambda x: distanceCalculate(x[0], np.array([1350,570])), approx_cnt))
				))

				id_external_points = sorted(external_points_dict.items(), key=lambda x:x[1])[-2:]
    
				middle_point = find_middle_pint(
        			approx_cnt[id_external_points[0][0]][0],
           			approx_cnt[id_external_points[1][0]][0]
              	)
    
				hull = cv.convexHull(cnt, returnPoints=False)

				# Find the convexity defects in order to find the concave vertex
				defects = cv.convexityDefects(cnt, hull)

				# Check for concave corners
				for i in range(defects.shape[0]):
					_, _, f, d = defects[i, 0]

					if d > 1000: # capire perche' funziona
						     
						cv.drawContours(image, [approx_cnt], 0, (0, 0, 255), 2, cv.LINE_AA)
         
						A = cnt[f][0]
      
        		 		# Calculate the refined corner locations
						A_refined = cv.cornerSubPix(imgray, np.float32(cnt), winSize, zeroZone, criteria)[f][0]
					
						cv.line(image, (A[0], A[1] - 10), (A[0], A[1] + 10), (0,255,0), 1)
						cv.line(image, (A[0] - 10, A[1]), (A[0] + 10, A[1]), (0,255,0), 1)

						cv.line(image, A, np.int32(middle_point), (0, 255, 255), 1, cv.LINE_AA)
      
						dist_A_Ext_Mid = distanceCalculate(A, np.int32(middle_point))
      
						dist_A_Cir_Ctr = [(dist_A_Ext_Mid*((i*4.5) + 5) / 28) for i in range(5)]
      
						circles_ctr_coords = []
			
						for dist in dist_A_Cir_Ctr:
							rateo = dist / dist_A_Ext_Mid
							dx = middle_point[0] - A[0]
							dy = middle_point[1] - A[1]
							new_point_x = A[0] + (rateo * dx)
							new_point_y = A[1] + (rateo * dy)
							circles_ctr_coords.append(((new_point_y, new_point_x)))

							cv.circle(image, [np.int32(new_point_x), np.int32(new_point_y)], 2, (255,0,0), 2)

						bit_index = [1 if thresh[np.int32(coords[0]), np.int32(coords[1])] == 0 else 0 for coords in circles_ctr_coords]
			
			
						index = int("".join(str(x) for x in reversed(bit_index)), 2)
			
						cv.putText(image, str(index), A, cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 8, cv.LINE_AA)
						cv.putText(image, str(index), A, cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA)
      
						X, Y, Z = marker_reference_coords[index]
      
						dict_stats_to_return.append({'frame': frame_cnt, 'mark_id': index, 'Px': A_refined[0], 'Py': A_refined[1], 'X': X, 'Y': Y, 'Z': Z})

      
	return image, dict_stats_to_return


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
		obj_id = obj.split('.')[0]
  
		dict_stats = []

		# Create output video writer
		output_video = cv.VideoWriter(f"../output_part2/{obj_id}/{obj_id}_mask.mp4", cv.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
  
		while input_video.isOpened():
			start = time.time()
			
			# Extract a frame
			ret, frame = input_video.read()

			if not ret:	break
   
	  
			edited_frame, dict_stats_to_extend = find_markers(frame, actual_fps)


			frame_with_fps = copy.deepcopy(edited_frame)
   
			end = time.time()
			fps = 1 / (end-start)
   
			cv.putText(frame_with_fps, f"{fps:.2f} FPS", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
			cv.imshow(f'Marker Detector of {obj}', frame_with_fps)
   
			# Save frame without the FPS count in it
			output_video.write(edited_frame)

			dict_stats.extend(dict_stats_to_extend)

			if cv.waitKey(1) == ord('q'):
				break

			actual_fps += 1

		
		save_stats(obj_id, dict_stats)

  
		input_video.release()
		output_video.release()
		cv.destroyAllWindows()


