import cv2 as cv
import numpy as np
import time
import copy

from board import Board
from utils import save_stats, set_marker_reference_coords, resize_for_laptop


using_laptop = False

# Dictionary of object file name that we have to process with associated parameters
parameters = {
	'obj01.mp4': {'circle_mask_size': 15, 'window_size': (10, 10)},
	'obj02.mp4': {'circle_mask_size': 13, 'window_size': (9, 9)},
	'obj03.mp4': {'circle_mask_size': 13, 'window_size': (9, 9)},
	'obj04.mp4': {'circle_mask_size': 15, 'window_size': (10, 10)},
}



def main():
	
	# Set the marker reference coordinates for the 24 polygonls
	marker_reference = set_marker_reference_coords()

	# Iterate for each object
	for obj, hyper_param in parameters.items():
	 
		print(f'Marker Detector for {obj}...')
		input_video = cv.VideoCapture(f"../../data/{obj}")
		
		# Get video properties
		frame_width = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
		frame_height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))

		actual_fps = 0
		avg_fps = 0.0
		obj_id = obj.split('.')[0]

		board = Board(n_polygons=24, circle_mask_size=hyper_param['circle_mask_size'])
  
		dict_stats = [] # Initialize the list of dictionary that we will save as .csv file
		
  		# Create output video writer
		output_video = cv.VideoWriter(f"../../output_part2/{obj_id}/{obj_id}_mask.mp4", cv.VideoWriter_fourcc(*"mp4v"), input_video.get(cv.CAP_PROP_FPS), (frame_width, frame_height))

		prev_frameg = None

		while True:
			#print('\n\n-------------------------------------', actual_fps, '-------------------------------------')
			start = time.time()
			
			# Extract a frame
			ret, frame = input_video.read()

			if not ret:	break

		 
			frameg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
			_, thresh = cv.threshold(frameg, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
			mask = np.zeros_like(frameg)
   
			
			if(actual_fps % 10 == 0): 
				# Each 10 frames recompute the whole features to track
				board.find_interesting_points(thresh, frameg, mask)
			else: 
				# The other frame use the Lucaks Kanade Optical Flow to estimate the postition of the traked features based on the previous frame
				board.apply_LF_OF(thresh, prev_frameg, frameg, mask, hyper_param['window_size'])
	
			
			#reshaped_clockwise = board.get_clockwise_vertices_initial()
			reshaped_clockwise = board.polygons_check_and_clockwise()
			  
			# Obtain the dictionary of statistics
			dict_stats_to_extend = board.compute_markers(thresh, reshaped_clockwise, actual_fps, marker_reference)

			edited_frame = board.draw_stuff(frame)

			end = time.time()
			fps = 1 / (end-start)
   
			avg_fps += fps

			# Get the resized frame
			frame_with_fps_resized = resize_for_laptop(using_laptop, copy.deepcopy(frame))
  
			# Output the frame with the FPS
			cv.putText(frame_with_fps_resized, f"{fps:.2f} FPS", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
			cv.imshow(f'Marker Detector of {obj}', frame_with_fps_resized)
			
			# Save the frame without the FPS count
			output_video.write(edited_frame)
			
			# Extends our list with the obtained dictionary
			dict_stats.extend(dict_stats_to_extend)
	
			prev_frameg = frameg
   
			actual_fps += 1

			key = cv.waitKey(1)
			if key == ord('p'):
				cv.waitKey(-1) 
   
			if key == ord('q'):
				return

			#cv.waitKey(-1)
			
		print(' DONE')
  
		print(f'Average FPS is: {str(avg_fps / int(input_video.get(cv.CAP_PROP_FRAME_COUNT)))}')
  
		print('Saving data...')
		save_stats(obj_id, dict_stats)
		print(' DONE\n')
  
		# Release the input and output streams
		input_video.release()
		output_video.release()
		cv.destroyAllWindows()



if __name__ == "__main__":
	main()
 
 
 
 
 #reshaped_clockwise = board.polygons_check_and_clockwise2()
   
'''			for poly in reshaped_clockwise:
				cv.drawContours(frame, [np.int32(poly)], 0, (0, 0, 255), 1, cv.LINE_AA) # controllo se nei frame sbaglaiti c'e' solo un punto
				poly_ordered = sort_vertices_clockwise(poly, board.centroid)
    
    
    
				cv.line(frame, np.int32(poly_ordered[0,:]), np.int32(poly_ordered[1,:]), (0, 255, 255), 2, cv.LINE_AA) 
				cv.line(frame, np.int32(poly_ordered[3,:]), np.int32(poly_ordered[4,:]), (255, 0, 255), 2, cv.LINE_AA)
    
    
    
				cv.circle(frame, np.int32(poly_ordered[1,:]), 3, (255,255,255), -1)
				cv.circle(frame, np.int32(poly_ordered[3,:]), 3, (0,0,0), -1)
    
				#parallel_y = are_lines_near_parallel_to_y_axis(np.array([poly_ordered[1,:], poly_ordered[0,:]]), np.array([poly_ordered[4,:], poly_ordered[3,:]]))
				parallel = are_lines_parallel(np.array([poly_ordered[1,:], poly_ordered[0,:]]), np.array([poly_ordered[4,:], poly_ordered[3,:]]))
				#print(parallel, parallel_y)
    
				#if(not parallel_y and  not parallel):
				if(parallel):# and not parallel):
					print('parallel')
					cv.drawContours(frame, [np.int32(poly)], 0, (255, 255, 0), 2, cv.LINE_AA)# controllo se nei frame sbaglaiti c'e' solo un punto

    
				#print('PARALLELE?? ', are_lines_parallel(np.array([poly[1,:], poly[0,:]]), np.array([poly[4,:], poly[3,:]])))'''
				
