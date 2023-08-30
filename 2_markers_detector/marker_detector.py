import cv2 as cv
import numpy as np
import time
import copy
import argparse
import os

from board import Board
from utils import save_stats, set_marker_reference_coords, resize_for_laptop

objs = ['obj01.mp4', 'obj02.mp4', 'obj03.mp4', 'obj04.mp4']


def main(using_laptop: bool) -> None:
	'''
	PURPOSE: function that start the whole computation
	ARGUMENTS:
		- using_laptop (bool): boolean variable to indicate the usage of a laptop or not
	RETURN: None
	'''
	
	# Set the marker reference coordinates for the 24 polygonls
	marker_reference = set_marker_reference_coords()
 
	# Check if the user run the camera calibration program before
	if not os.path.exists('../../3_pose_estimation/calibration_info/cameraMatrix.npy') or not os.path.exists('../../3_pose_estimation/calibration_info/dist.npy'):
		print('Please, before running the pose estimation execute the camera calibration program.')
		return

	camera_matrix = np.load('../../3_pose_estimation/calibration_info/cameraMatrix.npy')
	dist = np.load('../../3_pose_estimation/calibration_info/dist.npy')

	# Iterate for each object
	for obj in objs:
	 
		print(f'Marker Detector for {obj}...')
  
		# Create the VideoCapture object
		input_video = cv.VideoCapture(f"../../data/{obj}")
		
		# Get video properties
		frame_width = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
		frame_height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))
  
		# Get the new camera intrinsic matrix based on the free scaling parameter
		newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist, (frame_width, frame_height), 1, (frame_width, frame_height))

		actual_fps = 0
		avg_fps = 0.0
		obj_id = obj.split('.')[0]

		# Create the Board object
		board = Board(n_polygons=24)
  
		dict_stats = [] # Initialize the list of dictionary that we will save as .csv file
		
  		# Create output video writer
		output_video = None

		prev_frameg = None

		while True:
            
			start = time.time()
			
			# Extract a frame
			ret, frame = input_video.read()

			if not ret:	break

			# Get the undistorted frame
			frame = cv.undistort(frame, camera_matrix, dist, None, newCameraMatrix)	
			x, y, w, h = roi
			frame = frame[y:y+h, x:x+w]

			# Update the output_video and the centroid
			if output_video is None:
				frame_width, frame_height = frame.shape[1], frame.shape[0] 
				output_video = cv.VideoWriter(f"../../output_part2/{obj_id}/{obj_id}_marker.mp4", cv.VideoWriter_fourcc(*"mp4v"), input_video.get(cv.CAP_PROP_FPS), (frame_width, frame_height))
				board.set_centroid(np.array([1280, int(frame_height // 2)]))
				
		 
			frameg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
			_, thresh = cv.threshold(frameg, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
   
			
			if(actual_fps % 5 == 0):
				# Each 5 frames recompute tqhe whole features to track
				board.find_interesting_points(thresh, frameg)
			else: 
				# The other frame use the Lucas-Kanade Optical Flow to estimate the postition of the traked features based on the previous frame
				board.apply_LK_OF(prev_frameg, frameg, (20, 20))
	
			# Order the detected features in clockwise order to be able to print correctly
			reshaped_clockwise = board.get_clockwise_vertices_initial()

			# Obtain the dictionary of statistics
			dict_stats_to_extend = board.compute_markers(thresh, reshaped_clockwise, actual_fps, marker_reference)

			# Draw the marker detector stuff
			edited_frame = board.draw_stuff(frame)

			end = time.time()
			fps = 1 / (end-start)
   
			avg_fps += fps

			# Get the resized frame
			frame_with_fps_resized = resize_for_laptop(using_laptop, copy.deepcopy(frame))
  
			# Output the frame with the FPS
			cv.putText(frame_with_fps_resized, f"{fps:.2f} FPS", (20, 20), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
	    
    # Get the console arguments
	parser = argparse.ArgumentParser(prog='Assignment2-Marker_Detector', description="Marker Detector")
	parser.add_argument('--hd_laptop', dest='hd_laptop', default=False, action='store_true', help="Using a 720p resolution")
	args = parser.parse_args()
 
	main(args.hd_laptop)
