import numpy as np
import cv2 as cv
import time
import copy
#import sys

from utils import set_marker_reference_coords, resize_for_laptop, write_ply_file
from background_foreground_segmentation import apply_segmentation
from board import Board
from voxels_cube import VoxelsCube


parameters = {
	#'obj01.mp4': {'circle_mask_size': 14, 'window_size': (7, 7), 'undist_axis': 55},
	#'obj02.mp4': {'circle_mask_size': 13, 'window_size': (9, 9), 'undist_axis': 60},
	'obj03.mp4': {'circle_mask_size': 14, 'window_size': (7, 7), 'undist_axis': 70},
	#'obj04.mp4': {'circle_mask_size': 15, 'window_size': (10, 10), 'undist_axis': 55},
}



def main(using_laptop: bool, voxel_cube_dim: int) -> None:
	'''
	PURPOSE: function that start the whole computation
	ARGUMENTS:
		- using_laptop (bool): boolean variable to indicate the usage of a laptop or not
		- voxel_cube_dim (int): pixel dimension of a voxel cube edge
	RETURN: None
	'''
	 
	# Set the marker reference coordinates for the 24 polygonls
	marker_reference = set_marker_reference_coords()
	
	camera_matrix = np.load('./calibration_info/cameraMatrix.npy')
	dist = np.load('./calibration_info/dist.npy')
  
  
	# Iterate for each object
	for obj, hyper_param in parameters.items():
	  
		print(f'Marker Detector for {obj}...')
		input_video = cv.VideoCapture(f"../data/{obj}")
		  
		# Get video properties
		frame_width = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
		frame_height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))

		actual_fps = 0
		avg_fps = 0.0
		obj_id = obj.split('.')[0]

		undistorted_resolution = None
		prev_frameg = None
  
		half_axis_len = hyper_param['undist_axis']

		# Create the Board object
		board = Board(n_polygons=24, circle_mask_size=hyper_param['circle_mask_size'])

		# Create the VoxelsCube object
		voxels_cube = VoxelsCube(half_axis_len=half_axis_len, voxel_cube_dim=voxel_cube_dim, camera_matrix=camera_matrix, dist=dist, frame_width=frame_width, frame_height=frame_height)
  
		# Create output video writer initialized at None since we do not know the undistorted resolution
		output_video = None

		while True:
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
				board.apply_LK_OF(thresh, prev_frameg, frameg, mask, hyper_param['window_size'])

			# Remove the polygon that are convex, order clockwie and remove the alst polygon by area
			reshaped_clockwise = board.polygons_check_and_clockwise()
   
			# Obtain the dictionary of statistics
			pixsl_info = board.compute_markers(thresh, reshaped_clockwise, marker_reference)

			# Draw the marker detector stuff
			edited_frame = board.draw_stuff(frame)
   
   
			if pixsl_info.shape[0] >= 6:
				
				# Extract the 2D and 3D points
				twoD_points = pixsl_info[:,1:3]
				threeD_points = pixsl_info[:,3:6]
    
				# Find the rotation and translation vectors
				undist, imgpts_centroid, imgpts_cube, newCameraMatrix = voxels_cube.apply_projections(twoD_points, threeD_points, edited_frame)
    
				# Apply the segmentation
				undist_mask = apply_segmentation(obj, undist)


				# Update the undistorted_resolution and output_video the first time that the undistorted image resutn a valid shape
				if undistorted_resolution is None: 
					undistorted_resolution = undist.shape[:2]
					output_video = cv.VideoWriter(f'../output_project/{obj_id}/{obj_id}.mp4', cv.VideoWriter_fourcc(*'mp4v'), input_video.get(cv.CAP_PROP_FPS), np.flip(undistorted_resolution))
    
	
				undist = board.draw_origin(undist, (board.centroid[0], undistorted_resolution[0] // 2), np.int32(imgpts_centroid))
				undist = board.draw_cube(undist, np.int32(imgpts_cube))
    
				# Undistorting the segmented frame to analyze the voxels centre
				undist_b_f_image = cv.undistort(undist_mask, camera_matrix, dist, None, newCameraMatrix)
    
				# Update the binary array of foreground voxels and draw the background one
				edited_frame = voxels_cube.set_background_voxels(undistorted_resolution, undist_b_f_image, undist)
    
				
			end = time.time()
			fps = 1 / (end-start)
   
			avg_fps += fps

			# Get the resized frame
			frame_with_fps_resized = resize_for_laptop(using_laptop, copy.deepcopy(edited_frame))
  
			# Output the frame with the FPS   			
			cv.putText(frame_with_fps_resized, f"{fps:.2f} FPS", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
			cv.imshow(f'Pose Estiamtion of {obj}', frame_with_fps_resized)
			   
			# Save the frame without the FPS count in case of no error
			if pixsl_info.shape[0] >= 6: output_video.write(edited_frame)
   
	 
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

		# Get the voxel cube ciooirdinates and faces to write a PLY file
		voxels_cube_coords, voxel_cube_faces = voxels_cube.get_cubes_coords_and_faces()

		print(f'Saving PLY file with {voxels_cube_coords.shape[0]} vertices and {voxel_cube_faces.shape[0]} faces...')
		write_ply_file(obj_id, voxels_cube_coords, voxel_cube_faces)
		print(' DONE\n')

		# Release the input and output streams
		input_video.release()
		output_video.release()
		cv.destroyAllWindows()



if __name__ == "__main__":
	#using_laptop = bool(sys.argv[1])
	#voxel_cube_dim = int(sys.argv[2])
	#if(voxel_cube_dim < 2 or voxel_cube_dim % 2 == 1): print('Insert corrent arguments in command line')
	#else: main(using_laptop, voxel_cube_dim)
	main(False, 2)
 