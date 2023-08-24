import numpy as np
import cv2 as cv
import time
import copy
import argparse
import os

from utils import set_marker_reference_coords, resize_for_laptop, write_ply_file
from background_foreground_segmentation import apply_segmentation
from board import Board
from voxels_cube import VoxelsCube


# Objects undist_axis
parameters = {
	'obj01.mp4': {'undist_axis': 55},
	'obj02.mp4': {'undist_axis': 60},
	'obj03.mp4': {'undist_axis': 75},
	'obj04.mp4': {'undist_axis': 55},
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
	
	# Check if the user run the camera calibration program before
	if not os.path.exists('./calibration_info/cameraMatrix.npy') or not os.path.exists('./calibration_info/dist.npy'):
		print('Please, before running the project, execute the camera calibration program to obtatain the camera extrinsic parameters.')
		return

	# Load the camera matrix and distorsion
	camera_matrix = np.load('./calibration_info/cameraMatrix.npy')
	dist = np.load('./calibration_info/dist.npy')
  
  
	# Iterate for each object
	for obj, hyper_param in parameters.items():
		
		print(f'Marker Detector for {obj}...')
  
		# Create the VideoCapture object
		input_video = cv.VideoCapture(f"../data/{obj}")
		  
		# Get video properties
		frame_width = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
		frame_height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))

		actual_fps = 0
		avg_fps = 0.0
		avg_rmse = 0.0
		obj_id = obj.split('.')[0]

		undistorted_resolution = None
		prev_frameg = None
  
		half_axis_len = hyper_param['undist_axis']

		# Create the Board object
		board = Board(n_polygons=24)

		# Create the VoxelsCube object
		voxels_cube = VoxelsCube(half_axis_len=half_axis_len, voxel_cube_dim=voxel_cube_dim, camera_matrix=camera_matrix, dist=dist, frame_width=frame_width, frame_height=frame_height)
  
		# Create output video writer initialized at None since we do not know the undistorted resolution
		output_video = None

		while True:
      
			start = time.time()
			   
			# Extract a frame
			ret, frame = input_video.read()

			if not ret:	break
   
			# Get the undistorted frame and the new camera matrix
			undist, newCameraMatrix = voxels_cube.get_undistorted_frame(frame)

			# Update the undistorted_resolution, output_video and the centroid
			if undistorted_resolution is None: 
				undistorted_resolution = undist.shape[:2]
				output_video = cv.VideoWriter(f'../output_project/{obj_id}/{obj_id}.mp4', cv.VideoWriter_fourcc(*'mp4v'), input_video.get(cv.CAP_PROP_FPS), np.flip(undistorted_resolution))
				board.set_centroid(np.array([1280, int(undistorted_resolution[0] // 2)]))
			
		   
			frameg = cv.cvtColor(undist, cv.COLOR_BGR2GRAY)
			_, thresh = cv.threshold(frameg, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
   
			   
			if(actual_fps % 5 == 0): 
				# Each 5 frames recompute the whole features to track
				board.find_interesting_points(thresh, frameg)
			else:
				# The other frame use the Lucaks Kanade Optical Flow to estimate the postition of the traked features based on the previous frame
				board.apply_LK_OF(prev_frameg, frameg, (20, 20))

			# Order the detected features in clockwise order to be able to print correctly
			reshaped_clockwise = board.get_clockwise_vertices_initial()   

			# Obtain the np.array of markers information
			markers_info = board.compute_markers(thresh, reshaped_clockwise, marker_reference)
   
			edited_frame = undist
   

			if markers_info.shape[0] >= 6:

				# Draw the marker detector stuff
				edited_frame = board.draw_stuff(edited_frame)
				
				# Extract the indices ID, the 2D and 3D points
				indices_ID = markers_info[:,0]
				twoD_points = markers_info[:,1:3]
				threeD_points = markers_info[:,3:6]
    
    
				# Find the rotation and translation vectors
				imgpts_centroid, imgpts_cube = voxels_cube.apply_projections(twoD_points, threeD_points)

				# Get the RMSE for the actual frame
				avg_rmse += voxels_cube.compute_RMSE(indices_ID, marker_reference, twoD_points)
    
				# Apply the segmentation
				undist_mask = apply_segmentation(obj, edited_frame)

				# Draw the projected cube and centroid
				edited_frame = board.draw_origin(edited_frame, np.int32(imgpts_centroid))
				edited_frame = voxels_cube.draw_cube(edited_frame, np.int32(imgpts_cube))
    
				# Undistorting the segmented frame to analyze the voxels centre
				undist_b_f_image = cv.undistort(undist_mask, camera_matrix, dist, None, newCameraMatrix)
    
				# Update the binary array of foreground voxels and draw the background
				edited_frame = voxels_cube.set_background_voxels(undistorted_resolution, undist_b_f_image, edited_frame)
    
				
			end = time.time()
			fps = 1 / (end-start)
   
			avg_fps += fps

			# Get the resized frame
			frame_with_fps_resized = resize_for_laptop(using_laptop, copy.deepcopy(edited_frame))
  
			# Output the frame with the FPS   			
			cv.putText(frame_with_fps_resized, f"{fps:.2f} FPS", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
			cv.imshow(f'Pose Estiamtion of {obj}', frame_with_fps_resized)
			   
			# Save the frame without the FPS count
			output_video.write(edited_frame)
   
	 		# Update the previous gray frame
			prev_frameg = frameg
   
			actual_fps += 1

			key = cv.waitKey(1)
			if key == ord('p'):
				cv.waitKey(-1) 
   
			if key == ord('q'):
				return
		  


		print(' DONE')
		print(f'Average FPS is: {str(avg_fps / int(input_video.get(cv.CAP_PROP_FRAME_COUNT)))}')
		print(f'Average RMSE is: {str(avg_rmse / int(input_video.get(cv.CAP_PROP_FRAME_COUNT)))}')


		# Release the input and output streams
		input_video.release()
		output_video.release()
		cv.destroyAllWindows()

		print('Saving PLY file...')
  
		# Get the voxel cube coordinates and faces to write a PLY file
		voxels_cube_coords, voxel_cube_faces = voxels_cube.get_cubes_coords_and_faces()
		# Save in a .ply file
		write_ply_file(obj_id, voxels_cube_coords, voxel_cube_faces)
		print(' DONE\n')



if __name__ == "__main__":
    
    # Get the console arguments
	parser = argparse.ArgumentParser(prog='SpaceCarving', description="Space Carving Project")
	parser.add_argument('--hd_laptop', dest='hd_laptop', default=False, action='store_true', help="Using a 720p resolution")
	parser.add_argument("voxel_cube_dim", type=int, help="Dimension of a voxel cube edge")
	args = parser.parse_args()
	
	main(args.hd_laptop, args.voxel_cube_dim)
 