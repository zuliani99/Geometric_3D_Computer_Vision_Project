import numpy as np
import cv2 as cv
import time
import copy


from utils import set_marker_reference_coords, resize_for_laptop, get_cube_and_centroids_voxels, write_ply_file
from background_foreground_segmentation import apply_segmentation
from board import Board


parameters = {
	'obj01.mp4': {'circle_mask_size': 15, 'window_size': (10, 10), 'undist_axis': 55},
	#'obj02.mp4': {'circle_mask_size': 13, 'window_size': (9, 9), 'undist_axis': 60},
	#'obj03.mp4': {'circle_mask_size': 13, 'window_size': (9, 9), 'undist_axis': 70},
	#'obj04.mp4': {'circle_mask_size': 15, 'window_size': (10, 10), 'undist_axis': 55},
}


using_laptop = False
voxel_cube_dim = 4



def main():
	 
	# Set the marker reference coordinates for the 24 polygonls
	marker_reference = set_marker_reference_coords()
 
	camera_matrix = np.load('./calibration_info/cameraMatrix.npy')
	dist = np.load('./calibration_info/dist.npy')
 
	axis_centroid = np.float32([[20,0,0], [0,20,0], [0,0,30]]).reshape(-1,3)
	 
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
  
		unidst_axis = hyper_param['undist_axis']
  
		axis_vertical_edges = np.float32([
											[-unidst_axis, -unidst_axis, 70], [-unidst_axis, unidst_axis, 70],
											[unidst_axis ,unidst_axis, 70], [unidst_axis, -unidst_axis, 70],
											[-unidst_axis, -unidst_axis, 70 + unidst_axis * 2],[-unidst_axis, unidst_axis, 70 + unidst_axis * 2],
											[unidst_axis, unidst_axis, 70 + unidst_axis * 2],[unidst_axis, -unidst_axis, 70 + unidst_axis * 2]
										])


		# Get the coordinates of the voxel center (3D marker reference)
		center_voxels, cube_coords_centroid = get_cube_and_centroids_voxels(unidst_axis, voxel_cube_dim)
  
		# Initialize an index array to mark the mantained voxel that will determine the object volume
		binary_centroid_fore_back = np.ones((np.power(center_voxels.shape[0], 3), 1), dtype=np.int32)
		#print('binary_centroid_fore_back', binary_centroid_fore_back.shape)

		# create the board object
		board = Board(n_polygons=24, circle_mask_size=hyper_param['circle_mask_size'])
  
		# Create output video writer initialized at None since we do not know the undistorted resolution
		output_video = None

		while True:
			#print(f'------------------------------ {actual_fps} ------------------------------')
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
			pixsl_info = board.compute_markers(thresh, reshaped_clockwise, marker_reference)

			# Draw the marker detector stuff
			edited_frame = board.draw_stuff(frame)
   
   
			if pixsl_info.shape[0] >= 6:
       
				twoD_points = pixsl_info[:,1:3]
				threeD_points = pixsl_info[:,3:6]
    

				# Find the rotation and translation vectors
				ret, rvecs, tvecs = cv.solvePnP(objectPoints=threeD_points.astype('float32'), imagePoints=twoD_points.astype('float32'), cameraMatrix=camera_matrix, distCoeffs=dist, flags=cv.SOLVEPNP_IPPE)
  
				imgpts_cubes_centroid, _ = cv.projectPoints(objectPoints=np.reshape(center_voxels, (np.power(center_voxels.shape[0], 3), 3)), rvec=rvecs, tvec=tvecs, cameraMatrix=camera_matrix, distCoeffs=dist)

				imgpts_centroid, _ = cv.projectPoints(objectPoints=axis_centroid, rvec=rvecs, tvec=tvecs, cameraMatrix=camera_matrix, distCoeffs=dist)
				imgpts_cube, _ = cv.projectPoints(objectPoints=axis_vertical_edges, rvec=rvecs, tvec=tvecs, cameraMatrix=camera_matrix, distCoeffs=dist)
			   		  	 
				newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist, (frame_width, frame_height), 1, (frame_width, frame_height))
		  
				imgpts_cubes_centroid = np.squeeze(imgpts_cubes_centroid)

    
				# Undistort the image
				undist = cv.undistort(edited_frame, camera_matrix, dist, None, newCameraMatrix)	
				x, y, w, h = roi
				undist = undist[y:y+h, x:x+w] # Adjust the image resolution
    
    
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
				#cv.imshow('segmented', undist_b_f_image)
    
    
				for idx, centr_coords in enumerate(imgpts_cubes_centroid):
					if(centr_coords[0] < undistorted_resolution[1] and centr_coords[1]  < undistorted_resolution[0]):
						if(undist_b_f_image[int(centr_coords[1]), int(centr_coords[0])] == 0):
							binary_centroid_fore_back[idx] = 0
							cv.circle(undist, (int(centr_coords[0]), int(centr_coords[1])), 1, (255,255,255), -1)

							
					# ------------------------ NACORA DA AGGIUNGERE IL DISCORSO DI METTERE A ZERO SU TUTTO L'ASSE ------------------------
    

				edited_frame = undist
					
			
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
			   
		
		
		new_shape_bit_centroid = np.asarray(center_voxels.shape[:3])
		new_shape_bit_centroid = np.append(new_shape_bit_centroid, np.array([1]))
  
		binary_centroid_fore_back_reshaped = np.reshape(binary_centroid_fore_back, new_shape_bit_centroid)
		mantained_centroids_idx = np.argwhere(binary_centroid_fore_back_reshaped == 1)
  
		resulting_voxels = cube_coords_centroid[mantained_centroids_idx[:, 0], mantained_centroids_idx[:, 1], mantained_centroids_idx[:, 2]]
		
  
   
		print(' DONE')
		print(f'Average FPS is: {str(avg_fps / int(input_video.get(cv.CAP_PROP_FRAME_COUNT)))}\n')
  
		voxels_cube_coords = np.reshape(resulting_voxels, (resulting_voxels.shape[0] * 8, 3))
		voxel_cube_faces = np.zeros((0,5), dtype=np.int32)
	
		for idx in range(0, voxels_cube_coords.shape[0], 8):
			voxel_cube_faces = np.vstack((voxel_cube_faces, np.array([4, idx + 2, idx + 0, idx + 1, idx + 3])))
			voxel_cube_faces = np.vstack((voxel_cube_faces, np.array([4, idx + 6, idx + 4, idx + 5, idx + 7])))
			voxel_cube_faces = np.vstack((voxel_cube_faces, np.array([4, idx + 6, idx + 4, idx + 0, idx + 2])))
			voxel_cube_faces = np.vstack((voxel_cube_faces, np.array([4, idx + 7, idx + 5, idx + 1, idx + 3])))
			voxel_cube_faces = np.vstack((voxel_cube_faces, np.array([4, idx + 0, idx + 4, idx + 5, idx + 1])))
			voxel_cube_faces = np.vstack((voxel_cube_faces, np.array([4, idx + 2, idx + 6, idx + 7, idx + 3])))

		print(f'Saving PLY file with {voxels_cube_coords.shape[0]} vertices...')
		write_ply_file(obj_id, voxels_cube_coords, voxel_cube_faces)
		print(' DONE\n')

		# Release the input and output streams
		input_video.release()
		output_video.release()
		cv.destroyAllWindows()



if __name__ == "__main__":
	main()
 