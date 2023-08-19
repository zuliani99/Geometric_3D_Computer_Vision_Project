import numpy as np
import cv2 as cv

from typing import Tuple
import numpy.typing as npt



class VoxelsCube:
	def __init__(self, half_axis_len, voxel_cube_dim, camera_matrix, dist, frame_width, frame_height) -> None:
		self.frame_width = frame_width
		self.frame_height = frame_height
		self.camera_matrix = camera_matrix
		self.dist = dist
		self.voxel_cube_dim = voxel_cube_dim
		self.half_axis_len = half_axis_len
		self.axis_centroid = np.float32([[20,0,0], [0,20,0], [0,0,30]]).reshape(-1,3)
		self.axis_vertical_edges = np.float32([
											[-half_axis_len, -half_axis_len, 70], [-half_axis_len, half_axis_len, 70],
											[half_axis_len ,half_axis_len, 70], [half_axis_len, -half_axis_len, 70],
											[-half_axis_len, -half_axis_len, 70 + half_axis_len * 2],[-half_axis_len, half_axis_len, 70 + half_axis_len * 2],
											[half_axis_len, half_axis_len, 70 + half_axis_len * 2],[half_axis_len, -half_axis_len, 70 + half_axis_len * 2]
										])
		self.center_voxels, self.cube_coords_centroid = self.get_cube_and_centroids_voxels()
		self.binary_centroid_fore_back = np.ones((np.power(self.center_voxels.shape[0], 3), 1), dtype=np.int32)
  
	
	
	def get_cube_and_centroids_voxels(self) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
		'''
		PURPOSE: 
		ARGUMENTS: None
		RETURN: Tuple[np.NDArray[np.float32], np.NDArray[np.float32]]
			- center_voxels (np.NDArray[np.float32]): voxels centroid
			- cube_coords_voxels (np.NDArray[np.float32]) voxels cube coordinates
		'''	

		axis_length = (self.half_axis_len * 2) // self.voxel_cube_dim
		
		center_voxels = np.zeros((0, axis_length, axis_length, 3), dtype=np.float32)
		cube_coords_voxels = np.zeros((0, axis_length, axis_length, 8, 3), dtype=np.float32)
	
		for z in range (70 + (self.voxel_cube_dim // 2), 70 + (self.half_axis_len * 2) - self.voxel_cube_dim // 2 + 1, self.voxel_cube_dim):
			center_voxels_at_z = np.zeros((0, axis_length, 3), dtype=np.float32)
			cube_coords_voxels_at_z = np.zeros((0, axis_length, 8, 3), dtype=np.float32)
			
			for y in range (-self.half_axis_len + self.voxel_cube_dim // 2, self.half_axis_len - self.voxel_cube_dim // 2 + 1, self.voxel_cube_dim):
				rows = np.zeros((0,3), dtype=np.float32)
				cubes = np.zeros((0,8,3), dtype=np.float32)
				
				for x in range (-self.half_axis_len + self.voxel_cube_dim // 2, self.half_axis_len - self.voxel_cube_dim // 2 + 1, self.voxel_cube_dim):
					rows = np.vstack((rows, np.array([x, y, z], dtype=np.float32)))
					cube = np.array([
						np.array([y + self.voxel_cube_dim // 2, x + self.voxel_cube_dim // 2, z + self.voxel_cube_dim // 2]),
						np.array([y + self.voxel_cube_dim // 2, x + self.voxel_cube_dim // 2, z - self.voxel_cube_dim // 2]),
						np.array([y + self.voxel_cube_dim // 2, x - self.voxel_cube_dim // 2, z + self.voxel_cube_dim // 2]),
						np.array([y + self.voxel_cube_dim // 2, x - self.voxel_cube_dim // 2, z - self.voxel_cube_dim // 2]),
						np.array([y - self.voxel_cube_dim // 2, x + self.voxel_cube_dim // 2, z + self.voxel_cube_dim // 2]),
						np.array([y - self.voxel_cube_dim // 2, x + self.voxel_cube_dim // 2, z - self.voxel_cube_dim // 2]),
						np.array([y - self.voxel_cube_dim // 2, x - self.voxel_cube_dim // 2, z + self.voxel_cube_dim // 2]),
						np.array([y - self.voxel_cube_dim // 2, x - self.voxel_cube_dim // 2, z - self.voxel_cube_dim // 2]),
					], dtype=np.float32)
					cubes = np.vstack((cubes, np.expand_dims(cube, axis=0)))
				
				center_voxels_at_z = np.vstack((center_voxels_at_z, np.expand_dims(rows, axis=0)))
				cube_coords_voxels_at_z = np.vstack((cube_coords_voxels_at_z, np.expand_dims(cubes, axis=0)))
	
			center_voxels = np.vstack((center_voxels, np.expand_dims(center_voxels_at_z, axis=0)))
			cube_coords_voxels = np.vstack((cube_coords_voxels, np.expand_dims(cube_coords_voxels_at_z, axis=0)))

		return center_voxels, cube_coords_voxels
	
	
	def get_undistorted_frame(self, to_edit_frame: npt.NDArray[np.uint8]) -> Tuple[npt.NDArray[np.uint8], cv.typing.MatLike]:
		'''
		PURPOSE: allpy the projections of voxels centroid, cube and board centroid
		ARGUMENTS: 
			- to_edit_frame (npt.NDArray[np.uint8]): frame to edit
		RETURN: Tuple[npt.NDArray[np.uint8], cv.typing.MatLike]
			- undist (npt.NDArray[np.uint8]): undistorted edited image
			- newCameraMatrix (cv.typing.MatLike)
		'''	

		newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(self.camera_matrix, self.dist, (self.frame_width, self.frame_height), 1, (self.frame_width, self.frame_height))
		  
		# Undistort the image
		undist = cv.undistort(to_edit_frame, self.camera_matrix, self.dist, None, newCameraMatrix)	
		x, y, w, h = roi
		undist = undist[y:y+h, x:x+w] # Adjust the image resolution

		return undist, newCameraMatrix


	
	def apply_projections(self, twoD_points: npt.NDArray[np.float32], threeD_points: npt.NDArray[np.float32]) \
			-> Tuple[npt.NDArray[np.uint8], cv.typing.MatLike, cv.typing.MatLike, cv.typing.MatLike]:
		'''
		PURPOSE: apply the projections of voxels centroid, cube and board centroid
		ARGUMENTS: 
			- twoD_points (np.NDArray[np.float32])
			- threeD_points (np.NDArray[np.float32])
		RETURN: Tuple[np.NDArray[np.uint8], cv.typing.MatLike, cv.typing.MatLike, cv.typing.MatLike]
			- imgpts_centroid (cv.typing.MatLike): 2D image centroid coordinates
			- imgpts_cube (cv.typing.MatLike): 2D image cube coordinates
		'''	

		# Find the rotation and translation vectors
		_, rvecs, tvecs = cv.solvePnP(objectPoints=threeD_points.astype('float32'), imagePoints=twoD_points.astype('float32'), cameraMatrix=self.camera_matrix, distCoeffs=self.dist, flags=cv.SOLVEPNP_IPPE)
  
		imgpts_cubes_centroid, _ = cv.projectPoints(objectPoints=np.reshape(self.center_voxels, (np.power(self.center_voxels.shape[0], 3), 3)), rvec=rvecs, tvec=tvecs, cameraMatrix=self.camera_matrix, distCoeffs=self.dist)
		
		self.imgpts_cubes_centroid = np.squeeze(imgpts_cubes_centroid)

		imgpts_centroid, _ = cv.projectPoints(objectPoints=self.axis_centroid, rvec=rvecs, tvec=tvecs, cameraMatrix=self.camera_matrix, distCoeffs=self.dist)
		imgpts_cube, _ = cv.projectPoints(objectPoints=self.axis_vertical_edges, rvec=rvecs, tvec=tvecs, cameraMatrix=self.camera_matrix, distCoeffs=self.dist)
  
		return imgpts_centroid, imgpts_cube



	def set_background_voxels(self, undistorted_resolution: Tuple[int, int], undist_b_f_image: npt.NDArray[np.uint8], undist: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
		'''
		PURPOSE: apply the projections of voxels centroid, cube and board centroid
		ARGUMENTS: 
			- undistorted_resolution (Tuple[int, int]): undistorted image resolution
			- undist_b_f_image (npt.NDArray[np.uint8]): undistorted segmented frame
			- undist (npt.NDArray[np.uint8]): undistorted image to edit
		RETURN:
			- undist (npt.NDArray[np.uint8]): undistorted edited image
		'''	

		for idx, centr_coords in enumerate(self.imgpts_cubes_centroid):
			if centr_coords[0] < undistorted_resolution[1] and centr_coords[1] < undistorted_resolution[0] and \
					centr_coords[0] >= 0  and centr_coords[1] >= 0 and undist_b_f_image[int(centr_coords[1]), int(centr_coords[0])] == 0:
				self.binary_centroid_fore_back[idx] = 0
				cv.circle(undist, (int(centr_coords[0]), int(centr_coords[1])), 1, (255,255,255), -1)

		return undist



	def get_cubes_coords_and_faces(self) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
		'''
		PURPOSE: get the voxel cube ciooirdinates and faces to write a PLY file
		ARGUMENTS: None
		RETURN: Tuple[np.NDArray[npt.float32], npt.NDArray[np.float32]]
			- voxels_cube_coords (npt.NDArray[np.float32]): voxel centroid coordinates belonging to the foreground
			- voxel_cube_faces (npt.NDArray[np.float32]): voxel cube faces belonging to the foreground
		'''	

		new_shape_bit_centroid = np.asarray(self.center_voxels.shape[:3])
		new_shape_bit_centroid = np.append(new_shape_bit_centroid, np.array([1]))
  
		binary_centroid_fore_back_reshaped = np.reshape(self.binary_centroid_fore_back, new_shape_bit_centroid)
		# Get the ID of the centroid that belong to the foreground
		mantained_centroids_idx = np.argwhere(binary_centroid_fore_back_reshaped == 1)

		# Get their coordinates
		resulting_voxels = self.cube_coords_centroid[mantained_centroids_idx[:, 0], mantained_centroids_idx[:, 1], mantained_centroids_idx[:, 2]]
  
		voxels_cube_coords = np.reshape(resulting_voxels, (resulting_voxels.shape[0] * 8, 3))
		voxel_cube_faces = np.zeros((0,5), dtype=np.int32)

		# Full the faces vector with the correct vertices ID
		for idx in range(0, voxels_cube_coords.shape[0], 8):
			voxel_cube_faces = np.vstack((voxel_cube_faces, np.array([4, idx + 2, idx + 0, idx + 1, idx + 3])))
			voxel_cube_faces = np.vstack((voxel_cube_faces, np.array([4, idx + 6, idx + 4, idx + 5, idx + 7])))
			voxel_cube_faces = np.vstack((voxel_cube_faces, np.array([4, idx + 6, idx + 4, idx + 0, idx + 2])))
			voxel_cube_faces = np.vstack((voxel_cube_faces, np.array([4, idx + 7, idx + 5, idx + 1, idx + 3])))
			voxel_cube_faces = np.vstack((voxel_cube_faces, np.array([4, idx + 0, idx + 4, idx + 5, idx + 1])))
			voxel_cube_faces = np.vstack((voxel_cube_faces, np.array([4, idx + 2, idx + 6, idx + 7, idx + 3])))
   
		return voxels_cube_coords, voxel_cube_faces