import numpy as np
import cv2 as cv

from typing import Dict, Tuple

# VoxelsCube class that manege projection of markers points into the image

class VoxelsCube:
    
	def __init__(self, cube_half_edge, voxel_cube_edge_dim, camera_matrix, dist, frame_width, frame_height) -> None:
		self.__frame_width = frame_width
		self.__frame_height = frame_height
		self.__camera_matrix = camera_matrix
		self.__dist = dist
		self.__voxel_cube_edge_dim = voxel_cube_edge_dim
		self.__cube_half_edge = cube_half_edge
		self.__centroid_axes = np.float32([[20,0,0], [0,20,0], [0,0,30]]).reshape(-1,3)
		self.__cube_vertices = np.float32([
			[-cube_half_edge, -cube_half_edge, 70], [-cube_half_edge, cube_half_edge, 70],
			[cube_half_edge, cube_half_edge, 70], [cube_half_edge, -cube_half_edge, 70],
			[-cube_half_edge, -cube_half_edge, 70 + cube_half_edge * 2], [-cube_half_edge, cube_half_edge, 70 + cube_half_edge * 2],
			[cube_half_edge, cube_half_edge, 70 + cube_half_edge * 2], [cube_half_edge, -cube_half_edge, 70 + cube_half_edge * 2]
		])
		self.__voxels_center, self.__voxs_cubes_verts_coords = self.get_cube_and_centroids_voxels()
		self.__binary_centroids_fore_back = np.ones((np.power(self.__voxels_center.shape[0], 3), 1), dtype=np.int32)
  
	
 
	
	def get_cube_and_centroids_voxels(self) -> Tuple[np.ndarray[int, np.float32], np.ndarray[int, np.float32]]:
		'''
		PURPOSE: obtain the centre voxels centroid coordinates and their respective corners coordinates that form the voxel cube
		ARGUMENTS: None
		RETURN: Tuple[np.ndarray[int, np.float32], np.ndarray[int, np.float32]]
			- voxels_center (np.ndarray[int, np.float32]): voxels centroids
			- centroids_cubes_coords (np.ndarray[int, np.float32]) voxels cube coordinates
		'''	

		# Get the number of voxels in a single dimension 
		first_dim_voxel = (self.__cube_half_edge * 2) // self.__voxel_cube_edge_dim
		
		# Initialize the np.array to store the voxels centre coords and voxels cube vertices coords
		voxels_center = np.zeros((0, first_dim_voxel, first_dim_voxel, 3), dtype=np.float32)
		centroids_cubes_coords = np.zeros((0, first_dim_voxel, first_dim_voxel, 8, 3), dtype=np.float32)
	
		# Iterate through the z axis
		for z in np.arange(70 + (self.__voxel_cube_edge_dim / 2), 70 + (self.__cube_half_edge * 2) - self.__voxel_cube_edge_dim / 2 + 1, 
					 		self.__voxel_cube_edge_dim):
			
			voxels_center_at_z = np.zeros((0, first_dim_voxel, 3), dtype=np.float32)
			centroids_cubes_coords_at_z = np.zeros((0, first_dim_voxel, 8, 3), dtype=np.float32)
			
			# Iterate through the y axis
			for y in np.arange(-self.__cube_half_edge + self.__voxel_cube_edge_dim / 2, 
					  			self.__cube_half_edge - self.__voxel_cube_edge_dim / 2 + 1, self.__voxel_cube_edge_dim):
				
				rows = np.zeros((0,3), dtype=np.float32)
				cubes = np.zeros((0,8,3), dtype=np.float32)
				
				# Iterate through the x axis
				for x in np.arange(-self.__cube_half_edge + self.__voxel_cube_edge_dim / 2, 
									self.__cube_half_edge - self.__voxel_cube_edge_dim / 2 + 1, self.__voxel_cube_edge_dim):
					
					rows = np.vstack((rows, np.array([x, y, z], dtype=np.float32))) # Voxel centre coords
					cube = np.array([	# Voxel cube vertices coords
						np.array([y + self.__voxel_cube_edge_dim / 2, x + self.__voxel_cube_edge_dim / 2, z + self.__voxel_cube_edge_dim / 2]),
						np.array([y + self.__voxel_cube_edge_dim / 2, x + self.__voxel_cube_edge_dim / 2, z - self.__voxel_cube_edge_dim / 2]),
						np.array([y + self.__voxel_cube_edge_dim / 2, x - self.__voxel_cube_edge_dim / 2, z + self.__voxel_cube_edge_dim / 2]),
						np.array([y + self.__voxel_cube_edge_dim / 2, x - self.__voxel_cube_edge_dim / 2, z - self.__voxel_cube_edge_dim / 2]),
						np.array([y - self.__voxel_cube_edge_dim / 2, x + self.__voxel_cube_edge_dim / 2, z + self.__voxel_cube_edge_dim / 2]),
						np.array([y - self.__voxel_cube_edge_dim / 2, x + self.__voxel_cube_edge_dim / 2, z - self.__voxel_cube_edge_dim / 2]),
						np.array([y - self.__voxel_cube_edge_dim / 2, x - self.__voxel_cube_edge_dim / 2, z + self.__voxel_cube_edge_dim / 2]),
						np.array([y - self.__voxel_cube_edge_dim / 2, x - self.__voxel_cube_edge_dim / 2, z - self.__voxel_cube_edge_dim / 2]),
					], dtype=np.float32)
					cubes = np.vstack((cubes, np.expand_dims(cube, axis=0)))
				
				voxels_center_at_z = np.vstack((voxels_center_at_z, np.expand_dims(rows, axis=0)))
				centroids_cubes_coords_at_z = np.vstack((centroids_cubes_coords_at_z, np.expand_dims(cubes, axis=0)))
	
			voxels_center = np.vstack((voxels_center, np.expand_dims(voxels_center_at_z, axis=0)))
			centroids_cubes_coords = np.vstack((centroids_cubes_coords, np.expand_dims(centroids_cubes_coords_at_z, axis=0)))

		return voxels_center, centroids_cubes_coords
	
	
 

	def get_newCameraMatrix(self) -> None:
		'''
		PURPOSE: get the new camera intrinsic matrix
		ARGUMENTS: None
		RETURN: None
		'''	

		newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(self.__camera_matrix, self.__dist, (self.__frame_width, self.__frame_height), 1, (self.__frame_width, self.__frame_height))
		self.__newCameraMatrix = newCameraMatrix
		self.__roi = roi



 
	def get_undistorted_frame(self, to_edit_frame: np.ndarray[int, np.uint8]) -> Tuple[np.ndarray[int, np.uint8], cv.typing.MatLike]:
		'''
		PURPOSE: obtain the undistorted image
		ARGUMENTS: 
			- to_edit_frame (np.ndarray[int, np.uint8]): frame to edit
		RETURN: Tuple[np.ndarray[int, np.uint8], cv.typing.MatLike]
			- undist (np.ndarray[int, np.uint8]): undistorted edited image
		'''	
		  
		# Undistort the image
		undist = cv.undistort(to_edit_frame, self.__camera_matrix, self.__dist, None, self.__newCameraMatrix)	
		x, y, w, h = self.__roi
		undist = undist[y:y+h, x:x+w] # Adjust the image resolution

		return undist
		



	def apply_projections(self, twoD_points: np.ndarray[int, np.float32], threeD_points: np.ndarray[int, np.float32]) \
			-> Tuple[cv.typing.MatLike, cv.typing.MatLike]:
		'''
		PURPOSE: apply the projections of voxels centroid, cube and board centroid
		ARGUMENTS: 
			- twoD_points (np.ndarray[int, np.float32])
			- threeD_points (np.ndarray[int, np.float32])
		RETURN: Tuple[cv.typing.MatLike, cv.typing.MatLike]
			- imgpts_centroid (cv.typing.MatLike): 2D image centroid coordinates
			- imgpts_cube (cv.typing.MatLike): 2D image cube coordinates
		'''	

		# Find the rotation and translation vectors
		_, rvecs, tvecs = cv.solvePnP(objectPoints=threeD_points.astype('float32'), imagePoints=twoD_points.astype('float32'),
								cameraMatrix=self.__camera_matrix, distCoeffs=self.__dist, flags=cv.SOLVEPNP_IPPE)
		self.__rvecs = rvecs
		self.__tvecs = tvecs

		# Obtain the projection of the vocels cubes centroid
		imgpts_voxels_cubes_centroid, _ = cv.projectPoints(objectPoints=np.reshape(self.__voxels_center, (np.power(self.__voxels_center.shape[0], 3), 3)),
                                            				rvec=self.__rvecs, tvec=self.__tvecs, cameraMatrix=self.__camera_matrix, distCoeffs=self.__dist)
		
		self.__imgpts_voxels_cubes_centroid = np.squeeze(imgpts_voxels_cubes_centroid)

	 	# Obtain the projection of the board centroid with axes and the cube that will inglobe the object
		imgpts_centroid, _ = cv.projectPoints(objectPoints=self.__centroid_axes, rvec=self.__rvecs, tvec=self.__tvecs, cameraMatrix=self.__camera_matrix, distCoeffs=self.__dist)
		imgpts_cube, _ = cv.projectPoints(objectPoints=self.__cube_vertices, rvec=self.__rvecs, tvec=self.__tvecs, cameraMatrix=self.__camera_matrix, distCoeffs=self.__dist)
	
		return imgpts_centroid, imgpts_cube




	def compute_RMSE(self, IDs_points: np.ndarray[int, np.float32], marker_reference: Dict[int, Tuple[int, int, int]], twoD_points: np.ndarray[int, np.float32]) -> np.float32:
		'''
		PURPOSE: compute the RMSE of the reporojection points
		ARGUMENTS: 
			- IDs_points: (np.ndarray[int, np.float32])
			- marker_reference: Dict[int, Tuple[int, int, int]]
			- twoD_points (np.ndarray[int, np.float32])
		RETURN:
			- rmse (np.float32): RMSE of the reprojected points in the actual frame
		'''	
    
		# Initializing the np array to store the marker reference coordinates of the actual detected index polygon
		marker_reference_points = np.zeros((0,3), dtype=np.float32)
		for idx in IDs_points:
			marker_reference_points = np.vstack((marker_reference_points, np.array(marker_reference[idx], dtype=np.float32)))
		
		# Compute the reprojection
		reprojections, _ = cv.projectPoints(marker_reference_points, self.__rvecs, self.__tvecs, self.__camera_matrix, self.__dist)

		# Compute the RMSE for all detected points
		return np.sqrt(np.mean(np.square(np.linalg.norm(twoD_points - np.squeeze(reprojections), axis=1))))




	def set_background_voxels(self, undistorted_resolution: Tuple[int, int], undist_b_f_image: np.ndarray[int, np.uint8], undist: np.ndarray[int, np.uint8]) -> np.ndarray[int, np.uint8]:
		'''
		PURPOSE: update the binary array of voxels centroid by analysing their position on the segmented image
		ARGUMENTS: 
			- undistorted_resolution (Tuple[int, int]): undistorted image resolution
			- undist_b_f_image (np.ndarray[int, np.uint8]): undistorted segmented frame
			- undist (np.ndarray[int, np.uint8]): undistorted image to edit
		RETURN:
			- undist (np.ndarray[int, np.uint8]): undistorted edited image
		'''	

		for idx, centr_coords in enumerate(self.__imgpts_voxels_cubes_centroid):
			if centr_coords[0] < undistorted_resolution[0] and centr_coords[1] < undistorted_resolution[1] and \
					centr_coords[0] >= 0  and centr_coords[1] >= 0 and undist_b_f_image[int(centr_coords[1]), int(centr_coords[0])] == 0:
				self.__binary_centroids_fore_back[idx] = 0
				cv.circle(undist, (int(centr_coords[0]), int(centr_coords[1])), 1, (255,255,255), -1)

		return undist




	def get_cubes_coords_and_faces(self) -> Tuple[np.ndarray[int, np.float32], np.ndarray[int, np.float32]]:
		'''
		PURPOSE: get the voxels cube coordinates and faces to write a PLY file
		ARGUMENTS: None
		RETURN: Tuple[np.ndarray[int, np.float32], np.ndarray[int, np.float32]]
			- voxels_cube_coords (np.ndarray[int, np.float32]): voxel centroid coordinates belonging to the foreground
			- voxels_cube_faces (np.ndarray[int, np.float32]): voxel cube faces belonging to the foreground
		'''	

		new_shape_bits_centroids = np.asarray(self.__voxels_center.shape[:3])
		new_shape_bits_centroids = np.append(new_shape_bits_centroids, np.array([1]))
  
		binary_centroids_fore_back_reshaped = np.reshape(self.__binary_centroids_fore_back, new_shape_bits_centroids)
		# Get the ID of the centroid that belong to the foreground
		mantained_centroids_idx = np.argwhere(binary_centroids_fore_back_reshaped == 1)

		# Get their cube coordinates
		resulting_voxels = self.__voxs_cubes_verts_coords[mantained_centroids_idx[:, 0], mantained_centroids_idx[:, 1], mantained_centroids_idx[:, 2]]

		# Reshaping the coordinates
		voxels_cube_coords = np.reshape(resulting_voxels, (resulting_voxels.shape[0] * 8, 3))
  
		# Initialize the np.array to store the faces of each voxel cube
		voxels_cube_faces = np.zeros((0,5), dtype=np.int32)

		# Full the faces vector with the correct voxel cube vertices ID
		for idx in range(0, voxels_cube_coords.shape[0], 8):
			voxels_cube_faces = np.vstack((voxels_cube_faces, np.array([4, idx + 2, idx + 0, idx + 1, idx + 3])))
			voxels_cube_faces = np.vstack((voxels_cube_faces, np.array([4, idx + 6, idx + 4, idx + 5, idx + 7])))
			voxels_cube_faces = np.vstack((voxels_cube_faces, np.array([4, idx + 6, idx + 4, idx + 0, idx + 2])))
			voxels_cube_faces = np.vstack((voxels_cube_faces, np.array([4, idx + 7, idx + 5, idx + 1, idx + 3])))
			voxels_cube_faces = np.vstack((voxels_cube_faces, np.array([4, idx + 0, idx + 4, idx + 5, idx + 1])))
			voxels_cube_faces = np.vstack((voxels_cube_faces, np.array([4, idx + 2, idx + 6, idx + 7, idx + 3])))
   
		return voxels_cube_coords, voxels_cube_faces




	def draw_cube(self, img: np.ndarray[int, np.uint8], imgpts: np.ndarray[int, np.int32]) -> np.ndarray[int, np.uint8]:
		'''
		PURPOSE: draw a red cube that inglobe the object
		ARGUMENTS:
			- img (np.ndarray[int, np.uint8]): image to modify
			- imgpts (np.ndarray[int, np.int32]): image points
		RETURN:
			- (np.ndarray[int, np.uint8]): modified frame
		'''	
		
		imgpts = np.int32(imgpts).reshape(-1,2)

		# Draw ground floor
		cv.drawContours(img, [imgpts[:4]], -1, (0,0,255), 3, cv.LINE_AA)

		# Draw pillars
		for i,j in zip(range(4), range(4,8)):
			cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (0,0,255), 3, cv.LINE_AA)

		# Draw top layer
		cv.drawContours(img, [imgpts[4:]], -1, (0,0,255), 3)

		return img
