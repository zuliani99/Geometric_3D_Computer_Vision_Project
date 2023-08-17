import numpy as np
import cv2 as cv

class VoxelsCube:
	def __init__(self, unidst_axis, voxel_cube_dim, camera_matrix, dist, frame_width, frame_height) -> None:
		self.frame_width = frame_width
		self.frame_height = frame_height
		self.camera_matrix = camera_matrix
		self.dist = dist
		self.voxel_cube_dim = voxel_cube_dim
		self.unidst_axis = unidst_axis
		self.axis_centroid = np.float32([[20,0,0], [0,20,0], [0,0,30]]).reshape(-1,3)
		self.axis_vertical_edges = np.float32([
											[-unidst_axis, -unidst_axis, 70], [-unidst_axis, unidst_axis, 70],
											[unidst_axis ,unidst_axis, 70], [unidst_axis, -unidst_axis, 70],
											[-unidst_axis, -unidst_axis, 70 + unidst_axis * 2],[-unidst_axis, unidst_axis, 70 + unidst_axis * 2],
											[unidst_axis, unidst_axis, 70 + unidst_axis * 2],[unidst_axis, -unidst_axis, 70 + unidst_axis * 2]
										])
		self.center_voxels, self.cube_coords_centroid = self.get_cube_and_centroids_voxels()
		self.binary_centroid_fore_back = np.ones((np.power(self.center_voxels.shape[0], 3), 1), dtype=np.int32)
  
	
	
	def get_cube_and_centroids_voxels(self):
		axis_length = (self.unidst_axis * 2) // self.voxel_cube_dim
		
		center_voxels = np.zeros((0, axis_length, axis_length, 3), dtype=np.float32)
		cube_coords_voxels = np.zeros((0, axis_length, axis_length, 8, 3), dtype=np.float32)
	
		for z in range (70 + (self.voxel_cube_dim // 2), 70 + (self.unidst_axis * 2) - self.voxel_cube_dim // 2 + 1, self.voxel_cube_dim):
			center_voxels_at_z = np.zeros((0, axis_length, 3), dtype=np.float32)
			cube_coords_voxels_at_z = np.zeros((0, axis_length, 8, 3), dtype=np.float32)
			
			for y in range (-self.unidst_axis + self.voxel_cube_dim // 2, self.unidst_axis - self.voxel_cube_dim // 2 + 1, self.voxel_cube_dim):
				rows = np.zeros((0,3), dtype=np.float32)
				cubes = np.zeros((0,8,3), dtype=np.float32)
				
				for x in range (-self.unidst_axis + self.voxel_cube_dim // 2, self.unidst_axis - self.voxel_cube_dim // 2 + 1, self.voxel_cube_dim):
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
	
	
	
	def apply_projections(self, twoD_points, threeD_points, edited_frame):
		# Find the rotation and translation vectors
		_, rvecs, tvecs = cv.solvePnP(objectPoints=threeD_points.astype('float32'), imagePoints=twoD_points.astype('float32'), cameraMatrix=self.camera_matrix, distCoeffs=self.dist, flags=cv.SOLVEPNP_IPPE)
  
		imgpts_cubes_centroid, _ = cv.projectPoints(objectPoints=np.reshape(self.center_voxels, (np.power(self.center_voxels.shape[0], 3), 3)), rvec=rvecs, tvec=tvecs, cameraMatrix=self.camera_matrix, distCoeffs=self.dist)

		imgpts_centroid, _ = cv.projectPoints(objectPoints=self.axis_centroid, rvec=rvecs, tvec=tvecs, cameraMatrix=self.camera_matrix, distCoeffs=self.dist)
		imgpts_cube, _ = cv.projectPoints(objectPoints=self.axis_vertical_edges, rvec=rvecs, tvec=tvecs, cameraMatrix=self.camera_matrix, distCoeffs=self.dist)
			   		  	 
		newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(self.camera_matrix, self.dist, (self.frame_width, self.frame_height), 1, (self.frame_width, self.frame_height))
		  
		self.imgpts_cubes_centroid = np.squeeze(imgpts_cubes_centroid)

		# Undistort the image
		undist = cv.undistort(edited_frame, self.camera_matrix, self.dist, None, newCameraMatrix)	
		x, y, w, h = roi
		undist = undist[y:y+h, x:x+w] # Adjust the image resolution
  
		return undist, imgpts_centroid, imgpts_cube, newCameraMatrix



	def set_background_voxels(self, undistorted_resolution, undist_b_f_image, undist):
		for idx, centr_coords in enumerate(self.imgpts_cubes_centroid):
			if centr_coords[0] < undistorted_resolution[1] and centr_coords[1] < undistorted_resolution[0] and \
					centr_coords[0] >= 0  and centr_coords[1] >= 0 and undist_b_f_image[int(centr_coords[1]), int(centr_coords[0])] == 0:
				self.binary_centroid_fore_back[idx] = 0
				cv.circle(undist, (int(centr_coords[0]), int(centr_coords[1])), 1, (255,255,255), -1)

				# ------------------------ NACORA DA AGGIUNGERE IL DISCORSO DI METTERE A ZERO SU TUTTO L'ASSE ------------------------

		return undist



	def get_cubes_coords_and_faces(self):
		new_shape_bit_centroid = np.asarray(self.center_voxels.shape[:3])
		new_shape_bit_centroid = np.append(new_shape_bit_centroid, np.array([1]))
  
		binary_centroid_fore_back_reshaped = np.reshape(self.binary_centroid_fore_back, new_shape_bit_centroid)
		mantained_centroids_idx = np.argwhere(binary_centroid_fore_back_reshaped == 1)
  
		resulting_voxels = self.cube_coords_centroid[mantained_centroids_idx[:, 0], mantained_centroids_idx[:, 1], mantained_centroids_idx[:, 2]]
  
		voxels_cube_coords = np.reshape(resulting_voxels, (resulting_voxels.shape[0] * 8, 3))
		voxel_cube_faces = np.zeros((0,5), dtype=np.int32)
	
		for idx in range(0, voxels_cube_coords.shape[0], 8):
			voxel_cube_faces = np.vstack((voxel_cube_faces, np.array([4, idx + 2, idx + 0, idx + 1, idx + 3])))
			voxel_cube_faces = np.vstack((voxel_cube_faces, np.array([4, idx + 6, idx + 4, idx + 5, idx + 7])))
			voxel_cube_faces = np.vstack((voxel_cube_faces, np.array([4, idx + 6, idx + 4, idx + 0, idx + 2])))
			voxel_cube_faces = np.vstack((voxel_cube_faces, np.array([4, idx + 7, idx + 5, idx + 1, idx + 3])))
			voxel_cube_faces = np.vstack((voxel_cube_faces, np.array([4, idx + 0, idx + 4, idx + 5, idx + 1])))
			voxel_cube_faces = np.vstack((voxel_cube_faces, np.array([4, idx + 2, idx + 6, idx + 7, idx + 3])))
   
		return voxels_cube_coords, voxel_cube_faces