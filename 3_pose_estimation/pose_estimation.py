import numpy as np
import cv2 as cv
import time
import copy


from utils import set_marker_reference_coords, resize_for_laptop
from board import Board


parameters = {
	'obj01.mp4': {'circle_mask_size': 15, 'window_size': (10, 10)},
	'obj02.mp4': {'circle_mask_size': 13, 'window_size': (9, 9)},
	'obj03.mp4': {'circle_mask_size': 13, 'window_size': (9, 9)},
	'obj04.mp4': {'circle_mask_size': 15, 'window_size': (10, 10)},
}

using_laptop = False


def draw_origin(img, corner, imgpts):
 
	cv.arrowedLine(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 4, cv.LINE_AA)
	cv.putText(img, 'Y', tuple(imgpts[0].ravel()), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv.LINE_AA)
 
	cv.arrowedLine(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 4, cv.LINE_AA)
	cv.putText(img, 'X', tuple(imgpts[1].ravel()), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv.LINE_AA)
 
	cv.arrowedLine(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 4, cv.LINE_AA)
	cv.putText(img, 'Z', tuple(imgpts[2].ravel()), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv.LINE_AA)

	return img



def draw_cube(img, imgpts):

    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,0,255),3, cv.LINE_AA)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(0,0,255),3, cv.LINE_AA)

    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img




def main():
	
	# Set the marker reference coordinates for the 24 polygonls
	marker_reference = set_marker_reference_coords()
 
	camera_matrix = np.load('./calibration_info/cameraMatrix.npy')
	dist = np.load('./calibration_info/dist.npy')
 
	axis_centroid = np.float32([[20,0,0], [0,20,0], [0,0,30]]).reshape(-1,3)
	axis_vertical_edges = np.float32([[-55,-55,80], [-55,55,80], [55,55,80], [55,-55,80],
                   					  [-55,-55,190],[-55,55,190],[55,55,190],[55,-55,190] ])
 
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

		board = Board(n_polygons=24, circle_mask_size=hyper_param['circle_mask_size'])
  
		
  		# Create output video writer
		output_video = cv.VideoWriter(f"../../output_part3/{obj_id}/{obj_id}_mask.mp4", cv.VideoWriter_fourcc(*"mp4v"), input_video.get(cv.CAP_PROP_FPS), (frame_width, frame_height))

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
			pixsl_info = board.compute_markers(thresh, reshaped_clockwise, marker_reference)
			#print(pixsl_info[:,3:6].shape, pixsl_info[:,1:3].shape)
   
			twoD_points = np.float32(pixsl_info[:,1:3])
			threeD_points = np.float32(pixsl_info[:,3:6])
   
   
			edited_frame = board.draw_stuff(frame)
   
			#print(twoD_points)


			# qui posso fare cose
   
   
   
   
   
			#if pixsl_info.shape[0] >= 6:
	   
			#print(camera_matrix, dist)

			# Find the rotation and translation vectors
			ret, rvecs, tvecs = cv.solvePnP(objectPoints=threeD_points, imagePoints=twoD_points, cameraMatrix=camera_matrix, distCoeffs=dist, flags=cv.SOLVEPNP_IPPE)
			# ------------------------------------------ ERROR HERE ------------------------------------------
			# pixsl_info deve essere np.float32

			# Project 3D points to image plane
			imgpts_centroid, _ = cv.projectPoints(objectPoints=axis_centroid, rvec=rvecs, tvec=tvecs, cameraMatrix=camera_matrix, distCoeffs=dist)
			imgpts_cube, _ = cv.projectPoints(objectPoints=axis_vertical_edges, rvec=rvecs, tvec=tvecs, cameraMatrix=camera_matrix, distCoeffs=dist)
    	
			#print(imgpts.shape)
	
			# ora il mio dubbio e' come capire come dirgli di disegnare il quadrato 
			# credo che se dal immagine non distorta rieesco a prendermi le coordinate die punti e poi qel punto che e' nascosto
			# riuscire a calcolarmi la sua posizione che dovrebbe essere possibile avendo l'immagine non distorta dovrei riuscire ad interpoalre
			# TUTTO DA VERIFICARE OVVIAMENTE
	
			# questa e' la semplificazione per l'assignment, poi devo prendermi i voxel che dovrebbero essere come una griglia su ogni faccia visibile del quadrato 3D
			# da li poi devo vedere il background e foreground se il relativo pixel/voxel tocca prima il background lo elimino dalla visualizzazione mentre
			# se tocca il foreground cioe' l'oggetto lo mostro
   
	
	
			newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist, (frame_width, frame_height), 1, (frame_width, frame_height))
	
			# Undistort the image
			undist = cv.undistort(edited_frame, camera_matrix, dist, None, newCameraMatrix)	
			x, y, w, h = roi
			undist = undist[y:y+h, x:x+w]
	

			undist_edited = draw_origin(undist, (board.centroid[0], int(undist.shape[0] / 2)), np.int32(imgpts_centroid))
			undist_edited = draw_cube(undist_edited, np.int32(imgpts_cube))
			#pixsl_info[np.where(pixsl_info[:, 0] == board.achor)[0]][:,1:3] # NON SERVE QUINID NON SERVE NEMMENO L'ANCHOR
    
    
			#print(pixsl_info)
    
    
			#print(imgpts_cube)


	
			#cv.imshow('Undistorted Image', undist_edited)
			#cv.imshow('Image Frame', frame)
				

			end = time.time()
			fps = 1 / (end-start)
   
			avg_fps += fps

			# Get the resized frame
			frame_with_fps_resized = resize_for_laptop(using_laptop, copy.deepcopy(undist_edited))
  
			# Output the frame with the FPS
			#cv.putText(frame_with_fps_resized, f"{fps:.2f} FPS", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
			#cv.imshow(f'Marker Detector of {obj}', frame_with_fps_resized)
   			
			cv.putText(frame_with_fps_resized, f"{fps:.2f} FPS", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
			cv.imshow(f'Marker Detector of {obj}', frame_with_fps_resized)
			
			# Save the frame without the FPS count
			output_video.write(undist_edited)
			
	
			prev_frameg = frameg
   
			actual_fps += 1

			key = cv.waitKey(1)
			if key == ord('p'):
				cv.waitKey(-1) 
   
			if key == ord('q'):
				return

			cv.waitKey(-1)
			
   
   
		print(' DONE')
  
		print(f'Average FPS is: {str(avg_fps / int(input_video.get(cv.CAP_PROP_FRAME_COUNT)))}')

  
		# Release the input and output streams
		input_video.release()
		output_video.release()
		cv.destroyAllWindows()




if __name__ == "__main__":
	main()
 