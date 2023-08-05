import cv2 as cv
import numpy as np
import random


#from utils import save_stats, set_marker_reference_coords
from board_copy import find_interesting_points, sort_vertices_clockwise


# List of object file name that we have to process
objs = ['obj01.mp4']#, 'obj02.mp4', 'obj03.mp4', 'obj04.mp4']

# se the Lucas Kanade parameters
criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 50, 0.01)
winsize = (10,10)
maxlevel = 3

using_laptop = False

circle_mask_size = 13 # 15 quasi ok


def random_bgr_color():
    blue = random.randint(0, 255)
    green = random.randint(0, 255)
    red = random.randint(0, 255)
    return (blue, green, red)


def resize_for_laptop(frame, mask):
	if using_laptop:
		frame = cv.resize(frame, (1080, 600))
		mask = cv.resize(mask, (1080, 600))
	return frame, mask


def main():

	# Iterate for each object
	for obj in objs:
		
		print(f'Marker Detector for {obj}...')
		input_video = cv.VideoCapture(f"../../data/{obj}")

		frame_width = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
		frame_height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))

		actual_fps = 0
		obj_id = obj.split('.')[0]
  
		output_video = cv.VideoWriter(f"../../output_part2/{obj_id}/{obj_id}_mask.mp4", cv.VideoWriter_fourcc(*"mp4v"), input_video.get(cv.CAP_PROP_FPS), (frame_width, frame_height))
  

		ret, frame = input_video.read()
		frameg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		mask = np.zeros_like(frameg)
		tracked_features = find_interesting_points(frameg, mask)
  
		for x, y in tracked_features:
			cv.circle(frame, (int(x), int(y)), 4, (0,0,255), -1)
			cv.circle(mask, (int(x), int(y)), circle_mask_size, 255, -1)
   
   
	
	
		centroid, tracked_features = sort_vertices_clockwise(tracked_features, x_centroid=1300, y_centroid=550)
		tracked_features = tracked_features[:int(tracked_features.shape[0]/5)*5,:]
   
   
		cv.drawMarker(frame, np.int32(centroid), color=(255,255,255), markerType=cv.MARKER_CROSS, thickness=2)
			


		reshaped_cloclwise = np.reshape(tracked_features, (int(tracked_features.shape[0]/5), 5, 2))
   
   
		_, last_poly_sorted = sort_vertices_clockwise(reshaped_cloclwise[-1,:,:])
   
		print(cv.contourArea(np.int32(last_poly_sorted)))
  
		if(cv.contourArea(np.int32(last_poly_sorted)) <= 1625.0):
			to_remove = np.all(tracked_features[:, None] == reshaped_cloclwise[-1,:,:], axis=2).any(axis=1)
			tracked_features = np.delete(tracked_features, to_remove, axis=0)
			reshaped_cloclwise = reshaped_cloclwise[:reshaped_cloclwise.shape[0]-1, :, :]
    
   
		for idx, poly in enumerate(reshaped_cloclwise):
			# random BGR
			color = random_bgr_color()
			for x, y in poly:
				cv.line(frame, (int(x), int(y)), np.int32(centroid), color, 1, cv.LINE_AA)
			_, print_order = sort_vertices_clockwise(poly)
			cv.drawContours(frame, np.int32([print_order]), 0, (0,0,255), 1, cv.LINE_AA)
			cv.putText(frame, f'{str(idx)}  {cv.contourArea(np.int32(print_order))}', np.int32(print_order[0]), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 1, cv.LINE_AA)
    
   
   
   
   
		actual_fps += 1

		frame_resized, mask_resized = resize_for_laptop(frame, mask)
   
		cv.imshow('frame', frame_resized)
		cv.imshow('mask', mask_resized)
   
		print(actual_fps, 'tracked_features', tracked_features.shape[0], int(tracked_features.shape[0]/5))
   
		#cv.waitKey(-1)  
   
		output_video.write(frame)
   
		prev_frameg = frameg

		while True:
			
			ret, frame = input_video.read()

			if not ret:	break

		 
			frameg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
			mask = np.zeros_like(frameg)
   
			
			if(actual_fps % 15 == 0): #10 sembra abbastanza buono
				tracked_features = find_interesting_points(frameg, mask)
				for x, y in tracked_features: 
					cv.circle(frame, (int(x), int(y)), 4, (0,255,255), -1)
					cv.circle(mask, (int(x), int(y)), circle_mask_size, 255, -1)
			else:
		  
				p1, st, _ = cv.calcOpticalFlowPyrLK(prev_frameg, frameg, tracked_features, None, winSize=winsize, maxLevel=maxlevel, criteria=criteria)#, flags=cv.OPTFLOW_LK_GET_MIN_EIGENVALS, minEigThreshold=0.01)
				assert(p1.shape[0] == tracked_features.shape[0])
				p0r, st0, _ = cv.calcOpticalFlowPyrLK(frameg, prev_frameg, p1, None, winSize=winsize, maxLevel=maxlevel, criteria=criteria)#, flags=cv.OPTFLOW_LK_GET_MIN_EIGENVALS, minEigThreshold=0.01)
		
				fb_good = (np.fabs(p0r-tracked_features) < 0.1).all(axis=1)
				fb_good = np.logical_and(np.logical_and(fb_good, st.flatten()), st0.flatten())
					
				tracked_features = p1[fb_good, :]
					
				for x, y in tracked_features: 
					cv.circle(frame, (int(x), int(y)), 4, (0,0,255), -1)
					cv.circle(mask, (int(x), int(y)), circle_mask_size, 255, -1)
					
				new_corner = find_interesting_points(frameg, mask)
				tracked_features = np.vstack((tracked_features, new_corner)) 

				for x, y in new_corner: 
					cv.circle(frame, (int(x), int(y)), 4, (0,255,255), -1)
					cv.circle(mask, (int(x), int(y)), circle_mask_size, 255, -1)
    
    
    
																					# correct centroid
			centroid, tracked_features = sort_vertices_clockwise(tracked_features, x_centroid=1300, y_centroid=550)

			#if(tracked_features.shape[0] < 95): tracked_features = tracked_features[:90,:]
			#else: tracked_features = tracked_features[:95,:]
			tracked_features = tracked_features[:int(tracked_features.shape[0]/5)*5,:]
   
   
			cv.drawMarker(frame, np.int32(centroid), color=(255,255,255), markerType=cv.MARKER_CROSS, thickness=2)
			
			'''if(tracked_features.shape[0] < 90):q
				print('going to crash')
				frame, mask = resize_for_laptop(frame, mask)
				cv.imshow('frame',frame)
				cv.imshow('mask',mask)
				print('tracked_features', tracked_features.shape[0], int(tracked_features.shape[0]/5))
				cv.waitKey(-1)'''

			reshaped_cloclwise = np.reshape(tracked_features, (int(tracked_features.shape[0]/5), 5, 2))
   
   
			_, last_poly_sorted = sort_vertices_clockwise(reshaped_cloclwise[-1,:,:])
   
			print(cv.contourArea(np.int32(last_poly_sorted)))
  
			if(cv.contourArea(np.int32(last_poly_sorted)) <= 1625.0):
				#to_remove = np.all(tracked_features[:, None] == reshaped_cloclwise[-1,:,:], axis=2).any(axis=1)
				#tracked_features = np.delete(tracked_features, to_remove, axis=0)
				reshaped_cloclwise = reshaped_cloclwise[:reshaped_cloclwise.shape[0]-1, :, :]
    
   
			for idx, poly in enumerate(reshaped_cloclwise):
				# random BGR
				color = random_bgr_color()
				for x, y in poly:
					cv.line(frame, (int(x), int(y)), np.int32(centroid), color, 1, cv.LINE_AA)
				_, print_order = sort_vertices_clockwise(poly)
				cv.drawContours(frame, np.int32([print_order]), 0, (0,0,255), 1, cv.LINE_AA)
				cv.putText(frame, f'{str(idx)}  {cv.contourArea(np.int32(print_order))}', np.int32(print_order[0]), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 1, cv.LINE_AA)
    
    
    
    
    
    
		
			prev_frameg = frameg

			frame_resized, mask_resized = resize_for_laptop(frame, mask)
   
			cv.imshow('frame', frame_resized)
			cv.imshow('mask', mask_resized)
   
   
			print(actual_fps, 'tracked_features', tracked_features.shape[0], int(tracked_features.shape[0]/5))
   
			#cv.waitKey(-1)
	  
			
			output_video.write(frame)

			key = cv.waitKey(1)
			if key == ord('p'):
				cv.waitKey(-1) #wait until any key is pressed
   
			if key == ord('q'):
				return

			actual_fps += 1

		print(' DONE\n')
  

		# Release the input and output streams
		input_video.release()
		output_video.release()
		cv.destroyAllWindows()




if __name__ == "__main__":
	main()