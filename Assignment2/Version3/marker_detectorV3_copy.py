import cv2 as cv
import numpy as np

#from utils import save_stats, set_marker_reference_coords
from board import find_interesting_points


# List of object file name that we have to process
objs = ['obj01.mp4', 'obj02.mp4', 'obj03.mp4', 'obj04.mp4']

# se the Lucas Kanade parameters
criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01)
winsize = (5,5)
maxlevel = 4

circle_mask_size = 10



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

		prev_frameg = None # Previosu gray frame
		tracked_features = np.zeros( (0,2), dtype=np.float32 ) 
  

		ret, frame = input_video.read()
		frameg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		mask = np.zeros_like(frameg)
		corners =  find_interesting_points(frameg, mask)
		tracked_features = np.vstack((tracked_features, corners))
  
	
		for x, y in tracked_features:
			pos = (int(x), int(y))		
			cv.circle(frame, pos, 3, (0,0,255), -1)
			cv.circle(mask, pos, circle_mask_size, 255, -1)
   
		output_video.write(frame)
   
		prev_frameg = frameg


		# Until the video is open
		while True:
			
			# Extract a frame
			ret, frame = input_video.read()

			if not ret:	break

   
			frameg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
			mask = np.zeros_like(frameg)
			
   
			if tracked_features.shape[0] > 0: # in case I have something within my array of features to track
       
				p1, st, _ = cv.calcOpticalFlowPyrLK(prev_frameg, frameg, tracked_features, None, winSize=winsize, maxLevel=maxlevel, criteria=criteria)
				assert(p1.shape[0] == tracked_features.shape[0])
				p0r, st0, _ = cv.calcOpticalFlowPyrLK(frameg, prev_frameg, p1, None, winSize=winsize, maxLevel=maxlevel, criteria=criteria)
				#good = abs(tracked_features - p0r).reshape(-1, 2).max(-1) < 0.5
				#tracked_features = p1[good, :]
				
    
				#print(p1[np.squeeze(st == 1)])
				#tracked_features = p1[np.squeeze(st == 1)]

    
				#print(len(tracked_features))



				fb_good = (np.fabs(p0r-tracked_features) < 0.1).all(axis=1)
				fb_good = np.logical_and(np.logical_and(fb_good, st.flatten()), st0.flatten())
				tracked_features = p1[fb_good, :]


				# Update tracks
				for x, y in tracked_features:
					pos = (int(x), int(y)) # set the pair of X and Y coordiantes
					cv.circle(frame, pos, 3, (0,0,255), -1)
					cv.circle(mask, pos, circle_mask_size, 255, -1)
			

			
			corners =  find_interesting_points(frameg, mask)
			if(corners.shape[0] > 0): tracked_features = np.vstack((tracked_features, corners)) # stack arrays in sequence vertically, for the coordinate of the cumulative features

			prev_frameg = frameg
   
   
			cv.imshow('frame',frame)
			#cv.imshow('mask',mask)
			
			output_video.write(frame)
   
			if cv.waitKey(1) == ord('q'):
				break

			actual_fps += 1

		print(' DONE\n')
  

		# Release the input and output streams
		input_video.release()
		output_video.release()
		cv.destroyAllWindows()




if __name__ == "__main__":
	main()