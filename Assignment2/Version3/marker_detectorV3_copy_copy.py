import time
from types import NoneType
import cv2 as cv
import numpy as np

#from utils import save_stats, set_marker_reference_coords
from board_copy import find_interesting_points


# List of object file name that we have to process
objs = ['obj01.mp4', 'obj02.mp4', 'obj03.mp4', 'obj04.mp4']

# se the Lucas Kanade parameters
criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01)
winsize = (5,5)
maxlevel = 4

circle_mask_size = 11 #10

def draw_mask_frame(polygons_points, frame=None, mask=None):
	for poly in polygons_points:
		for x, y in poly:
			if(x != 0 and y != 0):
				pos = (int(x), int(y))		
				if(type(frame) != NoneType): cv.circle(frame, pos, 3, (0,0,255), -1)
				if(type(mask) != NoneType): cv.circle(mask, pos, circle_mask_size, 255, -1)
		if(type(frame) != NoneType): cv.drawContours(frame, np.int32([poly]), 0, (0, 0, 255), 2, cv.LINE_AA)
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

		prev_frameg = None # Previosu gray frame
		#tracked_features = np.zeros( (0,2), dtype=np.float32 ) 
  

		ret, frame = input_video.read()
		frameg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		mask = np.zeros_like(frameg)
		tracked_features =  find_interesting_points(frameg, mask) # (18,5,2)
		#tracked_features = np.vstack((tracked_features, corners))
		#print(tracked_features.shape)

  
		frame, mask = draw_mask_frame(tracked_features, frame, mask)
   
		output_video.write(frame)
   
		prev_frameg = frameg


		# Until the video is open
		while True:
			
			# Extract a frame
			ret, frame = input_video.read()

			if not ret:	break

   
			frameg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
			mask = np.zeros_like(frameg)
	
			# spread int an array of 90, 2
			tf_OF = np.reshape(tracked_features, (tracked_features.shape[0]*tracked_features.shape[1], tracked_features.shape[2]))
   
			p1, st, _ = cv.calcOpticalFlowPyrLK(prev_frameg, frameg, tf_OF, None, winSize=winsize, maxLevel=maxlevel, criteria=criteria)
			assert(p1.shape[0] == tf_OF.shape[0])
			p0r, st0, _ = cv.calcOpticalFlowPyrLK(frameg, prev_frameg, p1, None, winSize=winsize, maxLevel=maxlevel, criteria=criteria)

			fb_good = (np.fabs(p0r-tf_OF) < 0.1).all(axis=1)
			fb_good = np.logical_and(np.logical_and(fb_good, st.flatten()), st0.flatten())
			# vector of true false
			#print(fb_good.shape)
			#return
   
			#print(np.where(fb_good == False)[0]) set to zero all bad features 
			tf_OF[np.squeeze(np.where(fb_good == False))] = np.array([0.,0.], dtype=np.float32)
			#print(tf_OF)
  
			
			# reconvert the array into a 18,5,2 array
			tracked_features = np.reshape(tf_OF, (int(tf_OF.shape[0]/tracked_features.shape[1]), tracked_features.shape[1], tracked_features.shape[2]))
			#print(tracked_features, tracked_features.shape)
   


			_, mask = draw_mask_frame(tracked_features, mask=mask)
		
			tracked_features = find_interesting_points(frameg, mask, tracked_features)
			#return
   
			print(tracked_features)
   
   
			#if(corners.shape[0] > 0):
				#frame, mask = draw_mask_frame(frame, mask, corners)
			frame, _ = draw_mask_frame(tracked_features, frame=frame)

				#tracked_features = np.vstack((tracked_features, corners)) # stack arrays in sequence vertically, for the coordinate of the cumulative features

			prev_frameg = frameg
   
   
			cv.imshow('frame',frame)
			#cv.imshow('mask',mask)
			
			output_video.write(frame)
   
			if cv.waitKey(1) == ord('q'):
				break

			actual_fps += 1
   
			time.sleep(3)

		print(' DONE\n')
  

		# Release the input and output streams
		input_video.release()
		output_video.release()
		cv.destroyAllWindows()




if __name__ == "__main__":
	main()