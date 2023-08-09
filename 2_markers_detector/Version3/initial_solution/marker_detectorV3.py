import cv2 as cv
import numpy as np

#from utils import save_stats, set_marker_reference_coords
from board import find_interesting_points


# List of object file name that we have to process
objs = ['obj01.mp4', 'obj02.mp4', 'obj03.mp4', 'obj04.mp4']


def main() -> None:

	# Iterate for each object
	for obj in objs:
		
		print(f'Marker Detector for {obj}...')
		input_video = cv.VideoCapture(f"../../data/{obj}")


		actual_fps = 0

		prev_frameg = None # Previosu gray frame
		#tracked_features = np.zeros( (0,5,2), dtype=np.float32 ) # 0 rows and 2 columsn for storing the track
		tracked_features = np.zeros( (0,2), dtype=np.float32 ) # 0 rows and 2 columsn for storing the track
		tracked_features_ids = np.zeros( (0,1), dtype=np.int64 ) # 0 row and 1 column to store the feature ID
	
		max_index = 0 # index of maximum id in the tracked_features_ids array	

		tracks = {} # dictioanry of track


		# Until the video is open
		while True:
			
			# Extract a frame
			ret, frame = input_video.read()

			if not ret:	break

   
			frameg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
			mask = np.zeros( frameg.shape, dtype=np.uint8)

   
			if tracked_features.shape[0]>0:		# in case I have something within my array of features to trak
				
				# se the Lucas Kanade parameters
				criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
				winsize = (10,10)
				maxlevel = 4
				fb_threshold = 0.5
    
				#print(tracked_features.shape, np.squeeze(tracked_features).shape)
				
				p1, st, _ = cv.calcOpticalFlowPyrLK(prev_frameg, # Old frame
													frameg, # New frame
													tracked_features, # Poins to track
													None, 
													winSize=winsize, 
													maxLevel=maxlevel, 
													criteria=criteria)
				
				# We compute the flow in both way forward and backward
    
				
				assert(p1.shape[0] == tracked_features.shape[0])
    
				
				p0r, st0, _ = cv.calcOpticalFlowPyrLK(frameg, 
														prev_frameg, 
														p1, 
														None, 
														winSize=winsize, 
														maxLevel=maxlevel, 
														criteria=criteria)


				# find the good ones


				fb_good = (np.fabs(p0r - tracked_features) < fb_threshold).all(axis=1)
				fb_good = np.logical_and(np.logical_and(fb_good, st.flatten()), st0.flatten())
				# is a true false array

				# ids of bad features
				#bad_ids = tracked_features_ids[np.logical_not( fb_good.flatten())]

				tracked_features = p1[fb_good, :]
				tracked_features_ids = tracked_features_ids[fb_good.flatten()]

				# Remove tracks with no match in the current frame
				#for id_bad in range(bad_ids.shape[0]): # for each bad id
					#if tracks.get(bad_ids[id_bad].item(0)): # check if the id is present in the dictioanry
					#del tracks[bad_ids[id_bad].item(0)]	# if yes I delete it
						#cv.circle(mask, (int(tracked_features[id_bad, 0]), int(tracked_features[id_bad, 1])), circle_size, 0, thickness=-1)


				circle_size = 20

				# Update tracks
				for feature_idx in range(tracked_features.shape[0]):
					#tidx = tracked_features_ids[feature_idx].item(0)
					#if tracks.get(tidx) is None: # in case there is not the id in the track dictionary initialize it to an empty list
					#	tracks[tidx] = []

					# Add current location to the track
					#tracks[tidx].append(tracked_features[feature_idx, :]) # append the coordinates of the new feature to tack

					# Render current feature position
					pos = (int(tracked_features[feature_idx, 0]), int(tracked_features[feature_idx, 1])) # set the pair of X and Y coordiantes
					
					cv.circle(frame, pos, circle_size, (255,255,255), thickness=-1)
					
					cv.drawMarker(frame, pos, (0,0,255), markerSize=5, thickness=2) # draw a marker in the specific position

					# Update a mask image so we will avoid extract new features too close to this one
					cv.circle(mask, pos, circle_size, 255, thickness=-1)
					# this maybe if I am correct will ensure that we do not extract new feature within the given mask 

			
			corners =  find_interesting_points(frameg, mask)#np.squeeze(find_interesting_points(frameg, mask))
			
			#print(actual_fps, corners.shape, type(corners))
   
			# And add to the list of tracked features
			new_indices = np.array(range(max_index,max_index+corners.shape[0])) # the new indices start from the current maximum idx up to the last corner id + max index
			tracked_features = np.vstack((tracked_features, corners)) # stack arrays in sequence vertically, for the coordinate of the cumulative features
			tracked_features_ids = np.vstack((tracked_features_ids, np.expand_dims(new_indices,axis=-1)) ) # stack arrays in sequence vertically, for the feature ids 
			max_index = np.int32(np.amax(tracked_features_ids) + 1) # update the maximum index id
			#print(new_indices, tracked_features, tracked_features_ids)
			#print(actual_fps, tracked_features.shape)
	
			# update the old gray frame
			prev_frameg = frameg
   
   
			cv.imshow('frame',frame)
			#cv.imshow('mask',mask)
   

			if cv.waitKey(1) == ord('q'):
				break

			actual_fps += 1

		print(' DONE')

		#print('Saving data...')
		#save_stats(obj_id, dict_stats)
		#print(' DONE\n')

		# Release the input and output streams
		input_video.release()
		#output_video.release()
		cv.destroyAllWindows()



if __name__ == "__main__":
	main()