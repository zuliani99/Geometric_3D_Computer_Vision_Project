import cv2 as cv
import numpy as np

# color function to select a random color
def colors(n):
	import random
	ret = []
	r = int(random.random() * 256)
	g = int(random.random() * 256)
	b = int(random.random() * 256)
	step = 256 / n
	for _ in range(n):
		r += step
		g += step
		b += step
		r = int(r) % 256
		g = int(g) % 256
		b = int(b) % 256
		ret.append((r,g,b))
	return ret


if __name__ == "__main__":

	cap = cv.VideoCapture(0)#('./car.mov')#('./obj01.mp4')

	history_size = 20 # lenght of the trak

	prev_frameg = None # Previosu gray frame
	tracked_features = np.zeros( (0,2), dtype=np.float32 ) # 0 rows and 2 columsn for storing the track
	tracked_features_ids = np.zeros( (0,1), dtype=np.int64 ) # 0 row and 1 column to store the feature ID
	
	max_index = 0 # index of maximum id in the tracked_features_ids array
	palette = colors(30) # 30 colors
	
	tracks = {} # dictioanry of track
	curr_shot = 0

	# see the Lucas Kanade parameters
	criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 50, 0.01)
	winsize =(5,5)
	maxlevel=1
	fb_threshold = 0.8
	

	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()
		if not ret: break

		frameg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # convert in gray
		#frameg_original = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # convert in gray
		#frameg = np.zeros(frameg_original.shape, dtype=np.uint8)
		#frameg[:, 1050:1600] = frameg_original[:, 1050:1600]

		mask = np.zeros( frameg.shape, dtype=np.uint8) # create a mask with the same size as the gray frame

		# Update tracked features
		if tracked_features.shape[0]>0:		# in case I have something within my array of features to trak
			
			
			p1, st, err = cv.calcOpticalFlowPyrLK(prev_frameg, # Old frame
												  frameg, # New frame
												  tracked_features, # Poins to track
												  None, 
												  winSize=winsize, 
												  maxLevel=maxlevel, 
												  criteria=criteria)
			
			# We compute the flow in both way forward and backward
			
			assert(p1.shape[0] == tracked_features.shape[0])
			
			p0r, st0, err = cv.calcOpticalFlowPyrLK(frameg, 
													prev_frameg, 
													p1, 
													None, 
													winSize=winsize, 
													maxLevel=maxlevel, 
													criteria=criteria)


			# find the good ones


			fb_good = (np.fabs(p0r-tracked_features) < fb_threshold).all(axis=1)
			fb_good = np.logical_and(np.logical_and(fb_good, st.flatten()), st0.flatten())
			# is a true false array

			# ids of bad features
			bad_ids = tracked_features_ids[np.logical_not( fb_good.flatten())]

			tracked_features = p1[fb_good, :]
			tracked_features_ids = tracked_features_ids[fb_good.flatten()]

			# Remove tracks with no match in the current frame
			for id_bad in range(bad_ids.shape[0]): # for each bad id
				if tracks.get(bad_ids[id_bad].item(0)): # check if the id is present in the dictioanry
					del tracks[bad_ids[id_bad].item(0)]	# if yes I delete it


			# Update tracks
			for feature_idx in range(tracked_features.shape[0]):
				tidx = tracked_features_ids[feature_idx].item(0)
				if tracks.get(tidx) is None: # in case there is not the id in the track dictionary initialize it to an empty list
					tracks[tidx] = []

				# Add current location to the track
				tracks[tidx].append(tracked_features[feature_idx, :]) # append the coordinates of the new feature to tack
				fcol = palette[tidx%len(palette)] # set the feature color
				
    
				history = tracks[ tidx ]
				if len(history)>history_size:
					# History too long, remove the last elements
					history=history[len(history)-history_size:]
					tracks[tidx] = history

				# Render feature history as line segments
				for hh in range(0, len(history)-2, 1 ):
					pos = (int(history[hh][0]),int(history[hh][1]))
					posn = (int(history[hh+1][0]),int(history[hh+1][1]))
					cv.line( frame, pos, posn , fcol, 1  )
    
    
				# Render current feature position
				pos = (int(tracked_features[feature_idx, 0]), int(tracked_features[feature_idx, 1])) # set the pair of X and Y coordiantes
				cv.drawMarker(frame, pos, fcol, markerSize=5, thickness=2) # draw a marker in the specific position

				# Update a mask image so we will avoid extract new features too close to this one
				cv.circle(mask, pos, 20, (255,255,255), thickness=-1)
				# this maybe if I am correct will ensure that we do not extract new feature in the given mask


		# Extract new features, so corners
		corners =  np.squeeze(cv.goodFeaturesToTrack(frameg, 
														20, 
														qualityLevel=0.0001, 
														minDistance=10, 
														blockSize=11, 
														mask=(255-mask)
														).astype(np.float32))

		# And add to the list of tracked features
		new_indices = np.array(range(max_index,max_index+corners.shape[0])) # the new indices start from the current maximum idx up to the last corner id + max index
		tracked_features = np.vstack((tracked_features, corners)) # stack arrays in sequence vertically, for the coordinate of the cumulative features
		tracked_features_ids = np.vstack((tracked_features_ids, np.expand_dims(new_indices,axis=-1)) ) # stack arrays in sequence vertically, for the feature ids 
		max_index = np.amax(tracked_features_ids) + 1 # update the maximum index id

  
		# update the old gray frame
		prev_frameg = frameg

		# Display the resulting frame
		cv.imshow('frame',frame)
		#cv.imshow('mask',mask)
		keypressed = cv.waitKey(1)
		if keypressed & 0xFF == ord('q'):
			break

		if keypressed & 0xFF == ord('s'):
			cv.imwrite( "shot_%04d.png"%(curr_shot), frameg )
			curr_shot = curr_shot+1

	# When everything done, release the capture
	cap.release()
	cv.destroyAllWindows()

	print("bye")
