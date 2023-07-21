import numpy as np
import cv2 as cv
import time


lk_params = dict(winSize  = (15, 15),
                maxLevel = 2,
                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners = 10,
                    qualityLevel = 0.3,
                    minDistance = 10,
                    blockSize = 7 )


trajectory_len = 20
detect_interval = 1
trajectories = []
frame_idx = 0

if __name__ == "__main__":
	
		#for obj in list('obj1.mp4'):
		obj = 'obj01.mp4'
  
		input_video = cv.VideoCapture(f"../data/{obj}")



		# Create output video writer
		#output_video = cv.VideoWriter(f"../output_part1/{obj.split('.')[0]}_mask.mp4", cv.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
  
		while True:
			start = time.time()
      
			# Extract a frame
			ret, frame = input_video.read()

			if not ret:	break
   
			frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
			img = frame.copy()

			# Calculate optical flow for a sparse feature set using the iterative Lucas-Kanade Method
			if len(trajectories) > 0:
				img0, img1 = prev_gray, frame_gray
				p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
				p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
				p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
				d = abs(p0-p0r).reshape(-1, 2).max(-1)
				good = d < 1

				new_trajectories = []

				# Get all the trajectories
				for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
					if not good_flag:
						continue
					trajectory.append((x, y))
					if len(trajectory) > trajectory_len:
						del trajectory[0]
					new_trajectories.append(trajectory)
					# Newest detected point
					cv.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

				trajectories = new_trajectories

				# Draw all the trajectories
				cv.polylines(img, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))
				cv.putText(img, 'track count: %d' % len(trajectories), (20, 50), cv.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)


			# Update interval - When to update and detect new features
			if frame_idx % detect_interval == 0:
				mask = np.zeros_like(frame_gray)
				mask[:] = 255

				# Lastest point in latest trajectory
				for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
					cv.circle(mask, (x, y), 5, 0, -1)

				# Detect the good features to track
				p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
				if p is not None:
					# If good features can be tracked - add that to the trajectories
					for x, y in np.float32(p).reshape(-1, 2):
						trajectories.append([(x, y)])


			frame_idx += 1
			prev_gray = frame_gray

			# End time
			end = time.time()
			# calculate the FPS for current frame detection
			fps = 1 / (end-start)
			
			# Show Results
			cv.putText(img, f"{fps:.2f} FPS", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
			cv.imshow('Optical Flow', img)
			cv.imshow('Mask', mask)
   
   

   
			if cv.waitKey(1) == ord('q'):
				break
			
			
		input_video.release()
		cv.destroyAllWindows()


