import cv2 as cv
import time
import copy
import numpy as np

# List of object file name that we have to process
objs = ['obj01.mp4', 'obj02.mp4', 'obj03.mp4', 'obj04.mp4']


feature_params = dict(maxCorners = 100,
                    qualityLevel = 0.01,
                    minDistance = 10,
                    blockSize = 11)


def main() -> None:
	
 	# Set the marker reference coordinates for the 24 polygonls
	#marker_reference = set_marker_reference_coords()
	
	# Iterate for each object
	for obj in objs:
		obj_id = obj.split('.')[0]
		print(f'Marker Detector for {obj}...')
		input_video = cv.VideoCapture(f"../../data/{obj}")

		fps = input_video.get(cv.CAP_PROP_FPS)
  
		actual_fps = 0
  
		while True:
			start = time.time() # Start the timer to compute the actual FPS 
			
			# Extract a frame
			ret, frame = input_video.read()

			if not ret:	break
			
   
			gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
      
			mask_thresh = np.zeros((1080, 1920), dtype=np.uint8)
			mask_thresh[:, 1050:1600] = gray[:, 1050:1600]
      
			corners = cv.goodFeaturesToTrack(mask_thresh, **feature_params)
			corners = np.int0(corners)
			for i in corners:
				x,y = i.ravel()
				cv.circle(frame,(x,y),3,(0,0,255),-1)


			#edited_frame = frame#board.draw_stuff(frame)




			frame_with_fps = copy.deepcopy(frame) 
   
			end = time.time()
			fps = 1 / (end-start) # Compute the FPS

			# Output the frame with the FPS
			cv.putText(frame_with_fps, f"{fps:.2f} FPS", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
			cv.imshow(f'Marker Detector of {obj}', frame_with_fps)
   

			if cv.waitKey(1) == ord('q'):
				break

			actual_fps += 1

		input_video.release()
		cv.destroyAllWindows()



if __name__ == "__main__":
	main()