import cv2 as cv
import time
import copy

from utils import save_stats, set_marker_reference_coords
from board import Board


# List of object file name that we have to process
objs = ['obj01.mp4', 'obj02.mp4', 'obj03.mp4', 'obj04.mp4']


def main() -> None:
	'''
	PURPOSE: main function
	ARGUMENTS: None
	RETURN: None
	'''	
	
 	# Set the marker reference coordinates for the 24 polygonls
	marker_reference = set_marker_reference_coords()
	
	# Iterate for each object
	for obj in objs:
		
		print(f'Marker Detector for {obj}...')
		input_video = cv.VideoCapture(f"../../data/{obj}")

		# Get video properties
		frame_width = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
		frame_height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))

		actual_fps = 0
		obj_id = obj.split('.')[0]
  
		board = Board(24)
  
		dict_stats = [] # Initialize the list of dictionary that we will save as .csv file

		# Create output video writer
		output_video = cv.VideoWriter(f"../../output_part2/{obj_id}/{obj_id}_mask.mp4", cv.VideoWriter_fourcc(*"mp4v"), input_video.get(cv.CAP_PROP_FPS), (frame_width, frame_height))
  
		# Until the video is open
		while True:
			start = time.time() # Start the timer to compute the actual FPS 
			
			# Extract a frame
			ret, frame = input_video.read()

			if not ret:	break

			# Obtain the edited frame and the dictionary of statistics
			dict_stats_to_extend = board.find_markers(frame, actual_fps, marker_reference)

			# Draw stuff on the image
			edited_frame = board.draw_stuff(frame)

			frame_with_fps = copy.deepcopy(edited_frame) 
   
			end = time.time()
			fps = 1 / (end-start) # Compute the FPS

			# Output the frame with the FPS
			cv.putText(frame_with_fps, f"{fps:.2f} FPS", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
			cv.imshow(f'Marker Detector of {obj}', frame_with_fps)
   
			# Save the frame without the FPS count
			output_video.write(edited_frame)

			# Extends our list with the obtained dictionary
			dict_stats.extend(dict_stats_to_extend)

			key = cv.waitKey(1)
			if key == ord('p'):
				cv.waitKey(-1) #wait until any key is pressed
    
			if key == ord('q'):
				return

			actual_fps += 1

		print(' DONE')

		print('Saving data...')
		save_stats(obj_id, dict_stats)
		print(' DONE\n')

		# Release the input and output streams
		input_video.release()
		output_video.release()
		cv.destroyAllWindows()



if __name__ == "__main__":
	main()