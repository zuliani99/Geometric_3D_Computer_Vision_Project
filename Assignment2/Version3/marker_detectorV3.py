import cv2 as cv
import numpy as np
import time
import copy

from board import Board
from utils import save_stats, set_marker_reference_coords, resize_for_laptop


# List of object file name that we have to process
objs = ['obj01.mp4']#, 'obj02.mp4', 'obj03.mp4', 'obj04.mp4']

using_laptop = True


def main():
    
    # Set the marker reference coordinates for the 24 polygonls
	marker_reference = set_marker_reference_coords()

	# Iterate for each object
	for obj in objs:
		
		print(f'Marker Detector for {obj}...')
		input_video = cv.VideoCapture(f"../../data/{obj}")

		frame_width = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
		frame_height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))

		actual_fps = 0
		obj_id = obj.split('.')[0]
  
		board = Board(n_polygons=24, circle_mask_size=13)
  
		dict_stats = [] # Initialize the list of dictionary that we will save as .csv file
  
		output_video = cv.VideoWriter(f"../../output_part2/{obj_id}/{obj_id}_mask.mp4", cv.VideoWriter_fourcc(*"mp4v"), input_video.get(cv.CAP_PROP_FPS), (frame_width, frame_height))

		prev_frameg = None

		while True:
			print('\n\n-------------------------------------', actual_fps, '-------------------------------------')
			start = time.time()
			
			ret, frame = input_video.read()

			if not ret:	break

		 
			frameg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
			_, thresh = cv.threshold(frameg, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
			mask = np.zeros_like(frameg)
   
			
			if(actual_fps % 15 == 0): board.find_interesting_points(thresh, frameg, mask)
			else: board.apply_LF_OF(thresh, prev_frameg, frameg, mask)
    

			reshaped_clockwise = board.get_clockwise_vertices()
			dict_stats_to_extend = board.compute_markers(thresh, reshaped_clockwise, actual_fps, marker_reference)
			edited_frame = board.draw_stuff(frame)

			end = time.time()
			fps = 1 / (end-start)
   
			frame_with_fps_resized, mask_resized = resize_for_laptop(using_laptop, copy.deepcopy(edited_frame), mask)
  
			cv.putText(frame_with_fps_resized, f"{fps:.2f} FPS", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
			cv.imshow(f'Marker Detector of {obj}', frame_with_fps_resized)
  
			output_video.write(edited_frame)

			dict_stats.extend(dict_stats_to_extend)
    
			prev_frameg = frameg
   
			actual_fps += 1

			key = cv.waitKey(1)
			if key == ord('p'):
				cv.waitKey(-1) 
   
			if key == ord('q'):
				return



			#cv.waitKey(-1)



			
		print(' DONE\n')
  
		print('Saving data...')
		save_stats(obj_id, dict_stats)
		print(' DONE\n')
  

		input_video.release()
		output_video.release()
		cv.destroyAllWindows()



if __name__ == "__main__":
	main()