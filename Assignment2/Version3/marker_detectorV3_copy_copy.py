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

circle_mask_size = 11 #10


def order_vertices_clockwise(vertices):
    # Calculate the centroid
    centroid = np.mean(vertices, axis=0)

    # Calculate polar angles with respect to the centroid
    angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])

    # Sort the vertices based on angles (counterclockwise order)
    sorted_indices = np.argsort(angles)

    return vertices[sorted_indices[::-1]]


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

		prev_frameg = None
		#tracked_poly_vertices = np.zeros( (0,5,2), dtype=np.float32 ) 


		ret, frame = input_video.read()
		frameg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		mask = np.zeros_like(frameg)
		_, tracked_poly_vertices =  find_interesting_points(frameg, mask) # -> it havbe to return 18,5,2
		#tracked_poly_vertices = np.vstack((tracked_poly_vertices, corners))

		# here I fund the initial vertex of each visible polygon using my corner detecotr
		for poly in tracked_poly_vertices:
			for x, y in poly:
				pos = (int(x), int(y))		
				cv.circle(frame, pos, 3, (0,0,255), -1) # modify the frame
				cv.circle(mask, pos, circle_mask_size, 255, -1) # set the mask to ingnore future near detected marker

		output_video.write(frame)

		prev_frameg = frameg
		
		#print(actual_fps, tracked_poly_vertices.shape)

		# Until the video is open
		while True:
			
			# Extract a frame
			ret, frame = input_video.read()

			if not ret:	break


			frameg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
			mask = np.zeros_like(frameg)

			# here I have to reshape the traked features into a 2 dimensional array
			tf_OF = np.reshape(tracked_poly_vertices, (tracked_poly_vertices.shape[0]*tracked_poly_vertices.shape[1], tracked_poly_vertices.shape[2]))

			print(tf_OF)

			p1, st, _ = cv.calcOpticalFlowPyrLK(prev_frameg, frameg, np.float32(tf_OF), None, winSize=winsize, maxLevel=maxlevel, criteria=criteria)
			assert(p1.shape[0] == tf_OF.shape[0])
			p0r, st0, _ = cv.calcOpticalFlowPyrLK(frameg, prev_frameg, p1, None, winSize=winsize, maxLevel=maxlevel, criteria=criteria)

			fb_good = (np.fabs(p0r - tf_OF) < 0.1).all(axis=1)
			fb_good = np.logical_and(np.logical_and(fb_good, st.flatten()), st0.flatten())


			new_features_bool = np.reshape(fb_good, (int(fb_good.shape[0]/tracked_poly_vertices.shape[1]), tracked_poly_vertices.shape[1], 1))
			lost_features_bool = np.logical_not(new_features_bool)
			
			actual_tracked_poly_vertices = tracked_poly_vertices * new_features_bool.astype(float)
			lose_tracked_poly_vertices = tracked_poly_vertices * lost_features_bool.astype(float)
   
			#print(actual_tracked_poly_vertices)
			#print(lose_tracked_poly_vertices)
   
			for poly in actual_tracked_poly_vertices:
				for x, y in poly:
					if x!= 0. and y != 0.: cv.circle(mask, (int(x), int(y)), circle_mask_size, 255, -1)
					# here i draw the point that LK predict 


			tracked_poly_vertices, new_polygon_coords = find_interesting_points(frameg, mask, actual_tracked_poly_vertices, lose_tracked_poly_vertices)
			index_new_polygon = -1
   
			for idx, poly in enumerate(tracked_poly_vertices):
				clock_wise_oreder = order_vertices_clockwise(poly)
				area = cv.contourArea(np.int32(clock_wise_oreder))
				if area < 1720:
					tracked_poly_vertices[idx]= np.zeros((5,2), dtype=np.float32)
					index_new_polygon = idx
					break
 
			# I found a new polygon
			if index_new_polygon != -1: tracked_poly_vertices[index_new_polygon] = np.squeeze(new_polygon_coords[0])
			

			for poly in tracked_poly_vertices:
				for x, y in poly:
					if x!= 0. and y != 0.: cv.circle(frame, (int(x), int(y)), 2, (0,0,255), -1)

			prev_frameg = frameg


			cv.imshow('frame',frame)
			#cv.imshow('mask',mask)

			output_video.write(frame)

			key = cv.waitKey(1)
			if key == ord('p'):
				cv.waitKey(-1) #wait until any key is pressed

			if key == ord('q'):
				break

			actual_fps += 1

		print(' DONE\n')


		# Release the input and output streams
		input_video.release()
		output_video.release()
		cv.destroyAllWindows()




if __name__ == "__main__":
	main()