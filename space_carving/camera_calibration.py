import numpy as np
import cv2 as cv


sampled_frames = 40
chessboard_size = (9,6)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


def main() -> None:
	print('Finding Chessboard Corners for Camera Calibration...')
	calibration_video = cv.VideoCapture('../data/calibration.mp4')


	# Get video properties
	frame_width = int(calibration_video.get(cv.CAP_PROP_FRAME_WIDTH))
	frame_height = int(calibration_video.get(cv.CAP_PROP_FRAME_HEIGHT))
	video_length = int(calibration_video.get(cv.CAP_PROP_FRAME_COUNT))
	skip_rate = video_length // (sampled_frames - 1)

	count_frame = 0

	while count_frame < video_length:

		calibration_video.set(cv.CAP_PROP_POS_FRAMES, count_frame)

		ret, frame = calibration_video.read()
		if not ret: break

		frameg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

		# Find the chess board corners
		ret, corners = cv.findChessboardCorners(frameg, chessboard_size, None)
		if ret == True:
			objpoints.append(objp)
			corners2 = cv.cornerSubPix(frameg, corners, (11,11), (-1,-1), criteria)
			imgpoints.append(corners)

			# Draw and display the corners
			cv.drawChessboardCorners(frame, chessboard_size, corners2, ret)
			cv.imshow('Chessboard Frame', frame)

		count_frame += skip_rate

		cv.waitKey(100)

	cv.destroyAllWindows()

	print(' DONE\n')


	print('Running Camera Calibration...')
	ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (frame_width, frame_height), None, None)
	print(' DONE\n')

	mean_error = 0
	for i in range(len(objpoints)):
		imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
		error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
		mean_error += error
	print(f'Total Error: {mean_error / len(objpoints)}\n')
	 

	print('Saving Intrinsic and Distortion matricies...')
	np.save('./calibration_info/cameraMatrix', cameraMatrix)
	np.save('./calibration_info/dist', dist)
	print(' DONE')
 
 
if __name__ == "__main__":
	main()
 