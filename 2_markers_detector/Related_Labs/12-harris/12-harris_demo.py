"""
    Real-time Harris corner detector demo

    Keyboard commands:
          i: cycle the underlying image (camera frame, gradient, harris response)
          s: save current frame
          m: change camera property
        a/z: increase/decrease the value of the current property
          q: quit

"""

SCALE = 7
K = 0.1
THRESH = 0.01
NMS_WINSIZE = 13

#####################################################################


import cv2 as cv
import numpy as np
from CamViewer import CameraViewer


def mainloop():
    print("Opening car.mov video")
    
    #cam = CameraViewer(0)		# uses the webcam instead
    
    cam = CameraViewer('./car.mov')
    print("Press q to exit")

    running = True
    image_to_show = 0
    while running:

        img = cam.get_frame()
        if not cam.is_valid():
            #running = False
            continue


        # Harris corner detection

        img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        
        
        
        # - Compute horizontal and vertical component of the gradient, using the SOBEL operator
        # - Apply directly SOBEL without blurring, since noise is mandled by the averaging in second moment matrix
        Ix = cv.Sobel(img_gray, cv.CV_32F, dx=1, dy=0, ksize=3) / 255.0
        Iy = cv.Sobel(img_gray, cv.CV_32F, dx=0, dy=1, ksize=3) / 255.0

		# Image gradient whcich correspond to the direnction of the edges
        grad_mag = np.sqrt( Ix**2 + Iy**2 )

		# - Generating the value of M IxIx, IyIy and only IxIy and not also IyIx sinche matrix M is symmetric
        # Calculate the 3 elements of the second moment matrix convolcing Iu^2, IuIv, Iv^2, with the gaussian kernel
        #	(calculate an average of the neighborhood 7x7 of each pixel)
        Hxx = cv.GaussianBlur(Ix * Ix, (SCALE,SCALE), sigmaX = 0, sigmaY = 0)
        Hxy = cv.GaussianBlur(Ix * Iy, (SCALE,SCALE), sigmaX = 0, sigmaY = 0)
        Hyy = cv.GaussianBlur(Iy * Iy, (SCALE,SCALE), sigmaX = 0, sigmaY = 0)

		# Harris Response
		#  R  =        det(M)       - k(trace(M))^2
        Hresp = (Hxx*Hyy - Hxy*Hxy) - K*(Hxx+Hyy)**2


        # Non-maxima suppression -> keeps obnly corners that exceed a certain threshold and are not lcoal maxima
		# Gray scale dilatation computes maximum of the pixel on the image
        Hrespd = cv.dilate(Hresp,
                           cv.getStructuringElement(cv.MORPH_ELLIPSE, (NMS_WINSIZE,NMS_WINSIZE)))
        
        corners = ((Hresp > THRESH) * (Hresp == Hrespd)).astype(np.uint8)*255
									# pixel that after dilatation do not change are local maxima
        
        corners_loc = np.argwhere(corners > 0)
        ##########################



        # Visualization Steps
        #
        shown_img = [img, grad_mag, Hresp][image_to_show]

        if shown_img.dtype != np.uint8:
            # Scale min-max to 0-1 if image is floating point
            cv.normalize(shown_img, shown_img, 0, 1, norm_type=cv.NORM_MINMAX)
        
        if shown_img.ndim < 3 or shown_img.shape[2] == 1:
            # Transform to color image so we can later plot coloured circles
            shown_img = cv.cvtColor(shown_img, cv.COLOR_GRAY2BGR)

        for corner in corners_loc:
            cv.circle(shown_img, (corner[1], corner[0]), 3, (0,0,255), 2, cv.LINE_AA)

        cv.imshow('camera', shown_img)
        keyp = cv.waitKey(1)
        cam.control(keyp)
        running = keyp != 113  # Press q to exit

        if keyp == 105: # Press i to change the displayed image
            image_to_show = (image_to_show + 1) % 3


if __name__ == "__main__":
    mainloop()