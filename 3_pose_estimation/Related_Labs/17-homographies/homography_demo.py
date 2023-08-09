import cv2 as cv
import numpy as np


def main():
    #I = cv.imread("./data/book.jpg")
    I = cv.imread("./data/facade.jpg")

    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            points.append( np.array( (x,y), dtype=np.float32 ))
        pass

    
    Id = np.array(I)
    while True:


        cv.imshow("img", Id)
        cv.setMouseCallback("img", mouse_callback)
        
        if cv.waitKey(16) == ord('q'):
            break

        for i in range( len(points) ):
            cv.line( Id, tuple(points[i].astype(int)), tuple(points[ (i+1)%len(points) ].astype(int)), (0,0,255), 2 )
        
        if len(points) == 4:

            cv.imshow("img", Id)
            dst_pts = np.array( [ [0,0], [400,0], [400,500], [0,500] ] ) 

            H = cv.findHomography( np.array(points), dst_pts )
            warped = cv.warpPerspective( I, H[0], (400,500) )
            cv.imshow("warped", warped )
            cv.waitKey(0)
            points = []
            cv.destroyWindow("warped")
            Id = np.array(I)



if __name__ == "__main__":
    main()
