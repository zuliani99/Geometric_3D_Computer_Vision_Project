import cv2 as cv
import numpy as np

th_low = 30
th_high = 130


control_pts = [[600,350],
               [600,375],
               [600,400],
               [600,425],
               [600,450],
               [600,475],
               [600,500],
               [600,550],
               [600,570]
               ]


def main():

    cap = cv.VideoCapture("car.mov")
    paused = True
    print("Press SPACE to play/pause")

    while cap.isOpened():
        ret,frame = cap.read()

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        frame_gray = cv.GaussianBlur(frame_gray, 
                                     None, 
                                     sigmaX=1.8, 
                                     sigmaY=1.8 ).astype(np.uint8)
        
        edge_img = cv.Canny(frame_gray,
                            threshold1=th_low,
                            threshold2=th_high,
                            L2gradient=True)

        edge_img_dbg = cv.cvtColor(edge_img, cv.COLOR_GRAY2BGR)

        left_lane = []
        right_lane = []
        
        for pt in control_pts:
            cv.drawMarker(edge_img_dbg, tuple( pt ), (0,0,255), markerSize=3, thickness=2)
            
            # Find the first edge element at the left of pt
            aux1 = np.nonzero( edge_img[pt[1],pt[0]::-1])
            # Find the first edge element at the right of pt
            aux2 = np.nonzero( edge_img[pt[1],pt[0]::1])


            if (len(aux1) > 0 and len(aux2) > 0 and len(aux1[0]) > 0 and len(aux2[0]) > 0):
                
                leftp = pt[0] - aux1[0][0] 
                rightp = pt[0] + aux2[0][0] 

                centerp = ( int( (rightp+leftp)/2 ) , pt[1])

                left_lane.append( (leftp,pt[1]))
                right_lane.append( (rightp,pt[1]))

                cv.drawMarker(edge_img_dbg, (leftp,pt[1]), (255,0,0), markerSize=13, thickness=2)
                cv.drawMarker(edge_img_dbg, (rightp,pt[1]), (0,255,0), markerSize=13, thickness=2)
                cv.drawMarker(edge_img_dbg, centerp, (255,255,0), markerSize=13, thickness=2)


        # Draw left lane
        for ii in range(len(left_lane)-1):
            cv.line(frame, left_lane[ii], left_lane[ii+1], (0,255,0), 8, cv.LINE_AA)

        # Draw right lane
        for ii in range(len(right_lane)-1):
            cv.line(frame, right_lane[ii], right_lane[ii+1], (0,0,255), 8, cv.LINE_AA)

        cv.imshow("Video", frame)
        cv.imshow("Canny", edge_img)
        cv.imshow("Lane detection", edge_img_dbg)

        
        key = cv.waitKey(0 if paused else 10)
        if key==32:
            paused = not paused



if __name__ == "__main__":
    main()