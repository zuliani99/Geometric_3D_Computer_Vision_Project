import cv2 as cv
import numpy as np


def apply_homography_helper(I, H):
    p1 = H@np.array([0,0,1])
    p1 = p1 / p1[2]
    p2 = H@np.array([I.shape[1],0,1] )
    p2 = p2 / p2[2]
    p3 = H@np.array([I.shape[1],I.shape[0],1] )
    p3 = p3 / p3[2]
    p4 = H@np.array([0,I.shape[0],1] )
    p4 = p4 / p4[2]

    #print(p1)
    #print(p2)
    #print(p3)
    #print(p4)

    top = np.amin( [p1[1],p2[1],p3[1],p4[1]] )
    left = np.amin( [p1[0],p2[0],p3[0],p4[0]] )
    bottom = np.amax( [p1[1],p2[1],p3[1],p4[1]] )
    right = np.amax( [p1[0],p2[0],p3[0],p4[0]] )
    
    #print("bbox:   %f %f - %f %f"%(left,right,top,bottom))

    midp = np.array( [ (right+left)/2, (bottom+top)/2 ] )

    scale = np.amax( [right-left, bottom-top] )

    HH = np.array( [ [1,0,I.shape[1]/2 ], [0,1,I.shape[0]/2], [0,0,1] ] ) @ np.array( [ [500/scale,0,0 ], [0,500/scale,0], [0,0,1] ] ) @ np.array( [ [1,0,-midp[0] ], [0,1,-midp[1]], [0,0,1] ] ) @ H

    return cv.warpPerspective( I, HH, ( int(I.shape[1]),int(I.shape[0]) ))


def line_from_points( p1, p2 ):
    p1h = np.array( [p1[0], p1[1],1] )
    p2h = np.array( [p2[0], p2[1],1] )
    l = np.cross(p1h,p2h)
    return l / np.sqrt( l[0]**2 + l[1]**2 )



def main():
    I = cv.imread("./data/pavement2.jpg")

    K = np.array( [ [I.shape[1],0,I.shape[1]/2],[0,I.shape[1],I.shape[0]/2],[0,0,1] ] )
    print(K)

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

        for i in range( 0,len(points)-1,2 ):
            cv.line( Id, tuple(points[i].astype(int)), tuple(points[ (i+1) ].astype(int)), (0,0,255), 2 )
        
        if len(points) == 8:
            l1 = line_from_points( points[0], points[1] )
            l2 = line_from_points( points[2], points[3] )

            vp1 = np.cross( l1, l2 )
            print( "Vanishing point 1" )
            print(vp1)

            l1 = line_from_points( points[4], points[5] )
            l2 = line_from_points( points[6], points[7] )

            vp2 = np.cross( l1, l2 )
            print( "Vanishing point 2" )
            print(vp2)


            l = np.cross(vp1,vp2)
            l = l / np.sqrt( l[0]**2 + l[1]**2 )
            print( "Vanishing line" )
            print(l)

            n = K.T @ l
            n = n / np.linalg.norm(n)
            print("Plane normal: ")
            print(n)

            r1 = np.cross( n, np.array([1,0,0])  )
            r1 = r1 / np.linalg.norm(r1)
            r2 = np.cross( r1,n )

            R = np.vstack( (r2,r1,n) )
            print(R)
            print( np.linalg.det(R) )

            print( "R n:")
            print( R@n )

            H = K @ R @ np.linalg.inv(K)

            print("H:")
            print(H)
            print("H^-T l:")
            print( np.linalg.inv(H).T @ l )

            warped = apply_homography_helper(I,H)

            cv.imshow("warped", warped )
            cv.waitKey(0)
            points = []
            cv.destroyWindow("warped")

            points = []



if __name__ == "__main__":
    main()
