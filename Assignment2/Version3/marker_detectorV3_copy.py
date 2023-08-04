import cv2 as cv
import numpy as np

#from utils import save_stats, set_marker_reference_coords
from board import find_interesting_points


# List of object file name that we have to process
objs = ['obj01.mp4', 'obj02.mp4', 'obj03.mp4', 'obj04.mp4']

# se the Lucas Kanade parameters
criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.01)
winsize = (3,3)
maxlevel = 3

circle_mask_size = 9


def sort_vertices_clockwise(vertices, centroid=None):
    if centroid is None: centroid = np.mean(vertices, axis=0)
    angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    return centroid, vertices[sorted_indices]


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
  

        ret, frame = input_video.read()
        frameg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        mask = np.zeros_like(frameg)
        tracked_features = find_interesting_points(frameg, mask)
  
        for x, y in tracked_features:
            cv.circle(frame, (int(x), int(y)), 3, (0,0,255), -1)
            cv.circle(mask, (int(x), int(y)), circle_mask_size, 255, -1)
   
        output_video.write(frame)
   
        prev_frameg = frameg

        while True:
            
            ret, frame = input_video.read()

            if not ret:	break

         
            frameg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            mask = np.zeros_like(frameg)
            
          
            p1, st, _ = cv.calcOpticalFlowPyrLK(prev_frameg, frameg, tracked_features, None, winSize=winsize, maxLevel=maxlevel, criteria=criteria)
            assert(p1.shape[0] == tracked_features.shape[0])
            p0r, st0, _ = cv.calcOpticalFlowPyrLK(frameg, prev_frameg, p1, None, winSize=winsize, maxLevel=maxlevel, criteria=criteria)

            fb_good = (np.fabs(p0r-tracked_features) < 0.1).all(axis=1)
            fb_good = np.logical_and(np.logical_and(fb_good, st.flatten()), st0.flatten())
            
            tracked_features = p1[fb_good, :]
            
            
            
            
            

            for x, y in tracked_features: 
                cv.circle(frame, (int(x), int(y)), 3, (0,0,255), -1)
                cv.circle(mask, (int(x), int(y)), circle_mask_size, 255, -1)
            
   
            new_corner = find_interesting_points(frameg, mask)
            tracked_features = np.vstack((tracked_features, new_corner)) 

            for x, y in new_corner: 
                cv.circle(frame, (int(x), int(y)), 3, (0,255,255), -1)
                cv.circle(mask, (int(x), int(y)), circle_mask_size, 255, -1)
    

            centroid, clockwise = sort_vertices_clockwise(tracked_features, [1290,540])
   
            
   
   
            cv.drawMarker(frame, np.int32(centroid), color=(255,255,255), markerType=cv.MARKER_CROSS, thickness=2)
   
            if tracked_features.shape[0] < 90: 
                print('ALT!!!!!!!!!!!!')
                cv.imshow('frame',frame)
                cv.waitKey(-1)


            if(clockwise.shape[0] < 95): clockwise = np.reshape(clockwise[:90,:], (18,5,2))
            else: clockwise = np.reshape(clockwise[:95,:], (19,5,2))
            
   
            print('area', clockwise.shape[0], cv.contourArea(clockwise[-1,:,:]))

            if(cv.contourArea(clockwise[-1,:,:]) < 800):
                print('area', cv.contourArea(clockwise[-1,:,:]))
                to_remove = np.all(tracked_features[:, None] == clockwise[-1,:,:], axis=2).any(axis=1)
                tracked_features = np.delete(tracked_features, to_remove, axis=0)
                clockwise = clockwise[:clockwise.shape[0]-1,:,:]
    
    
            for poly in clockwise:
                _, print_order = sort_vertices_clockwise(poly)
                cv.drawContours(frame, np.int32([print_order]), 0, (255,255,0), 1, cv.LINE_AA)







            prev_frameg = frameg

            cv.imshow('frame',frame)
            #cv.imshow('mask',mask)
   
   
            print('tracked_features', tracked_features.shape[0], int(tracked_features.shape[0]/5))
   
            cv.waitKey(-1)
      
            
            output_video.write(frame)

            key = cv.waitKey(1)
            #if key == ord('p'):
                #cv.waitKey(-1) #wait until any key is pressed
   
            if key == ord('q'):
                return

            actual_fps += 1

        print(' DONE\n')
  

        # Release the input and output streams
        input_video.release()
        output_video.release()
        cv.destroyAllWindows()




if __name__ == "__main__":
    main()