import numpy as np
import cv2


class CameraViewer:
    modes = [ ["EXPOSURE",cv2.CAP_PROP_EXPOSURE,0.1], ["BRIGHTNESS",cv2.CAP_PROP_BRIGHTNESS,1], ["CONTRAST",cv2.CAP_PROP_CONTRAST,1], ["SHARPNESS",cv2.CAP_PROP_SHARPNESS,1], ["GAIN",cv2.CAP_PROP_GAIN,1], ["SATURATION",cv2.CAP_PROP_SATURATION,1] ]

    def __init__( self, device = 0, rot90=False ):
        self.cap = cv2.VideoCapture( device )
        self.cap.set( cv2.CAP_PROP_FRAME_WIDTH, 1280 )
        self.cap.set( cv2.CAP_PROP_FRAME_HEIGHT, 1280 )
        self.cap.set( cv2.CAP_PROP_SHARPNESS, 0 )
        self.mode = 0
        self.rot90 = rot90
        self.img_idx = 0
        self.last_frame = None
        self.denoise = 0.0

    def get_size( self ):
        return ( self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame( self ):
        self.ret, self.img = self.cap.read()
        if not self.img is None:
            if self.rot90:
                self.img = cv2.rotate(self.img, cv2.ROTATE_90_COUNTERCLOCKWISE )

            if not self.last_frame is None:
                self.img = (1.0-self.denoise)*self.img.astype(np.float32) + self.denoise*self.last_frame

            self.last_frame = self.img.astype(np.float32)
            self.img = self.img.astype(np.uint8)

        return self.img

    def is_valid( self ):
        return self.ret

    def control( self, keyp ):
        """
        Give a command (usually from a keypress) to the CameraViewer

        Parameters:
        keyp:  the integer keyboard code pressed

        Commands:
                "m": change the current camera property mode (Ex. Brightness, Sharpness, Constrast, etc)
          UP or "a": Increase the value of the current property
        DOWN or "z": Decrease the value of the current property
                "s": Save the current frame to snaps/img_%05d.png
                "r": Rotate the camera 90° CCW
        """
        if keyp == ord('m'):
            self.mode = (self.mode + 1) % len(CameraViewer.modes) 
            print("Current mode: %s" % CameraViewer.modes[self.mode][0])

        if keyp == 38 or keyp == ord('a'):
            self.cap.set( CameraViewer.modes[ self.mode][1], self.cap.get(CameraViewer.modes[ self.mode][1])+CameraViewer.modes[self.mode][2] )

        if keyp == 40 or keyp == ord('z'):
            self.cap.set( CameraViewer.modes[ self.mode][1], self.cap.get(CameraViewer.modes[ self.mode][1])-CameraViewer.modes[self.mode][2] )

        if keyp == ord('r'):
            self.rot90 = not self.rot90
            self.last_frame = None

        if keyp == ord('+'):
            self.denoise = self.denoise+0.1
            if self.denoise>1.0:
                self.denoise=1.0
            print("Denoise level: ",self.denoise)

        if keyp == ord('-'):
            self.denoise = self.denoise-0.1
            if self.denoise<0.0:
                self.denoise=0.0
            print("Denoise level: ",self.denoise)

        if keyp == ord('s'):
            filename = "snaps/img_%05d.png" % self.img_idx
            print( "Saving %s"%filename )
            cv2.imwrite(filename, self.img )
            self.img_idx = self.img_idx+1
        




def main():
    img_idx = 0

    print("Opening capture device")
    cam = CameraViewer()

    running = True
    while running:

        img = cam.get_frame()
        if not cam.is_valid():
            running = False
            continue

        cv2.imshow('camera', img)
        keyp = cv2.waitKey(1)
        cam.control(keyp)
        running = keyp != 113  # Press q to exit




if __name__ == "__main__":
    print("Using OpenCV %s" % (cv2.__version__))
    main()
