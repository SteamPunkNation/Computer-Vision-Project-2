import cv2
import time

class Camera:
    def __init__(self, source=0, width=640, height=480, fps=30):
        self.source = source
        
        # On Windows, cv2.CAP_DSHOW (DirectShow) is often much more stable than the default MSMF backend, 
        # especially for high-end webcams like the Logitech Brio which can freeze on initialization.
        if isinstance(source, int):
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(source)
        
        # Request specific resolution and fps
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Read the first frame to ensure it's working
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
            
    def read(self):
        """Reads a single frame from the camera."""
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, frame
        
    def release(self):
        """Releases the camera resource."""
        self.cap.release()
