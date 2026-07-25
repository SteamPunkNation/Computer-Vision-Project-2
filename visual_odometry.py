import cv2
import numpy as np

class VisualOdometry:
    def __init__(self, focal_length=715.0, pp=(320.0, 240.0)):
        # Camera intrinsic matrix parameters (approximate for typical webcams if uncalibrated)
        self.focal = focal_length
        self.pp = pp
        
        # Internal state
        self.cur_R = None
        self.cur_t = None
        
        # Trajectory storage for plotting
        self.trajectory = []
        
        # Scale for monocular VO
        # The user requested assuming the camera is at a traditional height for a webcam
        # Height of a person sitting at a desk is ~1.2m
        # However, for pure monocular VO without IMU, scale is usually ambiguous. 
        # For testing, we use a constant scale factor, or calculate it based on camera height assumption.
        self.absolute_scale = 1.0 

    def update(self, q1, q2):
        """
        Calculates the camera pose given matched keypoints q1 (prev) and q2 (current).
        """
        if q1 is None or q2 is None or len(q1) < 8 or len(q2) < 8:
            return self.cur_R, self.cur_t, None, None
            
        # Calculate Essential Matrix using RANSAC
        E, mask = cv2.findEssentialMat(q2, q1, focal=self.focal, pp=self.pp, 
                                       method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        if E is None or E.shape != (3, 3):
            return self.cur_R, self.cur_t, None, None

        # Recover relative camera rotation and translation from Essential Matrix
        _, R, t, mask = cv2.recoverPose(E, q2, q1, focal=self.focal, pp=self.pp)
        
        # If this is the first frame, initialize current pose
        if self.cur_R is None or self.cur_t is None:
            self.cur_R = R
            self.cur_t = t
            self.trajectory.append((self.cur_t[0][0], self.cur_t[2][0])) # x, z
        else:
            # Update the global pose
            # We assume constant scale=1.0 for monocular VO without external scale reference.
            # In a real system, scale could be computed from known camera height and ground plane.
            
            # Simple heuristic to prevent erratic jumps (if translation is too large)
            if np.linalg.norm(t) > 0.1: # Threshold to filter noise
                self.cur_t = self.cur_t + self.absolute_scale * self.cur_R.dot(t)
                self.cur_R = R.dot(self.cur_R)
                
            self.trajectory.append((self.cur_t[0][0], self.cur_t[2][0]))

        return self.cur_R, self.cur_t, R, t
