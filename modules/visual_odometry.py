import cv2
import numpy as np
from scipy.optimize import least_squares

class VisualOdometry:
    def __init__(self, focal_length=715.0, pp=(320.0, 240.0), camera_height=1.2, window_size=5):
        # Camera intrinsic matrix parameters (approximate for typical webcams if uncalibrated)
        self.focal = focal_length
        self.pp = pp
        self.K = np.array([[focal_length, 0, pp[0]],
                           [0, focal_length, pp[1]],
                           [0, 0, 1]])
        
        # Internal state
        self.cur_R = None
        self.cur_t = None
        
        # Trajectory storage for plotting
        self.trajectory = []
        
        # Dynamic scale based on camera height assumption
        self.camera_height = camera_height 
        self.absolute_scale = 1.0
        
        # Windowed Bundle Adjustment state
        self.window_size = window_size
        self.poses = [] # Stores (R, t) tuples

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
            self.poses.append((self.cur_R.copy(), self.cur_t.copy()))
            self.trajectory.append((self.cur_t[0][0], self.cur_t[2][0])) # x, z
            return self.cur_R, self.cur_t, R, t
            
        # --- 1. Dynamic Scale Estimation ---
        # Triangulate points to estimate scale based on camera height
        # Projection matrices
        P1 = np.hstack((self.cur_R, self.cur_t))
        
        new_R = R.dot(self.cur_R)
        new_t = self.cur_t + self.cur_R.dot(t) # Unscaled new translation for triangulation
        P2 = np.hstack((new_R, new_t))
        
        # Triangulate
        pts_4d = cv2.triangulatePoints(self.K.dot(P1), self.K.dot(P2), q1.T, q2.T)
        pts_3d = pts_4d[:3, :] / pts_4d[3, :] # Normalize homogeneous coordinates
        pts_3d = pts_3d.T
        
        # Filter for points that are likely on the ground (lower half of image)
        ground_points = []
        for i, pt in enumerate(q1):
            if pt[1] > self.pp[1]: # y-coordinate greater than principal point y (lower half)
                ground_points.append(pts_3d[i])
                
        if len(ground_points) > 10:
            # Assume average y of these points corresponds to the ground, relative to camera height
            avg_y = np.mean([p[1] for p in ground_points])
            if abs(avg_y) > 1e-5:
                # Estimate scale: real_height / observed_height
                estimated_scale = self.camera_height / abs(avg_y)
                # Apply smoothing to scale to avoid erratic jumps
                self.absolute_scale = 0.8 * self.absolute_scale + 0.2 * estimated_scale
        
        # --- 2. Update Global Pose with Scale ---
        if np.linalg.norm(t) > 0.1: # Threshold to filter noise
            self.cur_t = self.cur_t + self.absolute_scale * self.cur_R.dot(t)
            self.cur_R = R.dot(self.cur_R)
            
        self.poses.append((self.cur_R.copy(), self.cur_t.copy()))
        
        # Limit window size
        if len(self.poses) > self.window_size:
            self.poses.pop(0)
            
        # --- 3. Local Windowed Bundle Adjustment ---
        # We perform BA if we have enough poses in the window
        # For simplicity and performance in this demo, we run a very lightweight adjustment
        # on the last few frames translation vector only to smooth the trajectory.
        if len(self.poses) == self.window_size:
            self._local_bundle_adjustment()
            # Update current pose from optimized window
            self.cur_R, self.cur_t = self.poses[-1]
            
        self.trajectory.append((self.cur_t[0][0], self.cur_t[2][0]))

        return self.cur_R, self.cur_t, R, t

    def _local_bundle_adjustment(self):
        """
        A lightweight windowed adjustment to smooth translation drift over the window.
        Full nonlinear BA over 3D points and SE(3) poses is too slow for Python real-time loop,
        so we apply a spline/moving average optimization to the translation window.
        """
        # Extract translations
        translations = np.array([pose[1].flatten() for pose in self.poses])
        
        def error_func(t_flat, original_t):
            t_reshaped = t_flat.reshape(-1, 3)
            # Smoothness penalty (second derivative approximation)
            smoothness = np.sum(np.diff(np.diff(t_reshaped, axis=0), axis=0)**2)
            # Data term (stay close to original)
            data_term = np.sum((t_reshaped - original_t)**2)
            return data_term + 5.0 * smoothness
            
        res = least_squares(error_func, translations.flatten(), args=(translations,))
        optimized_t = res.x.reshape(-1, 3)
        
        # Update poses with optimized translations
        for i in range(len(self.poses)):
            self.poses[i] = (self.poses[i][0], optimized_t[i].reshape(3, 1))
