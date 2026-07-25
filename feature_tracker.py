import cv2
import numpy as np

class FeatureTracker:
    def __init__(self, max_corners=1000, quality_level=0.01, min_distance=10):
        # Parameters for Shi-Tomasi corner detection
        self.feature_params = dict(maxCorners=max_corners,
                                   qualityLevel=quality_level,
                                   minDistance=min_distance,
                                   blockSize=3)
        
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
                              
        self.prev_gray = None
        self.p0 = None
        self.track_len = 10
        self.tracks = []
        self.frame_idx = 0

    def track(self, frame, mask):
        """
        Extracts and tracks features using LK optical flow.
        Uses the provided mask to avoid detecting features on dynamic objects.
        Returns:
            good_new (np.array): Tracked points in the current frame.
            good_old (np.array): Corresponding points in the previous frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        good_new, good_old = None, None
        
        if self.prev_gray is None:
            # First frame, initialize features
            self.p0 = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
            self.prev_gray = gray
            return None, None
            
        if self.p0 is not None and len(self.p0) > 0:
            # Calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.p0, None, **self.lk_params)
            
            # Select good points
            if p1 is not None:
                good_new_temp = p1[st == 1]
                good_old_temp = self.p0[st == 1]
                
                # Actively filter out any points that have moved into a masked area (dynamic objects)
                valid_points = []
                for i, (x, y) in enumerate(good_new_temp):
                    if 0 <= int(x) < mask.shape[1] and 0 <= int(y) < mask.shape[0]:
                        if mask[int(y), int(x)] == 255: # 255 means static background
                            valid_points.append(i)
                            
                good_new = good_new_temp[valid_points]
                good_old = good_old_temp[valid_points]
        
        # Determine if we need to detect more features (if we lost too many)
        if self.p0 is None or len(good_new) < 500:
            # Extract new features in areas where we don't already have them
            # Create a mask that excludes areas around existing points and dynamic objects
            combined_mask = mask.copy()
            if good_new is not None and len(good_new) > 0:
                for x, y in np.int32(good_new):
                    cv2.circle(combined_mask, (x, y), self.feature_params['minDistance'], 0, -1)
            
            new_features = cv2.goodFeaturesToTrack(gray, mask=combined_mask, **self.feature_params)
            
            if new_features is not None:
                if good_new is not None:
                    good_new = np.vstack((good_new, new_features[:, 0, :]))
                    good_old = np.vstack((good_old, new_features[:, 0, :])) # Approximate for new points
                else:
                    good_new = new_features[:, 0, :]
                    good_old = good_new.copy()

        # Update previous frame and points
        self.prev_gray = gray
        if good_new is not None:
            self.p0 = good_new.reshape(-1, 1, 2)
        else:
            self.p0 = None
            
        return good_new, good_old
