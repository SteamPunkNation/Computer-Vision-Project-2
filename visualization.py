import cv2
import numpy as np

class Visualizer:
    def __init__(self, traj_size=800, traj_scale=100):
        self.traj_size = traj_size
        self.traj_scale = traj_scale # Scaling factor for the trajectory map
        self.traj_img = np.zeros((traj_size, traj_size, 3), dtype=np.uint8)
        self.center_x = traj_size // 2
        self.center_y = traj_size // 2

    def draw_features(self, frame, good_new, good_old):
        """Draws tracked features and optical flow tracks on the frame."""
        vis_frame = frame.copy()
        
        if good_new is not None and good_old is not None:
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                
                a, b, c, d = int(a), int(b), int(c), int(d)
                
                # Draw line between old and new position
                vis_frame = cv2.line(vis_frame, (a, b), (c, d), (0, 255, 0), 2)
                # Draw point at new position
                vis_frame = cv2.circle(vis_frame, (a, b), 3, (0, 0, 255), -1)
                
        return vis_frame

    def update_trajectory(self, cur_t):
        """Updates and returns the trajectory image."""
        if cur_t is not None:
            x, y, z = cur_t[0][0], cur_t[1][0], cur_t[2][0]
            
            # Map physical coordinates to image coordinates
            # Assuming motion is mostly in x and z plane (forward/backward, left/right)
            draw_x = int(x * self.traj_scale) + self.center_x
            draw_y = int(z * self.traj_scale) + self.center_y
            
            # Draw a circle at the current position
            cv2.circle(self.traj_img, (draw_x, draw_y), 2, (0, 255, 0), 1)
            
            # Draw camera view indicator
            cv2.rectangle(self.traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
            text = f"Coordinates: x={x:02f}m y={y:02f}m z={z:02f}m"
            cv2.putText(self.traj_img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
            
        return self.traj_img
        
    def add_mask_overlay(self, frame, mask):
        """Overlays the static mask on the frame for visualization (red tint on dynamic objects)."""
        colored_mask = np.zeros_like(frame)
        colored_mask[mask == 0] = [0, 0, 255] # Red for dynamic objects
        
        # Blend the mask with the frame
        return cv2.addWeighted(frame, 1.0, colored_mask, 0.4, 0)
