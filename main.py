import cv2
import time
import argparse
import numpy as np

from camera import Camera
from segmentation import SemanticSegmenter
from feature_tracker import FeatureTracker
from visual_odometry import VisualOdometry
from visualization import Visualizer

import csv
import datetime

def main(source=0, use_masking=True, camera_height=1.2, window_size=5):
    # Initialize components
    cam = Camera(source=source)
    
    # Initialize logger
    log_filename = f"metrics_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_file = open(log_filename, 'w', newline='')
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(['Timestamp', 'FPS', 'Delay_ms', 'TrackedFeatures', 'Pose_X', 'Pose_Z', 'Motion', 'Rotation'])
    
    # Optional semantic masking
    segmenter = SemanticSegmenter() if use_masking else None
    
    tracker = FeatureTracker()
    vo = VisualOdometry(camera_height=camera_height, window_size=window_size)
    vis = Visualizer()
    
    print("Starting Semantic-Aware Visual Odometry Pipeline...")
    print("Press 'q' to quit.")
    
    # Variables for FPS calculation
    prev_time = time.time()
    fps = 0
    
    # Variables for direction averaging
    last_update_time = time.time()
    accum_t = np.zeros(3)
    accum_r = np.zeros(3)
    t_count = 0
    r_count = 0
    display_direction_str = "Stationary"
    display_rotation_str = "Stable"
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame or video ended.")
            break
            
        start_time = time.time()
        
        # 1. Semantic Segmentation (Foreground Masking)
        if use_masking:
            mask = segmenter.get_static_mask(frame)
        else:
            mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
            
        # 2. Feature Tracking
        good_new, good_old = tracker.track(frame, mask)
        
        # 3. Visual Odometry (Pose Estimation)
        cur_R, cur_t, R, t = vo.update(good_old, good_new)
        
        if t is not None and np.linalg.norm(t) > 0.1:
            tx, ty, tz = t.flatten()
            accum_t += np.array([tx, ty, tz])
            t_count += 1
                
        if R is not None:
            rvec, _ = cv2.Rodrigues(R)
            rx, ry, rz = rvec.flatten()
            rx, ry, rz = np.degrees([rx, ry, rz])
            accum_r += np.array([rx, ry, rz])
            r_count += 1
            
        # Calculate performance metrics
        curr_time = time.time()
        process_delay = (curr_time - start_time) * 1000 # ms
        fps = 1.0 / (curr_time - prev_time) if curr_time - prev_time > 0 else 0
        prev_time = curr_time
        
        # Average the direction and rotation every 1.0 seconds to prevent flickering
        if curr_time - last_update_time >= 1.0:
            if t_count > 0:
                avg_tx, avg_ty, avg_tz = accum_t / t_count
                if abs(avg_tx) > max(abs(avg_ty), abs(avg_tz)) and abs(avg_tx) > 0.05:
                    display_direction_str = "Right" if avg_tx > 0 else "Left"
                elif abs(avg_ty) > max(abs(avg_tx), abs(avg_tz)) and abs(avg_ty) > 0.05:
                    display_direction_str = "Down" if avg_ty > 0 else "Up"
                elif abs(avg_tz) > max(abs(avg_tx), abs(avg_ty)) and abs(avg_tz) > 0.05:
                    display_direction_str = "Forward" if avg_tz > 0 else "Backward"
                else:
                    display_direction_str = "Stationary"
            else:
                display_direction_str = "Stationary"
                
            if r_count > 0:
                avg_rx, avg_ry, avg_rz = accum_r / r_count
                if abs(avg_rx) > max(abs(avg_ry), abs(avg_rz)) and abs(avg_rx) > 0.5:
                    display_rotation_str = "Tilt Down" if avg_rx > 0 else "Tilt Up"
                elif abs(avg_ry) > max(abs(avg_rx), abs(avg_rz)) and abs(avg_ry) > 0.5:
                    display_rotation_str = "Pan Left" if avg_ry > 0 else "Pan Right"
                elif abs(avg_rz) > max(abs(avg_rx), abs(avg_ry)) and abs(avg_rz) > 0.5:
                    display_rotation_str = "Clockwise" if avg_rz > 0 else "Counter-Clockwise"
                else:
                    display_rotation_str = "Stable"
            else:
                display_rotation_str = "Stable"
                
            # Reset accumulators
            accum_t = np.zeros(3)
            accum_r = np.zeros(3)
            t_count = 0
            r_count = 0
            last_update_time = curr_time
        
        # 4. Visualization
        # Overlay mask on frame
        if use_masking:
            vis_frame = vis.add_mask_overlay(frame, mask)
        else:
            vis_frame = frame.copy()
            
        # Draw features
        vis_frame = vis.draw_features(vis_frame, good_new, good_old)
        
        # Update trajectory image
        traj_img = vis.update_trajectory(cur_t)
        
        tracked_count = len(good_new) if good_new is not None else 0
        
        # Display metrics on frame
        cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Delay: {process_delay:.1f} ms", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Features: {tracked_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Motion: {display_direction_str}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(vis_frame, f"Rot: {display_rotation_str}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Log to file
        x_val = cur_t[0][0] if cur_t is not None else 0.0
        z_val = cur_t[2][0] if cur_t is not None else 0.0
        csv_writer.writerow([datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3], f"{fps:.1f}", f"{process_delay:.1f}", tracked_count, f"{x_val:.4f}", f"{z_val:.4f}", display_direction_str, display_rotation_str])
        
        # Show windows
        cv2.imshow('Semantic-Aware Visual Odometry', vis_frame)
        cv2.imshow('Trajectory', traj_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cam.release()
    log_file.close()
    cv2.destroyAllWindows()

def list_available_cameras():
    print("Scanning for available video sources...")
    import platform
    import subprocess
    
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(['system_profiler', 'SPCameraDataType'], capture_output=True, text=True)
            output = result.stdout
            cameras = []
            for line in output.split('\n'):
                line = line.strip()
                if line and not line.startswith('Camera:') and not line.startswith('Model ID:') and not line.startswith('Unique ID:'):
                    if line.endswith(':'):
                        cameras.append(line[:-1])
            if cameras:
                print("\nAvailable cameras:")
                for idx, name in enumerate(cameras):
                    print(f"  [{idx}] {name}")
                print("")
                return list(range(len(cameras)))
        except Exception:
            pass

    # Windows (pygrabber) or generic fallback
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        devices = graph.get_input_devices()
        
        if not devices:
            print("No cameras found by pygrabber.")
            return []
            
        print("\nAvailable cameras:")
        for idx, name in enumerate(devices):
            print(f"  [{idx}] {name}")
        print("")
        return list(range(len(devices)))
    except ImportError:
        print("Install 'pygrabber' (pip install pygrabber) to see actual camera names on Windows.")
        print("Falling back to scanning indices... (this may take a few seconds)")
        
        available_cameras = []
        # Attempt to silence OpenCV warnings during probe
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
        except AttributeError:
            pass
            
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap is not None and cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
                
        if available_cameras:
            print(f"Available camera indices found: {available_cameras}")
        else:
            print("No available cameras found via standard indices.")
        return available_cameras

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Semantic-Aware Monocular Visual Odometry")
    parser.add_argument('--source', type=str, default='0', help='Video source (0 for webcam or path to video file)')
    parser.add_argument('--no-mask', action='store_true', help='Disable semantic masking for comparison')
    parser.add_argument('--camera-height', type=float, default=1.2, help='Assumed camera height in meters for scale estimation')
    parser.add_argument('--window-size', type=int, default=5, help='Window size for local bundle adjustment')
    
    args = parser.parse_args()
    
    # List available cameras before prompting
    list_available_cameras()
    
    # Prompt for video source
    print("Enter a camera index from the list above, or a path to a video file.")
    user_input = input("Enter video source (press Enter for default 0): ").strip()
    
    if not user_input:
        user_input = args.source
        
    # Parse source as int if it's a digit (webcam index)
    src = int(user_input) if user_input.isdigit() else user_input
    
    main(source=src, use_masking=not args.no_mask, camera_height=args.camera_height, window_size=args.window_size)
