import cv2
import numpy as np
from ultralytics import YOLO
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class SemanticSegmenter:
    def __init__(self, model_name='generated/models/yolov8n-seg.pt'):
        # Automatically use MPS on Mac, CUDA on Windows if available, else CPU
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            
        print(f"Loading YOLOv8 segmentation model on {self.device}...")
        self.model = YOLO(model_name)
        
        # Initialize MediaPipe Hands for explicit hand masking
        print("Loading MediaPipe Hands model...")
        base_options = python.BaseOptions(model_asset_path='generated/models/hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                               num_hands=4,
                                               min_hand_detection_confidence=0.4,
                                               min_tracking_confidence=0.4)
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Classes we consider "dynamic" and want to mask out
        # COCO classes: 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
        self.dynamic_classes = [0, 1, 2, 3, 5, 7]

    def get_static_mask(self, frame):
        """
        Returns a binary mask of the frame where:
        255 = Static background (safe for feature tracking)
        0 = Dynamic foreground (e.g., people, vehicles, hands) to be ignored
        """
        height, width = frame.shape[:2]
        # Default mask is all 255 (everything is static)
        static_mask = np.ones((height, width), dtype=np.uint8) * 255
        
        # 1. Run YOLOv8 Segmentation
        results = self.model.predict(source=frame, device=self.device, classes=self.dynamic_classes, verbose=False, conf=0.3)
        
        if results and len(results) > 0:
            result = results[0]
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                combined_mask = np.zeros((masks.shape[1], masks.shape[2]), dtype=bool)
                for mask in masks:
                    combined_mask = np.logical_or(combined_mask, mask > 0.5)
                
                if combined_mask.shape[0] != height or combined_mask.shape[1] != width:
                    combined_mask = cv2.resize(combined_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
                
                # Set YOLO dynamic areas to 0
                static_mask[combined_mask] = 0
                
        # 2. Run MediaPipe Hands
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)
        
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Get the points of the hand
                points = []
                for lm in hand_landmarks:
                    x = int(lm.x * width)
                    y = int(lm.y * height)
                    points.append((x, y))
                    
                # Create a convex hull (polygon) around the hand
                if len(points) > 0:
                    hull = cv2.convexHull(np.array(points))
                    # Draw a filled polygon on the mask (0 = dynamic)
                    cv2.fillConvexPoly(static_mask, hull, 0)
                    
                    # Also draw large circles around each joint to create a generous safety buffer for the hand
                    for p in points:
                        cv2.circle(static_mask, p, radius=30, color=0, thickness=-1)
                
        return static_mask
