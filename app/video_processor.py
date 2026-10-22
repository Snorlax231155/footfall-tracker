import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import time
import threading
import os
from .density_analyzer import DensityAnalyzer
from .heatmap_generator import HeatmapGenerator
from pathlib import Path

class VideoProcessor:
    def __init__(self):
        # Use the smallest YOLO model for better performance
        self.model = YOLO('yolov8n.pt')
        self.storage_path = "static/heatmaps/"
        self.trackers = []
        self.track_ids = []
        
        # Performance optimization parameters
        self.frame_skip = 2  # Process every nth frame
        self.target_width = 640  # Reduced resolution width
        self.target_height = 480  # Reduced resolution height
        self.heatmap_update_interval = 2.0  # Update heatmap every n seconds
        self.last_heatmap_update = time.time()
        self.show_preview = os.environ.get('SHOW_PREVIEW', '0') == '1'

        # Additional tracking parameters
        self.min_detection_confidence = 0.5
        self.min_tracking_confidence = 0.3
        self.max_disappeared = 30  # Frames before removing track
        self.min_distance = 50     # Minimum pixels between centroids
        self.min_frames = 3        # Minimum frames to confirm as valid person

        self.should_stop = False
        self.processing_thread = None
        self.current_count = 0
        self.total_cumulative_count = 0
        self.zone_counts = {}
        self.lock = threading.Lock()  # Thread-safe counter updates
        self.processing_complete = False

        self.zones = {}
        self.zone_data = {}  # Store detection data per zone
        self.heatmap_points = {}  # Store points for heatmap per zone
        
        # Ensure static directory exists
        os.makedirs('static/heatmaps', exist_ok=True)

        # Create static directory with proper path
        self.static_dir = Path('static/heatmaps')
        self.static_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize heatmap data
        self.heatmap_data = np.zeros((self.target_height, self.target_width), dtype=np.float32)

    def process_stream(self, video_url, zones):
        # Start processing in a separate thread
        self.processing_thread = threading.Thread(
            target=self._process_stream_internal,
            args=(video_url, zones)
        )
        self.processing_thread.start()
        return self.processing_thread

    def _process_stream_internal(self, video_url, zones):
        if video_url == "0":
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(video_url)

        if not cap.isOpened():
            print(f"Error: Unable to open video stream {video_url}")
            return self._get_current_stats()

        # Initialize counters and data structures
        self.current_count = 0
        self.total_cumulative_count = 0
        self.zone_counts = {zone['zone_id']: 0 for zone in zones}
        heatmap_data = []
        last_time = time.time()
        frame_count = 0
        
        # Tracking state
        tracked_objects = {}       # {id: {'centroid': (x,y), 'frames': n, 'disappeared': n}}
        next_object_id = 1
        confirmed_ids = set()      # Set of IDs that have been confirmed as valid people

        try:
            while cap.isOpened() and not self.should_stop:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % self.frame_skip != 0:
                    continue

                frame = cv2.resize(frame, (self.target_width, self.target_height))

                # Run YOLO detection
                results = self.model(frame, 
                                   classes=[0],
                                   conf=self.min_detection_confidence,
                                   iou=0.45,
                                   max_det=20)

                # Process detections
                current_centroids = []
                current_detections = []
                
                for det in results[0].boxes.data:
                    x1, y1, x2, y2, conf, cls = det
                    if conf < self.min_detection_confidence:
                        continue
                        
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                    current_centroids.append((centroid, (x1, y1, x2, y2)))
                    current_detections.append((x1, y1, x2, y2))

                # Update zone tracking with current detections
                frame = self._process_zones(frame, current_detections, zones)

                # Update tracking with thread safety
                with self.lock:
                    if len(current_centroids) == 0:
                        for obj_id in tracked_objects:
                            tracked_objects[obj_id]['disappeared'] += 1
                    else:
                        # If we have no existing tracks, create new ones
                        if len(tracked_objects) == 0:
                            for centroid, bbox in current_centroids:
                                tracked_objects[next_object_id] = {
                                    'centroid': centroid,
                                    'frames': 1,
                                    'disappeared': 0,
                                    'bbox': bbox
                                }
                                next_object_id += 1
                        else:
                            # Match existing tracks to new detections
                            used_centroids = set()
                            for obj_id in list(tracked_objects.keys()):
                                if tracked_objects[obj_id]['disappeared'] > self.max_disappeared:
                                    del tracked_objects[obj_id]
                                    continue

                                obj_centroid = tracked_objects[obj_id]['centroid']
                                min_dist = float('inf')
                                closest_centroid = None
                                closest_bbox = None

                                for i, (centroid, bbox) in enumerate(current_centroids):
                                    if i in used_centroids:
                                        continue

                                    dist = np.sqrt((obj_centroid[0] - centroid[0])**2 + 
                                                 (obj_centroid[1] - centroid[1])**2)
                                    if dist < min_dist and dist < self.min_distance:
                                        min_dist = dist
                                        closest_centroid = centroid
                                        closest_bbox = bbox

                                if closest_centroid is not None:
                                    tracked_objects[obj_id]['centroid'] = closest_centroid
                                    tracked_objects[obj_id]['bbox'] = closest_bbox
                                    tracked_objects[obj_id]['frames'] += 1
                                    tracked_objects[obj_id]['disappeared'] = 0
                                    used_centroids.add(current_centroids.index((closest_centroid, closest_bbox)))

                                    # Check if this track should be confirmed
                                    if tracked_objects[obj_id]['frames'] >= self.min_frames and obj_id not in confirmed_ids:
                                        confirmed_ids.add(obj_id)
                                        self.total_cumulative_count += 1

                            # Create new tracks for unmatched detections
                            for i, (centroid, bbox) in enumerate(current_centroids):
                                if i not in used_centroids:
                                    tracked_objects[next_object_id] = {
                                        'centroid': centroid,
                                        'frames': 1,
                                        'disappeared': 0,
                                        'bbox': bbox
                                    }
                                    next_object_id += 1

                        # Update counts safely
                        self.current_count = sum(1 for obj_id in tracked_objects 
                                               if obj_id in confirmed_ids and 
                                               tracked_objects[obj_id]['disappeared'] == 0)
                        
                        # Update total count when new IDs are confirmed
                        for obj_id in tracked_objects:
                            if (tracked_objects[obj_id]['frames'] >= self.min_frames and 
                                obj_id not in confirmed_ids):
                                confirmed_ids.add(obj_id)
                                self.total_cumulative_count += 1

                # Visualization
                if not self.should_stop:
                    current_time = time.time()
                    for obj_id, obj_data in tracked_objects.items():
                        if obj_data['disappeared'] > 0:
                            continue
                            
                        x1, y1, x2, y2 = obj_data['bbox']
                        color = (0, 255, 0) if obj_id in confirmed_ids else (0, 165, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f'ID: {obj_id}', (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Display counts
                    if current_time - last_time >= 1:
                        cv2.putText(frame, f'Current Count: {self.current_count}', (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f'Total Count: {self.total_cumulative_count}', (10, 70), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        last_time = current_time

                    if getattr(self, 'show_preview', False):
                        try:
                            cv2.imshow('Footfall Detection', frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        except Exception:
                            pass

        finally:
            cap.release()
            if getattr(self, 'show_preview', False):
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            self.processing_complete = True
            return self._get_current_stats()

    def _get_current_stats(self):
        with self.lock:
            current_time = time.time()
            # Update heatmaps periodically or upon completion to avoid heavy disk operations on every 1s poll
            should_update_heatmaps = (
                self.processing_complete or 
                (current_time - self.last_heatmap_update >= self.heatmap_update_interval) or
                not hasattr(self, 'cached_heatmap_urls')
            )
            
            if should_update_heatmaps:
                heatmap_generator = HeatmapGenerator()
                heatmap_urls = []
                for zone_id, data in self.zone_data.items():
                    if data['points']:
                        heatmap_url = heatmap_generator.generate_for_zone(
                            data['points'],
                            self.target_width,
                            self.target_height,
                            zone_id
                        )
                        heatmap_urls.append(heatmap_url)
                if heatmap_urls or not hasattr(self, 'cached_heatmap_urls'):
                    self.cached_heatmap_urls = heatmap_urls
                self.last_heatmap_update = current_time

            # Create response
            response = {
                'footfall_summary': {
                    'total_footfall': self.total_cumulative_count,
                    'zone_footfall': {
                        zone_id: data['current_count'] 
                        for zone_id, data in self.zone_data.items()
                    },
                    'high_density_times': []
                },
                'heatmap_urls': getattr(self, 'cached_heatmap_urls', []),
                'processing_complete': self.processing_complete
            }

            # Add density analysis
            if self.zone_data:
                density_analyzer = DensityAnalyzer()
                density_data = density_analyzer.analyze({
                    'total_count': self.total_cumulative_count,
                    'zone_counts': {
                        zone_id: len(data['points'])
                        for zone_id, data in self.zone_data.items()
                    }
                })
                response['footfall_summary']['high_density_times'] = density_data['high_density_periods']

            return response

    def generate_heatmap(self, heatmap_data, width, height):
        # Create a blank heatmap
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Populate the heatmap with detection points
        for x, y in heatmap_data:
            heatmap[y, x] += 1  # Increment the heatmap at the detection point
        
        # Normalize the heatmap
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = np.uint8(heatmap)
        
        # Create a color map
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Save the heatmap
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"{self.storage_path}latest_heatmap.png", heatmap_color)

    def stop_processing(self):
        self.should_stop = True
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=3)
        
        self.processing_complete = True
        return self._get_current_stats()
    def _process_zones(self, frame, detections, zones):
        frame_height, frame_width = frame.shape[:2]
        
        # Initialize zones if not already done
        if not self.zones:
            if zones:
                for idx, zone in enumerate(zones):
                    zone_id = str(zone.get('zone_id', f'zone_{idx+1}'))
                    # Support both coordinates: {x_min, y_min, x_max, y_max} and bounds: {x1, y1, x2, y2}
                    coords = zone.get('coordinates', zone.get('bounds', {}))
                    x1_val = coords.get('x_min', coords.get('x1', 0))
                    y1_val = coords.get('y_min', coords.get('y1', 0))
                    x2_val = coords.get('x_max', coords.get('x2', frame_width))
                    y2_val = coords.get('y_max', coords.get('y2', frame_height))
                    
                    x1 = int(x1_val * frame_width) if (isinstance(x1_val, float) and x1_val <= 1.0) else int(x1_val)
                    y1 = int(y1_val * frame_height) if (isinstance(y1_val, float) and y1_val <= 1.0) else int(y1_val)
                    x2 = int(x2_val * frame_width) if (isinstance(x2_val, float) and x2_val <= 1.0) else int(x2_val)
                    y2 = int(y2_val * frame_height) if (isinstance(y2_val, float) and y2_val <= 1.0) else int(y2_val)
                    
                    self.zones[zone_id] = (x1, y1, x2, y2)
                    self.zone_data[zone_id] = {
                        'current_count': 0,
                        'total_count': 0,
                        'points': []
                    }
            else:
                # Default zones (Left Half: zone_1, Right Half: zone_2)
                mid_x = frame_width // 2
                self.zones['zone_1'] = (0, 0, mid_x, frame_height)
                self.zones['zone_2'] = (mid_x, 0, frame_width, frame_height)
                for zid in ['zone_1', 'zone_2']:
                    self.zone_data[zid] = {
                        'current_count': 0,
                        'total_count': 0,
                        'points': []
                    }

        # Reset current counts
        for zone_id in self.zones:
            self.zone_data[zone_id]['current_count'] = 0

        # Process each detection
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Check each zone
            for zone_id, (zx1, zy1, zx2, zy2) in self.zones.items():
                if (zx1 <= center_x <= zx2 and zy1 <= center_y <= zy2):
                    self.zone_data[zone_id]['current_count'] += 1
                    self.zone_data[zone_id]['points'].append((center_x, center_y))

        # Draw zone rectangles and current count labels
        for zone_id, (zx1, zy1, zx2, zy2) in self.zones.items():
            count = self.zone_data[zone_id]['current_count']
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 200, 0), 2)
            cv2.putText(frame, f'{zone_id}: {count}', (zx1 + 10, min(zy1 + 25, frame_height - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        return frame

    def save_heatmap(self, zone_id, points):
        """Save heatmap for a specific zone"""
        try:
            if not points:
                return None
                
            # Create heatmap data
            heatmap = np.zeros((self.target_height, self.target_width), dtype=np.float32)
            
            # Add Gaussian for each point
            for x, y in points:
                if 0 <= x < self.target_width and 0 <= y < self.target_height:
                    y_grid, x_grid = np.ogrid[max(0, y-30):min(self.target_height, y+30),
                                            max(0, x-30):min(self.target_width, x+30)]
                    gaussian = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * 15**2))
                    heatmap[max(0, y-30):min(self.target_height, y+30),
                           max(0, x-30):min(self.target_width, x+30)] += gaussian
            
            # Normalize and convert to uint8
            heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap = heatmap.astype(np.uint8)
            
            # Apply color map
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"heatmap_{zone_id}_{timestamp}.png"
            filepath = self.static_dir / filename
            
            # Save the file
            cv2.imwrite(str(filepath), heatmap_colored)
            
            # Return the URL path
            return f"/static/heatmaps/{filename}"
            
        except Exception as e:
            print(f"Error saving heatmap: {e}")
            return None

    def process_frame(self, frame):
        """Process frame with enhanced heatmap visualization"""
        height, width = frame.shape[:2]
        mid_x = width // 2
        
        # Get detections
        results = self.model(frame, classes=[0], conf=0.3)
        detections = results[0].boxes.data
        
        # Reset current counts
        for zone in self.zones.values():
            zone["current_count"] = 0
        
        # Create temporary heatmap for current frame
        current_heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Process each detection
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Update zone counts
            zone_id = "zone_id_1" if center_x < mid_x else "zone_id_2"
            self.zones[zone_id]["current_count"] += 1
            self.zones[zone_id]["count"] += 1
            self.zones[zone_id]["points"].append((center_x, center_y))
            
            # Add heat point with Gaussian distribution
            y_grid, x_grid = np.ogrid[max(0, center_y-50):min(height, center_y+50),
                                    max(0, center_x-50):min(width, center_x+50)]
            dist_from_center = ((x_grid - center_x)**2 + (y_grid - center_y)**2)
            gaussian = 2.0 * np.exp(-dist_from_center / (2 * (20**2)))
            
            current_heatmap[max(0, center_y-50):min(height, center_y+50),
                           max(0, center_x-50):min(width, center_x+50)] += gaussian
            
            # Draw detection box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Update cumulative heatmap
        self.heatmap_data = cv2.addWeighted(self.heatmap_data, 0.8, current_heatmap, 1.0, 0)
        
        # Apply decay to cumulative heatmap
        self.heatmap_data *= 0.95
        
        # Normalize and colorize heatmap
        heatmap_display = cv2.normalize(self.heatmap_data, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_display = heatmap_display.astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_display, cv2.COLORMAP_JET)
        
        # Create alpha mask for smoother blending
        alpha = heatmap_display.astype(float) / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        alpha = np.repeat(alpha, 3, axis=2)
        
        # Blend heatmap with original frame using alpha mask
        blended = frame.copy()
        mask = alpha > 0.1  # Only blend where heatmap is significant
        blended[mask] = cv2.addWeighted(frame, 0.4, heatmap_colored, 0.6, 0)[mask]
        
        # Draw zone divider
        cv2.line(blended, (mid_x, 0), (mid_x, height), (255, 255, 255), 2)
        
        # Draw counts with better visibility
        for i, (zone_id, data) in enumerate(self.zones.items()):
            x = 10 if i == 0 else mid_x + 10
            text = f"{zone_id}: {data['current_count']} (Total: {data['count']})"
            
            # White outline for better visibility
            cv2.putText(blended, text, (x, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
            # Green text
            cv2.putText(blended, text, (x, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        
        # Show the result
        cv2.imshow('Footfall Detection with Heatmap', blended)
        
        return blended
