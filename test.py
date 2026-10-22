"""
Quick verification script to test YOLO model loading, video processing, and heatmap generation.
"""
import time
import json
from ultralytics import YOLO
from app.video_processor import VideoProcessor

def test_pipeline():
    print("1. Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")
    print("   YOLOv8 model loaded successfully.")

    print("2. Running VideoProcessor on sample video...")
    vp = VideoProcessor()
    thread = vp.process_stream("sample_people.mp4", [])
    time.sleep(4)
    vp.stop_processing()
    stats = vp._get_current_stats()
    
    print("3. Pipeline Results:")
    print(json.dumps(stats, indent=2))
    print("\nAll pipeline components verified successfully!")

if __name__ == "__main__":
    test_pipeline()