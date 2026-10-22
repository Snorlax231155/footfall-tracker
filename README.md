# 🚶 CCTV Footfall Tracker & Density Analyzer

> **Team Vigilantes** | Real-time CCTV stream analysis for footfall counting, zone tracking, and spatial density heatmaps.

---

## 📌 About The Project

In retail stores, malls, and public facilities, understanding customer movement and foot traffic patterns is crucial for layout optimization, staffing, and crowd safety.

This project provides an end-to-end computer vision solution and REST API that parses CCTV camera streams (or pre-recorded video / webcam feeds) to:
- **Track Footfall**: Accurately count total visitors entering and moving through the space.
- **Zone-Level Analytics**: Monitor footfall and active counts inside user-defined or automatic store zones.
- **Crowd Density Detection**: Identify high-density congestion periods and flag peak activity times.
- **Dynamic Heatmaps**: Generate Gaussian spatial heatmaps visually depicting traffic concentration.
- **Cyberpunk Web Dashboard**: An interactive, real-time web interface for starting/stopping analysis, monitoring metrics, and visualizing heatmaps.

---

## 🛠️ What We Used

- **Deep Learning / Detection**: [YOLOv8](https://github.com/ultralytics/ultralytics) (`yolov8n.pt`) for real-time person detection.
- **Computer Vision**: [OpenCV](https://opencv.org/) for video frame decoding, spatial transformation, bounding box annotations, and heatmap rendering.
- **Backend API**: [Python 3](https://www.python.org/) & [Flask](https://flask.palletsprojects.com/) with CORS support for RESTful endpoints.
- **Tracking Algorithm**: Centroid-based multi-object tracking with track confirmation and disappearance handling.
- **Data & Visualizations**: [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/), and [Seaborn](https://seaborn.pydata.org/) for Gaussian density distribution matrices and heatmap colormaps.
- **Frontend Dashboard**: HTML5, modern CSS3 (matrix/cyberpunk styling), and JavaScript Fetch API with real-time status polling.

---

## ⚙️ How It Works

```
  [ CCTV Stream / Video / Webcam ]
                 │
                 ▼
     [ YOLOv8 Person Detection ]
                 │
                 ▼
  [ Centroid Tracking & Zone Mapping ]
        │                   │
        ▼                   ▼
 [ Footfall Counts ]   [ Density Matrix ]
        │                   │
        ▼                   ▼
 [ JSON Summary API ]  [ Gaussian Heatmaps ]
                 │
                 ▼
    [ Interactive Web Dashboard ]
```

1. **Video Ingestion**: The system connects to a CCTV video stream URL, local video file (e.g. `sample_people.mp4`), or live webcam (`0`).
2. **Object Detection**: YOLOv8 analyzes video frames with configurable frame skipping to detect people (`class 0`) with high accuracy and speed.
3. **Centroid Tracking**: Detection coordinates are tracked across frames. IDs that persist across multiple consecutive frames are confirmed as unique visitors to eliminate false positives.
4. **Zone Mapping**: The frame is divided into monitored zones (custom bounding coordinates or default left/right halves). The system records momentary and cumulative traffic per zone.
5. **Density Analysis & Heatmaps**: A 2D spatial Gaussian distribution is applied over all detected coordinates to generate smooth heatmaps (saved in `static/heatmaps/`). High-density intervals are recorded when zone counts exceed thresholds.
6. **API & Dashboard Output**: A Flask REST API serves metrics (`/status`), video controls (`/analyze`, `/stop`), and the interactive dashboard (`/`).

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+ (Python 3.10, 3.11, or 3.12 supported)
- Webcam (optional, for live camera testing)

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
# Clone the repository
git clone https://github.com/<your-username>/footfall-tracker.git
cd footfall-tracker

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Setup (Smoke Test)

Run the included verification script to confirm YOLO model inference, tracking, and heatmap generation:

```bash
python test.py
```

### 4. Run the Application

Start the Flask server:

```bash
python run.py
```

The server will start at `http://localhost:5000` (or `http://0.0.0.0:5000`).

---

## 🖥️ Usage

### Web Dashboard
1. Open `http://localhost:5000` in your web browser.
2. Enter your video source:
   - Type `sample_people.mp4` to run the included sample video.
   - Type `webcam` to use your computer's webcam.
   - Or paste any RTSP / HTTP video stream URL.
3. Click **Start Processing**.
4. Watch live footfall statistics, zone counts, and generated heatmaps update in real time!
5. Click **Stop Processing** to conclude the session and view the final summary.

---

## 📡 API Reference

### 1. Start Analysis
- **URL**: `/analyze`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "video_stream_url": "sample_people.mp4",
    "zones": [
      {
        "zone_id": "zone_1",
        "coordinates": {
          "x_min": 0,
          "y_min": 0,
          "x_max": 320,
          "y_max": 480
        }
      },
      {
        "zone_id": "zone_2",
        "coordinates": {
          "x_min": 320,
          "y_min": 0,
          "x_max": 640,
          "y_max": 480
        }
      }
    ]
  }
  ```
- **Response**:
  ```json
  {
    "status": "processing_started",
    "footfall_summary": {
      "total_footfall": 0,
      "zone_footfall": {},
      "high_density_times": []
    },
    "heatmap_urls": []
  }
  ```

### 2. Get Live Status
- **URL**: `/status`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "footfall_summary": {
      "total_footfall": 8,
      "zone_footfall": {
        "zone_1": 4,
        "zone_2": 4
      },
      "high_density_times": [
        {
          "start_time": "02:10",
          "end_time": "02:15",
          "zone_id": "zone_1"
        }
      ]
    },
    "heatmap_urls": [
      "/static/heatmaps/heatmap_zone_zone_1_20261023_021008.png"
    ],
    "processing_complete": false
  }
  ```

### 3. Stop Processing
- **URL**: `/stop`
- **Method**: `POST`
- **Response**: Returns final `footfall_summary`, `heatmap_urls`, and `processing_complete: true`.

---

## 📁 Project Structure

```
├── app/
│   ├── __init__.py           # Package initialization
│   ├── main.py               # Flask app & API endpoints
│   ├── video_processor.py    # YOLO detection & centroid tracking
│   ├── density_analyzer.py   # High/low density period detection
│   ├── heatmap_generator.py  # Gaussian spatial heatmap rendering
│   └── sort.py               # Kalman-based tracking utility
├── static/
│   └── heatmaps/             # Output folder for generated heatmaps
├── models/
│   └── mars-small128.pb      # Feature extraction weights
├── index.html                # Web dashboard UI
├── requirements.txt          # Python dependencies
├── run.py                    # Server entry point
├── test.py                   # Automated test script
├── sample_people.mp4         # Sample CCTV test video
├── yolov8n.pt                # YOLOv8 pre-trained weights
├── Dockerfile                # Containerization setup
└── README.md                 # Project documentation
```

---

## 👥 Authors
- **Team Vigilantes**
