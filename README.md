# ⚽ Player Re-Identification and Tracking using YOLO and DeepSORT

This project implements an **enhanced, real-time player re-identification system** that ensures consistent player IDs even when players leave and re-enter the video frame. The system uses a fine-tuned YOLO model for player detection and DeepSORT with appearance-based re-identification for robust tracking.

---

## 🚀 Project Overview

### Objective:
- Detect and consistently track players across a 15-second input video.
- Assign stable IDs that persist even after occlusion or players leaving/re-entering the frame.
- Provide enhanced visual overlays, player path tracing, and processing statistics.

### Key Features:
- **Robust ID Assignment:** Maintains ID consistency using both tracker output and spatial history.
- **Re-Identification:** Recovers lost players by comparing their re-entry positions to historical data.
- **Modular Class Design:** Clean, extensible Python class (`PlayerReidentifier`) structure.
- **Real-Time Visualization:** Visual overlays with player IDs, player paths, and live tracking statistics.
- **Optional Video Output:** Supports saving the output video.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/player-reidentification.git
cd player-reidentification
```
### 2. Create and activate Virtual Environment
## For Windows
```bash
python -m venv myenv
myenv\Scripts\activate
```
## For Linux/macOS
```bash
python3 -m venv myenv
source myenv/bin/activate
```
Also download the model from the following link and add it to the project folder
```bash
https://huggingface.co/Hari-1-0/player-re-identification
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
## Required files
- best.pt --> Fine-tuned YOLO model for player and ball detection
- 15sec_input_720p.mp4 --> Input video file for processing.
# How to Run
```bash
python main.py
```
# Optional:
To save the output video, you can pass the output_path parameter in process_video():
```bash
reid_system.process_video(
    output_path='output_video.mp4',
    show_display=True
)
```
