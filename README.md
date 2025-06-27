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
