# Real-Time Video Fire Detection with LLM Reporting

This project provides a real-time fire detection system that analyzes video streams to identify potential fire outbreaks. It uses a multi-stage computer vision pipeline to ensure accurate detection and can be integrated with a Large Language Model (LLM) for automated incident reporting.

---

## 📌 Overview

The system processes a video file frame-by-frame, applying a series of sophisticated filters to distinguish between genuine fires and other visual noise. When a fire is detected, it is highlighted with a bounding box. The system can then pass an annotated frame to a language model to generate a descriptive report, which could be saved as a PDF or used for other alert mechanisms.

---

## 🔑 Key Features

- **Real-Time Video Processing**: Analyzes video files to detect fires as they happen.  
- **Advanced Filtering Pipeline**: Utilizes a chain of filters to minimize false positives by checking for key fire characteristics.  
- **Spatio-Temporal Analysis**: Detects fires based on:  
  - *Minimum Area*: Ignores tiny, irrelevant hotspots.  
  - *Shape Irregularity*: Identifies the chaotic and non-uniform shape of flames.  
  - *Temporal Unstability*: Tracks the flickering and rapid shape-changing motion characteristic of fire.  
- **Clear Visualization**: Displays the original video alongside an annotated version showing detected fires in real-time.  
- **LLM Integration**: Automatically sends fire event data to a Large Language Model (e.g., for generating incident reports).  
- **Modular & Configurable**: The detection parameters are easily configurable to adapt to different environments and video conditions.

---

## ⚙️ How It Works

The detection logic is built on a sequence of checks. For a region in the video to be classified as a fire, it must pass through the entire filtering pipeline:

1. **Candidate Detection**: The system first identifies all potential "regions of interest" in a frame.  
2. **Area Filter**: It discards any regions that are too small to be considered a significant threat.  
3. **Irregularity Filter**: It then checks if the shape of the region is irregular. Solid, regular shapes (like a red car) are filtered out, while chaotic shapes (like flames) are kept.  
4. **Shape Unstability Filter**: Analyzes the region across consecutive frames. Static red objects fail this test, whereas a real fire passes.  
5. **Confirmation & Reporting**: If a region passes all three filters, it is confirmed as a fire. The system annotates the frame and, if enabled, triggers the `LLMController` to process the event.

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
python src/main.py

