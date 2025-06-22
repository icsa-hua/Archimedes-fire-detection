# 🔥 Fire Detection Video Processing Script

This Python script processes a video file to detect and highlight potential fire regions using computer vision techniques (OpenCV + NumPy). It applies multiple filters to ensure detections match fire-like behavior and displays results in real time.

---

## 📌 Features

- **Bright Region Detection:** Identifies very bright areas in video frames that could indicate fire.
- **Area Filtering:** Ignores small detections that are too small to be fire.
- **Irregularity Filtering:** Detects regions with irregular (non-solid) shapes, since fire tends to flicker and change shape.
- **Shape Unstability Filtering:** Checks how much a region’s shape changes between frames — fire shapes are unstable.
- **Visual Feedback:** Displays both original and annotated frames side-by-side with labeled regions:
  - 🔴 **Certain Fire** (in red)
  - 🟡 **Possible Fire-Hot Object** (in yellow)

---

## ⚙️ How It Works

1. Each frame is converted to grayscale and thresholded to isolate bright regions (brightness > 245).
2. Morphological dilation is applied to fill gaps in the bright regions.
3. Contours of bright regions are detected.
4. For each detected contour:
   - The **area** is checked against a minimum size threshold.
   - The **solidity** (compactness) is measured to check for irregular shapes.
   - The **shape change** is compared to previous frames to identify unstable regions.
5. Objects passing all filters are labeled as `Fire`. Stable bright objects are labeled as `Stable Hot Object`.
6. Frames are displayed side-by-side: original + output with overlaid labels and colors.

---