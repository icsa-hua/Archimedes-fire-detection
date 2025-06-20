import numpy as np
from alibi_detect.cd import MMDDrift
import cv2

# --- Utility: Load frames from video ---
def load_frames(video_path, n_frames=100, resize=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while count < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if resize:
            frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        count += 1
    cap.release()
    return np.array(frames)

# --- Example distortion: use from distortions.py ---
from distortions import add_motion_blur

# --- Load reference (original) and test (distorted) frames ---
ref_frames = load_frames("./data/1.MP4", n_frames=100, resize=(128, 128))
distorted_frames = np.array([add_motion_blur(f, degree=15, angle=45) for f in ref_frames])

# --- Flatten frames for drift detector (MMDDrift expects 2D samples) ---
ref_flat = ref_frames.reshape((ref_frames.shape[0], -1))
distorted_flat = distorted_frames.reshape((distorted_frames.shape[0], -1))

# --- Fit drift detector on reference frames ---
cd = MMDDrift(ref_flat, p_val=.05)

# --- Test for drift on distorted frames ---
preds = cd.predict(distorted_flat)
print("Drift detected?" , preds['data']['is_drift'])
print("p-value:", preds['data']['p_val'])

