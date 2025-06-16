import os
import sys
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

sys.path.append('./')

class VideoDataset(Dataset):
    def __init__(self, video_paths, frames_per_sample=30, target_size=None, transform=None):
        """
        Args:
            video_paths (list): List of video file paths.
            frames_per_sample (int): Number of frames per sample.
            target_size (tuple or None): (width, height) to resize frames, or None for original.
            transform (callable or None): Optional transform on each frame (applies to NumPy arrays).
        """
        self.video_paths = video_paths
        self.frames_per_sample = frames_per_sample
        self.target_size = target_size
        self.transform = transform
        self.samples = []

        for vid_idx, video_path in enumerate(video_paths):
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            for start in range(0, total_frames, frames_per_sample):
                self.samples.append((vid_idx, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_idx, start_frame = self.samples[idx]
        video_path = self.video_paths[vid_idx]

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for _ in range(self.frames_per_sample):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.target_size:
                frame = cv2.resize(frame, self.target_size)

            if self.transform:
                frame = self.transform(frame)

            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames could be read starting from frame {start_frame} in {video_path}")

        frames = np.stack(frames, axis=0)
        return frames

    
if __name__ == "__main__":

    video_file = "./data/video.mp4"

    dataset = VideoDataset(video_paths=[video_file], frames_per_sample=11,  target_size=(512, 512))

    for i, frames in enumerate(dataset):
        print(f"Sample {i} shape: {frames.shape}")  
        break