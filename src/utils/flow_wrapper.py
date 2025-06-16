import cv2
import sys
import numpy as np

sys.path.append('./')

from src.utils.itterator import VideoDataset

class OpticalFlowFireDetector:
    def __init__(self, base_dataset, grid_step=16, mag_var_thresh=2.0, ang_var_thresh=1.0, temp_thresh=200):
        """
        Fire detector using optical flow turbulence and thermal mask.

        Args:
            base_dataset: VideoDataset instance
            grid_step (int): size of grid cell
            mag_var_thresh (float): min variance of magnitude to consider fire
            ang_var_thresh (float): min variance of angle to consider fire
            temp_thresh (int): threshold for thermal intensity (0-255)
        """
        self.base_dataset = base_dataset
        self.grid_step = grid_step
        self.mag_var_thresh = mag_var_thresh
        self.ang_var_thresh = ang_var_thresh
        self.temp_thresh = temp_thresh

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        frames = self.base_dataset[idx]
        if frames.ndim == 4 and frames.shape[-1] == 3:
            frames_gray = np.array([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames])
        else:
            frames_gray = frames[..., 0] if frames.ndim == 4 else frames

        prev_gray = frames_gray[-2]
        next_gray = frames_gray[-1]

        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        hot_mask = next_gray > self.temp_thresh

        h, w = flow.shape[:2]
        step = self.grid_step
        y_idx, x_idx = np.mgrid[step//2:h:step, step//2:w:step]
        grid_points = np.stack((x_idx, y_idx), axis=-1)
        grid_flows = flow[y_idx, x_idx]

        fire_cells = []

        for i in range(y_idx.shape[0]):
            for j in range(y_idx.shape[1]):
                cy, cx = y_idx[i, j], x_idx[i, j]
                y0 = max(cy - step//2, 0)
                y1 = min(cy + step//2, h)
                x0 = max(cx - step//2, 0)
                x1 = min(cx + step//2, w)

                local_mag = mag[y0:y1, x0:x1]
                local_ang = ang[y0:y1, x0:x1]
                local_hot = hot_mask[y0:y1, x0:x1]

                if np.count_nonzero(local_hot) < 0.3 * local_hot.size:
                    continue

                var_mag = np.var(local_mag)
                var_ang = np.var(local_ang)

                if var_mag > self.mag_var_thresh and var_ang > self.ang_var_thresh:
                    fire_cells.append((cx, cy))

        return {
            "frames": frames,
            "flow": flow,
            "grid_points": grid_points,
            "grid_flows": grid_flows,
            "fire_cells": fire_cells
        }

if __name__ == "__main__":
    
    dataset = VideoDataset("./data/video2.mp4", frames_per_sample=2, target_size=(512, 512))
    fire_detector = OpticalFlowFireDetector(dataset, grid_step=16, mag_var_thresh=2.0, ang_var_thresh=1.0, temp_thresh=200)

    for i in range(len(fire_detector)):
        sample = fire_detector[i]
        frame_vis = sample["frames"][-1].copy()

        if frame_vis.ndim == 2 or frame_vis.shape[-1] == 1:
            frame_vis = cv2.cvtColor(frame_vis, cv2.COLOR_GRAY2BGR)

        for pt, flow_vec in zip(sample["grid_points"].reshape(-1, 2),
                                sample["grid_flows"].reshape(-1, 2)):
            x, y = int(pt[0]), int(pt[1])
            fx, fy = flow_vec
            cv2.arrowedLine(frame_vis, (x, y), (int(x + fx), int(y + fy)),
                            (0, 255, 0), 1, tipLength=0.3)

        for cx, cy in sample["fire_cells"]:
            cv2.rectangle(frame_vis, (cx - 8, cy - 8), (cx + 8, cy + 8), (0, 0, 255), 2)

        cv2.imshow("Fire Detection Grid", frame_vis)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
