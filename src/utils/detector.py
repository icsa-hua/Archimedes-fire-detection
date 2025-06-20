import cv2
import sys
import numpy as np
from collections import deque
from scipy.spatial.distance import cdist

sys.path.append('./')

from src.utils.utils import *

#TODO: refactor, not in final stage
#TODO: connect with image itterator for efficiency
#TODO: grid-search params
#TODO: multi-processing on image ops
#TODO: multi-threading into loading, writting, showing op 

'''
Ranking System (BFSD):
±±±±±±±±±±±±±±±±±±±±±±
F - how much motion there is (from optical flow),
B - how many blobs are seen,
S - how much the blob shapes change over time (fires flicker and change shape),
D - how spread out the blobs are (fire usually stays together).
'''


class OpticalFlowCalculator:
    def compute(self, prev_gray, gray):
        return cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=2, winsize=9, iterations=1,
            poly_n=5, poly_sigma=1.1, flags=0
        )

class BlobAnalyzer:
    def __init__(self, max_blob_distance=50):
        self.max_blob_distance = max_blob_distance

    def extract_blobs(self, bright_mask):
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        centroids = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
                blobs.append({'centroid': (cx, cy), 'hu': calculate_hu_moments(cnt), 'contour': cnt})
                centroids.append([cx, cy])
        return blobs, np.array(centroids) if centroids else np.array([])

    def match_blobs(self, prev_blobs, curr_centroids):
        if not prev_blobs or not curr_centroids.any():
            return [-1] * len(curr_centroids)
        prev_points = np.array([b['centroid'] for b in prev_blobs])
        dists = cdist(prev_points, curr_centroids)
        matches = [-1] * len(curr_centroids)
        used = set()
        for i in range(len(curr_centroids)):
            sorted_idx = np.argsort(dists[:, i])
            for j in sorted_idx:
                if dists[j, i] < self.max_blob_distance and j not in used:
                    matches[i] = j
                    used.add(j)
                    break
        return matches

class RankingSystem:
    def __init__(self, shape_change_sensitivity=4.0, dispersion_penalty_factor=0.05, fire_detection_threshold=5000):
        self.shape_change_sensitivity = shape_change_sensitivity
        self.dispersion_penalty_factor = dispersion_penalty_factor
        self.fire_detection_threshold = fire_detection_threshold
        self.recent_scores = deque(maxlen=5)

    def compute(self, flow_mag, num_blobs, avg_shape_diff, blob_dispersion):
        base = flow_mag * num_blobs
        shape_bonus = 1.0 + (0.5 * avg_shape_diff * self.shape_change_sensitivity / 8.0)
        disp_pen = 1.0 + self.dispersion_penalty_factor * blob_dispersion
        raw_score = (base * shape_bonus) / disp_pen
        self.recent_scores.append(raw_score)
        mean_recent = np.mean(self.recent_scores)
        temp_mult = 1.0 + (mean_recent / (self.fire_detection_threshold + 1e-5)) * 0.5
        final_score = raw_score * temp_mult
        return final_score, raw_score

class FireDetector:
    def __init__(self):
        self.flow_calc = OpticalFlowCalculator()
        self.blob_analyzer = BlobAnalyzer()
        self.ranker = RankingSystem()
        self.prev_gray = None
        self.prev_blobs = []

    def process(self, frame_rgb):
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        out = frame_rgb.copy()

        if self.prev_gray is None:
            self.prev_gray = gray
            return out, [], False

        flow = self.flow_calc.compute(self.prev_gray, gray)
        _, bright_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

        sum_flow_mag = self._compute_sum_flow_mag(flow, bright_mask)
        curr_blobs, centroids = self.blob_analyzer.extract_blobs(bright_mask)
        matches = self.blob_analyzer.match_blobs(self.prev_blobs, centroids)

        avg_shape_diff = self._compute_avg_shape_diff(curr_blobs, matches)
        blob_disp = self._compute_blob_dispersion(centroids)

        final_score, raw_score = self.ranker.compute(sum_flow_mag, len(curr_blobs), avg_shape_diff, blob_disp)
        is_fire = final_score > self.ranker.fire_detection_threshold

        self._annotate_frame(out, curr_blobs, sum_flow_mag, avg_shape_diff, blob_disp, raw_score, final_score, is_fire)

        bboxes = self._get_fire_bboxes(curr_blobs) if is_fire else []

        self.prev_gray = gray
        self.prev_blobs = curr_blobs
        return out, bboxes, is_fire

    def _compute_sum_flow_mag(self, flow, bright_mask):
        return sum(
            (fx * fx + fy * fy) ** 0.5
            for y in range(0, flow.shape[0], 8)
            for x in range(0, flow.shape[1], 8)
            if bright_mask[y, x] > 0
            for fx, fy in [flow[y, x]]
        )

    def _compute_avg_shape_diff(self, curr_blobs, matches):
        total_shape_diff, matched = 0.0, 0
        for i, blob in enumerate(curr_blobs):
            m = matches[i]
            if m != -1:
                diff = np.linalg.norm(blob['hu'] - self.prev_blobs[m]['hu'])
                total_shape_diff += diff
                matched += 1
        return total_shape_diff / matched if matched > 0 else 0.0

    def _compute_blob_dispersion(self, centroids):
        if len(centroids) > 1:
            return (np.std(centroids[:, 0]) + np.std(centroids[:, 1])) / 2.0
        return 0.0

    def _annotate_frame(self, out, curr_blobs, sum_flow_mag, avg_shape_diff, blob_disp, raw_score, final_score, is_fire):
        cv2.putText(out, f"B:{len(curr_blobs)} F:{sum_flow_mag:.1f} S:{avg_shape_diff:.2f} D:{blob_disp:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(out, f"Raw:{raw_score:.1f} Final:{final_score:.1f}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if is_fire:
            cv2.putText(out, "FIRE DETECTED", (10, 80), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)
            for blob in curr_blobs:
                cv2.drawContours(out, [blob['contour']], -1, (0, 0, 255), 2)

    def _get_fire_bboxes(self, curr_blobs):
        bboxes = []
        for blob in curr_blobs:
            x, y, w, h = cv2.boundingRect(blob['contour'])
            bboxes.append([x, y, x + w, y + h])
        return bboxes



if __name__ == "__main__":
    cap = cv2.VideoCapture("./data/1.MP4")
    ret, frame = cap.read()
    small_size = (frame.shape[1]//2, frame.shape[0]//2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    detector = FireDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_small = cv2.resize(frame, small_size)
        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        out_frame, bboxes, is_fire = detector.process(frame_rgb)

        bbox_img = frame_small.copy()
        if is_fire and bboxes:
            nms = perform_non_maximum_suppression(bboxes)
            for x1,y1,x2,y2 in nms:
                cv2.rectangle(bbox_img, (x1,y1), (x2,y2), (0,0,255), 2)

        h = 360
        combined = cv2.hconcat([
            resize_image_to_height(frame_small, h),
            resize_image_to_height(cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR), h),
            resize_image_to_height(bbox_img, h)
        ])
        cv2.imshow("Fire Detection", combined)
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
