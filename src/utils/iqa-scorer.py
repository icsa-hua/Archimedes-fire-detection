from pypiqe import piqe
import cv2

if __name__ == "__main__":
    cap = cv2.VideoCapture("./data/2.MP4")
    ret, frame = cap.read()
    small_size = (frame.shape[1]//2, frame.shape[0]//2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_small = cv2.resize(frame, small_size)
        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        # Calculate PIQE score
        score, activityMask, noticeableArtifactMask, noiseMask = piqe(frame_gray)
        print(f"PIQE Score: {score:.4f}")