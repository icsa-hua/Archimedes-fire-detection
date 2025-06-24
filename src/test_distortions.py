import ctypes
import cv2
from iqa.fiqe import fiqe_score
from utils.distortions import *
from pypiqe import piqe


if __name__ == "__main__":
    # Get screen size (Windows)
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)
    print(f"Screen size: {screen_width}x{screen_height}")

    cap = cv2.VideoCapture("./data/2.MP4")
    ret, frame = cap.read()
    small_size = (frame.shape[1]//2, frame.shape[0]//2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    distortions = {
        "Original": lambda img: img,
        "Lens Blur": lambda img: add_lens_blur(img, ksize=15),
        "Motion Blur": lambda img: add_motion_blur(img, degree=15, angle=45),
        "Blackout": add_blackout,
        "Overexposure": lambda img: add_overexposure(img, factor=2.5),
        "Underexposure": lambda img: add_underexposure(img, factor=0.3),
        "Noise": lambda img: add_noise(img, mean=0, std=25),
        "Compression": lambda img: add_compression_artifacts(img, quality=5),
        "Color Distortion": add_color_distortion,
        "Glare": add_glare,
        "Ghosting": lambda img: add_ghosting(img, shift=10, alpha=0.6),
        "Flicker": lambda img: add_flicker(img, factor=1.8),
        "Freeze": add_frame_freeze,
        "Obstruction": add_obstruction,
        "Crop": add_crop,
        "Aliasing": lambda img: add_aliasing(img, factor=4)
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_small = cv2.resize(frame, small_size)
        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

        # Apply distortions and annotate
        annotated_versions = []
        for name, func in distortions.items():
            img = func(frame_rgb)
            annotated = img.copy()
            # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img.astype(np.float32)

            # # NIQE score
            # niqe_score = calculate_niqe(img, niqe_params='niqe_image_params.mat')
            # niqe_text = f"NIQE: {niqe_score:.4f}"

            # FIQE score
            fiqe = fiqe_score(img, model_path='fiqe_model.npz')
            fiqe_text = f"FIQE: {fiqe:.4f}"

            # PIQE score
            score, activityMask, noticeableArtifactMask, noiseMask = piqe(img)
            piqe_text = f"PIQE: {score:.4f}"

            # Distortion name
            name_text = name

            # Draw with black outline for visibility
            y0 = 30
            dy = 35
            for i, text in enumerate([name_text, piqe_text, fiqe_text]):
                y = y0 + i * dy
                # Outline
                cv2.putText(
                    annotated, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA
                )
                # Foreground
                cv2.putText(
                    annotated, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
                )

            annotated_versions.append(annotated)

        # Arrange in a 4x4 grid (16 cells, last one black)
        grid_size = 4
        cell_h, cell_w = annotated_versions[0].shape[:2]
        grid = []
        for i in range(grid_size):
            row = []
            for j in range(grid_size):
                idx = i * grid_size + j
                if idx < len(annotated_versions):
                    row.append(annotated_versions[idx])
                else:
                    # Fill empty cell with black
                    row.append(np.zeros_like(annotated_versions[0]))
            grid.append(np.hstack(row))
        combined_frame = np.vstack(grid)

        # Maintain aspect ratio while fitting to screen
        ch, cw = combined_frame.shape[:2]
        scale = min(screen_width / cw, screen_height / ch)
        new_w, new_h = int(cw * scale), int(ch * scale)
        resized = cv2.resize(combined_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to center
        top = (screen_height - new_h) // 2
        bottom = screen_height - new_h - top
        left = (screen_width - new_w) // 2
        right = screen_width - new_w - left
        combined_frame_padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

        cv2.imshow("Distorted Frames", combined_frame_padded)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()