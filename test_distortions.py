import ctypes
import cv2
import torch
from distortions import *


if __name__ == "__main__":
    # Get screen size (Windows)
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)
    print(f"Screen size: {screen_width}x{screen_height}")

    cap = cv2.VideoCapture("./data/4.MP4")
    ret, frame = cap.read()
    small_size = (frame.shape[1]//2, frame.shape[0]//2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    distortions = {
        "Original": lambda img: img,
        "Lens Blur": lambda img: LensBlur(ksize=11)(img)[0],
        "Motion Blur": lambda img: MotionBlur(degree=15, angle=45)(img)[0],
        # "Blackout": lambda img: Blackout()(img)[0],
        "Overexposure": lambda img: Overexposure(factor=2.5)(img)[0],
        "Underexposure": lambda img: Underexposure(factor=0.3)(img)[0],
        "Noise": lambda img: GaussianNoise(mean=0, std=25)(img)[0],
        "Compression": lambda img: Compression(quality=5)(img)[0],
        # "Color Distortion": lambda img: ColorDistortion()(img)[0],
        # "Glare": lambda img: Glare()(img)[0],
        "Ghosting": lambda img: Ghosting(shift=10, alpha=0.6)(img)[0],
        # "Flicker": lambda img: Flicker(factor=1.8)(img)[0],
        # "Freeze": lambda img: FrameFreeze()(img)[0],
        # "Obstruction": lambda img: Obstruction()(img)[0],
        # "Crop": lambda img: Crop()(img)[0],
        "Aliasing": lambda img: Aliasing(factor=4)(img)[0]
    }
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")    

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
            img = img.astype(np.float32)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
            # print("img_tensor shape:", img_tensor.shape)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_gray = img_gray.astype('float32') / 255.0
            img_gray_tensor = torch.from_numpy(img_gray).unsqueeze(0).unsqueeze(0).to(device)
            # print("img_gray_tensor shape:", img_gray_tensor.shape)

            # # NIQE score
            # niqe = NIQE()
            # niqe_score = niqe(img_gray_tensor)
            # niqe_text = f"NIQE: {float(niqe_score):.4f}"

            # # PIQE score
            # piqe = PIQE()
            # piqe_score = piqe(img_gray_tensor)
            # piqe_text = f"PIQE: {float(piqe_score):.4f}"

            # # ARNIQA score
            # arniqa = ARNIQA()
            # arniqa_score = arniqa(img_tensor)
            # arniqa_text = f"ARNIQA: {float(arniqa_score):.4f}"

            # # PAQ2PIQ score
            # paq2piq = PAQ2PIQ()
            # paq2piq.eval()  # Ensure model is in evaluation mode
            # paq2piq_score = paq2piq(img_tensor)
            # paq2piq_text = f"PAQ2PIQ: {float(paq2piq_score):.4f}"

            # Distortion name
            name_text = name

            # Draw with black outline for visibility
            y0 = 30
            dy = 35
            for i, text in enumerate([name_text]):
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
        grid_size = 3
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
            # cv2.imwrite("distorted_frames.png", combined_frame_padded)  # Uncomment to save the image
            break
        
    cap.release()
    cv2.destroyAllWindows()