import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
import ctypes
from pypiqe import piqe

# Sorbel mask for edge detection

# Utility: Convert between PIL and OpenCV
def cv2_to_pil(img): return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
def pil_to_cv2(img): return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# 1. Lens Blur (Gaussian blur)
def add_lens_blur(img, ksize=11):
    '''Uniform blur across the image due to out-of-focus lens.'''
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

# 2. Motion Blur (Directional)
def add_motion_blur(img, degree=15, angle=45):
    '''Directional blur caused by object/camera motion.'''
    k = np.zeros((degree, degree))
    k[int((degree - 1)/2), :] = np.ones(degree)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1.0), (degree, degree))
    k /= degree
    return cv2.filter2D(img, -1, k)

# 3. Blackout
def add_blackout(img):
    '''Frame is completely or mostly dark; no visible content.'''
    return np.zeros_like(img)

# 4. Overexposure
def add_overexposure(img, factor=2.5):
    '''Washed-out, overly bright regions with loss of detail (high pixel intensities).'''
    pil_img = cv2_to_pil(img)
    enhancer = ImageEnhance.Brightness(pil_img)
    return pil_to_cv2(enhancer.enhance(factor))

# 5. Underexposure
def add_underexposure(img, factor=0.3):
    '''Very dark regions, detail lost due to low light (but not a complete blackout).'''
    pil_img = cv2_to_pil(img)
    enhancer = ImageEnhance.Brightness(pil_img)
    return pil_to_cv2(enhancer.enhance(factor))

# 6. Noise (Gaussian)
def add_noise(img, mean=0, std=25):
    '''Grainy appearance due to high ISO, compression artifacts, or sensor issues.'''
    noise = np.random.normal(mean, std, img.shape).astype(np.uint8)
    noisy_img = cv2.add(img, noise)
    return noisy_img

# 7. Compression Artifacts (JPEG)
def add_compression_artifacts(img, quality=5):
    '''Blockiness, ringing, or banding caused by aggressive JPEG or video compression.'''
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(enc, 1)

# 8. Color Distortion (Hue/Saturation Shift)
def add_color_distortion(img):
    '''Unnatural colors due to white balance issues or color profile corruption.'''
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[..., 0] = (hsv[..., 0] + 50) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.5, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# 9. Glare (Simulated white patch)
def add_glare(img):
    '''Bright reflections or glares that obscure parts of the image.'''
    h, w = img.shape[:2]
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.circle(mask, (w//2, h//3), int(w*0.2), (255, 255, 255), -1)
    return cv2.addWeighted(img, 0.8, mask, 0.5, 0)

# 10. Ghosting / Double Exposure
def add_ghosting(img, shift=10, alpha=0.6):
    '''Multiple offset copies of objects due to sync issues or reflections.'''
    overlay = np.roll(img, shift, axis=1)
    return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

# 11. Flicker (brightness pulse)
def add_flicker(img, factor=1.8):
    '''Sudden brightness change in a single frame (common in videos).'''
    return add_overexposure(img, factor)

# 12. Frame Freeze (return unchanged copy)
def add_frame_freeze(img): 
    '''Frame repeated multiple times, possibly due to dropped frames.'''
    return img.copy()

# 13. Obstruction (hand/fog simulation)
def add_obstruction(img):
    '''Something blocks the camera (e.g., hand, finger, foggy lens).'''
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (w//3, h//3), (2*w//3, 2*h//3), 255, -1)
    obstruction = cv2.GaussianBlur(np.full_like(img, (80, 80, 80)), (51, 51), 0)
    return np.where(mask[:, :, None] == 255, obstruction, img)

# 14. Cropped / Partial Frame
def add_crop(img):
    '''Only part of the scene is captured due to misaligned camera or error.'''
    h, w = img.shape[:2]
    return cv2.copyMakeBorder(img[:h//2, :w//2], h//2, 0, w//2, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

# 15. Aliasing / Moiré (simulate via downsampling and upsampling)
def add_aliasing(img, factor=4):
    '''Jagged edges or pattern interference, especially on repeated textures.'''
    h, w = img.shape[:2]
    small = cv2.resize(img, (w//factor, h//factor), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


if __name__ == "__main__":
    # Get screen size (Windows)
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)
    print(f"Screen size: {screen_width}x{screen_height}")

    cap = cv2.VideoCapture("./data/12.MP4")
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

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            score, activityMask, noticeableArtifactMask, noiseMask = piqe(img_gray)

            # Annotate with PIQE score
            name = f"{name} (PIQE: {score:.4f})"
            cv2.putText(
                annotated, name, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
            )

            # Black outline for visibility
            cv2.putText(
                annotated, name, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA
            )
            cv2.putText(
                annotated, name, (10, 30),
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