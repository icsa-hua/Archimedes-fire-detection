from skimage.measure import shannon_entropy
from skimage.filters import sobel
import numpy as np
import cv2

def extract_fiqe_features(img):
    """
    Extracts FLIR Image Quality Features (FIQE) from a given image patch.

    Feature vector captures various distortions:
        - Information loss
        - Blur
        - Low contrast / dynamic range compression
        - Noise and sensor degradation
        - Flatness or uniformity (over-smoothing, low texture)
    """
    img = img.astype(np.float32)
    img = np.clip(img, 0, 1)

    # 1. Entropy (↓ entropy → poor information richness, compression artifacts, underexposure)
    entropy_val = shannon_entropy((img * 255).astype(np.uint8))  

    # 2. CNR - Contrast-to-Noise Ratio (↓ CNR → poor contrast or excessive noise)
    signal = np.mean(img)
    noise = np.std(img)
    cnr_val = signal / (noise + 1e-8)  

    # 3. Sharpness via Sobel edges (↓ sharpness → blur, motion blur, focus loss)
    sobel_edges = sobel(img)
    sharpness = np.mean(sobel_edges)  

    # 4. Thermal Noise Estimation (↑ noise_est → sensor noise, temperature instability)
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    diff = img - blurred
    noise_mask = np.abs(diff) < np.percentile(np.abs(diff), 25)  # flat regions
    noise_est = np.std(img[noise_mask])

    # 5. Dynamic Range (↓ dyn_range → clipping, underexposure, flat contrast)
    dyn_range = np.max(img) - np.min(img)

    # # 6. Gradient Magnitude (↓ grad_mag → smooth/blurry image, structure loss)
    grad_mag = np.mean(np.abs(sobel_edges))

    # 7. Flat Area Ratio (↑ flat_ratio → texture loss, blur, over-smoothing)
    lap_var = cv2.Laplacian(img, cv2.CV_32F, ksize=3)**2
    flat_ratio = np.mean(lap_var < np.percentile(lap_var, 10))

    # Final 7-D feature vector
    features = np.array([
        entropy_val,   # ↓ entropy → information loss
        cnr_val,       # ↓ CNR → low contrast or noise
        sharpness,     # ↓ sharpness → blur
        noise_est,     # ↑ noise → thermal sensor noise
        dyn_range,     # ↓ dynamic range → clipping or contrast compression
        grad_mag,      # ↓ gradient strength → blur or low structure
        flat_ratio     # ↑ flat regions → over-smoothing or low detail
    ], dtype=np.float32)

    return features

