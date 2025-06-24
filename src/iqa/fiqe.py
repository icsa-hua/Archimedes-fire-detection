import os
import cv2
import numpy as np
import scipy
from iqa.fiqe_features import extract_fiqe_features
from iqa.niqe_features import extract_niqe_features


def extract_patches(image, patch_size=96, stride=96):
    h, w = image.shape
    hoffset = (h % patch_size)
    woffset = (w % patch_size)
    if hoffset > 0:
        image = image[:-hoffset, :]
    if woffset > 0:
        image = image[:, :-woffset]
    patches = []
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = image[i:i + patch_size, j:j + patch_size]
            patches.append(patch)

    return patches


def train_fiqe(inputVideoPath, frame_step=1, patch_size=96, stride=96, output_path='fiqe_model.npz'):
    cap = cv2.VideoCapture(inputVideoPath)
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    all_features = []

    for t in range(0, T, frame_step):
        percent = (t + 1) / T * 100
        print(f"Processing frame {t + 1}/{T} ({percent:.1f}%)", end='\r', flush=True)
        ret, frame = cap.read()
        if not ret:
            break
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.astype(np.float32) / 255.0

        patches = extract_patches(frame, patch_size, stride)
        for patch in patches:            
            # niqe_feats = extract_niqe_features(patch)
            # all_features.append(niqe_feats)
            fiqe_feats = extract_fiqe_features(patch)
            all_features.append(fiqe_feats)
            # all_features.append(np.hstack((niqe_feats, fiqe_feats)))

    all_features = np.vstack(all_features)
    mu = np.mean(all_features, axis=0)
    cov = np.cov(all_features.T)

    print(f"Training complete. Mean shape: {mu.shape}, Covariance shape: {cov.shape}")
    print(f"Mean: {mu[:5]}... Covariance: {cov[:5, :5]}...")
    cap.release()

    np.savez(output_path, mu=mu, cov=cov, patch_size=patch_size, stride=stride)
    print(f"FIQE model saved to {output_path}")


def compute_mahalanobis(x, mu, cov_inv):
    """Mahalanobis distance between feature vector x and pristine model"""
    diff = x - mu
    return np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T))


def fiqe_score(image, model_path='fiqe_model.npz'):
    """Compute FIQE score for a given image using the trained model."""
    data = np.load(model_path)
    mu = data['mu']
    cov = data['cov']
    patch_size = data['patch_size']
    stride = data['stride']

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32) / 255.0

    patches = extract_patches(image, patch_size, stride)
    scores = []

    # covmat = ((pop_cov+sample_cov)/2.0)
    pinvmat = scipy.linalg.pinv(cov)

    for patch in patches:
        feats = extract_fiqe_features(patch)
        if feats is not None:
            score = compute_mahalanobis(feats, mu, pinvmat)
            scores.append(score)

    return np.mean(scores) if scores else float('inf')
