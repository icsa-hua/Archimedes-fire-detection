import numpy as np
import cv2
import scipy.io
from os.path import join, dirname, exists
from fiqe.mscn import compute_image_mscn_transform
from fiqe.stats import aggd_features, paired_product

# only used during training
#from skimage.util.shape import view_as_windows

def _niqe_extract_subband_feats(mscncoefs):
    # alpha_m,  = extract_ggd_features(mscncoefs)
    alpha_m, N, bl, br, lsq, rsq = aggd_features(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
    return np.array([alpha_m, (bl+br)/2.0,
            alpha1, N1, bl1, br1,  # (V)
            alpha2, N2, bl2, br2,  # (H)
            alpha3, N3, bl3, br3,  # (D1)
            alpha4, N4, bl4, br4,  # (D2)
    ])

def get_patches_train_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 1, stride)

def get_patches_test_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 0, stride)

def extract_on_patches(img, patch_size):
    h, w = img.shape
    patch_size = int(patch_size)
    patches = []
    for j in range(0, h-patch_size+1, patch_size):
        for i in range(0, w-patch_size+1, patch_size):
            patch = img[j:j+patch_size, i:i+patch_size]
            patches.append(patch)

    patches = np.array(patches)

    patch_features = []
    for p in patches:
        patch_features.append(_niqe_extract_subband_feats(p))
    patch_features = np.array(patch_features)

    return patch_features

def _get_patches_generic(img, patch_size, is_train, stride):
    h, w = np.shape(img)
    if h < patch_size or w < patch_size:
        print("Input image is too small")
        exit(0)

    # ensure that the patch divides evenly into img
    hoffset = (h % patch_size)
    woffset = (w % patch_size)

    if hoffset > 0:
        img = img[:-hoffset, :]
    if woffset > 0:
        img = img[:, :-woffset]


    img = img.astype(np.float32)
    img2 = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    mscn1, var, mu = compute_image_mscn_transform(img)
    mscn1 = mscn1.astype(np.float32)

    mscn2, _, _ = compute_image_mscn_transform(img2)
    mscn2 = mscn2.astype(np.float32)


    feats_lvl1 = extract_on_patches(mscn1, patch_size)
    feats_lvl2 = extract_on_patches(mscn2, patch_size/2)

    feats = np.hstack((feats_lvl1, feats_lvl2))# feats_lvl3))

    #if is_train:
    #    variancefield = view_as_windows(var, (patch_size, patch_size), step=patch_size)
    #    variancefield = variancefield.reshape(-1, patch_size, patch_size)
    #    avg_variance = np.mean(np.mean(variancefield, axis=2), axis=1)
    #    avg_variance /= np.max(avg_variance)
    #    feats = feats[avg_variance > 0.75]

    return feats

def train_niqe(inputVideoPath, frame_step=1, niqe_params='niqe_image_params.mat'):
    patch_size = 96

    cap = cv2.VideoCapture(inputVideoPath)
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    assert H > (patch_size*2+1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"
    assert W > (patch_size*2+1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"

    all_features = []
    for t in range(0, T, frame_step):
        percent = (t + 1) / T * 100
        print(f"Processing frame {t + 1}/{T} ({percent:.1f}%)", end='\r', flush=True)
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        norm_gray_frame = gray_frame.astype(np.float32) / 255.0
        feats = get_patches_train_features(norm_gray_frame, patch_size)
        all_features.append(feats)

    cap.release()
    all_features = np.vstack(all_features)
    module_path = dirname(__file__)
    scipy.io.savemat(join(module_path, niqe_params), {
        'pop_mu': np.mean(all_features, axis=0), 
        'pop_cov': np.cov(all_features.T),
        'N_total': np.array([[all_features.shape[0]]])})
    

def train_niqe_online(inputVideoPath, frame_step=1, niqe_params='niqe_image_params.mat'):
    patch_size = 96

    cap = cv2.VideoCapture(inputVideoPath)
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    assert H > (patch_size * 2 + 1), "Frame height too small, requires > 192"
    assert W > (patch_size * 2 + 1), "Frame width too small, requires > 192"

    module_path = dirname(__file__)
    param_path = join(module_path, niqe_params)

    # Load or initialize
    if exists(param_path):
        print("Loading existing niqe parameters...")
        params = scipy.io.loadmat(param_path)
        pop_mu = params['pop_mu'].flatten()
        pop_cov = params['pop_cov']
        N_total = params.get('N_total', np.array([[0]])).item()
        print(f"Old parameters: mu={pop_mu.shape}, cov={pop_cov.shape}, N_total={N_total}")
    else:
        print("No existing parameters found. Initializing...")
        pop_mu = None
        pop_cov = None
        N_total = 0

    for t in range(0, T, frame_step):
        percent = (t + 1) / T * 100
        print(f"Processing frame {t + 1}/{T} ({percent:.1f}%)", end='\r', flush=True)
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        norm_gray_frame = gray_frame.astype(np.float32) / 255.0
        feats = get_patches_train_features(norm_gray_frame, patch_size)  # [num_patches, num_features]

        if feats.size == 0:
            continue

        N_batch = feats.shape[0]
        batch_mu = np.mean(feats, axis=0)
        batch_cov = np.cov(feats.T)

        if pop_mu is None:
            pop_mu = batch_mu
            pop_cov = batch_cov
            N_total = N_batch
        else:
            # Online mean update
            delta = batch_mu - pop_mu
            new_N_total = N_total + N_batch
            new_mu = pop_mu + (N_batch / new_N_total) * delta

            # Online covariance update
            cov_correction = (N_total * N_batch) / new_N_total * np.outer(delta, delta)
            new_cov = ((N_total * pop_cov) + (N_batch * batch_cov) + cov_correction) / new_N_total

            # Update state
            pop_mu = new_mu
            pop_cov = new_cov
            N_total = new_N_total

    cap.release()

    print(f"New parameters: mu={pop_mu.shape}, cov={pop_cov.shape}, N_total={N_total}")
    scipy.io.savemat(param_path, {
        'pop_mu': pop_mu,
        'pop_cov': pop_cov,
        'N_total': N_total
    })

    print(f"\nOnline training complete. Parameters saved to {param_path}")


def calculate_niqe(img, niqe_params='niqe_image_params.mat'):
    patch_size = 96
    module_path = dirname(__file__)

    params = scipy.io.loadmat(join(module_path, niqe_params))
    pop_mu = np.ravel(params["pop_mu"])
    pop_cov = params["pop_cov"]

    H = img.shape[0]
    W = img.shape[1]
    
    assert H > (patch_size*2+1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"
    assert W > (patch_size*2+1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"

    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm_gray_frame = gray_frame.astype(np.float32) / 255.0
    feats = get_patches_test_features(norm_gray_frame, patch_size)
    sample_mu = np.mean(feats, axis=0)
    sample_cov = np.cov(feats.T)

    X = sample_mu - pop_mu
    covmat = ((pop_cov+sample_cov)/2.0)
    pinvmat = scipy.linalg.pinv(covmat)
    niqe = np.sqrt(np.dot(np.dot(X, pinvmat), X))

    return niqe


if __name__ == "__main__":
    video_path = "./data/1.MP4"
    train_niqe_online(video_path, frame_step=10)
    