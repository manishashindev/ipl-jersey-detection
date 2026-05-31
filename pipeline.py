"""
pipeline.py — IPL Jersey Team Detection Inference
Usage: python pipeline.py <image_path> [model_pkl_path]
"""

import os, sys
import cv2
import numpy as np
import joblib
from skimage.feature import hog, local_binary_pattern


# ── Feature extraction (must match training) ───────────────────────────────
def extract_features(cell_bgr):
    """Extract 338 hand-crafted features from a 100×75 BGR cell."""
    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [36], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
    n_px = cell_bgr.shape[0] * cell_bgr.shape[1]
    h_hist /= (n_px + 1e-7)
    s_hist /= (n_px + 1e-7)
    v_hist /= (n_px + 1e-7)

    moments = []
    for ch_img in [cell_bgr, hsv]:
        for ch in range(3):
            v = ch_img[:, :, ch].astype(np.float32)
            moments.extend([v.mean(), v.std()])

    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(gray, orientations=9,
                   pixels_per_cell=(25, 25), cells_per_block=(2, 2),
                   feature_vector=True).astype(np.float32)

    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)

    return np.concatenate([h_hist, s_hist, v_hist,
                           np.array(moments, dtype=np.float32),
                           hog_feat,
                           lbp_hist.astype(np.float32)])


# ── Inference ──────────────────────────────────────────────────────────────
def predict_image(image_path, model_pkl_path):
    """
    Predict IPL team labels for all 64 grid cells of a single image.

    Args:
        image_path     : path to input image (any resolution)
        model_pkl_path : path to model_<teamname>.pkl

    Returns:
        dict:
            fname        – filename of the image
            predictions  – list[64] of int (0–10)
            labels       – list[64] of team name strings
            csv_row      – dict ready for CSV output
    """
    bundle = joblib.load(model_pkl_path)
    model         = bundle['model']
    label_to_team = bundle['label_to_team']
    img_w, img_h  = bundle['img_w'],  bundle['img_h']
    cell_w, cell_h = bundle['cell_w'], bundle['cell_h']
    grid_r, grid_c = bundle['grid_r'], bundle['grid_c']

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    img = cv2.resize(img, (img_w, img_h))

    cells = [
        img[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
        for r in range(grid_r) for c in range(grid_c)
    ]
    feats = np.array([extract_features(c) for c in cells], dtype=np.float32)
    preds = model.predict(feats).tolist()

    fname = os.path.basename(image_path)
    csv_row = {'Image File Name': fname, 'Train Or Test': 'Test'}
    for i, p in enumerate(preds):
        csv_row[f'c{i+1:02d}'] = int(p)

    return {
        'fname':       fname,
        'predictions': preds,
        'labels':      [label_to_team[p] for p in preds],
        'csv_row':     csv_row,
    }


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <image_path> [model_pkl_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.join(os.path.dirname(__file__), 'output', 'model_TeamName.pkl')

    result = predict_image(image_path, model_path)

    print(f"\nImage : {result['fname']}")
    print(f"Teams detected: {set(result['labels']) - {'NO_PLAYER'}}")
    print("\nGrid predictions (row-major, c01 top-left → c64 bottom-right):")
    preds = result['predictions']
    labels = result['labels']
    for row in range(8):
        row_str = '  '.join(f"c{row*8+col+1:02d}:{labels[row*8+col]:>10}" for col in range(8))
        print(row_str)

    print("\nCSV row:")
    row = result['csv_row']
    print(','.join(str(row[k]) for k in sorted(row.keys())))
