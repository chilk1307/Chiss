"""
Classical content-based image retrieval (CBIR) baseline with hand-crafted features (plan 01, section 1.9).

Every image is resized to 128 x 128 and described by four feature families:
    color   HSV histogram, 8 x 4 x 4 bins, normalised to sum 1                   (colour spaces, histograms)
    lbp     uniform LBP (P = 8, R = 1) on the grey image, one 10-bin histogram
            per cell of a 4 x 4 grid, normalised to sum 1                          (texture)
    hog     HOG on the grey image, 16 x 16 px cells, 2 x 2 cells per block,
            9 orientations, L2 normalised                                          (gradients, shape)
    gabor   Gabor filter bank, 4 orientations x 3 wavelengths (4, 8, 16 px);
            mean and standard deviation of the response magnitude, L2 normalised  (linear filtering, texture)

Similarity between a query and a gallery image, per family:
    histograms (color, lbp): 1 - chi2 / 2, with chi2 = sum((a - b)^2 / (a + b)) in [0, 2]
    vectors (hog, gabor):    cosine similarity
With several families the similarities are averaged, which for cosine similarities is the same as concatenating the
normalised vectors.
"""
import math
from typing import Dict, List, Union

import cv2
import numpy as np
import PIL.Image
import torch
from skimage.feature import hog, local_binary_pattern

CBIR_FEATURES = ['color', 'lbp', 'hog', 'gabor']
HISTOGRAM_FEATURES = ['color', 'lbp']
IMAGE_SIZE = 128

COLOR_BINS = (8, 4, 4)
LBP_POINTS, LBP_RADIUS, LBP_GRID = 8, 1, 4
HOG_CELL, HOG_BLOCK, HOG_ORIENTATIONS = 16, 2, 9
GABOR_WAVELENGTHS = (4, 8, 16)
GABOR_ORIENTATIONS = (0, math.pi / 4, math.pi / 2, 3 * math.pi / 4)

FEATURE_DIMS = {
    'color': int(np.prod(COLOR_BINS)),
    'lbp': (LBP_POINTS + 2) * LBP_GRID * LBP_GRID,
    'hog': ((IMAGE_SIZE // HOG_CELL - HOG_BLOCK + 1) ** 2) * HOG_BLOCK * HOG_BLOCK * HOG_ORIENTATIONS,
    'gabor': 2 * len(GABOR_WAVELENGTHS) * len(GABOR_ORIENTATIONS),
}
CBIR_VERSION = 1  # bump when the feature definitions change, so cached features are recomputed


def parse_cbir_features(value: Union[str, List[str]]) -> List[str]:
    names = value.split(',') if isinstance(value, str) else list(value)
    names = [n.strip() for n in names if n.strip()]
    if names == ['all']:
        return list(CBIR_FEATURES)
    unknown = [n for n in names if n not in CBIR_FEATURES]
    if unknown or not names:
        raise ValueError(f"--cbir-features should be 'all' or a comma separated subset of {CBIR_FEATURES}")
    return names


def feature_slices() -> Dict[str, slice]:
    """
    Position of each family in the vector returned by CBIRFeatureExtractor (always all four families)
    """
    slices, start = {}, 0
    for name in CBIR_FEATURES:
        slices[name] = slice(start, start + FEATURE_DIMS[name])
        start += FEATURE_DIMS[name]
    return slices


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def color_histogram(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)  # H in [0, 180), S and V in [0, 256) for uint8 images
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(COLOR_BINS), [0, 180, 0, 256, 0, 256]).flatten()
    return hist / max(float(hist.sum()), 1e-12)


def lbp_histogram(gray: np.ndarray) -> np.ndarray:
    codes = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method='uniform').astype(np.int64)
    n_bins = LBP_POINTS + 2
    cell = IMAGE_SIZE // LBP_GRID
    hists = []
    for i in range(LBP_GRID):
        for j in range(LBP_GRID):
            block = codes[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell]
            h = np.bincount(block.ravel(), minlength=n_bins).astype(np.float64)
            hists.append(h / max(h.sum(), 1e-12))
    return np.concatenate(hists) / (LBP_GRID * LBP_GRID)


def hog_descriptor(gray: np.ndarray) -> np.ndarray:
    v = hog(gray, orientations=HOG_ORIENTATIONS, pixels_per_cell=(HOG_CELL, HOG_CELL),
            cells_per_block=(HOG_BLOCK, HOG_BLOCK), block_norm='L2-Hys', feature_vector=True)
    return _l2_normalize(v)


_gabor_bank = None


def gabor_bank():
    """
    Quadrature pairs (even, odd) of Gabor kernels; sigma = 0.56 * wavelength gives a bandwidth of about one octave
    """
    global _gabor_bank
    if _gabor_bank is None:
        _gabor_bank = []
        for wavelength in GABOR_WAVELENGTHS:
            sigma = 0.56 * wavelength
            ksize = int(6 * sigma) | 1
            for theta in GABOR_ORIENTATIONS:
                even = cv2.getGaborKernel((ksize, ksize), sigma, theta, wavelength, 0.5, 0, ktype=cv2.CV_32F)
                odd = cv2.getGaborKernel((ksize, ksize), sigma, theta, wavelength, 0.5, math.pi / 2,
                                         ktype=cv2.CV_32F)
                even -= even.mean()  # remove the DC response so flat regions give no energy
                _gabor_bank.append((even, odd))
    return _gabor_bank


def gabor_energy(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32) / 255.0
    feats = []
    for even, odd in gabor_bank():
        re = cv2.filter2D(g, cv2.CV_32F, even, borderType=cv2.BORDER_REFLECT)
        im = cv2.filter2D(g, cv2.CV_32F, odd, borderType=cv2.BORDER_REFLECT)
        magnitude = np.sqrt(re * re + im * im)
        feats.extend([magnitude.mean(), magnitude.std()])
    return _l2_normalize(np.array(feats, dtype=np.float64))


FEATURE_FUNCTIONS = {
    'color': lambda rgb, gray: color_histogram(rgb),
    'lbp': lambda rgb, gray: lbp_histogram(gray),
    'hog': lambda rgb, gray: hog_descriptor(gray),
    'gabor': lambda rgb, gray: gabor_energy(gray),
}


class CBIRFeatureExtractor:
    """
    Callable usable as a dataset 'preprocess': PIL image -> float tensor with all four feature families
    concatenated in the order of CBIR_FEATURES (see feature_slices)
    """

    def __call__(self, image: PIL.Image.Image) -> torch.Tensor:
        rgb = np.asarray(image.convert('RGB'))
        rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        parts = [FEATURE_FUNCTIONS[name](rgb, gray) for name in CBIR_FEATURES]
        return torch.from_numpy(np.concatenate(parts).astype(np.float32))


def chi_square_similarity(query: torch.Tensor, gallery: torch.Tensor, max_elements: int = 2 ** 25) -> torch.Tensor:
    """
    1 - chi2 / 2 between every query histogram and every gallery histogram (both normalised to sum 1)
    """
    n_query, n_gallery, dim = query.shape[0], gallery.shape[0], query.shape[1]
    chunk = max(1, max_elements // max(1, n_gallery * dim))
    out = torch.empty(n_query, n_gallery, device=query.device)
    for i in range(0, n_query, chunk):
        q = query[i:i + chunk, None, :]
        diff = q - gallery[None]
        total = (q + gallery[None]).clamp_min(1e-10)
        out[i:i + chunk] = 1 - 0.5 * (diff * diff / total).sum(-1)
    return out


def cbir_similarity(query: torch.Tensor, gallery: torch.Tensor, features: List[str]) -> torch.Tensor:
    """
    Average over the selected families of the per-family similarity (see the module docstring)
    """
    slices = feature_slices()
    total = torch.zeros(query.shape[0], gallery.shape[0], device=query.device)
    for name in features:
        q, g = query[:, slices[name]], gallery[:, slices[name]]
        if name in HISTOGRAM_FEATURES:
            total += chi_square_similarity(q, g)
        else:
            total += q @ g.T
    return total / len(features)
