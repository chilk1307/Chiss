"""
Blind estimation of the degradation of an image and automatic choice of the restoration (plan 03, section 3.3).

Measured quantities (all classical image processing):
    noise_sigma     noise standard deviation from the median absolute deviation of the finest wavelet coefficients
    impulse_ratio   fraction of pixels at 0 or 255 that differ strongly from their 3 x 3 median (salt and pepper)
    laplacian_var   variance of the Laplacian of the grey image (low = blurred)
    hf_ratio        share of the spectrum energy above half the Nyquist frequency (low = blurred / low resolution)
    blockiness      mean absolute step across the 8 x 8 block boundaries / mean step inside the blocks (JPEG)
    l_mean, l_p50   mean and median of L* in CIELAB (low = dark)
    a_mean, b_mean  mean a*, b* in CIELAB; cast = their norm (grey-world assumption: about 0 without colour cast)

--restore auto      if-then rules on these quantities, in priority order, with thresholds stored in
                    restore_rules.json (calibrate them with tools/calibrate_restore_rules.py on CIRR *train* images,
                    never on the evaluation sets). The first rule that fires gives the restoration; none = keep.
--restore auto_iqa  try a few restorations and keep the one with the best (lowest) BRISQUE score, a no-reference
                    image quality measure (library piq). Candidates are scored on a 320 x 320 central crop at the
                    original resolution.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from restorations import estimate_noise_sigma, restore

RULES_FILE = Path(__file__).with_name('restore_rules.json')
IQA_CANDIDATES = ['none', 'median', 'nlm', 'deblock', 'clahe', 'gray_world', 'unsharp']
FEATURES = ['noise_sigma', 'impulse_ratio', 'laplacian_var', 'hf_ratio', 'blockiness', 'l_mean', 'l_p50', 'a_mean',
            'b_mean', 'cast']


def blockiness(gray: np.ndarray, block: int = 8) -> float:
    g = gray.astype(np.float64)
    ratios = []
    for axis in (1, 0):
        steps = np.abs(np.diff(g, axis=axis))  # step between pixel i and i + 1 along the axis
        positions = np.arange(steps.shape[axis])
        at_boundary = (positions % block) == block - 1  # step between the last pixel of a block and the next one
        boundary = np.take(steps, np.nonzero(at_boundary)[0], axis=axis).mean()
        inside = np.take(steps, np.nonzero(~at_boundary)[0], axis=axis).mean()
        ratios.append(boundary / max(inside, 1e-6))
    return float(np.mean(ratios))


def high_frequency_ratio(gray: np.ndarray, cutoff: float = 0.5) -> float:
    g = gray.astype(np.float64)
    g = (g - g.mean()) * np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    power = np.abs(np.fft.rfft2(g)) ** 2
    fy = np.fft.fftfreq(g.shape[0])[:, None] / 0.5
    fx = np.fft.rfftfreq(g.shape[1])[None, :] / 0.5
    radius = np.sqrt(fx ** 2 + fy ** 2)
    power[0, 0] = 0
    return float(power[radius > cutoff].sum() / max(power.sum(), 1e-12))


def impulse_ratio(gray: np.ndarray, extreme: int = 5, difference: int = 50) -> float:
    med = cv2.medianBlur(gray, 3).astype(np.int16)
    g = gray.astype(np.int16)
    extreme_pixels = (g <= extreme) | (g >= 255 - extreme)
    return float((extreme_pixels & (np.abs(g - med) > difference)).mean())


def estimate_features(image: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float64)
    l_star = lab[..., 0] * 100.0 / 255.0
    a_mean = float(lab[..., 1].mean() - 128.0)
    b_mean = float(lab[..., 2].mean() - 128.0)
    return {
        'noise_sigma': estimate_noise_sigma(image),
        'impulse_ratio': impulse_ratio(gray),
        'laplacian_var': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        'hf_ratio': high_frequency_ratio(gray),
        'blockiness': blockiness(gray),
        'l_mean': float(l_star.mean()),
        'l_p50': float(np.median(l_star)),
        'a_mean': a_mean,
        'b_mean': b_mean,
        'cast': float(np.hypot(a_mean, b_mean)),
    }


_rules = None


def load_rules(path: Optional[Path] = None) -> List[dict]:
    global _rules
    if path is not None:
        with open(path) as f:
            return json.load(f)['rules']
    if _rules is None:
        with open(RULES_FILE) as f:
            _rules = json.load(f)['rules']
    return _rules


def choose_by_rules(features: Dict[str, float], rules: Optional[List[dict]] = None) -> Optional[str]:
    """
    First rule (in order) whose condition holds; a rule is {feature, op ('>' or '<'), threshold, method}
    """
    for rule in rules or load_rules():
        value = features[rule['feature']]
        if (rule['op'] == '>' and value > rule['threshold']) or (rule['op'] == '<' and value < rule['threshold']):
            return rule['method']
    return None


_brisque = None


def brisque_score(image: np.ndarray) -> float:
    global _brisque
    if _brisque is None:
        import piq
        import torch
        _brisque = (piq, torch)
    piq, torch = _brisque
    x = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        return float(piq.brisque(x, data_range=1.0, reduction='none').item())


def choose_by_iqa(image: np.ndarray, candidates: List[str] = IQA_CANDIDATES, crop: int = 320) -> Optional[str]:
    """
    Restoration whose output has the best BRISQUE score, computed on a central crop at the original resolution
    (downscaling would average away the noise and blocking that BRISQUE has to see)
    """
    h, w = image.shape[:2]
    top, left = max(0, (h - crop) // 2), max(0, (w - crop) // 2)
    patch = np.ascontiguousarray(image[top:top + crop, left:left + crop])
    scores = {}
    for method in candidates:
        candidate = patch if method == 'none' else restore(patch, method)
        scores[method] = brisque_score(candidate)
    best = min(scores, key=scores.get)
    return None if best == 'none' else best


def choose_restoration(image: np.ndarray, mode: str) -> Optional[str]:
    if mode == 'auto':
        return choose_by_rules(estimate_features(image))
    if mode == 'auto_iqa':
        return choose_by_iqa(image)
    raise ValueError(mode)
