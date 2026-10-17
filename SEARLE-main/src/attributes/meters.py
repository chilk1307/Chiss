"""
Attribute meters with classical image processing (plan 02, section 2.2). Images are resized so that the long side is
256 px.

Object mask (--attr-mask)
    otsu      white-background images (FashionIQ): Otsu threshold on max(saturation, 1 - value), opening / closing,
              largest connected component, holes filled
    grabcut   GrabCut initialised with a rectangle covering the centre of the image (5% margin)
    saliency  spectral residual saliency (Hou & Zhang, CVPR 2007), Otsu threshold, largest component
    auto      otsu when the image border is almost white, grabcut otherwise (spectral residual saliency marks the
              unusual details of an image rather than whole objects, see E2.1)
    none      the whole image
Measurements (on the object pixels, and on the whole image for constraints about the background)
    dominant colours   k-means (k = 4) in CIELAB: centres and pixel shares
    brightness         median L*
    colourfulness      Hasler & Suesstrunk (2003) M = sigma_rgyb + 0.3 mu_rgyb; mean HSV saturation
    pattern            Canny edge density inside the (eroded) mask; peak-to-background ratio of the strongest peak of
                       the Fourier spectrum within its frequency ring (periodic stripes give a sharp peak) and its
                       angle; entropy of the uniform LBP histogram; energy of 4 Gabor orientations
    zoom               mask area / image area

Satisfaction c in [0, 1] of a constraint (satisfaction()):
    colour (absolute)       sum_k p_k exp(-dE2000(centre_k, colour) / TAU_COLOR), dE to the closest prototype
    darker / brighter       sigmoid((L_ref - L_cand - DELTA_L) / TAU_L) and the opposite
    more / less colourful   sigmoid((M_cand - M_ref - DELTA_M) / TAU_M) and the opposite
    black and white         sigmoid((GRAY_M - M_cand) / TAU_GRAY)
    striped                 sigmoid((log10 peak ratio - PEAK0) / TAU_PEAK), times cos^2 of the angle error if an
                            orientation is requested
    patterned, floral...    max(sigmoid((edge density - EDGE0) / TAU_EDGE), striped); solid = 1 - patterned
    closer / farther        sigmoid((area_cand - area_ref - DELTA_A) / TAU_A) and the opposite
    negated constraint      1 - c
"""
import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from attributes.parser import Constraint

MASK_MODES = ['none', 'otsu', 'grabcut', 'saliency', 'auto']
LONG_SIDE = 256
N_COLORS = 4
METERS_VERSION = 1  # bump when a measurement changes, so cached attributes are recomputed

# sRGB prototypes of every colour name (several shades), converted to CIELAB below
COLOR_PROTOTYPES_RGB = {
    'red': [(200, 30, 40), (150, 20, 30), (230, 60, 60)],
    'maroon': [(110, 20, 35), (90, 30, 40)],
    'blue': [(40, 80, 200), (30, 60, 140), (110, 160, 220), (70, 130, 200)],
    'navy': [(25, 35, 80), (35, 45, 100)],
    'teal': [(0, 128, 128), (60, 180, 180), (64, 224, 208)],
    'green': [(40, 140, 60), (20, 90, 40), (110, 190, 110)],
    'olive': [(110, 110, 50), (85, 90, 40)],
    'yellow': [(240, 220, 50), (220, 190, 40), (250, 240, 140)],
    'gold': [(212, 175, 55), (200, 160, 60)],
    'orange': [(240, 140, 40), (230, 110, 30), (250, 170, 90)],
    'pink': [(240, 150, 180), (230, 100, 150), (250, 200, 210)],
    'purple': [(120, 60, 150), (90, 40, 120), (170, 120, 200)],
    'brown': [(120, 80, 50), (90, 60, 35), (150, 105, 70)],
    'beige': [(220, 200, 170), (200, 175, 140), (235, 220, 195)],
    'black': [(20, 20, 20), (40, 40, 45)],
    'white': [(245, 245, 245), (230, 230, 225)],
    'gray': [(128, 128, 128), (90, 90, 90), (180, 180, 180)],
}

# Parameters of the satisfaction functions (chosen with the synthetic tests of tools/test_meters.py, E2.1)
TAU_COLOR = 15.0
DELTA_L, TAU_L = 2.0, 4.0
DELTA_M, TAU_M = 2.0, 5.0
GRAY_M, TAU_GRAY = 10.0, 3.0
PEAK0, TAU_PEAK = 3.0, 0.5  # on log10 of the spectral peak-to-background ratio
EDGE0, TAU_EDGE = 0.05, 0.015
CANNY_LOW, CANNY_HIGH = 30, 90
DELTA_A, TAU_A = 0.02, 0.05

# Layout of the measurement vector (cached as a float array)
LAYOUT = {
    'obj_centers': N_COLORS * 3, 'obj_weights': N_COLORS, 'img_centers': N_COLORS * 3, 'img_weights': N_COLORS,
    'l_obj': 1, 'l_img': 1, 'm_obj': 1, 'm_img': 1, 'sat_obj': 1, 'sat_img': 1,
    'edge_density': 1, 'fft_peak': 1, 'fft_angle': 1, 'lbp_entropy': 1, 'gabor': 4, 'area': 1, 'mask_kind': 1,
}
VECTOR_SIZE = sum(LAYOUT.values())
MASK_KIND_CODE = {'none': 0, 'otsu': 1, 'grabcut': 2, 'saliency': 3}


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


PROTOTYPES_LAB = {name: _rgb_to_lab(np.array([shades], dtype=np.uint8))[0]
                  for name, shades in COLOR_PROTOTYPES_RGB.items()}


def resize_long_side(rgb: np.ndarray, size: int = LONG_SIDE) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    if scale >= 1:
        return rgb
    return cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


# Masks ----------------------------------------------------------------------------------------------------------------

def _largest_component(mask: np.ndarray, fill_holes: bool = True) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask.astype(bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = labels == largest
    if fill_holes:
        inverse = (~out).astype(np.uint8)
        _, bg_labels, _, _ = cv2.connectedComponentsWithStats(inverse, connectivity=4)
        border = set(np.unique(np.concatenate([bg_labels[0], bg_labels[-1], bg_labels[:, 0], bg_labels[:, -1]])))
        holes = (inverse == 1) & ~np.isin(bg_labels, list(border))
        out = out | holes
    return out


def _clean(mask: np.ndarray, close_fraction: float = 0.03) -> np.ndarray:
    """
    Opening (removes specks), closing with a kernel proportional to the image size (joins parts of the object), then
    the largest connected component with its holes filled
    """
    size = max(7, int(round(close_fraction * max(mask.shape))) | 1)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_open)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_close)
    return _largest_component(m > 0)


def otsu_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    score = np.maximum(hsv[..., 1], 1.0 - hsv[..., 2])  # far from white: saturated or dark
    score8 = np.clip(score * 255, 0, 255).astype(np.uint8)
    threshold, _ = cv2.threshold(score8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # a large closing joins light parts of the garment (white stripes, prints) that look like the background
    return _clean(score8 > max(threshold, 20), close_fraction=0.08)


def grabcut_mask(rgb: np.ndarray, iterations: int = 3) -> np.ndarray:
    h, w = rgb.shape[:2]
    mx, my = max(1, int(0.05 * w)), max(1, int(0.05 * h))
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(0)
    try:
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, (mx, my, w - 2 * mx, h - 2 * my), bgd, fgd,
                    iterations, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.ones((h, w), bool)
    return _clean((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD))


def spectral_residual_saliency(rgb: np.ndarray, size: int = 64) -> np.ndarray:
    """
    Hou & Zhang (2007): the saliency map is the inverse Fourier transform of the spectral residual (log amplitude minus
    its local average) with the original phase, squared and smoothed
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    spectrum = np.fft.fft2(small)
    log_amplitude = np.log(np.abs(spectrum) + 1e-8)
    phase = np.angle(spectrum)
    residual = log_amplitude - cv2.blur(log_amplitude, (3, 3))
    saliency = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
    saliency = cv2.GaussianBlur(saliency.astype(np.float32), (0, 0), 3)
    saliency = cv2.resize(saliency, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return saliency / max(float(saliency.max()), 1e-12)


def saliency_mask(rgb: np.ndarray) -> np.ndarray:
    """
    Otsu threshold of the saliency map (the 3 x mean threshold of the paper keeps only a few small peaks once the map is
    smoothed, see E2.1), then the largest component
    """
    saliency8 = (spectral_residual_saliency(rgb) * 255).astype(np.uint8)
    threshold, _ = cv2.threshold(saliency8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _clean(saliency8 > threshold)


def white_border(rgb: np.ndarray, width: float = 0.04) -> bool:
    h, w = rgb.shape[:2]
    bh, bw = max(1, int(width * h)), max(1, int(width * w))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    border = np.concatenate([hsv[:bh].reshape(-1, 3), hsv[-bh:].reshape(-1, 3), hsv[:, :bw].reshape(-1, 3),
                             hsv[:, -bw:].reshape(-1, 3)])
    return float(((border[:, 2] > 0.9) & (border[:, 1] < 0.12)).mean()) > 0.7


def object_mask(rgb: np.ndarray, mode: str = 'auto') -> Tuple[np.ndarray, str]:
    """
    Boolean object mask and the method actually used; falls back to the whole image when the mask is implausible
    """
    if mode == 'auto':
        mode = 'otsu' if white_border(rgb) else 'grabcut'
    if mode == 'none':
        return np.ones(rgb.shape[:2], bool), 'none'
    mask = {'otsu': otsu_mask, 'grabcut': grabcut_mask, 'saliency': saliency_mask}[mode](rgb)
    fraction = float(mask.mean())
    if fraction < 0.02 or fraction > 0.98:
        return np.ones(rgb.shape[:2], bool), 'none'
    return mask, mode


# Measurements -------------------------------------------------------------------------------------------------------

def dominant_colors(lab_pixels: np.ndarray, k: int = N_COLORS, max_pixels: int = 4000) -> Tuple[np.ndarray, np.ndarray]:
    """
    k-means in CIELAB; returns (k x 3 centres, k shares) sorted by share; empty clusters have share 0
    """
    pixels = lab_pixels.reshape(-1, 3).astype(np.float32)
    if len(pixels) > max_pixels:
        pixels = pixels[np.random.default_rng(0).choice(len(pixels), max_pixels, replace=False)]
    centers = np.zeros((k, 3), np.float32)
    weights = np.zeros(k, np.float32)
    if len(pixels) == 0:
        return centers, weights
    n = min(k, len(pixels))
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, found = cv2.kmeans(pixels, n, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    shares = np.bincount(labels.ravel(), minlength=n).astype(np.float32) / len(pixels)
    order = np.argsort(-shares)
    centers[:n], weights[:n] = found[order], shares[order]
    return centers, weights


def colorfulness(rgb_pixels: np.ndarray) -> float:
    """
    Hasler & Suesstrunk (2003) colourfulness on 0-255 RGB pixels
    """
    p = rgb_pixels.reshape(-1, 3).astype(np.float64)
    if len(p) == 0:
        return 0.0
    rg = p[:, 0] - p[:, 1]
    yb = 0.5 * (p[:, 0] + p[:, 1]) - p[:, 2]
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def fft_peak(gray: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """
    Periodicity of the texture: log10 of the peak-to-background ratio of the strongest spectral peak, i.e. the mean
    power around the peak divided by the median power of the ring of frequencies with the same radius (excluding the
    peak and its symmetric twin). Natural images spread the energy of a ring over all orientations (log10 ratio about
    0-2.5 on the sample photos); stripes concentrate it in one direction (> 5 on synthetic stripes). Also returns the
    angle of the peak frequency in degrees in [0, 180): vertical stripes give an angle near 0, horizontal near 90.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) < 64:
        return 0.0, 0.0
    crop = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float64)
    crop_mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    if min(h, w) < 24:
        return 0.0, 0.0
    # Soft window: Hann window times the blurred mask, so that the object outline does not create spectral lines
    weight = cv2.GaussianBlur(crop_mask.astype(np.float32), (0, 0), 4).astype(np.float64) * \
        np.outer(np.hanning(h), np.hanning(w))
    crop = (crop - crop[crop_mask].mean()) * weight
    power = np.abs(np.fft.fftshift(np.fft.fft2(crop))) ** 2
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[:h, :w]
    radius = np.hypot((yy - cy) / h, (xx - cx) / w)
    valid = radius > max(2.0 / min(h, w), 0.02)  # at least two periods in the crop; no DC / illumination
    if not valid.any():
        return 0.0, 0.0
    py, px = np.unravel_index(np.argmax(np.where(valid, power, 0)), power.shape)
    r = radius[py, px]
    near = (np.abs(yy - py) <= 3) & (np.abs(xx - px) <= 3)
    twin = (np.abs(yy - (2 * cy - py)) <= 3) & (np.abs(xx - (2 * cx - px)) <= 3)
    ring = valid & (radius > 0.75 * r) & (radius < 1.25 * r) & ~near & ~twin
    if ring.sum() < 20:
        return 0.0, 0.0
    peak = power[max(0, py - 1):py + 2, max(0, px - 1):px + 2].mean()
    ratio = peak / max(float(np.median(power[ring])), 1e-12)
    angle = math.degrees(math.atan2((py - cy) / h, (px - cx) / w)) % 180.0
    return float(np.log10(max(ratio, 1.0))), float(angle)


def lbp_entropy(gray: np.ndarray, mask: np.ndarray) -> float:
    from skimage.feature import local_binary_pattern
    codes = local_binary_pattern(gray, 8, 1, method='uniform').astype(np.int64)[mask]
    if len(codes) == 0:
        return 0.0
    p = np.bincount(codes, minlength=10).astype(np.float64)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def gabor_orientations(gray: np.ndarray, mask: np.ndarray, wavelength: float = 8.0) -> np.ndarray:
    g = gray.astype(np.float32) / 255.0
    energies = []
    for theta in (0, math.pi / 4, math.pi / 2, 3 * math.pi / 4):
        kernel = cv2.getGaborKernel((31, 31), 0.56 * wavelength, theta, wavelength, 0.5, 0, ktype=cv2.CV_32F)
        kernel -= kernel.mean()
        energies.append(float(np.abs(cv2.filter2D(g, cv2.CV_32F, kernel))[mask].mean()) if mask.any() else 0.0)
    e = np.array(energies)
    return e / max(e.sum(), 1e-12)


def measure(rgb: np.ndarray, mask_mode: str = 'auto') -> np.ndarray:
    """
    All measurements of an RGB uint8 image, as a float vector with layout LAYOUT
    """
    rgb = resize_long_side(np.ascontiguousarray(rgb))
    mask, used = object_mask(rgb, mask_mode)
    lab = _rgb_to_lab(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv_s = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32) / 255.0
    inner = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if inner.sum() < 64:
        inner = mask
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH) > 0
    obj_centers, obj_weights = dominant_colors(lab[mask])
    img_centers, img_weights = dominant_colors(lab)
    peak, angle = fft_peak(gray, inner)
    values = {
        'obj_centers': obj_centers.ravel(), 'obj_weights': obj_weights,
        'img_centers': img_centers.ravel(), 'img_weights': img_weights,
        'l_obj': [np.median(lab[..., 0][mask])], 'l_img': [np.median(lab[..., 0])],
        'm_obj': [colorfulness(rgb[mask])], 'm_img': [colorfulness(rgb)],
        'sat_obj': [hsv_s[mask].mean()], 'sat_img': [hsv_s.mean()],
        'edge_density': [edges[inner].mean()], 'fft_peak': [peak], 'fft_angle': [angle],
        'lbp_entropy': [lbp_entropy(gray, mask)], 'gabor': gabor_orientations(gray, mask),
        'area': [mask.mean()], 'mask_kind': [MASK_KIND_CODE[used]],
    }
    return np.concatenate([np.asarray(values[k], dtype=np.float32).ravel() for k in LAYOUT])


SLICES = {}
_start = 0
for _key, _size in LAYOUT.items():
    SLICES[_key] = slice(_start, _start + _size)
    _start += _size


def unpack(vector: np.ndarray) -> Dict[str, np.ndarray]:
    out = {k: np.asarray(vector[s]) for k, s in SLICES.items()}
    out['obj_centers'] = out['obj_centers'].reshape(N_COLORS, 3)
    out['img_centers'] = out['img_centers'].reshape(N_COLORS, 3)
    return out


def pack(values: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(values[k], dtype=np.float32).ravel() for k in LAYOUT])


# Satisfaction -------------------------------------------------------------------------------------------------------
# Vectorised over K candidates: cand is a (K, VECTOR_SIZE) array of measurements, ref one vector (or None)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def color_distance(centers: np.ndarray, color: str) -> np.ndarray:
    """
    CIEDE2000 distance of each centre (array (..., 3) in CIELAB) to the closest prototype of the colour
    """
    from skimage.color import deltaE_ciede2000
    prototypes = PROTOTYPES_LAB[color].astype(np.float64)
    a, b = np.broadcast_arrays(centers[..., None, :].astype(np.float64), prototypes)
    return deltaE_ciede2000(a, b).min(axis=-1)


def _column(cand: np.ndarray, key: str) -> np.ndarray:
    return cand[:, SLICES[key]][:, 0]


def color_scores(cand: np.ndarray, color: str, region: str = 'object') -> np.ndarray:
    prefix = 'obj' if region == 'object' else 'img'
    centers = cand[:, SLICES[f'{prefix}_centers']].reshape(-1, N_COLORS, 3)
    weights = cand[:, SLICES[f'{prefix}_weights']]
    return (weights * np.exp(-color_distance(centers, color) / TAU_COLOR)).sum(axis=1)


def striped_scores(cand: np.ndarray, orientation: Optional[str] = None) -> np.ndarray:
    score = _sigmoid((_column(cand, 'fft_peak') - PEAK0) / TAU_PEAK)
    if orientation is not None:
        target = 0.0 if orientation == 'vertical' else 90.0
        score = score * np.cos(np.radians(_column(cand, 'fft_angle') - target)) ** 2
    return score


def patterned_scores(cand: np.ndarray) -> np.ndarray:
    """
    Any texture: many edges, or a periodic structure (smooth wide stripes have few edges but a sharp Fourier peak)
    """
    return np.maximum(_sigmoid((_column(cand, 'edge_density') - EDGE0) / TAU_EDGE), striped_scores(cand))


def satisfaction_batch(c: Constraint, cand: np.ndarray, ref: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Degree in [0, 1] to which each candidate satisfies the constraint (None if it cannot be measured, e.g. a relative
    constraint without measurements of the reference image)
    """
    cand = np.atleast_2d(np.asarray(cand, dtype=np.float64))
    ref = None if ref is None else np.asarray(ref, dtype=np.float64)[None]
    obj = c.region == 'object'
    if c.family == 'color':
        value = color_scores(cand, c.value, c.region)
    elif c.family == 'brightness':
        if ref is None:
            return None
        key = 'l_obj' if obj else 'l_img'
        diff = _column(ref, key) - _column(cand, key)  # > 0 when the candidate is darker
        value = _sigmoid((diff - DELTA_L) / TAU_L) if c.value == 'darker' else _sigmoid((-diff - DELTA_L) / TAU_L)
    elif c.family == 'colorfulness':
        key = 'm_obj' if obj else 'm_img'
        if c.value == 'grayscale':
            value = _sigmoid((GRAY_M - _column(cand, 'm_img')) / TAU_GRAY)
        else:
            if ref is None:
                return None
            diff = _column(cand, key) - _column(ref, key)
            value = _sigmoid((diff - DELTA_M) / TAU_M) if c.value == 'more' else _sigmoid((-diff - DELTA_M) / TAU_M)
    elif c.family == 'pattern':
        if c.value == 'striped':
            value = striped_scores(cand, c.orientation)
        elif c.value == 'solid':
            value = 1.0 - patterned_scores(cand)
        else:  # patterned, floral, plaid, dotted
            value = patterned_scores(cand)
    elif c.family == 'zoom':
        if ref is None:
            return None
        diff = _column(cand, 'area') - _column(ref, 'area')
        value = _sigmoid((diff - DELTA_A) / TAU_A) if c.value == 'closer' else _sigmoid((-diff - DELTA_A) / TAU_A)
    else:
        return None
    return 1.0 - value if c.negated else value


def satisfaction(c: Constraint, cand: Dict[str, np.ndarray], ref: Optional[Dict[str, np.ndarray]]) -> Optional[float]:
    """
    Single-image version of satisfaction_batch, on measurements unpacked with unpack()
    """
    values = satisfaction_batch(c, pack(cand)[None], None if ref is None else pack(ref))
    return None if values is None else float(values[0])


def color_score(m: Dict[str, np.ndarray], color: str, region: str = 'object') -> float:
    return float(color_scores(pack(m)[None].astype(np.float64), color, region)[0])


def explain(c: Constraint, cand: Dict[str, np.ndarray], ref: Optional[Dict[str, np.ndarray]]) -> str:
    """
    Short human-readable reason, e.g. "darker: L* 62 -> 48 (-23%)"
    """
    obj = c.region == 'object'
    if c.family == 'brightness' and ref is not None:
        key = 'l_obj' if obj else 'l_img'
        before, after = float(ref[key][0]), float(cand[key][0])
        return f"{c.describe()}: L* {before:.0f} -> {after:.0f} ({100 * (after - before) / max(before, 1):+.0f}%)"
    if c.family == 'colorfulness':
        key = 'm_obj' if obj else 'm_img'
        if ref is not None and c.value != 'grayscale':
            return f"{c.describe()}: colourfulness {float(ref[key][0]):.0f} -> {float(cand[key][0]):.0f}"
        return f"{c.describe()}: colourfulness {float(cand['m_img'][0]):.0f}"
    if c.family == 'color':
        prefix = 'obj' if obj else 'img'
        d = color_distance(cand[f'{prefix}_centers'], c.value)
        k = int(np.argmin(d))
        return f"{c.describe()}: closest colour dE {d[k]:.0f} on {100 * float(cand[f'{prefix}_weights'][k]):.0f}% " \
               f"of the pixels"
    if c.family == 'pattern':
        return f"{c.describe()}: edges {100 * float(cand['edge_density'][0]):.1f}%, Fourier peak " \
               f"{float(cand['fft_peak'][0]):.2f} at {float(cand['fft_angle'][0]):.0f} deg"
    if c.family == 'zoom' and ref is not None:
        return f"{c.describe()}: object area {100 * float(ref['area'][0]):.0f}% -> {100 * float(cand['area'][0]):.0f}%"
    return c.describe()
