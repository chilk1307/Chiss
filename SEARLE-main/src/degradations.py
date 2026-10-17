"""
Image degradations applied to the reference (query) images (plan 03, section 3.1).

    --degrade none | <type>:<level> | mixed          e.g. --degrade gauss_noise:2

Level 1 = mild, level 2 = strong. Images are degraded at their original resolution, before the CLIP preprocessing.
The random parts (noise, salt and pepper positions, motion direction, the choice made by 'mixed') are seeded by the
image name, so a run is reproducible and every method sees exactly the same degraded images.

    type          level 1 / level 2                                   degradation model
    gauss_noise   sigma 10 / 25 (0-255 scale)                         additive noise
    salt_pepper   2% / 10% of the pixels                              impulse noise
    gauss_blur    sigma 1 / 2 px                                      convolution with a Gaussian PSF
    motion_blur   kernel length 9 / 21 px, random angle               convolution with a known line PSF
    jpeg          quality 30 / 10                                     DCT quantisation
    low_light     gamma 2.0 + noise sigma 10 / gamma 3.0 + sigma 20   intensity transform
    low_res       downscale x2 / x4 then bicubic upscale              resampling
    color_cast    R x 1.2, B x 0.8 / R x 1.4, B x 0.6                 wrong white balance
"""
import hashlib
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

DEGRADATIONS = ['gauss_noise', 'salt_pepper', 'gauss_blur', 'motion_blur', 'jpeg', 'low_light', 'low_res',
                'color_cast']
LEVELS = {
    'gauss_noise': {1: {'sigma': 10.0}, 2: {'sigma': 25.0}},
    'salt_pepper': {1: {'amount': 0.02}, 2: {'amount': 0.10}},
    'gauss_blur': {1: {'sigma': 1.0}, 2: {'sigma': 2.0}},
    'motion_blur': {1: {'length': 9}, 2: {'length': 21}},
    'jpeg': {1: {'quality': 30}, 2: {'quality': 10}},
    'low_light': {1: {'gamma': 2.0, 'sigma': 10.0}, 2: {'gamma': 3.0, 'sigma': 20.0}},
    'low_res': {1: {'factor': 2}, 2: {'factor': 4}},
    'color_cast': {1: {'r': 1.2, 'b': 0.8}, 2: {'r': 1.4, 'b': 0.6}},
}
DEGRADATIONS_VERSION = 1  # bump when a degradation changes, so cached features are recomputed


def parse_degrade(spec: Optional[str]) -> Optional[Tuple[str, int]]:
    """
    'none' -> None; 'mixed' -> ('mixed', 0); '<type>:<level>' -> (type, level)
    """
    if spec is None or spec == 'none':
        return None
    if spec == 'mixed':
        return 'mixed', 0
    name, _, level = spec.partition(':')
    if name not in DEGRADATIONS or level not in ('1', '2'):
        raise ValueError(f"--degrade should be 'none', 'mixed' or <type>:<1|2> with type in {DEGRADATIONS}, "
                         f"got '{spec}'")
    return name, int(level)


def seeded_rng(*parts) -> np.random.Generator:
    """
    Random generator seeded by the image name (and the operation), stable across runs and machines
    """
    digest = hashlib.sha1('|'.join(str(p) for p in parts).encode()).hexdigest()
    return np.random.default_rng(int(digest[:16], 16))


def gaussian_psf(sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1)
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    psf = np.outer(g, g)
    return psf / psf.sum()


def motion_psf(length: int, angle_deg: float) -> np.ndarray:
    """
    Line of `length` pixels through the centre of a length x length kernel, rotated by `angle_deg`
    """
    psf = np.zeros((length, length), dtype=np.float64)
    psf[length // 2, :] = 1.0
    rotation = cv2.getRotationMatrix2D(((length - 1) / 2, (length - 1) / 2), angle_deg, 1.0)
    psf = cv2.warpAffine(psf, rotation, (length, length), flags=cv2.INTER_LINEAR)
    return psf / psf.sum()


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def degrade(image: np.ndarray, kind: str, level: int, name: str = '') -> Tuple[np.ndarray, Dict]:
    """
    Degrade an RGB uint8 image. Returns the degraded image and a context dict (the degradation and, for blurs, the
    PSF, which the non-blind restorations wiener / richardson_lucy use).
    """
    p = LEVELS[kind][level]
    rng = seeded_rng(name, kind, level)
    context = {'degradation': f"{kind}:{level}"}
    x = image.astype(np.float64)
    if kind == 'gauss_noise':
        out = _clip(x + rng.normal(0, p['sigma'], x.shape))
    elif kind == 'salt_pepper':
        out = image.copy()
        mask = rng.random(image.shape[:2]) < p['amount']
        salt = rng.random(image.shape[:2]) < 0.5
        out[mask & salt] = 255
        out[mask & ~salt] = 0
    elif kind == 'gauss_blur':
        psf = gaussian_psf(p['sigma'])
        out = cv2.filter2D(image, -1, psf, borderType=cv2.BORDER_REFLECT)
        context['psf'] = psf
    elif kind == 'motion_blur':
        psf = motion_psf(p['length'], float(rng.uniform(0, 180)))
        out = cv2.filter2D(image, -1, psf, borderType=cv2.BORDER_REFLECT)
        context['psf'] = psf
    elif kind == 'jpeg':
        ok, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                  [cv2.IMWRITE_JPEG_QUALITY, int(p['quality'])])
        out = cv2.cvtColor(cv2.imdecode(buffer, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    elif kind == 'low_light':
        out = _clip(255.0 * (x / 255.0) ** p['gamma'] + rng.normal(0, p['sigma'], x.shape))
    elif kind == 'low_res':
        h, w = image.shape[:2]
        small = cv2.resize(image, (max(1, w // p['factor']), max(1, h // p['factor'])), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    elif kind == 'color_cast':
        out = _clip(x * np.array([p['r'], 1.0, p['b']]))
    else:
        raise ValueError(f"unknown degradation {kind}")
    return out, context


def apply_degrade_spec(image: np.ndarray, spec: Tuple[str, int], name: str) -> Tuple[np.ndarray, Dict]:
    """
    Apply a parsed --degrade spec; 'mixed' picks one degradation and one level per image (seeded by the name)
    """
    kind, level = spec
    if kind == 'mixed':
        rng = seeded_rng(name, 'mixed')
        kind = DEGRADATIONS[int(rng.integers(len(DEGRADATIONS)))]
        level = int(rng.integers(1, 3))
    return degrade(image, kind, level, name)
