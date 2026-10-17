"""
Classical image restorations applied to the reference (query) images (plan 03, section 3.2).

    --restore none | <method> | oracle | auto | auto_iqa

    method           for                     image-processing topic
    median           salt and pepper noise   non-linear filtering
    gaussian         noise, JPEG             linear spatial filtering
    bilateral        noise, JPEG             edge-preserving filtering
    nlm              Gaussian noise          non-local means denoising (strength from the estimated noise sigma)
    wiener           blur (known PSF)        frequency-domain deconvolution
    richardson_lucy  blur (known PSF)        iterative deconvolution
    unsharp          blur, low resolution    sharpening
    histeq           low light               histogram equalisation (luma channel)
    clahe            low light               contrast-limited adaptive histogram equalisation (L* channel)
    gamma            low light               gamma correction, gamma chosen so the mean luma becomes 0.5
    retinex          low light               single-scale Retinex
    gray_world       colour cast             colour constancy, grey-world assumption
    white_patch      colour cast             colour constancy, white-patch (99th percentile) assumption
    deblock          JPEG                    smoothing across the 8 x 8 block boundaries

wiener and richardson_lucy are non-blind: they use the PSF of the degradation when it is known (gauss_blur,
motion_blur); otherwise a Gaussian PSF with sigma 1 px is assumed.
'oracle' uses, for each degradation, the method listed in restore_oracle.json (filled from experiment E3.2).
'auto' and 'auto_iqa' are implemented in degradation_estimation.py.
"""
import json
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from degradations import gaussian_psf

RESTORATIONS = ['median', 'gaussian', 'bilateral', 'nlm', 'wiener', 'richardson_lucy', 'unsharp', 'histeq', 'clahe',
                'gamma', 'retinex', 'gray_world', 'white_patch', 'deblock']
SPECIAL_RESTORATIONS = ['oracle', 'auto', 'auto_iqa']
RESTORATIONS_VERSION = 1  # bump when a restoration changes, so cached features are recomputed
ORACLE_FILE = Path(__file__).with_name('restore_oracle.json')


def check_restore(spec: Optional[str]) -> Optional[str]:
    if spec is None or spec == 'none':
        return None
    if spec not in RESTORATIONS + SPECIAL_RESTORATIONS:
        raise ValueError(f"--restore should be 'none', one of {RESTORATIONS} or one of {SPECIAL_RESTORATIONS}, "
                         f"got '{spec}'")
    return spec


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def _per_channel(image: np.ndarray, fn) -> np.ndarray:
    return np.stack([fn(image[..., c]) for c in range(image.shape[2])], axis=-1)


def estimate_noise_sigma(image: np.ndarray) -> float:
    """
    Noise standard deviation (0-255 scale) from the median absolute deviation of the finest wavelet detail
    coefficients (Donoho), as in skimage.restoration.estimate_sigma
    """
    from skimage.restoration import estimate_sigma
    return float(estimate_sigma(image.astype(np.float64), channel_axis=-1, average_sigmas=True))


# Spatial filters ----------------------------------------------------------------------------------------------------

def median(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.medianBlur(image, ksize)


def gaussian(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), sigma)


def bilateral(image: np.ndarray, diameter: int = 7, sigma_color: float = 50, sigma_space: float = 7) -> np.ndarray:
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def nlm(image: np.ndarray, h: Optional[float] = None) -> np.ndarray:
    """
    Non-local means; the filtering strength follows the estimated noise level unless h is given
    """
    if h is None:
        h = float(np.clip(0.8 * estimate_noise_sigma(image), 3, 30))
    return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)


def unsharp(image: np.ndarray, sigma: float = 1.5, amount: float = 1.0) -> np.ndarray:
    blurred = cv2.GaussianBlur(image.astype(np.float64), (0, 0), sigma)
    return _clip(image + amount * (image - blurred))


# Deconvolution ------------------------------------------------------------------------------------------------------

def _psf_to_otf(psf: np.ndarray, shape) -> np.ndarray:
    """
    Zero-pad the PSF to the image size and move its centre to (0, 0), so that the FFT gives the transfer function
    """
    padded = np.zeros(shape, dtype=np.float64)
    kh, kw = psf.shape
    padded[:kh, :kw] = psf
    padded = np.roll(padded, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return np.fft.rfft2(padded)[..., None]


def _pad_for_fft(image: np.ndarray, pad: int) -> np.ndarray:
    """
    Reflection padding of `pad` pixels (limits the wrap-around of the circular convolution), extended at the bottom
    and right to sizes whose FFT is fast
    """
    from scipy.fft import next_fast_len
    h, w = image.shape[:2]
    fast_h, fast_w = next_fast_len(h + 2 * pad, real=True), next_fast_len(w + 2 * pad, real=True)
    return cv2.copyMakeBorder(image, pad, fast_h - h - pad, pad, fast_w - w - pad, cv2.BORDER_REFLECT)


def wiener(image: np.ndarray, psf: Optional[np.ndarray] = None, k: float = 0.01) -> np.ndarray:
    """
    Wiener deconvolution F = conj(H) / (|H|^2 + K) G, with the constant K approximating the noise-to-signal power
    ratio
    """
    psf = gaussian_psf(1.0) if psf is None else psf
    pad = max(psf.shape)
    h, w = image.shape[:2]
    x = _pad_for_fft(image, pad).astype(np.float64) / 255.0
    otf = _psf_to_otf(psf, x.shape[:2])
    gain = np.conj(otf) / (np.abs(otf) ** 2 + k)
    restored = np.fft.irfft2(np.fft.rfft2(x, axes=(0, 1)) * gain, s=x.shape[:2], axes=(0, 1))
    return _clip(restored[pad:pad + h, pad:pad + w] * 255.0)


def richardson_lucy(image: np.ndarray, psf: Optional[np.ndarray] = None, iterations: int = 20) -> np.ndarray:
    """
    Richardson-Lucy iterations u <- u * (flipped PSF (*) (g / (PSF (*) u))), computed with FFT convolutions
    """
    from scipy import fft  # single-precision FFTs: about twice as fast as numpy.fft, accurate enough here
    psf = gaussian_psf(1.0) if psf is None else psf
    pad = max(psf.shape)
    h, w = image.shape[:2]
    g = _pad_for_fft(image, pad).astype(np.float32) / 255.0 + 1e-3
    otf = _psf_to_otf(psf, g.shape[:2]).astype(np.complex64)
    otf_conj = np.conj(otf)
    u = np.full_like(g, 0.5)
    for _ in range(iterations):
        blurred = fft.irfft2(fft.rfft2(u, axes=(0, 1)) * otf, s=g.shape[:2], axes=(0, 1))
        ratio = g / np.maximum(blurred, 1e-6)
        u = u * fft.irfft2(fft.rfft2(ratio, axes=(0, 1)) * otf_conj, s=g.shape[:2], axes=(0, 1))
    return _clip(u[pad:pad + h, pad:pad + w] * 255.0)


# Intensity transforms -----------------------------------------------------------------------------------------------

def histeq(image: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def clahe(image: np.ndarray, clip_limit: float = 2.0, tiles: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tiles, tiles)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def gamma(image: np.ndarray, target_mean: float = 0.5) -> np.ndarray:
    """
    Brightening gamma correction: gamma = log(target) / log(mean luma), limited to [0.3, 1]
    """
    luma = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    mean = float(np.clip(luma.mean(), 1e-3, 0.999))
    g = float(np.clip(np.log(target_mean) / np.log(mean), 0.3, 1.0))
    table = np.clip(255.0 * (np.arange(256) / 255.0) ** g, 0, 255).astype(np.uint8)
    return cv2.LUT(image, table)


def _stretch(channel: np.ndarray, low: float = 1, high: float = 99) -> np.ndarray:
    lo, hi = np.percentile(channel, [low, high])
    return np.clip((channel - lo) / max(hi - lo, 1e-6), 0, 1)


def retinex(image: np.ndarray, sigma: float = 80) -> np.ndarray:
    """
    Single-scale Retinex: log(I) - log(Gaussian * I) per channel, then a 1%-99% contrast stretch
    """
    x = image.astype(np.float64) + 1.0
    reflectance = np.log(x) - np.log(cv2.GaussianBlur(x, (0, 0), sigma))
    return _clip(_per_channel(reflectance, _stretch) * 255.0)


# Colour constancy ---------------------------------------------------------------------------------------------------

def gray_world(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float64)
    means = x.reshape(-1, 3).mean(axis=0)
    return _clip(x * (means.mean() / np.maximum(means, 1e-6)))


def white_patch(image: np.ndarray, percentile: float = 99) -> np.ndarray:
    x = image.astype(np.float64)
    whites = np.percentile(x.reshape(-1, 3), percentile, axis=0)
    return _clip(x * (255.0 / np.maximum(whites, 1.0)))


# Compression artefacts ----------------------------------------------------------------------------------------------

def deblock(image: np.ndarray, block: int = 8, threshold: float = 20) -> np.ndarray:
    """
    Simple deblocking filter: across every 8 x 8 block boundary, when the step between the two pixels next to the
    boundary is small (a blocking artefact rather than a real edge), move the two pixels towards each other and their
    neighbours a little less
    """
    x = image.astype(np.float64)
    for axis in (1, 0):  # vertical boundaries (columns), then horizontal boundaries (rows)
        x = np.moveaxis(x, axis, 0)
        for b in range(block, x.shape[0] - 1, block):
            p1, p0, q0, q1 = x[b - 2], x[b - 1], x[b], x[b + 1]
            delta = q0 - p0
            mask = np.abs(delta) < threshold
            x[b - 1] = np.where(mask, p0 + delta / 4, p0)
            x[b] = np.where(mask, q0 - delta / 4, q0)
            x[b - 2] = np.where(mask, p1 + delta / 8, p1)
            x[b + 1] = np.where(mask, q1 - delta / 8, q1)
        x = np.moveaxis(x, 0, axis)
    return _clip(x)


FUNCTIONS = {
    'median': lambda im, ctx: median(im),
    'gaussian': lambda im, ctx: gaussian(im),
    'bilateral': lambda im, ctx: bilateral(im),
    'nlm': lambda im, ctx: nlm(im),
    'wiener': lambda im, ctx: wiener(im, ctx.get('psf')),
    'richardson_lucy': lambda im, ctx: richardson_lucy(im, ctx.get('psf')),
    'unsharp': lambda im, ctx: unsharp(im),
    'histeq': lambda im, ctx: histeq(im),
    'clahe': lambda im, ctx: clahe(im),
    'gamma': lambda im, ctx: gamma(im),
    'retinex': lambda im, ctx: retinex(im),
    'gray_world': lambda im, ctx: gray_world(im),
    'white_patch': lambda im, ctx: white_patch(im),
    'deblock': lambda im, ctx: deblock(im),
}


def restore(image: np.ndarray, method: str, context: Optional[Dict] = None) -> np.ndarray:
    return FUNCTIONS[method](image, context or {})


_oracle = None


def oracle_method(context: Dict) -> Optional[str]:
    """
    Best fixed restoration for the degradation that was applied (from restore_oracle.json); None if no degradation
    """
    global _oracle
    if _oracle is None:
        with open(ORACLE_FILE) as f:
            _oracle = json.load(f)['methods']
    degradation = context.get('degradation')
    if degradation is None:
        return None
    kind, _, level = degradation.partition(':')
    method = _oracle.get(f"{kind}:{level}", _oracle.get(kind))
    return None if method in (None, 'none') else method
