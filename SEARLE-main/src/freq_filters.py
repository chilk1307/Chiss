"""
Frequency-domain filtering of the reference images (plan 03, experiment E3.5).

    --freq-filter none | lowpass:<r> | highpass:<r>      r = cut-off frequency as a fraction of the Nyquist frequency

Gaussian filters (Gonzalez & Woods): H_lp(u, v) = exp(-D(u, v)^2 / (2 D0^2)) with D the distance to the centre of the
spectrum in units of the Nyquist frequency and D0 = r; H_hp = 1 - H_lp. The high-pass filter keeps the DC component
(the mean colour), otherwise the image would be almost black after clipping to [0, 255].
"""
from typing import Optional, Tuple

import numpy as np

FREQ_FILTERS_VERSION = 1


def parse_freq_filter(spec: Optional[str]) -> Optional[Tuple[str, float]]:
    if spec is None or spec == 'none':
        return None
    kind, _, value = spec.partition(':')
    try:
        cutoff = float(value)
    except ValueError:
        cutoff = -1
    if kind not in ('lowpass', 'highpass') or not 0 < cutoff <= 1:
        raise ValueError(f"--freq-filter should be 'none', 'lowpass:<r>' or 'highpass:<r>' with 0 < r <= 1, "
                         f"got '{spec}'")
    return kind, cutoff


def gaussian_transfer(shape, cutoff: float, highpass: bool) -> np.ndarray:
    h, w = shape
    fy = np.fft.fftfreq(h)[:, None] / 0.5  # frequencies in units of the Nyquist frequency (0.5 cycle / pixel)
    fx = np.fft.rfftfreq(w)[None, :] / 0.5
    lowpass = np.exp(-(fx ** 2 + fy ** 2) / (2 * cutoff ** 2))
    if not highpass:
        return lowpass
    transfer = 1 - lowpass
    transfer[0, 0] = 1.0  # keep the mean
    return transfer


def frequency_filter(image: np.ndarray, kind: str, cutoff: float) -> np.ndarray:
    x = image.astype(np.float64)
    transfer = gaussian_transfer(x.shape[:2], cutoff, kind == 'highpass')
    out = np.stack([np.fft.irfft2(np.fft.rfft2(x[..., c]) * transfer, s=x.shape[:2]) for c in range(x.shape[2])],
                   axis=-1)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
