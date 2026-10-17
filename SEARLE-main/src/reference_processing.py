"""
Processing chain applied to the reference (query) images only, never to the gallery (plan 03):

    original --degrade--> degraded --restore--> restored --freq-filter--> image given to CLIP

--save-quality adds the PSNR and SSIM of the final image with respect to the original one. The per-image information
(applied degradation, chosen restoration, PSNR, SSIM) ends up in dump.jsonl and its averages in metrics.json.
"""
import hashlib
from typing import Dict, Optional, Tuple

import numpy as np
import PIL.Image

from degradations import DEGRADATIONS_VERSION, apply_degrade_spec, parse_degrade
from freq_filters import FREQ_FILTERS_VERSION, frequency_filter, parse_freq_filter
from restorations import RESTORATIONS_VERSION, check_restore, oracle_method, restore

MAX_PSNR = 100.0  # PSNR of an unchanged image is infinite; stored as 100 dB


def image_quality(original: np.ndarray, processed: np.ndarray) -> Tuple[float, float]:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    if np.array_equal(original, processed):
        return MAX_PSNR, 1.0
    psnr = min(float(peak_signal_noise_ratio(original, processed, data_range=255)), MAX_PSNR)
    ssim = float(structural_similarity(original, processed, channel_axis=2, data_range=255))
    return psnr, ssim


class ReferenceProcessor:
    def __init__(self, degrade: Optional[str] = None, restore_spec: Optional[str] = None,
                 freq_filter: Optional[str] = None, save_quality: bool = False):
        self.degrade_spec = parse_degrade(degrade)
        self.restore_spec = check_restore(restore_spec)
        self.freq_spec = parse_freq_filter(freq_filter)
        self.save_quality = save_quality
        self.degrade_text, self.restore_text, self.freq_text = degrade or 'none', restore_spec or 'none', \
            freq_filter or 'none'

    @classmethod
    def from_args(cls, args) -> Optional['ReferenceProcessor']:
        processor = cls(args.degrade, args.restore, args.freq_filter, args.save_quality)
        return processor if processor.active else None

    @property
    def active(self) -> bool:
        return bool(self.degrade_spec or self.restore_spec or self.freq_spec or self.save_quality)

    @property
    def key(self) -> str:
        """
        Identifies the processing in feature cache keys (including the content of the oracle table or of the rules
        of 'auto', so that editing them invalidates the cache)
        """
        restore_text = self.restore_text
        if self.restore_spec in ('oracle', 'auto'):
            from degradation_estimation import RULES_FILE
            from restorations import ORACLE_FILE
            table = ORACLE_FILE if self.restore_spec == 'oracle' else RULES_FILE
            restore_text += '@' + hashlib.sha1(table.read_bytes()).hexdigest()[:8]
        return (f"degrade={self.degrade_text}-v{DEGRADATIONS_VERSION};restore={restore_text}-v"
                f"{RESTORATIONS_VERSION};freq={self.freq_text}-v{FREQ_FILTERS_VERSION}")

    def process_array(self, rgb: np.ndarray, name: str) -> Tuple[np.ndarray, Dict]:
        original = rgb
        info, context = {}, {}
        if self.degrade_spec:
            rgb, context = apply_degrade_spec(rgb, self.degrade_spec, name)
            info['degradation'] = context['degradation']
        if self.restore_spec:
            if self.restore_spec == 'oracle':
                method = oracle_method(context)
            elif self.restore_spec in ('auto', 'auto_iqa'):
                from degradation_estimation import choose_restoration
                method = choose_restoration(rgb, self.restore_spec)
            else:
                method = self.restore_spec
            if method:
                rgb = restore(rgb, method, context)
            info['restoration'] = method or 'none'
        if self.freq_spec:
            rgb = frequency_filter(rgb, *self.freq_spec)
        if self.save_quality:
            info['psnr'], info['ssim'] = image_quality(original, rgb)
        return rgb, info

    def __call__(self, image: PIL.Image.Image, name: str) -> Tuple[PIL.Image.Image, Dict]:
        rgb, info = self.process_array(np.asarray(image.convert('RGB')), name)
        return PIL.Image.fromarray(rgb), info
