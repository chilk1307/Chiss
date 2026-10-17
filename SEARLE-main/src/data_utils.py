from pathlib import Path

import PIL
import torch
import torchvision.transforms.functional as FT
from torch.utils.data import Dataset
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize, Resize
from torchvision.transforms import InterpolationMode

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def collate_fn(batch):
    '''
    function which discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    '''
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return None
    return torch.utils.data.dataloader.default_collate(batch)


class TargetPad:
    """
    If an image aspect ratio is above a target ratio, pad the image to match such target ratio.
    For more details see Baldrati et al. 'Effective conditioned and composed image retrieval combining clip-based features.' Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2022).
    """

    def __init__(self, target_ratio: float, size: int):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return FT.pad(image, padding, 0, 'constant')


INTERPOLATIONS = {
    'nearest': InterpolationMode.NEAREST,
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}


def targetpad_transform(target_ratio: float, dim: int, interpolation: str = 'bicubic') -> torch.Tensor:
    """
    CLIP-like preprocessing transform computed after using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :param interpolation: resize interpolation, one of INTERPOLATIONS
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=INTERPOLATIONS[interpolation]),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def clip_transform(dim: int, interpolation: str = 'bicubic') -> torch.Tensor:
    """
    Same pipeline as clip.load()'s preprocess (resize, center crop, no padding) with a configurable interpolation
    """
    return Compose([
        Resize(dim, interpolation=INTERPOLATIONS[interpolation]),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def build_preprocess(preprocess_type: str, dim: int, clip_preprocess: callable = None, targetpad_ratio: float = 1.25,
                     interpolation: str = 'bicubic') -> callable:
    """
    Image preprocessing used for evaluation. The defaults (targetpad, 1.25, bicubic) are the original SEARLE pipeline.
    """
    if preprocess_type == 'targetpad':
        print(f'Target pad preprocess pipeline is used (ratio {targetpad_ratio}, {interpolation})')
        return targetpad_transform(targetpad_ratio, dim, interpolation)
    elif preprocess_type == 'clip':
        print(f'CLIP preprocess pipeline is used ({interpolation})')
        if interpolation == 'bicubic' and clip_preprocess is not None:
            return clip_preprocess
        return clip_transform(dim, interpolation)
    raise ValueError("Preprocess type not supported")


def preprocess_key(preprocess_type: str, targetpad_ratio: float = 1.25, interpolation: str = 'bicubic') -> str:
    """
    Short string identifying a preprocessing configuration, used in feature cache keys
    """
    if preprocess_type == 'targetpad':
        return f"targetpad{targetpad_ratio:g}-{interpolation}"
    return f"{preprocess_type}-{interpolation}"
