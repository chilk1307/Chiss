from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Union, Dict, Literal

import PIL
import PIL.Image
from torch.utils.data import Dataset

# Images that fail to load are skipped (the sample becomes None and collate_fn drops it), but never silently:
# each failure is printed with the file name and the error. Callers compare the number of loaded samples with
# len(dataset) to report the total (see retrieval_core.encode_images).
MAX_PRINTED_LOAD_ERRORS = 20  # per process; DataLoader workers each have their own counter
_load_errors = 0


def warn_load_error(what: str, error: Exception) -> None:
    global _load_errors
    _load_errors += 1
    if _load_errors <= MAX_PRINTED_LOAD_ERRORS:
        print(f"[datasets] WARNING: cannot load {what}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    if _load_errors == MAX_PRINTED_LOAD_ERRORS:
        print(f"[datasets] further load errors in this process are not printed", file=sys.stderr, flush=True)


def open_image(path: Union[Path, str], attempts: int = 3) -> PIL.Image.Image:
    """
    Open and decode an image. Transient read errors (common when reading from Google Drive) are retried;
    a missing file is not.
    """
    for attempt in range(attempts):
        try:
            image = PIL.Image.open(path)
            image.load()
            return image
        except FileNotFoundError:
            raise
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(1)


class ImageListDataset(Dataset):
    """
    Dataset over an explicit list of images, used e.g. to encode the unique reference images of a split.
    Yields a dict with keys ['image', 'image_name'].
    """

    def __init__(self, names: List[str], paths: List[Union[Path, str]], preprocess: callable, split: str = ''):
        assert len(names) == len(paths)
        self.names = names
        self.paths = paths
        self.preprocess = preprocess
        self.split = split

    def __getitem__(self, index) -> Optional[dict]:
        try:
            return {
                'image': self.preprocess(open_image(self.paths[index])),
                'image_name': self.names[index]
            }
        except Exception as e:
            warn_load_error(str(self.paths[index]), e)
            return None

    def __len__(self):
        return len(self.names)


class FashionIQDataset(Dataset):
    """
    FashionIQ dataset class for PyTorch.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield :a dict with keys ['image', 'image_name']
        - In 'relative' mode the dataset yield dict with keys:
            - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_captions'] when
             split in ['train', 'val']
            - ['reference_image', 'reference_name', 'relative_captions'] when split == test
    About 2,431 FashionIQ images are dead links: triplets and gallery images whose files are missing are filtered
    out at construction time and the counts are printed.
    """

    def __init__(self, dataset_path: Union[Path, str], split: Literal['train', 'val', 'test'], dress_types: List[str],
                 mode: Literal['relative', 'classic'], preprocess: callable, no_duplicates: Optional[bool] = False):
        """
        :param dataset_path: path to the FashionIQ dataset
        :param split: dataset split, should be in ['train, 'val', 'test']
        :param dress_types: list of fashionIQ categories, each category should be in ['dress', 'shirt', 'toptee']
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield a dict with keys ['image', 'image_name']
            - In 'relative' mode the dataset yield dict with keys:
                - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_captions']
                 when split in ['train', 'val']
                - ['reference_image', 'reference_name', 'relative_captions'] when split == test
        :param preprocess: function which preprocesses the image
        :param no_duplicates: if True, the dataset will not yield duplicate images in relative mode, does not affect classic mode
        """
        dataset_path = Path(dataset_path)
        self.dataset_path = dataset_path
        self.mode = mode
        self.dress_types = dress_types
        self.split = split
        self.no_duplicates = no_duplicates

        # Validate the inputs
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")

        self.preprocess = preprocess

        # get triplets made by (reference_image, target_image, a pair of relative captions), keeping only the
        # triplets whose images exist on disk
        self.triplets: List[dict] = []
        n_declared_triplets = 0
        for dress_type in dress_types:
            with open(dataset_path / 'captions' / f'cap.{dress_type}.{split}.json') as f:
                raw_triplets = json.load(f)
            n_declared_triplets += len(raw_triplets)
            for t in raw_triplets:
                ref_ok = self.image_path(t['candidate']).exists()
                tgt_ok = self.image_path(t['target']).exists() if split in ['train', 'val'] else True
                if ref_ok and tgt_ok:
                    self.triplets.append(t)
        n_valid_triplets = len(self.triplets)

        # Remove duplicats from
        if self.no_duplicates:
            seen = set()
            new_triplets = []
            for triplet in self.triplets:
                if triplet['candidate'] not in seen:
                    seen.add(triplet['candidate'])
                    new_triplets.append(triplet)
            self.triplets = new_triplets

        # get the image names, keeping only the images that exist on disk
        self.image_names: list = []
        for dress_type in dress_types:
            with open(dataset_path / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
                self.image_names.extend(json.load(f))
        n_declared_images = len(self.image_names)
        self.image_names = [n for n in self.image_names if self.image_path(n).exists()]

        print(f"FashionIQ {split} - {dress_types}: kept {n_valid_triplets}/{n_declared_triplets} triplets "
              f"and {len(self.image_names)}/{n_declared_images} gallery images with files on disk")
        print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")

    def image_path(self, image_name: str) -> Path:
        return self.dataset_path / 'images' / f"{image_name}.jpg"

    def __getitem__(self, index) -> Optional[dict]:
        try:
            if self.mode == 'relative':
                relative_captions = self.triplets[index]['captions']
                reference_name = self.triplets[index]['candidate']

                if self.split in ['train', 'val']:
                    reference_image = self.preprocess(open_image(self.image_path(reference_name)))
                    target_name = self.triplets[index]['target']
                    target_image = self.preprocess(open_image(self.image_path(target_name)))

                    return {
                        'reference_image': reference_image,
                        'reference_name': reference_name,
                        'target_image': target_image,
                        'target_name': target_name,
                        'relative_captions': relative_captions
                    }

                elif self.split == 'test':
                    reference_image = self.preprocess(open_image(self.image_path(reference_name)))

                    return {
                        'reference_image': reference_image,
                        'reference_name': reference_name,
                        'relative_captions': relative_captions
                    }

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image = self.preprocess(open_image(self.image_path(image_name)))

                return {
                    'image': image,
                    'image_name': image_name
                }

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            warn_load_error(f"FashionIQ {self.split} {self.mode} sample {index}", e)
            return None

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class CIRRDataset(Dataset):
    """
   CIRR dataset class for PyTorch dataloader.
   The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield a dict with keys ['image', 'image_name']
        - In 'relative' mode the dataset yield dict with keys:
            - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_caption', 'group_members']
             when split in ['train', 'val']
            - ['reference_image', 'reference_name' 'relative_caption', 'group_members', 'pair_id'] when split == test
    """

    def __init__(self, dataset_path: Union[Path, str], split: Literal['train', 'val', 'test'],
                 mode: Literal['relative', 'classic'], preprocess: callable, no_duplicates: Optional[bool] = False):
        """
        :param dataset_path: path to the CIRR dataset
        :param split: dataset split, should be in ['train', 'val', 'test']
        :param mode: dataset mode, should be in ['relative', 'classic']:
                - In 'classic' mode the dataset yield a dict with keys ['image', 'image_name']
                - In 'relative' mode the dataset yield dict with keys:
                    - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_caption',
                    'group_members'] when split in ['train', 'val']
                    - ['reference_image', 'reference_name' 'relative_caption', 'group_members', 'pair_id'] when split == test
        :param preprocess: function which preprocesses the image
        :param no_duplicates: if True, the dataset will not yield duplicate images in relative mode, does not affect classic mode
        """
        dataset_path = Path(dataset_path)
        self.dataset_path = dataset_path
        self.preprocess = preprocess
        self.mode = mode
        self.split = split
        self.no_duplicates = no_duplicates

        if split == "test":
            split = "test1"
            self.split = "test1"

        # Validate inputs
        if split not in ['test1', 'train', 'val']:
            raise ValueError("split should be in ['test1', 'train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        # get triplets made by (reference_image, target_image, relative caption)
        with open(dataset_path / 'cirr' / 'captions' / f'cap.rc2.{split}.json') as f:
            self.triplets = json.load(f)

        # Remove duplicates from triplets
        if self.no_duplicates:
            seen = set()
            new_triplets = []
            for triplet in self.triplets:
                if triplet['reference'] not in seen:
                    seen.add(triplet['reference'])
                    new_triplets.append(triplet)
            self.triplets = new_triplets

        # get a mapping from image name to relative path
        with open(dataset_path / 'cirr' / 'image_splits' / f'split.rc2.{split}.json') as f:
            self.name_to_relpath = json.load(f)
        self.image_names = list(self.name_to_relpath.keys())

        print(f"CIRR {split} dataset in {mode} mode initialized")

    def image_path(self, image_name: str) -> Path:
        return self.dataset_path / self.name_to_relpath[image_name]

    def __getitem__(self, index) -> Optional[dict]:
        try:
            if self.mode == 'relative':
                group_members = self.triplets[index]['img_set']['members']
                reference_name = self.triplets[index]['reference']
                relative_caption = self.triplets[index]['caption']

                if self.split in ['train', 'val']:
                    reference_image = self.preprocess(open_image(self.image_path(reference_name)))
                    target_hard_name = self.triplets[index]['target_hard']
                    target_image = self.preprocess(open_image(self.image_path(target_hard_name)))

                    return {
                        'reference_image': reference_image,
                        'reference_name': reference_name,
                        'target_image': target_image,
                        'target_name': target_hard_name,
                        'relative_caption': relative_caption,
                        'group_members': group_members
                    }

                elif self.split == 'test1':
                    pair_id = self.triplets[index]['pairid']
                    reference_image = self.preprocess(open_image(self.image_path(reference_name)))
                    return {
                        'reference_image': reference_image,
                        'reference_name': reference_name,
                        'relative_caption': relative_caption,
                        'group_members': group_members,
                        'pair_id': pair_id
                    }

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image = self.preprocess(open_image(self.image_path(image_name)))

                return {
                    'image': image,
                    'image_name': image_name
                }

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            warn_load_error(f"CIRR {self.split} {self.mode} sample {index}", e)
            return None

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.name_to_relpath)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class CIRCODataset(Dataset):
    """
    CIRCO dataset class for PyTorch.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield a dict with keys ['image', 'image_name']
        - In 'relative' mode the dataset yield dict with keys:
            - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_captions', 'shared_concept',
             'gt_img_ids', 'query_id'] when split == 'val'
            - ['reference_image', 'reference_name', 'relative_captions', 'shared_concept', 'query_id'] when split == test
    """

    def __init__(self, dataset_path: Union[str, Path], split: Literal['val', 'test'],
                 mode: Literal['relative', 'classic'], preprocess: callable):
        """
        Args:
            dataset_path (Union[str, Path]): path to CIRCO dataset
            split (str): dataset split, should be in ['test', 'val']
            mode (str): dataset mode, should be in ['relative', 'classic']
            preprocess (callable): function which preprocesses the image
        """

        # Set dataset paths and configurations
        dataset_path = Path(dataset_path)
        self.mode = mode
        self.split = split
        self.preprocess = preprocess
        self.data_path = dataset_path

        # Ensure input arguments are valid
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'val']:
            raise ValueError("split should be in ['test', 'val']")

        # Load COCO images information
        with open(dataset_path / 'COCO2017_unlabeled' / "annotations" / "image_info_unlabeled2017.json", "r") as f:
            imgs_info = json.load(f)

        self.img_paths = [dataset_path / 'COCO2017_unlabeled' / "unlabeled2017" / img_info["file_name"] for img_info in
                          imgs_info["images"]]
        self.img_ids = [img_info["id"] for img_info in imgs_info["images"]]
        self.img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(self.img_ids)}

        # get CIRCO annotations
        with open(dataset_path / 'annotations' / f'{split}.json', "r") as f:
            self.annotations: List[dict] = json.load(f)

        # Get maximum number of ground truth images (for padding when loading the images)
        self.max_num_gts = 23  # Maximum number of ground truth images

        print(f"CIRCODataset {split} dataset in {mode} mode initialized")

    def image_path(self, img_id: str) -> Path:
        return self.img_paths[self.img_ids_indexes_map[str(img_id)]]

    def get_target_img_ids(self, index) -> Dict[str, int]:
        """
        Returns the id of the target image and ground truth images for a given query

        Args:
            index (int): id of the query

        Returns:
             Dict[str, int]: dictionary containing target image id and a list of ground truth image ids
        """

        return {
            'target_img_id': self.annotations[index]['target_img_id'],
            'gt_img_ids': self.annotations[index]['gt_img_ids']
        }

    def __getitem__(self, index) -> Optional[dict]:
        """
        Returns a specific item from the dataset based on the index.

        In 'classic' mode, the dataset yields a dictionary with the following keys: [img, img_id]
        In 'relative' mode, the dataset yields dictionaries with the following keys:
            - [reference_img, reference_img_id, target_img, target_img_id, relative_caption, shared_concept, gt_img_ids,
            query_id]
            if split == val
            - [reference_img, reference_img_id, relative_caption, shared_concept, query_id]  if split == test
        """
        try:
            if self.mode == 'relative':
                # Get the query id
                query_id = str(self.annotations[index]['id'])

                # Get relative caption and shared concept
                relative_caption = self.annotations[index]['relative_caption']
                shared_concept = self.annotations[index]['shared_concept']

                # Get the reference image
                reference_img_id = str(self.annotations[index]['reference_img_id'])
                reference_img = self.preprocess(open_image(self.image_path(reference_img_id)))

                if self.split == 'val':
                    # Get the target image and ground truth images
                    target_img_id = str(self.annotations[index]['target_img_id'])
                    gt_img_ids = [str(x) for x in self.annotations[index]['gt_img_ids']]
                    target_img = self.preprocess(open_image(self.image_path(target_img_id)))

                    # Pad ground truth image IDs with zeros for collate_fn
                    gt_img_ids += [''] * (self.max_num_gts - len(gt_img_ids))

                    return {
                        'reference_image': reference_img,
                        'reference_name': reference_img_id,
                        'target_image': target_img,
                        'target_name': target_img_id,
                        'relative_caption': relative_caption,
                        'shared_concept': shared_concept,
                        'gt_img_ids': gt_img_ids,
                        'query_id': query_id,
                    }

                elif self.split == 'test':
                    return {
                        'reference_image': reference_img,
                        'reference_name': reference_img_id,
                        'relative_caption': relative_caption,
                        'shared_concept': shared_concept,
                        'query_id': query_id,
                    }

            elif self.mode == 'classic':
                # Get image ID and image path
                img_id = str(self.img_ids[index])
                img_path = self.img_paths[index]

                # Preprocess image and return
                img = self.preprocess(open_image(img_path))
                return {
                    'image': img,
                    'image_name': img_id
                }
        except Exception as e:
            warn_load_error(f"CIRCO {self.split} {self.mode} sample {index}", e)
            return None

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        if self.mode == 'relative':
            return len(self.annotations)
        elif self.mode == 'classic':
            return len(self.img_ids)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class ImageNetDataset(Dataset):
    """
    Dataset class for ImageNet dataset
    The dataset yields a dictionary with the following keys: ['image', 'image_name']
    """

    def __init__(self, dataset_path: Union[Path, str], split: Literal['train', 'val', 'test'], preprocess: callable):
        dataset_path = Path(dataset_path)
        self.dataset_path = dataset_path
        self.split = split
        self.preprocess = preprocess

        if split not in ['train', 'val', 'test']:
            raise ValueError("split should be in ['train', 'val', 'test']")

        self.images_path = sorted(p for p in (dataset_path / split).rglob("*") if p.suffix.lower() in [".jpeg", ".jpg", ".png"])

    def __getitem__(self, index) -> dict:
        img_path = self.images_path[index]
        img = self.preprocess(open_image(img_path))
        return {
            'image': img,
            'image_name': img_path.name
        }

    def __len__(self):
        return len(self.images_path)
