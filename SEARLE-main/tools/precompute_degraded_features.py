"""
CLIP features of degraded versions of the ImageNet training images, for the robust phi (plan 03, section 3.4).

    python tools/precompute_degraded_features.py --imagenet-path $DATA/ImageNet1K --oti-exp-name oti_B32_s0 \
        --degrade none,gauss_noise:1,gauss_noise:2,gauss_blur:1,gauss_blur:2,jpeg:1,jpeg:2,low_light:1,low_light:2 \
        --out-dir $DRIVE/cache/imagenet_s0

Every image is degraded at its original resolution (degradations.py, seeded by the image name), then goes through the
same preprocessing as the training (targetpad 1.25 by default) and the same CLIP model as train_phi.py. Images are
processed in parts of --part-size images saved in <out-dir>/parts/, so an interrupted run continues where it stopped.
At the end <out-dir> contains one file per degradation (<type><level>.pt, float16, N x 512), image_names.pkl and
meta.json, the format read by train_phi.py --aug-feature-dir.
"""
import json
import pickle
import sys
from argparse import ArgumentParser
from pathlib import Path

import clip
import numpy as np
import PIL.Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data_utils import PROJECT_ROOT, collate_fn, targetpad_transform  # noqa: E402
from datasets import open_image, warn_load_error  # noqa: E402
from degradations import DEGRADATIONS_VERSION, degrade, parse_degrade  # noqa: E402
from utils import device  # noqa: E402


class DegradedImages(Dataset):
    def __init__(self, paths, specs, preprocess):
        self.paths, self.specs, self.preprocess = paths, specs, preprocess

    def __getitem__(self, index):
        path = self.paths[index]
        try:
            rgb = np.asarray(open_image(path).convert('RGB'))
            versions = []
            for spec in self.specs:
                parsed = parse_degrade(spec)
                image = rgb if parsed is None else degrade(rgb, parsed[0], parsed[1], path.name)[0]
                versions.append(self.preprocess(PIL.Image.fromarray(image)))
            return {'image': torch.stack(versions), 'image_name': path.name}
        except Exception as e:
            warn_load_error(str(path), e)
            return None

    def __len__(self):
        return len(self.paths)


@torch.no_grad()
def main():
    parser = ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--imagenet-path', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--oti-exp-name', default=None, help='Only the images with a token in this OTI experiment')
    parser.add_argument('--degrade', required=True, help="Comma separated degradations ('none' = clean image)")
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--clip-model-name', default='ViT-B/32')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--part-size', type=int, default=5000)
    args = parser.parse_args()

    specs = [s.strip() for s in args.degrade.split(',') if s.strip()]
    for spec in specs:
        parse_degrade(spec)  # validates
    if 'mixed' in specs:
        raise SystemExit("'mixed' is not a fixed degradation; list the degradations explicitly")
    paths = sorted(p for p in (Path(args.imagenet_path) / args.split).rglob('*')
                   if p.suffix.lower() in ('.jpeg', '.jpg', '.png'))
    if args.oti_exp_name:
        with open(PROJECT_ROOT / 'data' / 'oti_pseudo_tokens' / 'imagenet' / args.split / args.oti_exp_name /
                  'image_names.pkl', 'rb') as f:
            wanted = set(pickle.load(f))
        paths = [p for p in paths if p.name in wanted]
    print(f"{len(paths)} images x {len(specs)} versions")

    clip_model, _ = clip.load(args.clip_model_name, device=device, jit=False)
    clip_model = clip_model.float().eval()
    preprocess = targetpad_transform(1.25, clip_model.visual.input_resolution)

    out = Path(args.out_dir)
    parts_dir = out / 'parts'
    parts_dir.mkdir(parents=True, exist_ok=True)
    n_parts = (len(paths) + args.part_size - 1) // args.part_size
    for part in range(n_parts):
        part_file = parts_dir / f"part_{part:04d}.pt"
        if part_file.exists():
            continue
        chunk = paths[part * args.part_size:(part + 1) * args.part_size]
        loader = DataLoader(DegradedImages(chunk, specs, preprocess), batch_size=args.batch_size,
                            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
        names, features = [], []
        for batch in tqdm(loader, desc=f"part {part + 1}/{n_parts}"):
            if batch is None:
                continue
            images = batch['image'].to(device)  # (B, V, 3, H, W)
            b, v = images.shape[:2]
            encoded = clip_model.encode_image(images.flatten(0, 1)).reshape(b, v, -1)
            features.append(encoded.half().cpu())
            names.extend(batch['image_name'])
        torch.save({'names': names, 'features': torch.cat(features)}, part_file)

    names, features = [], []
    for part in range(n_parts):
        saved = torch.load(parts_dir / f"part_{part:04d}.pt", map_location='cpu')
        names += saved['names']
        features.append(saved['features'])
    features = torch.cat(features)  # (N, V, D)
    for i, spec in enumerate(specs):
        torch.save(features[:, i].clone(), out / f"{spec.replace(':', '')}.pt")
    with open(out / 'image_names.pkl', 'wb') as f:
        pickle.dump(names, f)
    meta = {'degradations': specs, 'clip_model': args.clip_model_name, 'preprocess': 'targetpad1.25-bicubic',
            'n_images': len(names), 'oti_exp_name': args.oti_exp_name, 'degradations_version': DEGRADATIONS_VERSION,
            'dtype': 'float16'}
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(f"Wrote {len(specs)} feature files of {len(names)} images to {out}")


if __name__ == '__main__':
    main()
