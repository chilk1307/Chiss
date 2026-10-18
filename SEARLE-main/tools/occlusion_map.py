"""
Occlusion maps (plan 01, section 1.13; experiment E1.13): which parts of the reference image the pseudo token "looks
at".

    python tools/occlusion_map.py --run R/E1.7d-phiB32-cirr --dataset cirr --dataset-path $DATA/CIRR \
        --n 10 --seed 0 --out R/E1.13-occlusion-cirr

For successful queries (a correct image in the top-5), every 32 x 32 patch of the preprocessed reference image
(stride 16, about 170 positions at 224 px) is replaced by neutral grey, the query is recomputed with the run's method
(phi, searle, image or sum) and the score of the target image is compared with the unoccluded score. The map of the
score drops is drawn over the image: red = the score falls when this region is hidden.
"""
import random
import sys
import textwrap
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_utils import (captions_of, correct_names, image_locator, load_image, load_run,  # noqa: E402
                       pipeline_from_run, query_from_record, score_images)

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def occluded_batch(image: torch.Tensor, patch: int, stride: int):
    """
    The image followed by one copy per patch position with the patch set to neutral grey (0.5 before normalisation)
    """
    _, h, w = image.shape
    grey = ((0.5 - CLIP_MEAN) / CLIP_STD).to(image.dtype)
    ys = list(range(0, h - patch + 1, stride))
    xs = list(range(0, w - patch + 1, stride))
    batch = [image]
    for y in ys:
        for x in xs:
            occluded = image.clone()
            occluded[:, y:y + patch, x:x + patch] = grey
            batch.append(occluded)
    return torch.stack(batch), ys, xs


def main():
    parser = ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--run', required=True)
    parser.add_argument('--dataset', required=True, choices=['cirr', 'circo', 'fashioniq'])
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--patch', type=int, default=32)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--out', required=True)
    parser.add_argument('--cache-dir', default=None)
    args = parser.parse_args()

    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    _, records = load_run(args.run)
    successes = [r for r in records if r.get('top') and correct_names(r) & {n for n, _ in r['top'][:5]}
                 and r.get('target')]
    chosen = random.Random(args.seed).sample(successes, min(args.n, len(successes)))
    pipeline = pipeline_from_run(args.run, args.dataset_path, args.cache_dir)
    locate = image_locator(args.dataset, args.dataset_path)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    galleries = {}
    for i, record in enumerate(chosen, 1):
        category = record.get('category')
        if category not in galleries:
            galleries[category] = pipeline.gallery_features(category)
        gallery, names = galleries[category]
        target = gallery[names.index(record['target'])][None]
        image = pipeline.preprocess(load_image(locate(record['reference'])))
        batch, ys, xs = occluded_batch(image, args.patch, args.stride)
        scores = []
        for start in range(0, len(batch), args.batch_size):
            chunk = batch[start:start + args.batch_size]
            queries = [query_from_record(record, reference=f"variant{start + j}") for j in range(len(chunk))]
            scores.append(score_images(pipeline, chunk, queries, target)[:, 0].cpu())
        scores = torch.cat(scores).numpy()
        drops = (scores[0] - scores[1:]).reshape(len(ys), len(xs))
        heat = np.zeros(image.shape[1:], dtype=np.float64)
        count = np.zeros_like(heat)
        for a, y in enumerate(ys):
            for b, x in enumerate(xs):
                heat[y:y + args.patch, x:x + args.patch] += drops[a, b]
                count[y:y + args.patch, x:x + args.patch] += 1
        heat /= np.maximum(count, 1)
        shown = (image * CLIP_STD + CLIP_MEAN).clamp(0, 1).permute(1, 2, 0).numpy()

        fig, axes = plt.subplots(1, 3, figsize=(9, 3.6))
        axes[0].imshow(shown)
        axes[0].set_title('reference (preprocessed)', fontsize=8)
        axes[1].imshow(shown)
        limit = max(abs(heat).max(), 1e-6)
        overlay = axes[1].imshow(cv2.GaussianBlur(heat, (0, 0), 4), cmap='RdBu_r', alpha=0.55, vmin=-limit,
                                 vmax=limit)
        axes[1].set_title(f"score drop when hidden (max {heat.max():+.3f})", fontsize=8)
        fig.colorbar(overlay, ax=axes[1], fraction=0.046)
        axes[2].imshow(load_image(locate(record['target'])))
        axes[2].set_title(f"target (score {scores[0]:.3f})", fontsize=8)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle('\n'.join(textwrap.wrap(' || '.join(captions_of(record)), 100)), fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"{i:02d}_{record['qid'].replace(':', '-')}.png", dpi=110)
        plt.close(fig)
        print(f"{i}/{len(chosen)} {record['qid']}: {len(scores) - 1} occluded images, max drop {heat.max():+.3f}")


if __name__ == '__main__':
    main()
