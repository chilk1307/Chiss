"""
Figures of degraded / restored reference images (plan 03; experiment E3.7, and the visual check of task C9).

    python tools/visualize_degradations.py --dataset cirr --dataset-path $DATA/CIRR --run R/E1.7d-phiB32-cirr \
        --degrade gauss_noise:2,low_light:2,jpeg:2,color_cast:2 --restore auto --n 8 --seed 0 --out R/E3.7-figures

For each of --n reference images (random, --seed): the original, then for each degradation the degraded image and its
restoration (--restore: a method, oracle, auto or auto_iqa; 'none' to skip), with PSNR / SSIM against the original.
With --run, every version is also used as the query of the run's method (phi, searle, image or sum) and its top-K
results are shown (green frame = correct image) with the rank of the target. Without --run only the images are drawn.
"""
import random
import sys
import textwrap
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import PIL.Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_utils import (captions_of, correct_names, image_locator, load_image, load_run,  # noqa: E402
                       pipeline_from_run, query_from_record, score_images)
from reference_processing import ReferenceProcessor, image_quality  # noqa: E402


def versions_of(rgb: np.ndarray, name: str, degradations, restore: str):
    """
    [(label, image, psnr, ssim)] for the original, every degradation and its restoration
    """
    out = [('original', rgb, None, None)]
    for spec in degradations:
        degraded, _ = ReferenceProcessor(degrade=spec).process_array(rgb, name)
        out.append((spec, degraded, *image_quality(rgb, degraded)))
        if restore != 'none':
            restored, info = ReferenceProcessor(degrade=spec, restore_spec=restore).process_array(rgb, name)
            out.append((f"{spec} + {info.get('restoration', restore)}", restored, *image_quality(rgb, restored)))
    return out


def main():
    parser = ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--dataset', required=True, choices=['cirr', 'circo', 'fashioniq'])
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--run', default=None, help='Run whose method retrieves with every version')
    parser.add_argument('--degrade', required=True, help='Comma separated degradations, e.g. gauss_noise:2,jpeg:2')
    parser.add_argument('--restore', default='auto')
    parser.add_argument('--n', type=int, default=8)
    parser.add_argument('--top', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', required=True)
    parser.add_argument('--cache-dir', default=None)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    degradations = [d.strip() for d in args.degrade.split(',') if d.strip()]
    locate = image_locator(args.dataset, args.dataset_path)
    if args.run:
        _, records = load_run(args.run)
        records = [r for r in records if r.get('top')]
        pipeline = pipeline_from_run(args.run, args.dataset_path, args.cache_dir)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from attribute_stats import load_queries
        pairs, _, _ = load_queries(args.dataset, args.dataset_path, 'val')
        records = [{'qid': q.qid, 'reference': q.reference, 'captions': q.captions, 'target': q.target,
                    'category': q.category} for _, q in pairs]
        pipeline = None
    chosen = random.Random(args.seed).sample(records, min(args.n, len(records)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    galleries = {}
    for i, record in enumerate(chosen, 1):
        rgb = np.asarray(load_image(locate(record['reference'])))
        versions = versions_of(rgb, record['reference'], degradations, args.restore)
        columns = 1 + (args.top if pipeline is not None else 0)
        fig, axes = plt.subplots(len(versions), columns, figsize=(2.2 * columns + 0.4, 2.3 * len(versions) + 0.6),
                                 squeeze=False)
        fig.suptitle('\n'.join(textwrap.wrap(' || '.join(captions_of(record)), 100)), fontsize=8)
        if pipeline is not None:
            category = record.get('category')
            if category not in galleries:
                galleries[category] = pipeline.gallery_features(category)
            gallery, names = galleries[category]
            images = torch.stack([pipeline.preprocess(PIL.Image.fromarray(v[1])) for v in versions])
            queries = [query_from_record(record, reference=f"version{k}") for k in range(len(versions))]
            similarity = score_images(pipeline, images, queries, gallery).cpu()
            exclude = names.index(record['reference']) if record['reference'] in names else None
            if exclude is not None:
                similarity[:, exclude] = float('-inf')  # the reference image itself is not a result
        correct = correct_names(record)
        for row, (label, image, psnr, ssim) in enumerate(versions):
            ax = axes[row][0]
            ax.imshow(image)
            title = label if psnr is None else f"{label}\nPSNR {psnr:.1f} dB, SSIM {ssim:.2f}"
            if pipeline is not None and record.get('target') in names:
                order = similarity[row].argsort(descending=True)
                rank = int((order == names.index(record['target'])).nonzero()[0]) + 1
                title += f"\ntarget rank {rank}"
                for col in range(1, columns):
                    name = names[int(order[col - 1])]
                    ax2 = axes[row][col]
                    ax2.imshow(load_image(locate(name)))
                    good = name in correct
                    for spine in ax2.spines.values():
                        spine.set_edgecolor('limegreen' if good else 'lightgray')
                        spine.set_linewidth(4 if good else 1)
                    ax2.set_title(f"#{col}", fontsize=7)
            ax.set_title(title, fontsize=7)
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(out / f"{i:02d}_{record['qid'].replace(':', '-')}.png", dpi=110)
        plt.close(fig)
    print(f"Written {len(chosen)} figures to {out}")


if __name__ == '__main__':
    main()
