"""
Calibrate the rules of --restore auto (plan 03, section 3.3) and check the degradation estimator.

    python tools/calibrate_restore_rules.py --images-dir $DATA/CIRR/train --n 300 --out src/restore_rules.json \
        --report-dir $DRIVE/results/E3.3-calibration

Use training images only (CIRR train), never the evaluation sets. Every sampled image is measured clean and with
each degradation at both levels (degradations.py). For each rule the threshold that maximises the balanced accuracy
(its degradation vs everything else, clean images included) is chosen; the rules are then applied in order and the
report gives the confusion matrix of the identified degradation and the error of the noise estimate.

Rules, in priority order (first match wins):
    impulse_ratio > t  -> median       (salt and pepper)
    l_mean        < t  -> clahe        (low light)
    blockiness    > t  -> deblock      (JPEG)
    noise_sigma   > t  -> nlm          (Gaussian noise)
    hf_ratio      < t  -> unsharp      (blur, motion blur, low resolution: the PSF is unknown, so no Wiener)
    cast          > t  -> gray_world   (colour cast)
"""
import json
import random
import sys
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from degradation_estimation import choose_by_rules, estimate_features  # noqa: E402
from degradations import DEGRADATIONS, LEVELS, degrade  # noqa: E402

RULES = [  # feature, op, detected degradations, method (the blur rule comes before the colour cast rule: a blurred
    # photo of a tinted scene would otherwise be "corrected" with gray-world)
    ('impulse_ratio', '>', ['salt_pepper'], 'median'),
    ('l_mean', '<', ['low_light'], 'clahe'),
    ('blockiness', '>', ['jpeg'], 'deblock'),
    ('noise_sigma', '>', ['gauss_noise'], 'nlm'),
    ('hf_ratio', '<', ['gauss_blur', 'motion_blur', 'low_res'], 'unsharp'),
    ('cast', '>', ['color_cast'], 'gray_world'),
]
METHOD_TO_GROUP = {'median': 'salt_pepper', 'clahe': 'low_light', 'deblock': 'jpeg', 'nlm': 'gauss_noise',
                   'gray_world': 'color_cast', 'unsharp': 'blur', None: 'clean'}


def group_of(kind: str) -> str:
    return 'blur' if kind in ('gauss_blur', 'motion_blur', 'low_res') else kind


def best_threshold(values: np.ndarray, positive: np.ndarray, op: str):
    """
    Threshold maximising (TPR + TNR) / 2 for the rule 'value op threshold'
    """
    v = values if op == '>' else -values
    order = np.argsort(v)
    v, pos = v[order], positive[order]
    n_pos, n_neg = pos.sum(), (~pos).sum()
    # classify value > candidate as positive; candidates between consecutive sorted values
    tp = n_pos - np.cumsum(pos)  # positives strictly above position i
    tn = np.cumsum(~pos)
    balanced = (tp / max(n_pos, 1) + tn / max(n_neg, 1)) / 2
    i = int(np.argmax(balanced[:-1])) if len(v) > 1 else 0
    threshold = (v[i] + v[min(i + 1, len(v) - 1)]) / 2
    return float(threshold if op == '>' else -threshold), float(balanced[i])


def main():
    parser = ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--images-dir', required=True, help='Training images (searched recursively)')
    parser.add_argument('--n', type=int, default=300, help='Number of sampled images')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', required=True, help='Rules file to write (src/restore_rules.json)')
    parser.add_argument('--report-dir', default=None)
    parser.add_argument('--max-side', type=int, default=640, help='Larger images are downscaled first')
    args = parser.parse_args()

    paths = sorted(p for p in Path(args.images_dir).rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    random.Random(args.seed).shuffle(paths)
    paths = paths[:args.n]
    rows = []  # (true kind, level, features)
    for i, path in enumerate(paths):
        image = Image.open(path).convert('RGB')
        image.thumbnail((args.max_side, args.max_side))
        rgb = np.asarray(image)
        rows.append(('clean', 0, estimate_features(rgb)))
        for kind in DEGRADATIONS:
            for level in (1, 2):
                rows.append((kind, level, estimate_features(degrade(rgb, kind, level, path.name)[0])))
        print(f"\r{i + 1}/{len(paths)} images measured", end='', flush=True)
    print()

    rules, rule_scores = [], {}
    kinds = np.array([r[0] for r in rows])
    for feature, op, detected, method in RULES:
        values = np.array([r[2][feature] for r in rows])
        threshold, balanced = best_threshold(values, np.isin(kinds, detected), op)
        rules.append({'feature': feature, 'op': op, 'threshold': round(threshold, 6), 'method': method,
                      'detects': detected})
        rule_scores[feature] = round(balanced, 3)

    confusion = defaultdict(Counter)
    for kind, level, features in rows:
        confusion[group_of(kind)][METHOD_TO_GROUP[choose_by_rules(features, rules)]] += 1
    groups = ['clean', 'salt_pepper', 'low_light', 'jpeg', 'gauss_noise', 'color_cast', 'blur']
    accuracy = sum(confusion[g][g] for g in groups) / max(len(rows), 1)
    noise_errors = {lv: [r[2]['noise_sigma'] - LEVELS['gauss_noise'][lv]['sigma'] for r in rows
                         if r[0] == 'gauss_noise' and r[1] == lv] for lv in (1, 2)}
    clean_sigma = [r[2]['noise_sigma'] for r in rows if r[0] == 'clean']

    result = {'source': f"{len(paths)} images from {args.images_dir} (seed {args.seed})", 'rules': rules,
              'balanced_accuracy_per_rule': rule_scores, 'identification_accuracy': round(accuracy, 3)}
    Path(args.out).write_text(json.dumps(result, indent=2), encoding='utf-8')

    lines = ["# Calibration of --restore auto", "", result['source'], "",
             "| Rule | Threshold | Balanced accuracy |", "|---|---|---|"]
    lines += [f"| {r['feature']} {r['op']} t -> {r['method']} | {r['threshold']:.4g} | {rule_scores[r['feature']]} |"
              for r in rules]
    lines += ["", f"Identification accuracy of the cascade (degradation type, clean = none): {100 * accuracy:.1f}%", "",
              "| True \\ identified | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    lines += [f"| {g} | " + " | ".join(str(confusion[g][h]) for h in groups) + " |" for g in groups]
    lines += ["", "Noise estimate (wavelet MAD): "
              + "; ".join(f"level {lv} (sigma {LEVELS['gauss_noise'][lv]['sigma']:g}): mean error "
                          f"{np.mean(e):+.1f}, mean abs error {np.mean(np.abs(e)):.1f}" for lv, e in noise_errors.items())
              + f"; clean images: estimated sigma {np.mean(clean_sigma):.1f} on average"]
    report = '\n'.join(lines) + '\n'
    print(report)
    if args.report_dir:
        out = Path(args.report_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'report.md').write_text(report, encoding='utf-8')
        (out / 'calibration.json').write_text(json.dumps({**result, 'confusion': {g: dict(confusion[g]) for g in groups}},
                                                         indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
