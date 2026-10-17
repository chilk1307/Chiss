"""
"Keep - edit" analysis of the errors (plan 01, section 1.11; experiment E1.11).

    python tools/keep_edit_analysis.py --dataset cirr --dataset-path $DATA/CIRR --runs R/E1.7d-phiB32-cirr \
        R/E1.8-sum-cirr --cache-dir $DRIVE/cache --out R/E1.11-keepedit-cirr

For every failed query (top-1 is not a correct image) the top-1 image is compared with the target in CLIP space:
    s_ref = cos(reference image, x)    how much x keeps the reference image
    s_txt = cos(caption, x)            how much x follows the caption (CLIP text features of the caption alone)
and the error is classified by the signs of d_ref = s_ref(top1) - s_ref(target), d_txt = s_txt(top1) - s_txt(target):
    copy_reference    d_ref > 0, d_txt <= 0   the top-1 is closer to the reference but follows the caption less
    ignore_reference  d_ref <= 0, d_txt > 0   the top-1 follows the caption but drops the reference
    both_higher       d_ref > 0, d_txt > 0    the top-1 is closer on both (ambiguous query or annotation)
    both_lower        d_ref <= 0, d_txt <= 0  the top-1 is worse on both (error of the composition itself)
Writes <out>/report.md, keep_edit.json, per_query.csv and a scatter plot per run.
"""
import csv
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import retrieval_core as core  # noqa: E402

CLASSES = ['copy_reference', 'ignore_reference', 'both_higher', 'both_lower']


def pipeline_for(args):
    from argparse import ArgumentParser as AP
    parser = AP()
    core.add_eval_args(parser, datasets=['cirr', 'fashioniq', 'circo'])
    eval_args = parser.parse_args(['--eval-type', 'image', '--dataset', args.dataset, '--dataset-path',
                                   args.dataset_path, '--clip-model-name', args.clip_model_name,
                                   '--num-workers', str(args.num_workers)] +
                                  (['--cache-dir', args.cache_dir] if args.cache_dir else []))
    return core.RetrievalPipeline(eval_args, 'val')


def classify(d_ref: float, d_txt: float) -> str:
    if d_ref > 0:
        return 'copy_reference' if d_txt <= 0 else 'both_higher'
    return 'ignore_reference' if d_txt > 0 else 'both_lower'


@torch.no_grad()
def main():
    parser = ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--dataset', required=True, choices=['cirr', 'fashioniq', 'circo'])
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--cache-dir', default=None)
    parser.add_argument('--clip-model-name', default='ViT-B/32')
    parser.add_argument('--num-workers', type=int, default=8)
    args = parser.parse_args()

    pipeline = pipeline_for(args)
    ref_features, ref_names = pipeline.reference_features()
    ref_index = {n: i for i, n in enumerate(ref_names)}
    ref_features = F.normalize(ref_features.float(), dim=-1)
    galleries = {}

    def gallery(category):
        if category not in galleries:
            features, names = pipeline.gallery_features(category)
            galleries[category] = (features.cpu(), {n: i for i, n in enumerate(names)})
        return galleries[category]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary, rows = {}, []
    for run_dir in map(Path, args.runs):
        with open(run_dir / 'dump.jsonl', encoding='utf-8') as f:
            records = [r for r in map(json.loads, f) if 'per_query' in r and r.get('target')]
        queries = [core.Query(qid=r['qid'], reference=r['reference'],
                              captions=r['captions'] if 'captions' in r else [r['caption']], target=r['target'],
                              category=r.get('category')) for r in records]
        fiq = args.dataset == 'fashioniq'
        text = F.normalize(core.encode_text_queries(pipeline.clip_model, queries, fiq).float().cpu(), dim=-1)
        counts = {c: 0 for c in CLASSES}
        points = {c: [] for c in CLASSES}
        n_success = 0
        for r, q, t in zip(records, queries, text):
            top1 = r['top'][0][0]
            correct = {r['target']} | set(r.get('gt') or [])
            if top1 in correct:
                n_success += 1
                continue
            features, index = gallery(q.category)
            if top1 not in index or q.target not in index or q.reference not in ref_index:
                continue
            v = ref_features[ref_index[q.reference]]
            g_top, g_target = features[index[top1]], features[index[q.target]]
            d_ref = float(v @ g_top - v @ g_target)
            d_txt = float(t @ g_top - t @ g_target)
            label = classify(d_ref, d_txt)
            counts[label] += 1
            points[label].append((d_ref, d_txt))
            rows.append({'run': run_dir.name, 'qid': q.qid, 'top1': top1, 'target': q.target, 'd_ref': round(d_ref, 4),
                         'd_txt': round(d_txt, 4), 'class': label})
        n_fail = sum(counts.values())
        summary[run_dir.name] = {'queries': len(records), 'success_top1': n_success, 'failures': n_fail,
                                 'rates': {c: round(100 * counts[c] / max(n_fail, 1), 1) for c in CLASSES}}
        plot(points, run_dir.name, out / f"scatter_{run_dir.name}.png")

    with open(out / 'per_query.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['run', 'qid', 'top1', 'target', 'd_ref', 'd_txt', 'class'])
        writer.writeheader()
        writer.writerows(rows)
    (out / 'keep_edit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    lines = [f"# Keep - edit analysis ({args.dataset})", "",
             "Share of each error type among the failed queries (top-1 not correct).", "",
             "| Run | Queries | Top-1 correct | Failures | " + " | ".join(CLASSES) + " |",
             "|---|---|---|---|" + "---|" * len(CLASSES)]
    for name, s in summary.items():
        lines.append(f"| {name} | {s['queries']} | {s['success_top1']} | {s['failures']} | "
                     + " | ".join(f"{s['rates'][c]}%" for c in CLASSES) + " |")
    report = '\n'.join(lines) + '\n'
    (out / 'report.md').write_text(report, encoding='utf-8')
    print(report)


def plot(points, title, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 5))
    colors = {'copy_reference': 'tab:red', 'ignore_reference': 'tab:blue', 'both_higher': 'tab:gray',
              'both_lower': 'tab:orange'}
    for label, pts in points.items():
        if pts:
            x, y = zip(*pts)
            ax.scatter(x, y, s=8, alpha=0.6, c=colors[label], label=f"{label} ({len(pts)})")
    ax.axhline(0, color='black', lw=0.8)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('s_ref(top-1) - s_ref(target)   (keeps the reference more ->)')
    ax.set_ylabel('s_txt(top-1) - s_txt(target)   (follows the caption more ->)')
    ax.set_title(f"Failed queries: {title}", fontsize=9)
    ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    main()
