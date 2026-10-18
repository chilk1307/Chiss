"""
Image quality (PSNR / SSIM) versus retrieval after restoration (plan 03; experiment E3.4).

    python tools/quality_vs_retrieval.py --runs R/E3.1-* R/E3.2-* --out R/E3.4-quality-cirr

The runs must have been made with --save-quality. They are grouped by dataset and degradation (--degrade); in each
group the run without restoration (--restore none, e.g. E3.1) is the reference, and for every restoration:
    gain in PSNR, gain in SSIM          (mean over the reference images)
    gain in retrieval                   (main metric: CIRR R@1, CIRCO mAP@10, FashionIQ average R@10)
The report gives, per degradation, the best restoration for PSNR, for SSIM and for retrieval ("best for the eye" vs
"best for the search engine"), the Spearman correlation between quality gain and retrieval gain over all
(degradation, restoration) pairs, and the per-query correlation between SSIM gain and target-rank improvement.
"""
import json
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_utils import load_run  # noqa: E402

MAIN = {'cirr': 'cirr_recall_at1', 'circo': 'circo_map_at10', 'fashioniq': 'average_fiq_recall_at10'}


def spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 3:
        return None, None
    rho, p = spearmanr(x, y)
    return (None, None) if np.isnan(rho) else (float(rho), float(p))


def main():
    parser = ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    groups = defaultdict(dict)
    for run in map(Path, args.runs):
        if not (run / 'metrics.json').exists():
            continue
        metrics, records = load_run(run)
        a = metrics['args']
        quality = metrics.get('info', {}).get('reference_processing', {})
        if a.get('degrade', 'none') == 'none' or 'mean_psnr' not in quality:
            print(f"skipped {run.name} (no degradation or no --save-quality)")
            continue
        groups[(metrics['dataset'], a['degrade'])][a.get('restore', 'none')] = {
            'run': run.name, 'metric': metrics['metrics'][MAIN[metrics['dataset']]],
            'psnr': quality['mean_psnr'], 'ssim': quality.get('mean_ssim'), 'records': records}

    rows, pairs, per_query = [], [], {'ssim_gain': [], 'rank_gain': []}
    for (dataset, degradation), runs in sorted(groups.items()):
        base = runs.get('none')
        if base is None:
            print(f"no run without restoration for {dataset} {degradation}: skipped")
            continue
        base_ranks = {r['qid']: (r.get('target_rank') or 101, r.get('reference_processing', {}).get('ssim'))
                      for r in base['records']}
        for method, r in sorted(runs.items()):
            row = {'dataset': dataset, 'degradation': degradation, 'restoration': method, 'run': r['run'],
                   'metric': r['metric'], 'psnr': r['psnr'], 'ssim': r['ssim'],
                   'metric_gain': r['metric'] - base['metric'], 'psnr_gain': r['psnr'] - base['psnr'],
                   'ssim_gain': (r['ssim'] - base['ssim']) if r['ssim'] is not None and base['ssim'] is not None
                   else None}
            rows.append(row)
            if method == 'none':
                continue
            pairs.append(row)
            for rec in r['records']:
                if rec['qid'] in base_ranks and 'reference_processing' in rec:
                    rank0, ssim0 = base_ranks[rec['qid']]
                    ssim1 = rec['reference_processing'].get('ssim')
                    if ssim0 is not None and ssim1 is not None:
                        per_query['ssim_gain'].append(ssim1 - ssim0)
                        per_query['rank_gain'].append(np.log(rank0) - np.log(rec.get('target_rank') or 101))

    rho_psnr = spearman([p['psnr_gain'] for p in pairs], [p['metric_gain'] for p in pairs])
    rho_ssim = spearman([p['ssim_gain'] for p in pairs if p['ssim_gain'] is not None],
                        [p['metric_gain'] for p in pairs if p['ssim_gain'] is not None])
    rho_query = spearman(per_query['ssim_gain'], per_query['rank_gain'])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# E3.4 Image quality vs retrieval after restoration", "",
             f"Spearman over the (degradation, restoration) pairs: PSNR gain vs retrieval gain rho = "
             f"{rho_psnr[0] if rho_psnr[0] is None else round(rho_psnr[0], 3)}; SSIM gain vs retrieval gain rho = "
             f"{rho_ssim[0] if rho_ssim[0] is None else round(rho_ssim[0], 3)} ({len(pairs)} pairs).",
             f"Per query: SSIM gain vs log target-rank improvement rho = "
             f"{rho_query[0] if rho_query[0] is None else round(rho_query[0], 3)} ({len(per_query['ssim_gain'])} "
             f"queries).", "",
             "| Dataset | Degradation | Restoration | Metric | Δ metric | PSNR | Δ PSNR | SSIM | Δ SSIM | Best for |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted({(r['dataset'], r['degradation']) for r in rows}):
        group = [r for r in rows if (r['dataset'], r['degradation']) == key]
        best = {'PSNR': max(group, key=lambda r: r['psnr'])['restoration'],
                'SSIM': max(group, key=lambda r: r['ssim'] or -1)['restoration'],
                'retrieval': max(group, key=lambda r: r['metric'])['restoration']}
        for r in group:
            tags = ', '.join(k for k, v in best.items() if v == r['restoration'])
            lines.append(f"| {r['dataset']} | {r['degradation']} | {r['restoration']} | {r['metric']:.2f} | "
                         f"{r['metric_gain']:+.2f} | {r['psnr']:.1f} | {r['psnr_gain']:+.1f} | "
                         f"{(r['ssim'] or 0):.3f} | {(r['ssim_gain'] or 0):+.3f} | {tags} |")
    report = '\n'.join(lines) + '\n'
    (out / 'report.md').write_text(report, encoding='utf-8')
    (out / 'quality_vs_retrieval.json').write_text(json.dumps(
        {'rows': rows and [{k: v for k, v in r.items()} for r in rows], 'spearman_psnr': rho_psnr,
         'spearman_ssim': rho_ssim, 'spearman_per_query': rho_query}, indent=2), encoding='utf-8')

    if pairs:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for ax, key, label in ((axes[0], 'psnr_gain', 'PSNR gain (dB)'), (axes[1], 'ssim_gain', 'SSIM gain')):
            points = [p for p in pairs if p[key] is not None]
            ax.scatter([p[key] for p in points], [p['metric_gain'] for p in points], s=18)
            for p in points:
                ax.annotate(f"{p['degradation']}+{p['restoration']}", (p[key], p['metric_gain']), fontsize=6)
            ax.axhline(0, color='gray', lw=0.8)
            ax.axvline(0, color='gray', lw=0.8)
            ax.set_xlabel(label)
            ax.set_ylabel('retrieval gain')
        fig.tight_layout()
        fig.savefig(out / 'quality_vs_retrieval.png', dpi=120)
        plt.close(fig)
    print(report)


if __name__ == '__main__':
    main()
