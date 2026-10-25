"""
Compare groups of runs that differ only by the training / optimisation seed (E10: OTI with several GPT phrase sets).

    python tools/seed_bootstrap.py --baseline GPTNeo27B \
        --set GPTNeo27B=R/E7.1a-...-ema_oti_pseudo_tokens,R/E10.1a-gptneo-seed43-circo-ema350,... \
        --set P2=R/E10.1b-P2-circo-ema350,R/E10.1b-P2-seed43-circo-ema350,... --out R/E10.1-seeds-circo

Each folder holds the dump.jsonl of validate.py (per-query metrics). For every set the per-query values are averaged
over its seeds; two 95% intervals of the difference with the baseline set are reported:
    queries        paired bootstrap of the queries only (the seed-averaged values), like tools/bootstrap_ci.py
    seeds+queries  two-level bootstrap: the seeds of each set are resampled with replacement, then the queries; this
                   also counts the seed noise, which bootstrap_ci.py does not
Writes <out>/seeds.md and <out>/seeds.json.
"""
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_ci import Run  # noqa: E402


def main():
    parser = ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--set', action='append', required=True, help='NAME=folder,folder,...')
    parser.add_argument('--baseline', required=True, help='NAME of the reference set')
    parser.add_argument('--metrics', default=None, help='Comma-separated per-query metrics (default: all)')
    parser.add_argument('--n-boot', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    sets = {}
    for spec in args.set:
        name, folders = spec.split('=', 1)
        runs = [Run(Path(f)) for f in folders.split(',') if f]
        sets[name] = [r for r in runs if r.records]
    qids = sorted(set.intersection(*[set(r.records) for runs in sets.values() for r in runs]))
    first = next(iter(sets.values()))[0]
    metrics = args.metrics.split(',') if args.metrics else first.metrics
    # values[name]: (n_seeds x n_queries x n_metrics)
    values = {n: np.stack([r.values(qids, metrics) for r in runs]) for n, runs in sets.items()}
    rng = np.random.default_rng(args.seed)
    base = values[args.baseline]
    rows = []
    for name, v in values.items():
        per_seed = v.mean(axis=1) * 100                      # n_seeds x n_metrics
        row = {'set': name, 'seeds': len(v), 'mean': per_seed.mean(0).tolist(),
               'sd_seeds': (per_seed.std(0, ddof=1) if len(v) > 1 else np.zeros(len(metrics))).tolist(),
               'per_seed': per_seed.tolist()}
        if name != args.baseline:
            diff_q = v.mean(0) - base.mean(0)                # n_queries x n_metrics, seed-averaged
            idx = rng.integers(0, len(qids), size=(args.n_boot, len(qids)))
            boot_q = diff_q[idx].mean(1) * 100
            boot_2 = np.empty((args.n_boot, len(metrics)))
            for b in range(args.n_boot):
                sa = v[rng.integers(0, len(v), len(v))].mean(0)
                sb = base[rng.integers(0, len(base), len(base))].mean(0)
                q = rng.integers(0, len(qids), len(qids))
                boot_2[b] = (sa[q] - sb[q]).mean(0) * 100
            row['delta'] = (diff_q.mean(0) * 100).tolist()
            row['ci_queries'] = np.percentile(boot_q, [2.5, 97.5], axis=0).T.tolist()
            row['ci_seeds_queries'] = np.percentile(boot_2, [2.5, 97.5], axis=0).T.tolist()
        rows.append(row)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'seeds.json').write_text(json.dumps({'metrics': metrics, 'queries': len(qids), 'rows': rows,
                                                'baseline': args.baseline}, indent=2), encoding='utf-8')
    lines = [f"# Seed groups vs {args.baseline} ({len(qids)} shared queries, {args.n_boot} resamples)", '',
             '| Set | Seeds | Metric | Mean | SD over seeds | Per seed | Δ | 95% CI (queries) | 95% CI (seeds+queries) |',
             '|---|---|---|---|---|---|---|---|---|']
    for row in rows:
        for i, m in enumerate(metrics):
            seeds = ' / '.join(f"{s[i]:.2f}" for s in row['per_seed'])
            if 'delta' in row:
                d = row['delta'][i]
                cq, c2 = row['ci_queries'][i], row['ci_seeds_queries'][i]
                star = ' *' if c2[0] > 0 or c2[1] < 0 else ''
                extra = f"{d:+.2f}{star} | [{cq[0]:+.2f}, {cq[1]:+.2f}] | [{c2[0]:+.2f}, {c2[1]:+.2f}]"
            else:
                extra = '– | – | –'
            lines.append(f"| {row['set']} | {row['seeds']} | {m} | {row['mean'][i]:.2f} | {row['sd_seeds'][i]:.2f} | "
                         f"{seeds} | {extra} |")
    lines += ['', '* the seeds+queries interval excludes 0.']
    (out / 'seeds.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
