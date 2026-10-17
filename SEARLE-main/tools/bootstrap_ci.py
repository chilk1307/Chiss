"""
Bootstrap 95% confidence intervals of the validation metrics (plan 01, section 1.4; experiment E1.4).

    python tools/bootstrap_ci.py --runs R/E1.7a-searle-cirr R/E1.8-*-cirr --baseline R/E1.7d-phiB32-cirr --out R/E1.4-ci-cirr

Each run folder must contain the dump.jsonl written by validate.py --output-dir (one line per query with the per-query
metrics). Queries are resampled with replacement --n-boot times; for FashionIQ the resampling is done inside each
category and the average over categories is reported, like the official metric.
With --baseline, every run is also compared with the baseline on the queries they share (paired bootstrap: the same
resampled queries for both): the difference is called significant when its 95% interval does not contain 0.
Bootstrap measures the noise due to the choice of queries only, not the noise due to training seeds.
Writes <out>/ci.md and <out>/ci.json.
"""
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class Run:
    def __init__(self, folder: Path):
        self.folder = folder
        self.name = folder.name
        self.records: Dict[str, dict] = {}
        with open(folder / 'dump.jsonl', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                if 'per_query' in record:
                    self.records[record['qid']] = record
        self.metrics = list(next(iter(self.records.values()))['per_query']) if self.records else []

    def categories(self, qids: List[str]) -> Dict[Optional[str], List[str]]:
        groups: Dict[Optional[str], List[str]] = {}
        for qid in qids:
            groups.setdefault(self.records[qid].get('category'), []).append(qid)
        return groups

    def values(self, qids: List[str], metrics: List[str]) -> np.ndarray:
        return np.array([[self.records[q]['per_query'][m] for m in metrics] for q in qids], dtype=np.float64)


def bootstrap_means(groups: Dict[Optional[str], np.ndarray], n_boot: int, seed: int) -> Dict[str, np.ndarray]:
    """
    groups: category -> (n_queries x n_metrics) values. Returns category (and 'average' when there are several
    categories) -> (n_boot x n_metrics) resampled means, in percent.
    """
    rng = np.random.default_rng(seed)
    result = {}
    for category, values in groups.items():
        idx = rng.integers(0, len(values), size=(n_boot, len(values)))
        result[category] = values[idx].mean(axis=1) * 100
    if len(groups) > 1:
        result['average'] = np.mean([result[c] for c in groups], axis=0)
    return result


def point_estimates(groups: Dict[Optional[str], np.ndarray]) -> Dict[str, np.ndarray]:
    result = {c: v.mean(axis=0) * 100 for c, v in groups.items()}
    if len(groups) > 1:
        result['average'] = np.mean([result[c] for c in groups], axis=0)
    return result


def metric_name(category: Optional[str], metric: str) -> str:
    return metric if category is None else f"{category}_{metric}"


def analyse(run: Run, baseline: Optional[Run], n_boot: int, seed: int) -> List[dict]:
    rows = []
    metrics = run.metrics
    qids = list(run.records)
    groups = {c: run.values(q, metrics) for c, q in run.categories(qids).items()}
    estimates, boots = point_estimates(groups), bootstrap_means(groups, n_boot, seed)
    for category in estimates:
        for j, m in enumerate(metrics):
            low, high = np.percentile(boots[category][:, j], [2.5, 97.5])
            rows.append({'run': run.name, 'metric': metric_name(category, m), 'n_queries': len(qids),
                         'value': round(float(estimates[category][j]), 2),
                         'ci_low': round(float(low), 2), 'ci_high': round(float(high), 2)})

    if baseline is None or baseline.folder.resolve() == run.folder.resolve():
        return rows
    shared_metrics = [m for m in metrics if m in baseline.metrics]
    common = [q for q in qids if q in baseline.records]
    if not common:
        print(f"WARNING: {run.name} and the baseline {baseline.name} share no query, no paired comparison")
        return rows
    if len(common) < max(len(qids), len(baseline.records)):
        print(f"WARNING: {run.name}: paired comparison on {len(common)} shared queries "
              f"(run {len(qids)}, baseline {len(baseline.records)})")
    common_groups = run.categories(common)
    diff_groups = {c: run.values(q, shared_metrics) - baseline.values(q, shared_metrics)
                   for c, q in common_groups.items()}
    base_groups = {c: baseline.values(q, shared_metrics) for c, q in common_groups.items()}
    diff_estimates, diff_boots = point_estimates(diff_groups), bootstrap_means(diff_groups, n_boot, seed)
    base_estimates = point_estimates(base_groups)
    by_name = {r['metric']: r for r in rows}
    for category in diff_estimates:
        for j, m in enumerate(shared_metrics):
            low, high = np.percentile(diff_boots[category][:, j], [2.5, 97.5])
            row = by_name[metric_name(category, m)]
            row.update({'baseline': baseline.name, 'n_shared': len(common),
                        'baseline_value': round(float(base_estimates[category][j]), 2),
                        'diff': round(float(diff_estimates[category][j]), 2),
                        'diff_low': round(float(low), 2), 'diff_high': round(float(high), 2),
                        'significant': bool(low > 0 or high < 0)})
    return rows


def to_markdown(rows: List[dict], baseline: Optional[str], n_boot: int) -> str:
    lines = [f"# Bootstrap 95% CI ({n_boot} resamples)", ""]
    if baseline:
        lines.append(f"Baseline: `{baseline}` (paired difference on the shared queries; * = 95% CI excludes 0)")
        lines.append("")
        lines.append("| Run | Metric | Value | 95% CI | Δ vs baseline | 95% CI of Δ | |")
        lines.append("|---|---|---|---|---|---|---|")
    else:
        lines.append("| Run | Metric | Value | 95% CI |")
        lines.append("|---|---|---|---|")
    for r in rows:
        line = f"| {r['run']} | {r['metric']} | {r['value']:.2f} | [{r['ci_low']:.2f}, {r['ci_high']:.2f}] |"
        if baseline:
            if 'diff' in r:
                line += f" {r['diff']:+.2f} | [{r['diff_low']:+.2f}, {r['diff_high']:+.2f}] | " \
                        f"{'*' if r['significant'] else ''} |"
            else:
                line += " — | — | |"
        lines.append(line)
    return '\n'.join(lines) + '\n'


def main():
    parser = ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--runs', nargs='+', required=True, help='Run folders (with dump.jsonl)')
    parser.add_argument('--baseline', default=None, help='Run folder to compare every run with')
    parser.add_argument('--out', required=True, help='Output folder')
    parser.add_argument('--n-boot', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    folders = [Path(p) for p in ([args.baseline] if args.baseline else []) + args.runs]
    runs, seen = [], set()
    for folder in folders:
        if folder.resolve() in seen:
            continue
        seen.add(folder.resolve())
        if not (folder / 'dump.jsonl').exists():
            print(f"WARNING: skipped {folder} (no dump.jsonl)")
            continue
        run = Run(folder)
        if not run.records:
            print(f"WARNING: skipped {folder} (no per-query metrics: test submission?)")
            continue
        runs.append(run)
    baseline = runs[0] if args.baseline and runs and runs[0].folder == Path(args.baseline) else None
    if args.baseline and baseline is None:
        raise SystemExit(f"Baseline {args.baseline} has no usable dump.jsonl")

    rows = [row for run in runs for row in analyse(run, baseline, args.n_boot, args.seed)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'ci.json', 'w', encoding='utf-8') as f:
        json.dump({'baseline': baseline.name if baseline else None, 'n_boot': args.n_boot, 'seed': args.seed,
                   'rows': rows}, f, indent=2)
    markdown = to_markdown(rows, baseline.name if baseline else None, args.n_boot)
    (out / 'ci.md').write_text(markdown, encoding='utf-8')
    print(markdown)
    print(f"Written {out / 'ci.md'}")


if __name__ == '__main__':
    main()
