"""
Metrics per modification category (plan 01, section 1.10; experiments E1.10 and E4.3b).

    python tools/query_taxonomy.py --dataset cirr --runs R/E1.7a-searle-cirr R/E1.7d-phiB32-cirr --out R/E1.10-taxonomy-cirr
    python tools/query_taxonomy.py --dataset cirr --categories spatial --baseline R/E1.7d-phiB32-cirr \
        --runs R/E4.3-flip-ref-cirr --out R/E4.3b-leftright-cirr

The captions are read from the runs' dump.jsonl (or from the dataset with --dataset-path when no run is given). Every
query is put in zero or more categories (src/attributes/taxonomy.py); each table cell is the mean of the per-query
metric in the category, with its bootstrap 95% CI, and with --baseline the paired difference with the baseline run.
FashionIQ queries of the three garment types are pooled inside a category.

Hand check of the rules (plan 01, 1.10 and plan 02, 2.1):
    --grading-sheet 100 [--seed 0]   writes grading_sheet.csv with random queries, the parsed constraints and the
                                     categories, and empty columns to fill in (y / n)
    --grade grading_sheet.csv        reads the filled sheet and reports the share of correct rows
"""
import csv
import json
import random
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from attributes.parser import parse_query  # noqa: E402
from attributes.taxonomy import CATEGORIES, categorize_query  # noqa: E402

MAIN_METRICS = {'cirr': ['cirr_recall_at1', 'cirr_recall_at5', 'cirr_group_recall_at1'],
                'circo': ['circo_map_at5', 'circo_map_at10'],
                'fashioniq': ['fiq_recall_at10', 'fiq_recall_at50']}


def captions_of(record: dict):
    return record['captions'] if 'captions' in record else [record['caption']]


def load_dump(run: Path):
    with open(run / 'dump.jsonl', encoding='utf-8') as f:
        return {r['qid']: r for r in map(json.loads, f) if 'per_query' in r}


def queries_from_dataset(dataset: str, path: str):
    from attribute_stats import load_queries
    pairs, _, _ = load_queries(dataset, path, 'val')
    return {q.qid: {'qid': q.qid, 'captions': q.captions} for _, q in pairs}


def bootstrap(values: np.ndarray, n_boot: int, rng: np.random.Generator):
    if len(values) == 0:
        return float('nan'), float('nan'), float('nan')
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1) * 100
    low, high = np.percentile(means, [2.5, 97.5])
    return float(values.mean() * 100), float(low), float(high)


def grading_sheet(records: dict, n: int, seed: int, path: Path) -> None:
    rng = random.Random(seed)
    chosen = rng.sample(sorted(records), min(n, len(records)))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['qid', 'caption', 'parsed_constraints', 'categories', 'constraints_ok (y/n)',
                         'missed_or_wrong_constraints', 'categories_ok (y/n)', 'notes'])
        for qid in chosen:
            caps = captions_of(records[qid])
            writer.writerow([qid, ' || '.join(caps), '; '.join(c.describe() for c in parse_query(caps)),
                             ', '.join(categorize_query(caps)), '', '', '', ''])
    print(f"Grading sheet with {len(chosen)} queries written to {path}")


def grade(path: Path) -> None:
    with open(path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for column in ('constraints_ok (y/n)', 'categories_ok (y/n)'):
        marks = [r[column].strip().lower() for r in rows if r[column].strip()]
        if marks:
            ok = sum(m.startswith('y') for m in marks)
            print(f"{column}: {ok}/{len(marks)} = {100 * ok / len(marks):.1f}% correct")


def main():
    parser = ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--dataset', required=True, choices=list(MAIN_METRICS))
    parser.add_argument('--dataset-path', default=None, help='Only needed without --runs')
    parser.add_argument('--runs', nargs='*', default=[])
    parser.add_argument('--baseline', default=None, help='Run compared (paired) with every run')
    parser.add_argument('--categories', default=None, help='Comma separated subset of ' + ','.join(CATEGORIES))
    parser.add_argument('--out', default=None)
    parser.add_argument('--n-boot', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--grading-sheet', type=int, default=0, help='Write a hand-grading sheet of N queries')
    parser.add_argument('--grade', default=None, help='Score a filled grading sheet')
    args = parser.parse_args()

    if args.grade:
        grade(Path(args.grade))
        return
    categories = CATEGORIES if not args.categories else [c.strip() for c in args.categories.split(',')]
    unknown = set(categories) - set(CATEGORIES)
    if unknown:
        raise SystemExit(f"unknown categories {sorted(unknown)}; choose among {CATEGORIES}")
    run_paths = ([Path(args.baseline)] if args.baseline else []) + [Path(r) for r in args.runs]
    run_paths = list(dict.fromkeys(run_paths))
    runs = {p.name: load_dump(p) for p in run_paths}
    if runs:
        records = next(iter(runs.values()))
    elif args.dataset_path:
        records = queries_from_dataset(args.dataset, args.dataset_path)
    else:
        raise SystemExit("give --runs or --dataset-path")
    if not args.out:
        raise SystemExit("--out is required")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.grading_sheet:
        grading_sheet(records, args.grading_sheet, args.seed, out / 'grading_sheet.csv')

    query_categories = {qid: categorize_query(captions_of(r)) for qid, r in records.items()}
    groups = {'all': list(records)}
    for c in categories:
        groups[c] = [qid for qid, cats in query_categories.items() if c in cats]
    if not args.categories:
        groups['other'] = [qid for qid, cats in query_categories.items() if not cats]
    with open(out / 'categories.jsonl', 'w', encoding='utf-8') as f:
        for qid, cats in query_categories.items():
            f.write(json.dumps({'qid': qid, 'categories': cats}) + '\n')

    metrics = MAIN_METRICS[args.dataset]
    rng = np.random.default_rng(args.seed)
    baseline = runs.get(Path(args.baseline).name) if args.baseline else None
    result = {'n_queries': len(records), 'groups': {}}
    lines = [f"# Metrics per modification category ({args.dataset})", "",
             f"{len(records)} queries; a query can be in several categories. Values: mean [95% CI]" +
             ("; Δ = paired difference with the baseline, * = CI excludes 0." if baseline else "."), ""]
    for metric in metrics:
        lines += [f"## {metric}", "", "| Category | Queries | " + " | ".join(runs) + " |",
                  "|---|---|" + "---|" * len(runs)]
        for group, qids in groups.items():
            cells = []
            for name, run in runs.items():
                shared = [q for q in qids if q in run]
                if not shared:
                    cells.append('—')
                    continue
                values = np.array([run[q]['per_query'][metric] for q in shared])
                mean, low, high = bootstrap(values, args.n_boot, rng)
                cell = f"{mean:.1f} [{low:.1f}, {high:.1f}]"
                entry = {'n': len(shared), 'mean': mean, 'ci': [low, high]}
                if baseline is not None and run is not baseline:
                    paired = [q for q in shared if q in baseline]
                    diff = np.array([run[q]['per_query'][metric] - baseline[q]['per_query'][metric] for q in paired])
                    d, dl, dh = bootstrap(diff, args.n_boot, rng)
                    cell += f" Δ {d:+.1f} [{dl:+.1f}, {dh:+.1f}]{'*' if dl > 0 or dh < 0 else ''}"
                    entry.update({'diff': d, 'diff_ci': [dl, dh]})
                result['groups'].setdefault(group, {}).setdefault(name, {})[metric] = entry
                cells.append(cell)
            lines.append(f"| {group} | {len(qids)} | " + " | ".join(cells) + " |")
        lines.append("")
    (out / 'taxonomy.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    report = '\n'.join(lines) + '\n'
    (out / 'report.md').write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
