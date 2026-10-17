"""
Collect every <results-dir>/<run>/metrics.json written by validate.py / generate_test_submission.py into one
Markdown summary, one table per dataset (plan 01, section 1.3).

    python tools/collect_results.py --results-dir $DRIVE/results --out $DRIVE/results/summary.md
"""
import json
from argparse import ArgumentParser
from pathlib import Path

COLUMNS = {
    'cirr': [('R@1', 'cirr_recall_at1'), ('R@5', 'cirr_recall_at5'), ('R@10', 'cirr_recall_at10'),
             ('R@50', 'cirr_recall_at50'), ('Rs@1', 'cirr_group_recall_at1'), ('Rs@2', 'cirr_group_recall_at2'),
             ('Rs@3', 'cirr_group_recall_at3')],
    'circo': [('mAP@5', 'circo_map_at5'), ('mAP@10', 'circo_map_at10'), ('mAP@25', 'circo_map_at25'),
              ('mAP@50', 'circo_map_at50')],
    'fashioniq': [('R@10', 'average_fiq_recall_at10'), ('R@50', 'average_fiq_recall_at50'),
                  ('shirt R@10', 'shirt_fiq_recall_at10'), ('dress R@10', 'dress_fiq_recall_at10'),
                  ('toptee R@10', 'toptee_fiq_recall_at10')],
}
HIDDEN_ARGS = {'eval_type', 'dataset', 'exp_name', 'phi_checkpoint_name', 'submission_name'}


def describe_method(m: dict) -> str:
    args = m.get('args', {})
    method = m.get('eval_type', '')
    if args.get('exp_name'):
        method += f" {args['exp_name']}"
    if args.get('phi_checkpoint_name'):
        method += f"/{args['phi_checkpoint_name']}"
    return method


def describe_flags(m: dict) -> str:
    flags = m.get('nondefault_args', {})
    parts = []
    for k, v in flags.items():
        if k in HIDDEN_ARGS:
            continue
        flag = '--' + k.replace('_', '-')
        parts.append(flag if v is True else f"{flag} {v}")
    return ' '.join(parts)


def main():
    parser = ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    runs = []
    for path in sorted(results_dir.glob('*/metrics.json')):
        try:
            runs.append(json.load(open(path, encoding='utf-8')))
        except (OSError, json.JSONDecodeError) as e:
            print(f"WARNING: cannot read {path}: {e}")

    lines = ["# Summary of runs", "", f"Source: `{results_dir}` ({len(runs)} runs). FIQ = FashionIQ val subset "
             "(images with dead links removed).", ""]
    for dataset, columns in COLUMNS.items():
        rows = [m for m in runs if m.get('dataset') == dataset and m.get('script') == 'validate']
        if not rows:
            continue
        lines += [f"## {dataset} val", "",
                  "| Code | Method | Options | " + " | ".join(c for c, _ in columns) + " | Queries | Runner | Date | Note |",
                  "|---|---|---|" + "---|" * len(columns) + "---|---|---|---|"]
        for m in sorted(rows, key=lambda r: r.get('exp_code', '')):
            values = [f"{m['metrics'][k]:.2f}" if k in m.get('metrics', {}) else '' for _, k in columns]
            info = m.get('info', {})
            warn = ' ⚠' if info.get('load_errors') or info.get('dropped_queries') else ''
            lines.append(f"| {m.get('exp_code', '')} | {describe_method(m)} | {describe_flags(m)} | "
                         + " | ".join(values)
                         + f" | {info.get('n_queries', '')}{warn} | {m.get('runner', '')} | "
                           f"{m.get('timestamp', '')[:10]} | {m.get('note', '')} |")
        lines.append("")

    submissions = [m for m in runs if m.get('script') == 'generate_test_submission']
    if submissions:
        lines += ["## Test submission files", "", "| Code | Dataset | Method | Options | Files | Runner | Date |",
                  "|---|---|---|---|---|---|---|"]
        for m in sorted(submissions, key=lambda r: r.get('exp_code', '')):
            files = ', '.join(Path(f).name for f in m.get('info', {}).get('submission_files', []))
            lines.append(f"| {m.get('exp_code', '')} | {m.get('dataset', '')} | {describe_method(m)} | "
                         f"{describe_flags(m)} | {files} | {m.get('runner', '')} | {m.get('timestamp', '')[:10]} |")
        lines.append("")
    lines.append("⚠ = some images could not be loaded or some queries were dropped (see `info` in metrics.json).")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print(f"\nWritten {out}")


if __name__ == '__main__':
    main()
