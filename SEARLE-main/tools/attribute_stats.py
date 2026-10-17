"""
Attribute statistics of a dataset (plan 02).

E2.0 (default): how many queries ask for each measurable attribute (captions only, no image needed)
    python tools/attribute_stats.py --dataset cirr --dataset-path $DATA/CIRR --out $R/E2.0-stats-cirr

E2.2 (--agreement): do the meters agree with the captions on the real (reference, target) pairs? For every query
with constraints, the reference and the target images are measured and, per constraint:
    relative constraint (darker, more colourful, closer...): agreement if c(target vs reference) > 0.5
    absolute constraint (red, striped, solid...): agreement if c(target) > c(reference), i.e. the target moved
                                                  towards the requested attribute
The same rule with a random gallery image instead of the target gives the chance level.
    python tools/attribute_stats.py --agreement --attr-mask auto --dataset fashioniq --dataset-path $DATA/FashionIQ \
        --cache-dir $DRIVE/cache --out $R/E2.2-agree-fiq

E3.6: add --degrade (and --restore) to process the reference images, e.g. --degrade color_cast:2 --restore gray_world

Writes <out>/stats.json and <out>/report.md.
"""
import json
import random
import sys
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from attributes.meters import MASK_MODES, satisfaction_batch  # noqa: E402
from attributes.parser import FAMILIES, parse_query  # noqa: E402

FIQ_CATEGORIES = ['dress', 'shirt', 'toptee']


def load_queries(dataset: str, path: str, split: str):
    """
    (group, query) pairs and functions giving the image files of references, targets and gallery images
    """
    from datasets import CIRRDataset, CIRCODataset, FashionIQDataset
    from retrieval_core import cirr_queries, circo_queries, fiq_queries
    if dataset == 'cirr':
        d = CIRRDataset(path, split, 'relative', None)
        return [('all', q) for q in cirr_queries(d)], {'all': d.image_path}, {'all': list(d.name_to_relpath)}
    if dataset == 'circo':
        d = CIRCODataset(path, split, 'relative', None)
        names = [str(i) for i in d.img_ids]
        return [('all', q) for q in circo_queries(d)], {'all': d.image_path}, {'all': names}
    pairs, paths, galleries = [], {}, {}
    for category in FIQ_CATEGORIES:
        d = FashionIQDataset(path, split, [category], 'relative', None)
        pairs += [(category, q) for q in fiq_queries(d, category)]
        paths[category] = d.image_path
        galleries[category] = d.image_names
    return pairs, paths, galleries


def frequency_stats(pairs) -> dict:
    groups = defaultdict(list)
    for group, q in pairs:
        groups[group].append(q)
        if group != 'all':
            groups['all'].append(q)
    stats = {}
    for group, queries in groups.items():
        n = len(queries)
        with_any, negated, relative = 0, 0, 0
        families, values = Counter(), Counter()
        for q in queries:
            constraints = parse_query(q.captions)
            with_any += bool(constraints)
            negated += any(c.negated for c in constraints)
            relative += any(c.relative for c in constraints)
            for family in {c.family for c in constraints}:
                families[family] += 1
            for c in constraints:
                values[f"{'not ' if c.negated else ''}{c.family}:{c.value}"] += 1
        stats[group] = {
            'queries': n, 'with_constraints_pct': round(100 * with_any / max(n, 1), 1),
            'negated_pct': round(100 * negated / max(n, 1), 1), 'relative_pct': round(100 * relative / max(n, 1), 1),
            'family_pct': {f: round(100 * families[f] / max(n, 1), 1) for f in FAMILIES},
            'top_values': values.most_common(25),
        }
    return stats


def agreement_stats(pairs, paths, galleries, args) -> dict:
    from attributes.rerank import AttributeStore
    from reference_processing import ReferenceProcessor
    processor = ReferenceProcessor(args.degrade, args.restore, None, False)
    processor = processor if processor.active else None
    clean = AttributeStore(args.cache_dir, args.dataset, args.attr_mask, args.num_workers)
    references = clean if processor is None else \
        AttributeStore(args.cache_dir, args.dataset, args.attr_mask, args.num_workers, processor)
    rng = random.Random(args.seed)

    work = [(g, q, parse_query(q.captions)) for g, q in pairs]
    work = [(g, q, cs) for g, q, cs in work if cs and q.target]
    randoms = {q.qid: rng.choice(galleries[g]) for g, q, _ in work}
    by_group = defaultdict(list)
    for g, q, cs in work:
        by_group[g].append((q, cs))
    target_m, ref_m = {}, {}
    for g, items in by_group.items():
        names = [q.target for q, _ in items] + [randoms[q.qid] for q, _ in items]
        target_m.update(clean.get(names, paths[g]))
        ref_m.update(references.get([q.reference for q, _ in items], paths[g]))

    rows = defaultdict(lambda: {'n': 0, 'agree': 0, 'agree_random': 0, 'c_target': []})
    for g, q, cs in work:
        if q.target not in target_m or q.reference not in ref_m or randoms[q.qid] not in target_m:
            continue
        ref = ref_m[q.reference]
        for c in cs:
            candidates = np.stack([target_m[q.target], target_m[randoms[q.qid]], ref])
            values = satisfaction_batch(c, candidates, ref)
            if values is None:
                continue
            c_target, c_random, c_reference = values
            if c.relative:
                agree, agree_random = c_target > 0.5, c_random > 0.5
            else:
                agree, agree_random = c_target > c_reference, c_random > c_reference
            for key in (f"{c.family}", f"{c.family}:{c.value}{' (not)' if c.negated else ''}", 'all'):
                rows[key]['n'] += 1
                rows[key]['agree'] += bool(agree)
                rows[key]['agree_random'] += bool(agree_random)
                rows[key]['c_target'].append(float(c_target))
    table = {}
    for key, r in rows.items():
        table[key] = {'n': r['n'], 'agreement_pct': round(100 * r['agree'] / max(r['n'], 1), 1),
                      'random_pct': round(100 * r['agree_random'] / max(r['n'], 1), 1),
                      'mean_c_target': round(float(np.mean(r['c_target'])), 3) if r['c_target'] else None}
    return {'mask': args.attr_mask, 'degrade': args.degrade, 'restore': args.restore,
            'failed_images': clean.n_failed + (references.n_failed if references is not clean else 0),
            'table': dict(sorted(table.items(), key=lambda kv: (kv[0] != 'all', kv[0])))}


def main():
    parser = ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--dataset', required=True, choices=['cirr', 'circo', 'fashioniq'])
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--split', default='val')
    parser.add_argument('--out', required=True)
    parser.add_argument('--agreement', action='store_true', help='E2.2: measure the (reference, target) pairs')
    parser.add_argument('--attr-mask', default='auto', choices=MASK_MODES)
    parser.add_argument('--cache-dir', default=None)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--degrade', default='none', help='E3.6: degradation of the reference images')
    parser.add_argument('--restore', default='none', help='E3.6: restoration of the reference images')
    parser.add_argument('--seed', type=int, default=0, help='seed of the random baseline images')
    args = parser.parse_args()

    pairs, paths, galleries = load_queries(args.dataset, args.dataset_path, args.split)
    result = {'dataset': args.dataset, 'split': args.split, 'frequency': frequency_stats(pairs)}
    lines = [f"# Attribute statistics: {args.dataset} {args.split}", "", "## E2.0 Share of queries per attribute", "",
             "| Group | Queries | Any constraint | " + " | ".join(FAMILIES) + " | Negated | Relative |",
             "|---|---|---|" + "---|" * len(FAMILIES) + "---|---|"]
    for group, s in result['frequency'].items():
        lines.append(f"| {group} | {s['queries']} | {s['with_constraints_pct']}% | "
                     + " | ".join(f"{s['family_pct'][f]}%" for f in FAMILIES)
                     + f" | {s['negated_pct']}% | {s['relative_pct']}% |")
    lines += ["", "Most frequent constraints: " + ", ".join(f"{v} ({n})" for v, n in
                                                          result['frequency']['all']['top_values'][:15])]
    if args.agreement:
        result['agreement'] = agreement_stats(pairs, paths, galleries, args)
        a = result['agreement']
        lines += ["", f"## E2.2 Agreement meters - captions (mask {a['mask']}, degrade {a['degrade']}, "
                      f"restore {a['restore']})", "",
                  "| Constraint | Pairs | Agreement | Random image | Mean c(target) |", "|---|---|---|---|---|"]
        for key, r in a['table'].items():
            lines.append(f"| {key} | {r['n']} | {r['agreement_pct']}% | {r['random_pct']}% | {r['mean_c_target']} |")
        lines += ["", "M3 rule (plan 02): keep the main contribution if constrained queries are >= 10% in CIRR and FIQ "
                      "and the agreement is >= 65% for most families."]
        if a['failed_images']:
            lines.append(f"\nWARNING: {a['failed_images']} images could not be loaded.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'stats.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    report = '\n'.join(lines) + '\n'
    (out / 'report.md').write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
