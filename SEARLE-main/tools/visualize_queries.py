"""
Qualitative examples (plan 01, section 1.12; experiments E1.12 and E2.8).

    python tools/visualize_queries.py --run R/E1.7d-phiB32-cirr --compare R/<final> --dataset cirr \
        --dataset-path $DATA/CIRR --n 20 --seed 0 --out R/E1.12-examples-cirr
    python tools/visualize_queries.py --explain-attributes --run R/<final> --dataset fashioniq \
        --dataset-path $DATA/FashionIQ --n 20 --seed 0 --out R/E2.8-explained-fiq

For each chosen query: the reference image with its caption and the top-K results of the run (green frame = correct
image), a second row for --compare, and with --explain-attributes the attribute satisfaction and the reason of each
re-ranked candidate (from the run's dump). --nearest-words adds the CLIP words closest to the pseudo token of the
reference (rebuilds the run's model, phi / searle only).
The queries are drawn at random (--seed) among successes (a correct image in the top-K) and failures, half each,
taking the modification categories in turn so that they are spread over the categories.
Writes one PNG per query and index.md in --out.
"""
import random
import sys
import textwrap
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from run_utils import captions_of, correct_names, image_locator, load_image, load_run, pipeline_from_run  # noqa: E402
from attributes.taxonomy import categorize_query  # noqa: E402


def choose(records, n, top, seed):
    """
    n queries, half successes / half failures, spread over the modification categories
    """
    rng = random.Random(seed)
    groups = {True: {}, False: {}}
    for r in records:
        success = bool(correct_names(r) & {name for name, _ in r['top'][:top]})
        for category in categorize_query(captions_of(r)) or ['other']:
            groups[success].setdefault(category, []).append(r)
    chosen, seen = [], set()
    for success, quota in ((True, n - n // 2), (False, n // 2)):
        pools = {c: rng.sample(v, len(v)) for c, v in sorted(groups[success].items())}
        picked = 0
        while picked < quota and any(pools.values()):
            for category in list(pools):
                while pools[category] and pools[category][0]['qid'] in seen:
                    pools[category].pop(0)
                if pools[category] and picked < quota:
                    r = pools[category].pop(0)
                    seen.add(r['qid'])
                    chosen.append((r, success, category))
                    picked += 1
    return chosen


def nearest_words(pipeline, reference: str, k: int = 5):
    import torch
    import torch.nn.functional as F
    from clip.simple_tokenizer import SimpleTokenizer
    token = pipeline.reference_tokens()[reference].to(pipeline.clip_model.token_embedding.weight.device).float()
    vocabulary = F.normalize(pipeline.clip_model.token_embedding.weight.float(), dim=-1)
    best = torch.topk(vocabulary @ F.normalize(token, dim=-1), k).indices.tolist()
    decoder = SimpleTokenizer().decoder
    return [decoder[i].replace('</w>', '') for i in best]


def main():
    parser = ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--run', required=True)
    parser.add_argument('--compare', default=None, help='Second run shown below the first one')
    parser.add_argument('--dataset', required=True, choices=['cirr', 'circo', 'fashioniq'])
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--top', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', required=True)
    parser.add_argument('--explain-attributes', action='store_true')
    parser.add_argument('--nearest-words', action='store_true')
    parser.add_argument('--cache-dir', default=None)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    _, records = load_run(args.run)
    records = [r for r in records if r.get('top') and (r.get('target') or r.get('gt'))]
    compare = {r['qid']: r for r in load_run(args.compare)[1]} if args.compare else {}
    locate = image_locator(args.dataset, args.dataset_path)
    pipeline = pipeline_from_run(args.run, args.dataset_path, args.cache_dir) if args.nearest_words else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    index = [f"# Examples of {Path(args.run).name}" + (f" vs {Path(args.compare).name}" if args.compare else ''), "",
             f"{args.n} queries, seed {args.seed}; green frame = correct image.", ""]
    rows_names = [Path(args.run).name] + ([Path(args.compare).name] if args.compare else [])
    for i, (record, success, category) in enumerate(choose(records, args.n, args.top, args.seed), 1):
        runs = [record] + ([compare[record['qid']]] if record['qid'] in compare else [])
        fig, axes = plt.subplots(len(runs), args.top + 1, figsize=(2.3 * (args.top + 1), 2.9 * len(runs) + 0.8),
                                 squeeze=False)
        caption = ' || '.join(captions_of(record))
        fig.suptitle('\n'.join(textwrap.wrap(caption, 110)), fontsize=9)
        correct = correct_names(record)
        for row, run_record in enumerate(runs):
            ax = axes[row][0]
            ax.imshow(load_image(locate(record['reference'])))
            title = f"reference\n{rows_names[row]}"
            if pipeline is not None and row == 0:
                try:
                    title += "\n≈ " + ', '.join(nearest_words(pipeline, record['reference']))
                except (ValueError, KeyError, AttributeError):
                    pass
            ax.set_title(title, fontsize=7)
            explained = {c['name']: c for c in (run_record.get('attributes') or {}).get('reranked_top', [])}
            for col, (name, score) in enumerate(run_record['top'][:args.top], 1):
                ax = axes[row][col]
                try:
                    ax.imshow(load_image(locate(name)))
                except Exception:
                    ax.text(0.5, 0.5, 'missing', ha='center')
                good = name in correct
                for spine in ax.spines.values():
                    spine.set_edgecolor('limegreen' if good else 'lightgray')
                    spine.set_linewidth(4 if good else 1)
                title = f"#{col} {score:.3f}"
                if args.explain_attributes and name in explained:
                    e = explained[name]
                    title += f"\nc={e['satisfaction']:.2f} (CLIP #{e['clip_rank']})"
                    if e.get('why'):
                        title += '\n' + '\n'.join(textwrap.wrap(e['why'][0], 30)[:2])
                ax.set_title(title, fontsize=6.5)
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        name = f"{i:02d}_{'success' if success else 'failure'}_{category}_{record['qid'].replace(':', '-')}.png"
        fig.savefig(out / name, dpi=110)
        plt.close(fig)
        index.append(f"- {'✓' if success else '✗'} [{category}] {caption[:120]} → `{name}`")
    (out / 'index.md').write_text('\n'.join(index) + '\n', encoding='utf-8')
    print(f"Written {len(index) - 4} figures to {out}")


if __name__ == '__main__':
    main()
