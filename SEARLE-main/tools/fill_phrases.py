"""
Give a phrase to every concept of a phrase set that has none, taking it from another set (E11).

    python tools/fill_phrases.py --phrases $GP/P3_qwen15i_all/concept_to_phrases.pkl \
        --fallback $GP/GPTNeo27B/concept_to_phrases.pkl --out-exp P3_qwen15i_allf --out-root $GP --n 16

Why: train_phi.py draws one concept per image among its top-150 and draws the whole batch again as long as one of the
256 concepts has no usable phrase. With the original set (2 such concepts) this never happens; with 601 empty
concepts (3%) a batch needs thousands of draws and an epoch takes 45 minutes instead of 3.5. Empty concepts get up to
--n phrases of the fallback set that contain the concept string (the legacy filter of train_phi.py); concepts that
still have none get one template phrase "a photo of {concept}". Writes <out-root>/<out-exp>/concept_to_phrases.pkl and
fill_stats.json.
"""
import json
import pickle
import random
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gpt_phrases_utils import ConceptPhrasesWriter, iter_concept_phrases  # noqa: E402


def main():
    parser = ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--phrases', required=True)
    parser.add_argument('--fallback', required=True)
    parser.add_argument('--out-exp', required=True)
    parser.add_argument('--out-root', required=True)
    parser.add_argument('--n', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    with open(args.phrases, 'rb') as f:
        phrases = pickle.load(f)
    empty = {c for c, v in phrases.items() if not v or any(c not in p for p in v)}
    rng = random.Random(args.seed)
    from_fallback = 0
    for concept, fb in iter_concept_phrases(args.fallback):
        if concept in empty:
            usable = [p for p in fb if concept in p]
            if usable:
                phrases[concept] = rng.sample(usable, min(args.n, len(usable)))
                from_fallback += 1
    template = 0
    for concept in empty:
        if not phrases[concept] or any(concept not in p for p in phrases[concept]):
            phrases[concept] = [f"a photo of {concept}"]
            template += 1
    out = Path(args.out_root) / args.out_exp
    out.mkdir(parents=True, exist_ok=True)
    writer = ConceptPhrasesWriter(out / 'concept_to_phrases.pkl')
    for concept, v in phrases.items():
        writer.add(concept, v)
    writer.close()
    stats = {'concepts': len(phrases), 'empty_before': len(empty), 'filled_from_fallback': from_fallback,
             'filled_with_template': template, 'fallback': args.fallback, 'n': args.n}
    (out / 'fill_stats.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
