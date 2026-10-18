"""
Quality statistics of a GPT phrase set (plan 06, section 6.1; experiment E6.1).

    python tools/phrases_stats.py --phrases data/GPT_phrases/GPTNeo27B/concept_to_phrases.pkl \
        --similar-concepts data/similar_concepts/imagenet/test/ViT-B32_top_250_ensembling_image_texts.csv \
        --out $DRIVE/results/E6.1-stats-P0

The file is read as a stream (gpt_phrases_utils.iter_concept_phrases), so a 700 MB set fits in a small memory.
Reported: concepts and phrases per concept; duplicates inside a concept (lower case, no punctuation); concepts dropped
by the original filter (one phrase without the exact concept string drops the whole concept); phrases where the
concept appears several times or only inside another word ("ant" in "giant"); length in words and in CLIP tokens
(on a sample); repeated words; non-ASCII characters; distinct-2 diversity; and, with --similar-concepts, the concepts
actually used (top-15 per image for OTI, top-150 for phi) and how many of them the original filter drops.
Writes <out>/stats.json and <out>/report.md.
"""
import json
import random
import re
import sys
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gpt_phrases_utils import concept_pattern, iter_concept_phrases  # noqa: E402

PUNCTUATION = re.compile(r"[^\w\s]")
SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    return SPACES.sub(' ', PUNCTUATION.sub(' ', text.lower())).strip()


def repeated_words(words) -> bool:
    if any(a == b for a, b in zip(words, words[1:])):
        return True
    counts = Counter(w for w in words if len(w) > 3)
    return bool(counts) and max(counts.values()) >= 3


def main():
    parser = ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--phrases', required=True, help='concept_to_phrases.pkl')
    parser.add_argument('--out', required=True)
    parser.add_argument('--similar-concepts', default=None, help='CSV of image_concepts_association.py')
    parser.add_argument('--top-k-oti', type=int, default=15)
    parser.add_argument('--top-k-phi', type=int, default=150)
    parser.add_argument('--token-sample', type=int, default=100000, help='Phrases tokenised with CLIP (sample)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    n_concepts = n_phrases = n_duplicates = n_multi = n_inside_only = n_repeated = n_non_ascii = 0
    per_concept, word_lengths = [], []
    legacy_dropped, empty_concepts = [], []
    distinct2 = []
    examples = {'duplicate': [], 'legacy_drop': [], 'inside_word': [], 'repeated_words': [], 'multiple': []}
    token_sample = []
    samples = {}
    for concept, phrases in iter_concept_phrases(args.phrases):
        n_concepts += 1
        n_phrases += len(phrases)
        per_concept.append(len(phrases))
        if not phrases:
            empty_concepts.append(concept)
            continue
        normalized = [normalize(p) for p in phrases]
        duplicates = len(phrases) - len(set(normalized))
        n_duplicates += duplicates
        if duplicates and len(examples['duplicate']) < 5:
            seen = Counter(normalized)
            examples['duplicate'].append(f"{concept}: {next(p for p, n in zip(phrases, normalized) if seen[n] > 1)}")
        if any(concept not in p for p in phrases):
            legacy_dropped.append(concept)
            if len(examples['legacy_drop']) < 5:
                examples['legacy_drop'].append(f"{concept}: {next(p for p in phrases if concept not in p)}")
        pattern = concept_pattern(concept)
        bigrams = []
        for p in phrases:
            words = p.lower().split()
            word_lengths.append(len(words))
            matches = len(pattern.findall(p))
            if matches >= 2:
                n_multi += 1
                if len(examples['multiple']) < 5:
                    examples['multiple'].append(f"{concept}: {p}")
            elif matches == 0 and concept.lower() in p.lower():
                n_inside_only += 1
                if len(examples['inside_word']) < 5:
                    examples['inside_word'].append(f"{concept}: {p}")
            if repeated_words(words):
                n_repeated += 1
                if len(examples['repeated_words']) < 5:
                    examples['repeated_words'].append(f"{concept}: {p}")
            if not p.isascii():
                n_non_ascii += 1
            bigrams.extend(zip(words, words[1:]))
        if bigrams:
            distinct2.append(len(set(bigrams)) / len(bigrams))
        if len(token_sample) < args.token_sample and rng.random() < 0.25:  # CLIP token lengths on a sample
            token_sample.extend(rng.sample(phrases, min(5, len(phrases))))
        if len(samples) < 8 and rng.random() < 0.001:
            samples[concept] = phrases[:3]
        if n_concepts % 2000 == 0:
            print(f"\r{n_concepts} concepts read", end='', flush=True)
    print()

    try:
        from clip.simple_tokenizer import SimpleTokenizer
        tokenizer = SimpleTokenizer()
        token_lengths = [len(tokenizer.encode(p)) for p in token_sample]
    except ImportError:
        token_lengths = []

    stats = {
        'file': args.phrases, 'concepts': n_concepts, 'phrases': n_phrases,
        'phrases_per_concept': {'min': int(np.min(per_concept)), 'mean': float(np.mean(per_concept)),
                                'max': int(np.max(per_concept))},
        'concepts_without_phrases': len(empty_concepts),
        'duplicate_phrases_pct': round(100 * n_duplicates / max(n_phrases, 1), 2),
        'concepts_dropped_by_original_filter': len(legacy_dropped),
        'concepts_dropped_by_original_filter_pct': round(100 * len(legacy_dropped) / max(n_concepts, 1), 2),
        'phrases_with_concept_several_times_pct': round(100 * n_multi / max(n_phrases, 1), 2),
        'phrases_with_concept_only_inside_a_word_pct': round(100 * n_inside_only / max(n_phrases, 1), 2),
        'phrases_with_repeated_words_pct': round(100 * n_repeated / max(n_phrases, 1), 2),
        'phrases_with_non_ascii_pct': round(100 * n_non_ascii / max(n_phrases, 1), 2),
        'words_per_phrase': {q: float(np.percentile(word_lengths, p)) for q, p in
                             (('p5', 5), ('median', 50), ('p95', 95))} if word_lengths else {},
        'clip_tokens_sample': {'n': len(token_lengths), 'median': float(np.median(token_lengths)),
                               'over_75_pct': round(100 * float(np.mean(np.array(token_lengths) > 75)), 2)}
        if token_lengths else None,
        'distinct2_mean': round(float(np.mean(distinct2)), 3) if distinct2 else None,
        'examples': examples, 'samples': samples,
    }

    if args.similar_concepts:
        import pandas as pd
        df = pd.read_csv(args.similar_concepts)
        oti_used, phi_used = set(), set()
        dropped = set(legacy_dropped)
        images_with_dropped_oti = 0
        for row in df.values.tolist():
            concepts = row[1:]
            top_oti = list(dict.fromkeys(str(w).strip() for w in concepts))[:args.top_k_oti]
            oti_used.update(top_oti)
            phi_used.update(str(w) for w in concepts[:args.top_k_phi])
            images_with_dropped_oti += any(c in dropped for c in top_oti)
        stats['used_concepts'] = {
            'images': len(df), 'oti_top_k': args.top_k_oti, 'phi_top_k': args.top_k_phi,
            'oti_concepts': len(oti_used), 'phi_concepts': len(phi_used), 'union': len(oti_used | phi_used),
            'union_dropped_by_original_filter': len((oti_used | phi_used) & dropped),
            'images_whose_oti_top_k_contains_a_dropped_concept_pct':
                round(100 * images_with_dropped_oti / max(len(df), 1), 1)}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'stats.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = [f"# E6.1 Phrase statistics", "", f"File: `{args.phrases}`", "", "| Measure | Value |", "|---|---|"]
    for key, value in stats.items():
        if key not in ('examples', 'samples', 'file'):
            lines.append(f"| {key} | {json.dumps(value, ensure_ascii=False)} |")
    lines += ["", "## Examples"] + [f"- **{k}**: " + " / ".join(v) for k, v in examples.items() if v]
    report = '\n'.join(lines) + '\n'
    (out / 'report.md').write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
