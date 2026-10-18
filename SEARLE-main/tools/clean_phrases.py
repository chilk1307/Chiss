"""
Build the cleaned phrase set P1 (plan 06, section 6.2).

    python tools/clean_phrases.py --phrases data/GPT_phrases/GPTNeo27B/concept_to_phrases.pkl --out-exp GPTNeo27B_clean

For every concept:
    1. the concept is searched in each phrase as a whole word, case insensitive, also when the decoder changed its
       quotes, dashes or spaces ("t - shirt", "jack-o’-lantern"); the occurrence found is rewritten with the exact
       concept string, so that both --phrase-match legacy and word find it ("normalised" in the report);
    2. phrases without the concept are dropped (one by one, not the whole concept);
    3. duplicates are removed (lower case, no punctuation);
    4. phrases longer than --max-tokens CLIP tokens are shortened word by word (CLIP would cut them anyway).
Concepts left without phrases are kept with an empty list (train_phi.py / oti_inversion.py then skip them).
The input is read as a stream and the output written as a stream (protocol 4, readable by Python 3.8), so the
800 MB set never has to be in memory. Writes data/GPT_phrases/<out-exp>/concept_to_phrases.pkl,
hyperparameters.json and clean_stats.json.
"""
import json
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data_utils import PROJECT_ROOT  # noqa: E402
from gpt_phrases_utils import ConceptPhrasesWriter, iter_concept_phrases, tolerant_pattern  # noqa: E402

PUNCTUATION = re.compile(r"[^\w\s]")
SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    return SPACES.sub(' ', PUNCTUATION.sub(' ', text.lower())).strip()


def main():
    parser = ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--phrases', required=True, help='Input concept_to_phrases.pkl (P0)')
    parser.add_argument('--out-exp', required=True, help='Output folder name in data/GPT_phrases/')
    parser.add_argument('--out-root', default=None, help='Root folder instead of data/GPT_phrases')
    parser.add_argument('--max-tokens', type=int, default=75, help='CLIP tokens (77 minus start / end tokens)')
    args = parser.parse_args()

    from clip.simple_tokenizer import SimpleTokenizer
    tokenizer = SimpleTokenizer()
    root = Path(args.out_root) if args.out_root else PROJECT_ROOT / 'data' / 'GPT_phrases'
    out = root / args.out_exp
    out.mkdir(parents=True, exist_ok=True)
    writer = ConceptPhrasesWriter(out / 'concept_to_phrases.pkl')
    counts = {'concepts': 0, 'phrases_in': 0, 'phrases_out': 0, 'dropped_no_concept': 0, 'dropped_duplicate': 0,
              'normalised': 0, 'shortened': 0, 'concepts_empty': 0, 'concepts_rescued': 0}
    for concept, phrases in iter_concept_phrases(args.phrases):
        counts['concepts'] += 1
        counts['phrases_in'] += len(phrases)
        pattern = tolerant_pattern(concept)
        kept, seen = [], set()
        legacy_drop = any(concept not in p for p in phrases)
        for phrase in phrases:
            match = pattern.search(phrase)
            if match is None:
                counts['dropped_no_concept'] += 1
                continue
            if match.group(0) != concept:
                phrase = phrase[:match.start()] + concept + phrase[match.end():]
                counts['normalised'] += 1
            phrase = SPACES.sub(' ', phrase).strip()
            words = phrase.split(' ')
            if len(tokenizer.encode(phrase)) > args.max_tokens:
                while len(words) > 1 and len(tokenizer.encode(' '.join(words))) > args.max_tokens:
                    words.pop()
                phrase = ' '.join(words)
                counts['shortened'] += 1
                if concept not in phrase:
                    counts['dropped_no_concept'] += 1
                    continue
            key = normalize(phrase)
            if key in seen:
                counts['dropped_duplicate'] += 1
                continue
            seen.add(key)
            kept.append(phrase)
        counts['phrases_out'] += len(kept)
        counts['concepts_empty'] += not kept
        counts['concepts_rescued'] += legacy_drop and bool(kept)
        writer.add(concept, kept)
        if counts['concepts'] % 2000 == 0:
            print(f"\r{counts['concepts']} concepts cleaned", end='', flush=True)
    writer.close()
    print()
    (out / 'clean_stats.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    (out / 'hyperparameters.json').write_text(json.dumps({'source': str(args.phrases), 'tool': 'clean_phrases.py',
                                                          'max_tokens': args.max_tokens}, indent=2), encoding='utf-8')
    print(json.dumps(counts, indent=2))
    print(f"Written {out / 'concept_to_phrases.pkl'}")


if __name__ == '__main__':
    main()
