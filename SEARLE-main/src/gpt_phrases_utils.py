"""
How the GPT phrases (data/GPT_phrases/<exp>/concept_to_phrases.pkl) are used by oti_inversion.py and train_phi.py
(plan 06, section 6.2). The defaults reproduce the original code exactly.

--phrase-match legacy   original behaviour: str.replace of the concept by ' $ ' (OTI replaces the first occurrence,
                        train_phi every occurrence), without word boundaries ("ant" in "giant" becomes "gi $ ")
--phrase-match word     the first occurrence of the concept as a whole word (case insensitive), in both scripts
--concept-filter legacy original behaviour: a concept is dropped as soon as one of its phrases does not contain it
--concept-filter per-phrase   only the phrases that do not contain the concept are dropped; a concept is dropped
                        when none of its phrases is left
"""
import re
from functools import lru_cache
from typing import Dict, List, Set, Tuple

from tqdm import tqdm

PHRASE_MATCH = ['legacy', 'word']
CONCEPT_FILTER = ['legacy', 'per-phrase']


@lru_cache(maxsize=None)
def concept_pattern(concept: str):
    return re.compile(r'(?<!\w)' + re.escape(concept) + r'(?!\w)', re.IGNORECASE)


def has_concept(phrase: str, concept: str, match: str) -> bool:
    if match == 'legacy':
        return concept in phrase
    return concept_pattern(concept).search(phrase) is not None


def replace_concept(phrase: str, concept: str, match: str, first_only: bool) -> str:
    """
    The phrase with the concept replaced by the pseudo-word ' $ '
    """
    if match == 'legacy':
        return phrase.replace(concept, " $ ", 1) if first_only else phrase.replace(concept, " $ ")
    return concept_pattern(concept).sub(" $ ", phrase, count=1)


def prepare_phrases(concept_to_phrases: Dict[str, List[str]], concept_filter: str = 'legacy',
                    match: str = 'legacy') -> Tuple[Dict[str, List[str]], Set[str]]:
    """
    Returns the phrases to use and the set of concepts that must not be used
    """
    print('filtering problematic concepts')
    if concept_filter == 'legacy':
        concepts_to_filter_out = set()
        for k, v in tqdm(concept_to_phrases.items()):
            for phrase in v:
                p = phrase.replace(k, " $ ", 1)
                if '$' not in p:
                    concepts_to_filter_out.add(k)
        return concept_to_phrases, concepts_to_filter_out
    kept = {k: [p for p in v if has_concept(p, k, match)] for k, v in tqdm(concept_to_phrases.items())}
    n_before = sum(len(v) for v in concept_to_phrases.values())
    n_after = sum(len(v) for v in kept.values())
    concepts_to_filter_out = {k for k, v in kept.items() if not v}
    print(f"per-phrase filter: kept {n_after}/{n_before} phrases, {len(concepts_to_filter_out)} concepts without "
          f"any usable phrase")
    return kept, concepts_to_filter_out
