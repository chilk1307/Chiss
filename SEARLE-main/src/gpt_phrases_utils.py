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
import pickletools
import re
import struct
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Set, Tuple, Union

from tqdm import tqdm

PHRASE_MATCH = ['legacy', 'word']
CONCEPT_FILTER = ['legacy', 'per-phrase']


@lru_cache(maxsize=None)
def concept_pattern(concept: str):
    return re.compile(r'(?<!\w)' + re.escape(concept) + r'(?!\w)', re.IGNORECASE)


APOSTROPHES = "'’‘`´"
DASHES = "-‐‑–—"


def tolerant_pattern(concept: str):
    """
    Whole-word, case-insensitive pattern of the concept that also accepts other quotes / dashes and extra spaces, as
    produced by the decoders of language models ("t - shirt", "jack-o’-lantern")
    """
    parts = []
    special = APOSTROPHES + DASHES
    for i, ch in enumerate(concept):
        if ch in APOSTROPHES:
            parts.append(r"\s*[" + re.escape(APOSTROPHES) + r"]\s*")
        elif ch in DASHES:
            parts.append(r"\s*[" + re.escape(DASHES) + r"]\s*")
        elif ch.isspace():
            # next to a quote or a dash the space may disappear ("Grevillea 'moonlight'" -> "Grevillea'moonlight'")
            near_special = (i > 0 and concept[i - 1] in special) or (i + 1 < len(concept) and concept[i + 1] in special)
            parts.append(r"\s*" if near_special else r"\s+")
        else:
            parts.append(re.escape(ch))
    return re.compile(r"(?<!\w)" + ''.join(parts) + r"(?!\w)", re.IGNORECASE)


def normalise_concept(phrase: str, concept: str):
    """
    The phrase with the first (tolerant) occurrence of the concept rewritten with the exact concept string, or None if
    the phrase does not contain the concept as a whole word
    """
    match = tolerant_pattern(concept).search(phrase)
    if match is None:
        return None
    return phrase[:match.start()] + concept + phrase[match.end():]


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
            if not v:  # a cleaned phrase set can leave a concept without phrases (never the case in GPTNeo27B)
                concepts_to_filter_out.add(k)
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


# Streaming access to concept_to_phrases.pkl (about 700 MB, several GB once unpickled) -------------------------------

_STRING_OPS = {'SHORT_BINUNICODE', 'BINUNICODE', 'BINUNICODE8', 'UNICODE'}


def iter_concept_phrases(path: Union[str, Path]) -> Iterator[Tuple[str, List[str]]]:
    """
    Yield (concept, phrases) from a pickled dict {str: [str, ...]} without loading the whole dictionary: the pickle
    opcodes are read one by one (pickletools.genops, no code is executed) and every key/value pair is released as soon
    as it is complete. Only the opcodes used by pickle.dump for such a dictionary are supported.
    """
    stack, marks, memo = [], [], {}
    with open(path, 'rb') as f:
        for opcode, arg, _ in pickletools.genops(f):
            name = opcode.name
            if name in ('PROTO', 'FRAME', 'STOP'):
                continue
            if name == 'EMPTY_DICT':
                stack.append({})
            elif name == 'EMPTY_LIST':
                stack.append([])
            elif name in _STRING_OPS:
                stack.append(arg)
            elif name in ('MEMOIZE', 'BINPUT', 'LONG_BINPUT', 'PUT'):
                index = len(memo) if name == 'MEMOIZE' else arg
                top = stack[-1]
                # keep only short strings (concept names) in the memo, so that memory stays small
                memo[index] = top if isinstance(top, str) and len(top) <= 64 else None
            elif name in ('BINGET', 'LONG_BINGET', 'GET'):
                if memo.get(arg) is None:
                    raise ValueError("unsupported pickle (reference to a long string or a container); "
                                     "load it with pickle.load instead")
                stack.append(memo[arg])
            elif name == 'MARK':
                marks.append(len(stack))
            elif name == 'APPEND':
                item = stack.pop()
                stack[-1].append(item)
            elif name == 'APPENDS':
                start = marks.pop()
                items = stack[start:]
                del stack[start:]
                stack[-1].extend(items)
            elif name == 'SETITEM':
                value = stack.pop()
                key = stack.pop()
                yield key, value
            elif name == 'SETITEMS':
                start = marks.pop()
                items = stack[start:]
                del stack[start:]
                for i in range(0, len(items), 2):
                    yield items[i], items[i + 1]
            else:
                raise ValueError(f"unsupported pickle opcode {name}")


class ConceptPhrasesWriter:
    """
    Write a dictionary {str: [str, ...]} as a protocol 4 pickle, a few concepts at a time, so that the whole output
    never has to be in memory. The file is read back with pickle.load (Python 3.8 included).
    """

    def __init__(self, path: Union[str, Path], batch: int = 1000):
        self.file = open(path, 'wb')
        self.file.write(b'\x80\x04}')  # PROTO 4, EMPTY_DICT
        self.batch, self.pending = batch, []

    def _string(self, s: str) -> None:
        data = s.encode('utf-8', 'surrogatepass')
        if len(data) < 256:
            self.file.write(b'\x8c' + bytes([len(data)]) + data)  # SHORT_BINUNICODE
        else:
            self.file.write(b'X' + struct.pack('<I', len(data)) + data)  # BINUNICODE

    def _flush(self) -> None:
        if not self.pending:
            return
        self.file.write(b'(')  # MARK
        for key, phrases in self.pending:
            self._string(key)
            self.file.write(b']')  # EMPTY_LIST
            if phrases:
                self.file.write(b'(')
                for p in phrases:
                    self._string(p)
                self.file.write(b'e')  # APPENDS
        self.file.write(b'u')  # SETITEMS
        self.pending = []

    def add(self, concept: str, phrases: Iterable[str]) -> None:
        self.pending.append((concept, list(phrases)))
        if len(self.pending) >= self.batch:
            self._flush()

    def close(self) -> None:
        self._flush()
        self.file.write(b'.')  # STOP
        self.file.close()
