"""
Rule-based parser of the visual constraints requested by a relative caption (plan 02, section 2.1). No language
model: a lexicon of words and phrases, longest match first, plus negation cues with a short scope.

    family        values                                   kind
    color         red, maroon, blue, navy, teal, green,    absolute: the candidate's dominant colour is close to it
                  olive, yellow, gold, orange, pink,
                  purple, brown, beige, black, white, gray
    brightness    darker, brighter                         relative to the reference image (L* in CIELAB)
    colorfulness  more, less (relative); grayscale (abs.)  colourfulness index / saturation
    pattern       striped, plaid, dotted, floral,          absolute: edge density, Fourier peak, LBP entropy
                  patterned, solid (+ vertical/horizontal)
    zoom          closer, farther                          relative: area of the object

Negation ('not', 'no', 'without', 'instead of', 'rather than', 'less'...) flips the constraints that start in the
next two words (three after 'remove'), stopping at punctuation, 'and', 'but', 'with'. A colour followed
or preceded by 'background' is measured on the whole image instead of the object (region 'image').
FashionIQ queries have two captions: their constraints are merged, and contradictory pairs are dropped.
"""
import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

FAMILIES = ['color', 'brightness', 'colorfulness', 'pattern', 'zoom']
RELATIVE_VALUES = {('brightness', 'darker'), ('brightness', 'brighter'), ('colorfulness', 'more'),
                   ('colorfulness', 'less'), ('zoom', 'closer'), ('zoom', 'farther')}

# Colour names; words that are also common objects in scene images (wine, coffee, cream, lemon, rose, peach...)
# are left out on purpose
COLOR_WORDS = {
    'red': 'red', 'redder': 'red', 'crimson': 'red', 'scarlet': 'red',
    'maroon': 'maroon', 'burgundy': 'maroon',
    'blue': 'blue', 'bluer': 'blue', 'cobalt': 'blue',
    'navy': 'navy',
    'teal': 'teal', 'turquoise': 'teal', 'aqua': 'teal', 'cyan': 'teal',
    'green': 'green', 'greener': 'green', 'emerald': 'green',
    'olive': 'olive',
    'yellow': 'yellow', 'yellower': 'yellow', 'mustard': 'yellow',
    'gold': 'gold', 'golden': 'gold',
    'orange': 'orange',
    'pink': 'pink', 'pinker': 'pink', 'magenta': 'pink', 'fuchsia': 'pink',
    'purple': 'purple', 'violet': 'purple', 'lilac': 'purple', 'lavender': 'purple', 'mauve': 'purple',
    'brown': 'brown', 'browner': 'brown',
    'beige': 'beige', 'tan': 'beige', 'khaki': 'beige', 'nude': 'beige', 'taupe': 'beige',
    'ivory': 'white', 'white': 'white', 'whiter': 'white',
    'black': 'black', 'blacker': 'black',
    'gray': 'gray', 'grey': 'gray', 'silver': 'gray', 'charcoal': 'gray', 'grayer': 'gray', 'greyer': 'gray',
}
COLOR_MODIFIERS = {'light', 'dark', 'pale', 'deep', 'bright', 'hot', 'baby', 'sky', 'royal', 'neon', 'pastel'}

# phrase -> (family, value); longest phrases are tried first
PHRASES: Dict[Tuple[str, ...], Tuple[str, str]] = {}


def _add(family: str, value: str, *phrases: str) -> None:
    for phrase in phrases:
        PHRASES[tuple(phrase.split())] = (family, value)


_add('colorfulness', 'grayscale', 'black and white', 'monochrome', 'monochromatic', 'grayscale', 'greyscale',
     'gray scale', 'grey scale', 'b w', 'sepia')
_add('colorfulness', 'more', 'more colorful', 'more colourful', 'colorful', 'colourful', 'multicolored',
     'multicoloured', 'multicolor', 'multi colored', 'multi coloured', 'multi color', 'vibrant', 'more vibrant',
     'more colors', 'more colours', 'brighter colors', 'brighter colours', 'more saturated', 'saturated')
_add('colorfulness', 'less', 'less colorful', 'less colourful', 'duller', 'dull', 'muted', 'less vibrant',
     'less saturated', 'desaturated', 'faded', 'fading', 'more muted')
_add('brightness', 'darker', 'darker', 'more dark', 'less bright', 'less light', 'not as bright', 'dimmer',
     'darkened', 'dimly lit', 'darker shade', 'darker tone')
_add('brightness', 'brighter', 'lighter', 'brighter', 'less dark', 'not as dark', 'more bright', 'brightened',
     'lightened', 'lighter shade', 'lighter tone', 'well lit', 'better lit', 'better lighting', 'brighter lighting',
     'more lighting', 'more light')
_add('pattern', 'striped', 'stripes', 'stripe', 'striped', 'stripy', 'pinstripe', 'pinstripes', 'pinstriped',
     'stripped', 'striped pattern', 'stripe pattern', 'stripes pattern', 'striped print')
_add('pattern', 'plaid', 'plaid', 'checkered', 'checked', 'tartan', 'gingham', 'checks', 'check pattern',
     'checkerboard', 'plaid pattern', 'plaid print', 'checkered pattern', 'checked pattern', 'tartan pattern')
_add('pattern', 'dotted', 'polka dot', 'polka dots', 'polka dotted', 'polkadot', 'polkadots', 'dots', 'dotted',
     'spotted', 'spots', 'polka dot pattern', 'polka dot print', 'dot pattern', 'dotted pattern', 'dot print')
_add('pattern', 'floral', 'floral', 'flowered', 'flowery', 'flower print', 'flower pattern', 'flower design',
     'floral print', 'floral pattern', 'flowers print')
_add('pattern', 'patterned', 'patterned', 'pattern', 'patterns', 'print', 'printed', 'prints', 'graphic',
     'graphics', 'animal print', 'leopard print', 'paisley', 'camo', 'camouflage', 'tie dye', 'designs', 'zigzag',
     'zig zag', 'zigzagged', 'chevron', 'visuals', 'busy pattern', 'busier')
_add('pattern', 'solid', 'solid', 'plain', 'solid color', 'solid colored', 'solid colour', 'solid coloured',
     'single color', 'single colored', 'one color', 'one colored', 'no pattern', 'no print', 'no design')
_add('zoom', 'closer', 'closer', 'close up', 'closeup', 'zoomed in', 'zoom in', 'zooms in', 'zoomed', 'zoom',
     'nearer', 'more zoomed', 'zoomed in more', 'tighter shot', 'close shot', 'closer view', 'closer shot')
_add('zoom', 'farther', 'zoomed out', 'zoom out', 'zooms out', 'further away', 'farther away', 'from further',
     'from farther', 'from a distance', 'from afar', 'wider view', 'wide view', 'wider shot', 'wide shot',
     'more distant', 'further back', 'farther back', 'less zoomed', 'farther', 'farther distance',
     'further distance', 'greater distance', 'larger distance', 'bigger distance', 'longer distance', 'far away',
     'from far away', 'from far')

# Phrases whose meaning already includes a negation (the cue is part of the phrase, so it is not applied again)
NEGATION_CUES = {('not',): 2, ('no',): 2, ('without',): 2, ('never',): 2, ('remove',): 3, ('removed',): 3,
                 ('removes',): 3, ('removing',): 3, ('minus',): 2, ('lacks',): 2, ('lacking',): 2,
                 ('instead', 'of'): 2, ('rather', 'than'): 2, ('less',): 2, ('nothing',): 2}
SCOPE_STOPS = {',', '.', ';', '!', '?', 'and', 'but', 'with', 'while', 'whereas', 'also'}
MAX_PHRASE = max(len(p) for p in PHRASES)


@dataclass(frozen=True)
class Constraint:
    family: str
    value: str
    negated: bool = False
    region: str = 'object'  # 'object' (masked) or 'image' (whole image, e.g. "white background")
    orientation: Optional[str] = None  # 'vertical' / 'horizontal' for stripes
    text: str = ''

    @property
    def relative(self) -> bool:
        return (self.family, self.value) in RELATIVE_VALUES

    def key(self) -> tuple:
        return self.family, self.value, self.region, self.orientation

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        words = {'darker': 'darker', 'brighter': 'brighter', 'more': 'more colorful', 'less': 'less colorful',
                 'grayscale': 'black and white', 'closer': 'closer view', 'farther': 'farther view'}
        text = words.get(self.value, self.value) if self.family != 'pattern' else self.value
        if self.orientation:
            text = f"{self.orientation} {text}"
        if self.region == 'image':
            text += ' (whole image)'
        return ('not ' if self.negated else '') + text


def tokenize(text: str) -> List[str]:
    text = text.lower().replace("n't", " not").replace('-', ' ').replace('/', ' ').replace('&', ' and ')
    return re.findall(r"[a-z]+|[0-9]+|[,.;!?]", text)


def parse_families(spec: str) -> List[str]:
    families = [f.strip() for f in spec.split(',') if f.strip()]
    unknown = [f for f in families if f not in FAMILIES]
    if unknown or not families:
        raise ValueError(f"--attr-families should be a comma separated subset of {FAMILIES}, got '{spec}'")
    return families


def _near(tokens: List[str], start: int, end: int, word: str, window: int = 2) -> bool:
    return word in tokens[max(0, start - window):start] or word in tokens[end:end + window]


def parse_caption(caption: str, negation: bool = True) -> List[Constraint]:
    tokens = tokenize(caption)
    found: List[Tuple[int, int, str, str, Optional[str]]] = []  # start, end, family, value, orientation
    anchors: List[int] = []  # position of the head word (the colour word, not its modifier), for negation scopes
    consumed = [False] * len(tokens)
    i = 0
    while i < len(tokens):
        match = None
        for n in range(min(MAX_PHRASE, len(tokens) - i), 0, -1):
            phrase = tuple(tokens[i:i + n])
            if phrase in PHRASES:
                match = (n, *PHRASES[phrase])
                break
        if match:
            n, family, value = match
            orientation = None
            if value == 'striped':
                for word in ('vertical', 'vertically', 'horizontal', 'horizontally'):
                    if _near(tokens, i, i + n, word):
                        orientation = 'vertical' if word.startswith('vert') else 'horizontal'
            # "a different pattern", "the same color pattern" say nothing measurable about the target
            vague = value == 'patterned' and any(w in tokens[max(0, i - 2):i] for w in ('different', 'same'))
            if not vague:
                found.append((i, i + n, family, value, orientation))
                anchors.append(i)
            consumed[i:i + n] = [True] * n
            i += n
            continue
        word = tokens[i]
        if word in COLOR_WORDS:
            start = i - 1 if i > 0 and tokens[i - 1] in COLOR_MODIFIERS and not consumed[i - 1] else i
            if word == 'navy' and i + 1 < len(tokens) and tokens[i + 1] == 'blue':  # "navy blue"
                found.append((start, i + 2, 'color', 'navy', None))
                anchors.append(i)
                consumed[start:i + 2] = [True] * (i + 2 - start)
                i += 2
                continue
            fruit = word == 'orange' and i > 0 and tokens[i - 1] in ('an', 'the') and \
                (i + 1 >= len(tokens) or tokens[i + 1] in SCOPE_STOPS | {'on', 'in', 'of', 'is'})
            if not fruit:
                found.append((start, i + 1, 'color', COLOR_WORDS[word], None))
                anchors.append(i)
                consumed[start:i + 1] = [True] * (i + 1 - start)
        i += 1

    negated = [False] * len(found)
    if negation:
        # "change the green beads to blue beads", "from red to blue": the first colour is the one to remove
        colors = [f for f, entry in enumerate(found) if entry[2] == 'color']
        for f in colors:
            for t in range(anchors[f] + 1, min(anchors[f] + 5, len(tokens))):
                if tokens[t] in ('to', 'into') and any(t < anchors[g] <= t + 4 for g in colors if g != f):
                    negated[f] = True
                    break
        for j in range(len(tokens)):
            for cue, scope in NEGATION_CUES.items():
                if tuple(tokens[j:j + len(cue)]) != cue or any(consumed[j:j + len(cue)]):
                    continue
                cue_end = j + len(cue)
                limit = cue_end + scope
                for k in range(cue_end, min(limit, len(tokens))):
                    if tokens[k] in SCOPE_STOPS:
                        limit = k
                        break
                for f, anchor in enumerate(anchors):
                    if cue_end <= anchor < limit:
                        negated[f] = not negated[f]

    constraints = []
    for (start, end, family, value, orientation), neg in zip(found, negated):
        region = 'image' if family in ('color', 'brightness', 'colorfulness') and \
            _near(tokens, start, end, 'background', 3) else 'object'
        constraints.append(Constraint(family, value, neg, region, orientation, ' '.join(tokens[start:end])))
    return merge_constraints([constraints])


def merge_constraints(groups: Iterable[List[Constraint]]) -> List[Constraint]:
    """
    Union of constraints; duplicates removed; a constraint present both negated and not negated is dropped
    """
    by_key: Dict[tuple, List[Constraint]] = {}
    for group in groups:
        for c in group:
            by_key.setdefault(c.key(), []).append(c)
    merged = []
    for key, items in by_key.items():
        polarities = {c.negated for c in items}
        if len(polarities) == 1:
            merged.append(items[0])
    return merged


def parse_query(captions: List[str], negation: bool = True,
                families: Optional[List[str]] = None) -> List[Constraint]:
    """
    Constraints of a query (one caption for CIRR / CIRCO, two for FashionIQ), optionally restricted to some families
    """
    constraints = merge_constraints(parse_caption(c, negation) for c in captions)
    if families is not None:
        constraints = [c for c in constraints if c.family in families]
    return constraints
