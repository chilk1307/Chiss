"""
Keyword rules that put a relative caption into modification categories (plan 01, section 1.10). A query can be in
several categories, or in none ('other').

    category     example words
    color        red, blue, color, colour (any colour word of parser.py)
    brightness   darker, lighter, brighter, colorful, black and white (brightness and colourfulness)
    pattern      striped, plaid, floral, pattern, print, solid, plain, materials (lace, denim, leather...)
    count        one, two, three, more <plural>, fewer, single, pair, digits
    spatial      left, right, top, bottom, behind, in front, next to, above, below, on top
    negation     remove, without, no, not, missing ("X instead of Y" is a replacement, not a negation)
    addition     add, holding, wearing, carrying, "with a ...", "has a ...", "put a ..."
    background   background, outdoors, indoors, beach, grass, snow, street, sky, water, room
    viewpoint    closer, zoom, angle, view, side, front, back, from above, perspective
    style        drawing, cartoon, painting, sketch, illustration, toy
"""
import re
from typing import Dict, List

from attributes.parser import COLOR_WORDS, PHRASES, tokenize

CATEGORIES = ['color', 'brightness', 'pattern', 'count', 'spatial', 'negation', 'addition', 'background',
              'viewpoint', 'style']

NUMBERS = {'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
           'single', 'pair', 'pairs', 'couple', 'several', 'many', 'multiple', 'fewer', 'dozen', 'both', 'lone',
           'alone', 'solo', 'twin', 'twins', 'triple', 'double', 'group', 'crowd', 'bunch', 'numerous', 'herd',
           'flock'}
MATERIALS = {'lace', 'lacy', 'leather', 'denim', 'silk', 'silky', 'satin', 'velvet', 'knit', 'knitted', 'wool',
             'woolen', 'cotton', 'linen', 'sheer', 'mesh', 'sequin', 'sequins', 'sequined', 'fur', 'furry',
             'metallic', 'shiny', 'glitter', 'glittery', 'sparkly', 'texture', 'textured', 'fabric', 'material',
             'chiffon', 'jersey', 'corduroy', 'suede', 'tweed', 'ruffles', 'ruffled', 'pleated', 'embroidered',
             'wooden', 'metal', 'plastic', 'glass'}
KEYWORDS: Dict[str, set] = {
    'color': {'color', 'colour', 'colored', 'coloured', 'colors', 'colours', 'hue'},
    # comparatives only: "dark green", "bright red" are colours, not brightness changes
    'brightness': {'darker', 'lighter', 'brighter', 'dimmer', 'colorful', 'colourful', 'monochrome', 'grayscale',
                   'greyscale', 'vibrant', 'dull', 'duller', 'faded', 'fading', 'saturated', 'sepia', 'brightness',
                   'lighting', 'darkness'},
    'pattern': {'striped', 'stripes', 'stripe', 'plaid', 'checkered', 'floral', 'pattern', 'patterned', 'patterns',
                'print', 'printed', 'solid', 'plain', 'dots', 'dotted', 'polka', 'graphic', 'graphics', 'logo',
                'design', 'designs', 'spotted', 'spots', 'tartan', 'paisley'} | MATERIALS,
    'spatial': {'left', 'right', 'top', 'bottom', 'behind', 'above', 'below', 'beneath', 'under', 'underneath',
                'beside', 'besides', 'between', 'near', 'nearby', 'center', 'centre', 'middle', 'corner',
                'foreground', 'atop', 'opposite', 'toward', 'towards'},
    'negation': {'no', 'not', 'without', 'remove', 'removed', 'removing', 'removes', 'none', 'nothing', 'never',
                 'lacks', 'lacking', 'minus', 'empty', 'unoccupied', 'nobody', 'except', 'excluding', 'missing',
                 'absent', 'gone'},
    # 'with' alone is too frequent ("white with shorter sleeves"); see ADDED_OBJECT for "with a dog"
    'addition': {'add', 'added', 'adding', 'adds', 'holding', 'holds', 'wearing', 'wears', 'carrying', 'carries',
                 'plus', 'additional', 'extra', 'accompanied', 'include', 'includes', 'including'},
    'background': {'background', 'backdrop', 'outdoors', 'outdoor', 'outside', 'indoors', 'indoor', 'inside',
                   'beach', 'grass', 'grassy', 'snow', 'snowy', 'street', 'road', 'field', 'sky', 'water', 'sea',
                   'ocean', 'lake', 'river', 'room', 'kitchen', 'bedroom', 'bathroom', 'forest', 'woods', 'park',
                   'city', 'sand', 'sandy', 'mountain', 'mountains', 'garden', 'yard', 'floor', 'wall', 'table',
                   'desk', 'shelf', 'trees', 'tree', 'scenery', 'setting', 'environment', 'surrounded', 'outdoors',
                   'night', 'daytime', 'sunset'},
    'viewpoint': {'closer', 'closeup', 'zoom', 'zoomed', 'angle', 'angles', 'view', 'viewed', 'perspective',
                  'aerial', 'overhead', 'distance', 'distant', 'farther', 'profile', 'upside', 'cropped', 'focus',
                  'focused', 'wider', 'portrait', 'rear', 'sideways', 'angled'},
    'style': {'drawing', 'drawn', 'cartoon', 'cartoonish', 'painting', 'painted', 'sketch', 'sketched',
              'illustration', 'illustrated', 'animated', 'anime', 'artistic', 'figurine', 'statue', 'sculpture',
              'stuffed', 'plush', 'lego', 'replica', 'clipart', 'emoji', 'graffiti', 'mural'},
}
MULTI_WORD = {
    'spatial': ['in front', 'next to', 'on top', 'to the left', 'to the right', 'on the left', 'on the right',
                'in the middle', 'close to', 'far from', 'in between', 'on the side of'],
    # "X instead of Y" is a replacement (CIRCO labels it compare_change), not a negation
    'negation': ['no longer', 'get rid', 'take away', 'taken away', 'take out', 'taken out'],
    'background': ['in the background'],
    'viewpoint': ['facing the camera', 'towards the camera', 'toward the camera', 'away from the camera',
                  'from above', 'from below', 'from the side', 'from behind', 'from the front', 'from the back',
                  'from its side', 'from the top', 'from the rear', 'from a different', 'from another', 'close up',
                  'zoom in', 'zoomed in', 'zoom out', 'zoomed out', 'different angle', 'side view', 'top view',
                  'front view', 'back view', 'rear view', 'bird s eye', 'birds eye', 'further away',
                  'farther away', 'shot from', 'taken from', 'seen from', 'photographed from', 'viewed from'],
    'brightness': ['black and white', 'more light', 'less light'],
    'count': ['more than', 'less than', 'a few', 'a lot of', 'lots of', 'a group of', 'a pair of', 'the same number',
              'fewer than'],
}
PLURAL_AFTER_COMPARATIVE = re.compile(r"\b(?:more|fewer|less|extra|additional|another)\s+(?:\w+\s+)?\w+s\b")
# "has a lighthouse", "shows two dogs", "there is a bench": an object is added to the scene
ADDED_OBJECT = re.compile(r"\b(?:has|have|shows|show|showing|includes|including|features|featuring|contains|"
                          r"there is|there are|now has|also has|with|put|place|placed|plus)\s+(?:a|an|some|two|three|"
                          r"four|several|many|multiple|another|his|her|their)\b(?!\s+(?:different|same|similar|"
                          r"smaller|bigger|larger|darker|lighter|brighter|lower|higher|longer|shorter|closer|wider|"
                          r"more|less|color|colour|shape|size|style|pattern|print|collar|neckline|hem|hemline|"
                          r"sleeve|sleeves|strap|straps|neck|cut|fit|length)\b)")


def categorize(caption: str) -> List[str]:
    tokens = tokenize(caption)
    words = set(tokens)
    padded = ' ' + ' '.join(tokens) + ' '
    found = set()
    for category, keywords in KEYWORDS.items():
        if words & keywords:
            found.add(category)
    for category, phrases in MULTI_WORD.items():
        if any(f' {p} ' in padded for p in phrases):
            found.add(category)
    if words & set(COLOR_WORDS):
        found.add('color')
    if ADDED_OBJECT.search(' '.join(tokens)):
        found.add('addition')
    if words & NUMBERS or any(t.isdigit() for t in tokens) or PLURAL_AFTER_COMPARATIVE.search(' '.join(tokens)):
        found.add('count')
    # attribute phrases of the parser (e.g. "zoomed in", "polka dots") also count for their categories
    family_category = {'color': 'color', 'brightness': 'brightness', 'colorfulness': 'brightness',
                       'pattern': 'pattern', 'zoom': 'viewpoint'}
    for n in range(1, 4):
        for i in range(len(tokens) - n + 1):
            phrase = tuple(tokens[i:i + n])
            if phrase in PHRASES:
                found.add(family_category[PHRASES[phrase][0]])
    if 'black and white' in padded:  # "black and white" is about colourfulness, not about the colours black, white
        if not (words & (set(COLOR_WORDS) - {'black', 'white'})) and padded.count(' black ') <= 1 \
                and padded.count(' white ') <= 1:
            found.discard('color')
    return [c for c in CATEGORIES if c in found]


def categorize_query(captions: List[str]) -> List[str]:
    found = set()
    for caption in captions:
        found.update(categorize(caption))
    return [c for c in CATEGORIES if c in found]
