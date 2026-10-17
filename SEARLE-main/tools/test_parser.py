"""
Unit test of the constraint parser (plan 02, section 2.1; task C11) on hand-written captions.

    python tools/test_parser.py

A constraint is written family:value, with '-' in front when negated, '/orientation' for stripes and '@image' when
it is measured on the whole image (e.g. "white background").
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from attributes.parser import parse_query  # noqa: E402

CASES = [
    ("is red", {"color:red"}),
    ("is darker", {"brightness:darker"}),
    ("is lighter in color", {"brightness:brighter"}),
    ("is a darker red", {"brightness:darker", "color:red"}),
    ("has stripes", {"pattern:striped"}),
    ("has no stripes", {"-pattern:striped"}),
    ("is not red", {"-color:red"}),
    ("is red instead of blue", {"color:red", "-color:blue"}),
    ("is more colorful", {"colorfulness:more"}),
    ("is less colorful", {"colorfulness:less"}),
    ("the photo is black and white", {"colorfulness:grayscale"}),
    ("is in black and white and zoomed in", {"colorfulness:grayscale", "zoom:closer"}),
    ("has vertical stripes", {"pattern:striped/vertical"}),
    ("has horizontal stripes and is blue", {"pattern:striped/horizontal", "color:blue"}),
    ("is a solid black", {"pattern:solid", "color:black"}),
    ("is plain white with no logo", {"pattern:solid", "color:white"}),
    ("has a floral print", {"pattern:floral"}),
    ("is polka dotted", {"pattern:dotted"}),
    ("has a plaid pattern", {"pattern:plaid"}),
    ("has a graphic", {"pattern:patterned"}),
    ("has a different pattern", set()),
    ("is a closer view of the dog", {"zoom:closer"}),
    ("is zoomed out", {"zoom:farther"}),
    ("shows the scene from further away", {"zoom:farther"}),
    ("has a white background", {"color:white@image"}),
    ("the background is green", {"color:green@image"}),
    ("is navy blue", {"color:navy"}),
    ("is light blue and sleeveless", {"color:blue"}),
    ("is an orange cat", {"color:orange"}),
    ("a boy eating an orange", set()),
    ("is less bright", {"brightness:darker"}),
    ("is not as dark", {"brightness:brighter"}),
    ("is brighter", {"brightness:brighter"}),
    ("has brighter colors", {"colorfulness:more"}),
    ("is grey and has no pattern", {"color:gray", "pattern:solid"}),
    ("remove the red car", {"-color:red"}),
    ("is darker and not striped", {"brightness:darker", "-pattern:striped"}),
    ("is pink with white polka dots", {"color:pink", "color:white", "pattern:dotted"}),
    ("two dogs instead of one", set()),
    ("is shiny and gold", {"color:gold"}),
    ("is a long dress", set()),
    ("is more revealing", set()),
    ("is beige and less fitted", {"color:beige"}),
    ("is less patterned", {"-pattern:patterned"}),
    ("has red and black plaid", {"color:red", "color:black", "pattern:plaid"}),
    ("is a t-shirt with a dog graphic", {"pattern:patterned"}),
    ("is in color rather than black and white", {"-colorfulness:grayscale"}),
    ("has a more colorful background", {"colorfulness:more@image"}),
    ("is a close up of a cat", {"zoom:closer"}),
    ("is darker and has a black background", {"brightness:darker", "color:black@image"}),
    ("is red, not blue", {"color:red", "-color:blue"}),
    ("without the stripes", {"-pattern:striped"}),
    ("doesn't have stripes", {"-pattern:striped"}),
    ("is a dark green", {"color:green"}),
    ("is a monochrome photo", {"colorfulness:grayscale"}),
    ("the dog is brown and white", {"color:brown", "color:white"}),
    ("has a checkered pattern", {"pattern:plaid"}),
    ("is solid and darker", {"pattern:solid", "brightness:darker"}),
    ("is no longer black and white", {"-colorfulness:grayscale"}),
    ("Change to a close-up of only a single dog, must have chocolate-brown coloured fur",
     {"zoom:closer", "color:brown"}),
    ("not revealing royal blue knee length dress", {"color:blue"}),
    ("is seen from a farther distance", {"zoom:farther"}),
    ("has the same color pattern", set()),
    ("has better lighting", {"brightness:brighter"}),
    ("Change the green beads to blue beads in safety pin", {"-color:green", "color:blue"}),
    ("from red to blue", {"-color:red", "color:blue"}),
    ("is pink chevron color", {"color:pink", "pattern:patterned"}),
    ("is pants with fading", {"colorfulness:less"}),
]
QUERY_CASES = [  # FashionIQ: two captions, constraints merged
    (["is red", "is not red"], set()),
    (["is darker", "has stripes"], {"brightness:darker", "pattern:striped"}),
    (["is pink and strapless", "is pink with no straps"], {"color:pink"}),
]


def describe(c) -> str:
    return f"{'-' if c.negated else ''}{c.family}:{c.value}{'/' + c.orientation if c.orientation else ''}" \
           f"{'@image' if c.region == 'image' else ''}"


def main():
    failures = 0
    cases = [([caption], expected) for caption, expected in CASES] + QUERY_CASES
    for captions, expected in cases:
        got = {describe(c) for c in parse_query(captions)}
        if got != expected:
            failures += 1
            print(f"FAIL {captions}: expected {sorted(expected)}, got {sorted(got)}")
    print(f"{len(cases) - failures}/{len(cases)} cases passed")
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
