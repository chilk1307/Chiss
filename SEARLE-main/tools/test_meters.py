"""
Experiment E2.1 (plan 02, section 2.2): check the attribute meters on synthetic images whose answer is known.

    python tools/test_meters.py --out results/E2.1-meters [--images-dir demo/sample_images]

Tests
    color        flat garment-like shapes of every colour shade (+ noise) on a white background: the right colour
                 name must get the highest colour satisfaction
    stripes      sinusoidal stripes (periods 6-32 px, 4 orientations) vs flat shapes vs random blobs: striped,
                 patterned and solid scores, and the vertical / horizontal decision
    brightness   real photos with gamma 0.5 ... 2.0: L* must decrease monotonically, "darker" must hold between
                 consecutive levels
    colourfulness real photos with saturation x0 ... x1.5: the colourfulness index must increase monotonically,
                 x0 must be "black and white"
    zoom         disks of growing radius on a white background: area must increase, "closer" must hold
    masks        IoU of each mask method with the true object mask, on a white background and on a textured one
Writes <out>/meters_test.json and <out>/report.md; prints the summary.
"""
import json
import math
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from attributes.meters import (COLOR_PROTOTYPES_RGB, MASK_MODES, measure, object_mask, satisfaction,  # noqa: E402
                               unpack, resize_long_side)
from attributes.parser import Constraint  # noqa: E402

RNG = np.random.default_rng(0)
SIZE = 224


def shape_mask(kind: str = 'ellipse', scale: float = 0.6) -> np.ndarray:
    mask = np.zeros((SIZE, SIZE), np.uint8)
    c = SIZE // 2
    if kind == 'ellipse':
        cv2.ellipse(mask, (c, c), (int(scale * SIZE * 0.4), int(scale * SIZE * 0.6)), 0, 0, 360, 1, -1)
    else:  # dress-like trapezoid
        pts = np.array([[c - 30, 30], [c + 30, 30], [c + 70, SIZE - 20], [c - 70, SIZE - 20]], np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def on_background(texture: np.ndarray, mask: np.ndarray, background=None) -> np.ndarray:
    bg = np.full((SIZE, SIZE, 3), 250, np.uint8) if background is None else background
    out = bg.copy()
    out[mask] = texture[mask]
    return out


def noisy_flat(rgb, sigma=6.0) -> np.ndarray:
    img = np.ones((SIZE, SIZE, 3)) * np.array(rgb, dtype=np.float64)
    return np.clip(img + RNG.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def stripes(period: float, angle_deg: float, colors=((30, 30, 120), (230, 230, 230))) -> np.ndarray:
    yy, xx = np.mgrid[:SIZE, :SIZE].astype(np.float64)
    theta = math.radians(angle_deg)
    phase = 2 * math.pi * (xx * math.cos(theta) + yy * math.sin(theta)) / period
    t = (np.sin(phase) + 1) / 2
    c0, c1 = np.array(colors[0], float), np.array(colors[1], float)
    return (c0 * t[..., None] + c1 * (1 - t[..., None])).astype(np.uint8)


def blobs(n: int = 60) -> np.ndarray:
    img = noisy_flat((200, 60, 80))
    for _ in range(n):
        color = tuple(int(v) for v in RNG.integers(0, 255, 3))
        cv2.circle(img, (int(RNG.integers(0, SIZE)), int(RNG.integers(0, SIZE))), int(RNG.integers(3, 12)), color, -1)
    return img


def m_of(rgb, mode='auto'):
    return unpack(measure(rgb, mode))


def test_colors():
    names = list(COLOR_PROTOTYPES_RGB)
    correct, total, confusions = 0, 0, []
    for name, shades in COLOR_PROTOTYPES_RGB.items():
        for shade in shades:
            img = on_background(noisy_flat(shade), shape_mask('dress'))
            m = m_of(img)
            scores = {n: satisfaction(Constraint('color', n), m, None) for n in names}
            best = max(scores, key=scores.get)
            total += 1
            correct += best == name
            if best != name:
                confusions.append(f"{name} {shade} -> {best} ({scores[best]:.2f} vs {scores[name]:.2f})")
    return {'accuracy': correct / total, 'n': total, 'confusions': confusions}


def test_stripes():
    rows, ok = [], 0
    checks = 0
    for period in (6, 10, 16, 32):
        for angle, orientation in ((0, 'vertical'), (90, 'horizontal'), (45, None), (135, None)):
            img = on_background(stripes(period, angle), shape_mask('dress'))
            m = m_of(img)
            s = satisfaction(Constraint('pattern', 'striped'), m, None)
            p = satisfaction(Constraint('pattern', 'patterned'), m, None)
            row = {'period': period, 'angle': angle, 'striped': round(s, 3), 'patterned': round(p, 3),
                   'fft_peak': round(float(m['fft_peak'][0]), 3), 'fft_angle': round(float(m['fft_angle'][0]), 1)}
            good = s > 0.5
            if orientation:
                right = satisfaction(Constraint('pattern', 'striped', orientation=orientation), m, None)
                other = 'horizontal' if orientation == 'vertical' else 'vertical'
                wrong = satisfaction(Constraint('pattern', 'striped', orientation=other), m, None)
                row['right_orientation'], row['wrong_orientation'] = round(right, 3), round(wrong, 3)
                good = good and right > wrong
            rows.append(row)
            ok += good
            checks += 1
    negatives = []
    for label, img in (('flat', on_background(noisy_flat((40, 90, 160)), shape_mask('dress'))),
                       ('flat_dark', on_background(noisy_flat((30, 30, 30)), shape_mask('dress'))),
                       ('sparse_blobs', on_background(blobs(60), shape_mask('dress'))),
                       ('dense_blobs', on_background(blobs(200), shape_mask('dress')))):
        m = m_of(img)
        s = satisfaction(Constraint('pattern', 'striped'), m, None)
        p = satisfaction(Constraint('pattern', 'patterned'), m, None)
        expected_patterned = label.endswith('blobs')
        good = s < 0.5 and ((p > 0.5) == expected_patterned)
        negatives.append({'image': label, 'striped': round(s, 3), 'patterned': round(p, 3),
                          'edge_density': round(float(m['edge_density'][0]), 3),
                          'fft_peak': round(float(m['fft_peak'][0]), 3), 'ok': good})
        ok += good
        checks += 1
    return {'accuracy': ok / checks, 'n': checks, 'stripes': rows, 'negatives': negatives}


def load_photos(images_dir: Path, n: int = 6):
    paths = sorted(p for p in images_dir.glob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'))[:n]
    return [(p.name, resize_long_side(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))) for p in paths]


def test_brightness(photos):
    gammas = [0.5, 0.7, 1.0, 1.4, 2.0]
    monotonic, pairs_ok, pairs = 0, 0, 0
    detail = []
    for name, img in photos:
        ms = [m_of(np.clip(255 * (img / 255.0) ** g, 0, 255).astype(np.uint8), 'none') for g in gammas]
        ls = [float(m['l_obj'][0]) for m in ms]
        monotonic += all(a > b for a, b in zip(ls, ls[1:]))
        for a, b in zip(ms, ms[1:]):
            pairs += 1
            pairs_ok += satisfaction(Constraint('brightness', 'darker'), b, a) > 0.5 and \
                satisfaction(Constraint('brightness', 'brighter'), b, a) < 0.5
        detail.append({'image': name, 'L*': [round(v, 1) for v in ls]})
    return {'monotonic_images': f"{monotonic}/{len(photos)}", 'pairs_ok': pairs_ok / max(pairs, 1),
            'detail': detail}


def test_colorfulness(photos):
    factors = [0.0, 0.5, 1.0, 1.5]
    monotonic, gray_ok = 0, 0
    detail = []
    for name, img in photos:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float64)
        ms = []
        for f in factors:
            h = hsv.copy()
            h[..., 1] = np.clip(h[..., 1] * f, 0, 255)
            ms.append(m_of(cv2.cvtColor(h.astype(np.uint8), cv2.COLOR_HSV2RGB), 'none'))
        values = [float(m['m_obj'][0]) for m in ms]
        monotonic += all(a < b for a, b in zip(values, values[1:]))
        gray = [satisfaction(Constraint('colorfulness', 'grayscale'), m, None) for m in ms]
        gray_ok += gray[0] > 0.9 and gray[2] < 0.5
        detail.append({'image': name, 'colorfulness': [round(v, 1) for v in values],
                       'grayscale_score': [round(v, 2) for v in gray]})
    return {'monotonic_images': f"{monotonic}/{len(photos)}", 'grayscale_ok': f"{gray_ok}/{len(photos)}",
            'detail': detail}


def test_zoom():
    radii = [20, 40, 60, 90]
    ms = []
    for r in radii:
        img = np.full((SIZE, SIZE, 3), 250, np.uint8)
        cv2.circle(img, (SIZE // 2, SIZE // 2), r, (180, 40, 40), -1)
        ms.append(m_of(img))
    areas = [float(m['area'][0]) for m in ms]
    closer = [satisfaction(Constraint('zoom', 'closer'), b, a) for a, b in zip(ms, ms[1:])]
    true_areas = [math.pi * r * r / SIZE ** 2 for r in radii]
    return {'areas': [round(a, 3) for a in areas], 'true_areas': [round(a, 3) for a in true_areas],
            'monotonic': all(a < b for a, b in zip(areas, areas[1:])), 'closer_scores': [round(c, 3) for c in closer]}


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def test_masks():
    truth = shape_mask('dress')
    white = on_background(noisy_flat((60, 110, 170)), truth)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    textured_bg = np.clip(np.stack([90 + 40 * np.sin(xx / 7.0), 120 + 30 * np.cos(yy / 9.0),
                                    80 + 20 * np.sin((xx + yy) / 11.0)], -1) + RNG.normal(0, 12, (SIZE, SIZE, 3)),
                          0, 255).astype(np.uint8)
    scene = on_background(noisy_flat((220, 40, 40)), truth, textured_bg)
    result = {}
    for label, img in (('white_background', white), ('textured_background', scene)):
        result[label] = {}
        for mode in MASK_MODES:
            mask, used = object_mask(img, mode)
            result[label][mode] = {'iou': round(iou(mask, truth), 3), 'used': used}
    return result


def main():
    parser = ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--out', required=True)
    parser.add_argument('--images-dir', default=str(Path(__file__).resolve().parents[1] / 'demo' / 'sample_images'))
    args = parser.parse_args()

    photos = load_photos(Path(args.images_dir))
    results = {'color': test_colors(), 'stripes': test_stripes(), 'brightness': test_brightness(photos),
               'colorfulness': test_colorfulness(photos), 'zoom': test_zoom(), 'masks': test_masks()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'meters_test.json').write_text(json.dumps(results, indent=2), encoding='utf-8')

    lines = ["# E2.1 Attribute meters on synthetic images", "",
             "| Test | Result |", "|---|---|",
             f"| Colour name of flat shapes (top-1) | {100 * results['color']['accuracy']:.0f}% of "
             f"{results['color']['n']} |",
             f"| Stripes detected (+ orientation) and no stripes on flat / blobs | "
             f"{100 * results['stripes']['accuracy']:.0f}% of {results['stripes']['n']} |",
             f"| L* monotonic with gamma | {results['brightness']['monotonic_images']} photos |",
             f"| 'darker' between consecutive gamma levels | {100 * results['brightness']['pairs_ok']:.0f}% |",
             f"| Colourfulness monotonic with saturation | {results['colorfulness']['monotonic_images']} photos |",
             f"| Saturation x0 recognised as black and white | {results['colorfulness']['grayscale_ok']} photos |",
             f"| Object area monotonic with the disk radius | {results['zoom']['monotonic']} |", ""]
    lines += ["## Masks (IoU with the true object)", "", "| Image | " + " | ".join(MASK_MODES) + " |",
              "|---|" + "---|" * len(MASK_MODES)]
    for label, modes in results['masks'].items():
        lines.append(f"| {label} | " + " | ".join(f"{v['iou']:.2f} ({v['used']})" for v in modes.values()) + " |")
    if results['color']['confusions']:
        lines += ["", "Colour confusions: " + "; ".join(results['color']['confusions'])]
    report = '\n'.join(lines) + '\n'
    (out / 'report.md').write_text(report, encoding='utf-8')
    print(report)
    print(json.dumps({k: results[k] for k in ('stripes', 'zoom')}, indent=1)[:3000])


if __name__ == '__main__':
    main()
