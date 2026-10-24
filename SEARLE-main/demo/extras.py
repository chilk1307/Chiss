"""
Demo extensions (plan 05, section 5.2, task C17), kept out of app.py so the original demo stays readable:

    1. attribute re-ranking with explanations (plan 02): constraints parsed from the caption, classical meters
       measured on the candidates, new score = z(CLIP) + w * satisfaction, measured attributes shown per result;
    2. degraded query -> estimated degradation -> restoration (plan 03), with the results before and after;
    3. more methods to compare: CLIP image + text sum and classical CBIR (colour histogram, LBP, HOG, Gabor);
    4. "Why?": occlusion map of the reference image for one result (plan 01, section 1.13).

Everything works on the gallery of the demo; measurements and CBIR features are computed on demand and cached in
demo/cache/.
"""
import base64
import io
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DEMO_DIR = Path(__file__).resolve().parent
SRC_DIR = DEMO_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from attributes.meters import N_COLORS, measure, object_mask, resize_long_side, satisfaction_batch, unpack, explain  # noqa: E402
from attributes.parser import FAMILIES, parse_query  # noqa: E402
from cbir import CBIR_FEATURES, CBIRFeatureExtractor, cbir_similarity  # noqa: E402
from degradations import DEGRADATIONS, LEVELS, degrade  # noqa: E402
from degradation_estimation import choose_by_rules, estimate_features  # noqa: E402
from restorations import RESTORATIONS, restore  # noqa: E402

CACHE_DIR = DEMO_DIR / "cache"
ATTR_CACHE = CACHE_DIR / "attributes.pt"
CBIR_CACHE = CACHE_DIR / "cbir.pt"
MASK_MODE = "auto"
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

_lock = threading.Lock()
_attrs: Dict[str, np.ndarray] = {}
_cbir: Dict[str, torch.Tensor] = {}
_loaded = False


def _load_caches():
    global _loaded
    if _loaded:
        return
    _loaded = True
    for path, store in ((ATTR_CACHE, _attrs), (CBIR_CACHE, _cbir)):
        if path.exists():
            try:
                store.update(torch.load(path, map_location="cpu", weights_only=False))
            except Exception as e:  # noqa: BLE001
                print("[demo] cache ignored:", path.name, e)


def _save(path: Path, store: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(dict(store), path)


# ----------------------------------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------------------------------
def to_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def data_url(rgb: np.ndarray, max_side: int = 512, fmt: str = "JPEG") -> str:
    im = Image.fromarray(rgb.astype(np.uint8))
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, fmt, quality=88)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def lab_to_hex(lab: np.ndarray) -> str:
    """CIELAB (L in [0, 100]) -> #rrggbb"""
    from skimage.color import lab2rgb
    rgb = np.clip(lab2rgb(np.asarray(lab, np.float64).reshape(1, 1, 3)).reshape(3), 0, 1)
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in rgb)


def describe_measures(vector: np.ndarray) -> dict:
    """Readable summary of a meters.measure() vector: colour swatches, L*, colourfulness, pattern, object size"""
    m = unpack(vector)
    order = np.argsort(-m["obj_weights"])
    swatches = [{"hex": lab_to_hex(m["obj_centers"][i]), "share": round(float(m["obj_weights"][i]), 2)}
                for i in order if m["obj_weights"][i] > 0.05][:N_COLORS]
    return {
        "swatches": swatches,
        "lightness": round(float(m["l_obj"][0]), 1),
        "colorfulness": round(float(m["m_obj"][0]), 1),
        "edge_density": round(float(m["edge_density"][0]), 3),
        "fft_peak": round(float(m["fft_peak"][0]), 2),
        "area": round(float(m["area"][0]), 2),
    }


# ----------------------------------------------------------------------------------------------------------------------
# 1. Attribute re-ranking
# ----------------------------------------------------------------------------------------------------------------------
def measures_for(engine, ids: List[str]) -> Dict[str, np.ndarray]:
    _load_caches()
    todo = [i for i in ids if i not in _attrs]
    for img_id in todo:
        path = engine.items[engine.id2idx[img_id]]["path"]
        try:
            with Image.open(path) as im:
                _attrs[img_id] = measure(to_rgb(im), MASK_MODE)
        except Exception as e:  # noqa: BLE001
            print("[demo] measure failed:", path, e)
    if todo:
        with _lock:
            _save(ATTR_CACHE, _attrs)
    return {i: _attrs[i] for i in ids if i in _attrs}


def attribute_rerank(engine, sims: torch.Tensor, ref_rgb: np.ndarray, caption: str, topk: int, weight: float,
                     rerank_k: int, families: Optional[List[str]] = None) -> dict:
    """
    Re-rank the top rerank_k of the similarity vector `sims` (gallery order) with the attribute meters.
    Returns the constraints, the re-ranked top-k (with CLIP rank, satisfaction and explanation) and the ranking moves.
    """
    constraints = parse_query([caption], negation=True, families=families or FAMILIES)
    k = min(max(rerank_k, topk), len(engine.items))
    vals, idx = sims.topk(k)
    ids = [engine.items[i]["id"] for i in idx.tolist()]
    base = [{"id": i, "name": engine.items[j]["name"], "score": round(float(v), 4), "clip_rank": r + 1}
            for r, (i, j, v) in enumerate(zip(ids, idx.tolist(), vals.tolist()))]
    if not constraints:
        return {"constraints": [], "results": base[:topk], "note": "Câu mô tả không có ràng buộc màu / độ sáng / "
                "độ rực / hoa văn / mức phóng to nên thứ hạng giữ nguyên."}
    measured = measures_for(engine, ids)
    ref_vector = measure(ref_rgb, MASK_MODE)
    vectors = np.stack([measured.get(i, np.zeros_like(ref_vector)) for i in ids])
    ok = np.array([i in measured for i in ids])
    per = []
    for c in constraints:
        v = satisfaction_batch(c, vectors, ref_vector)
        if v is not None:
            per.append((c, np.where(ok, v, 0.5)))
    if not per:
        return {"constraints": [c.describe() for c in constraints], "results": base[:topk],
                "note": "Không đo được ràng buộc nào trên các ảnh ứng viên."}
    satisfaction = np.mean([v for _, v in per], axis=0)
    clip_scores = vals.double().numpy()
    z = (clip_scores - clip_scores.mean()) / (clip_scores.std() + 1e-6)
    new = z + weight * satisfaction
    order = np.argsort(-new)
    ref_m = unpack(ref_vector)
    results = []
    for new_rank, o in enumerate(order[:topk]):
        item = dict(base[o])
        item.update({
            "rank": new_rank + 1, "move": int(base[o]["clip_rank"] - (new_rank + 1)),
            "satisfaction": round(float(satisfaction[o]), 3),
            "per_constraint": [{"constraint": c.describe(), "value": round(float(v[o]), 3)} for c, v in per],
            "why": [explain(c, unpack(vectors[o]), ref_m) for c, _ in per] if ok[o] else [],
            "measures": describe_measures(vectors[o]) if ok[o] else None,
        })
        results.append(item)
    return {"constraints": [c.describe() for c, _ in per], "results": results, "weight": weight, "rerank_k": k,
            "reference_measures": describe_measures(ref_vector)}


def mask_overlay(engine, img_id: str) -> str:
    """Object mask used by the meters, drawn over the image (green = object)"""
    path = engine.items[engine.id2idx[img_id]]["path"]
    with Image.open(path) as im:
        rgb = resize_long_side(np.ascontiguousarray(to_rgb(im)))
    mask, used = object_mask(rgb, MASK_MODE)
    over = rgb.astype(np.float32)
    over[~mask] = over[~mask] * 0.35
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    over = np.ascontiguousarray(over.astype(np.uint8))
    cv2.drawContours(over, contours, -1, (40, 220, 90), 2)
    return data_url(over, 320, "PNG"), used


# ----------------------------------------------------------------------------------------------------------------------
# 2. Degradation and restoration
# ----------------------------------------------------------------------------------------------------------------------
def degrade_and_restore(ref_rgb: np.ndarray, degrade_spec: str, restore_mode: str) -> dict:
    """
    degrade_spec: 'none' or '<type>:<level>'; restore_mode: 'none', 'auto' (rules on estimated features) or a method
    """
    out = {"degradation": degrade_spec, "restore_mode": restore_mode}
    image, context = ref_rgb, {}
    if degrade_spec and degrade_spec != "none":
        kind, _, level = degrade_spec.partition(":")
        if kind not in DEGRADATIONS or int(level or 0) not in LEVELS[kind]:
            raise ValueError(f"suy giảm không hợp lệ: {degrade_spec}")
        image, context = degrade(ref_rgb, kind, int(level), "demo")
    out["degraded"] = image
    feats = estimate_features(image)
    out["features"] = {k: round(v, 3) for k, v in feats.items()}
    method = None
    if restore_mode == "auto":
        method = choose_by_rules(feats)
    elif restore_mode in RESTORATIONS:
        method = restore_mode
    out["method"] = method or "none"
    out["restored"] = restore(image, method, context) if method else image
    return out


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


# ----------------------------------------------------------------------------------------------------------------------
# 3. CBIR
# ----------------------------------------------------------------------------------------------------------------------
def cbir_sims(engine, ref: Image.Image, features: List[str] = CBIR_FEATURES) -> torch.Tensor:
    _load_caches()
    extractor = CBIRFeatureExtractor()
    todo = [it for it in engine.items if it["id"] not in _cbir]
    for it in todo:
        try:
            with Image.open(it["path"]) as im:
                _cbir[it["id"]] = extractor(im)
        except Exception:  # noqa: BLE001
            pass
    if todo:
        with _lock:
            _save(CBIR_CACHE, _cbir)
    dim = extractor(ref).shape[0]
    gallery = torch.stack([_cbir.get(it["id"], torch.zeros(dim)) for it in engine.items])
    return cbir_similarity(extractor(ref)[None], gallery, features).squeeze(0)


# ----------------------------------------------------------------------------------------------------------------------
# 4. Occlusion map ("Why?")
# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def occlusion_map(engine, ref: Image.Image, caption: str, target_id: str, patch: int = 32, stride: int = 16) -> dict:
    """
    Every patch of the preprocessed reference is set to neutral grey; the drop of the target score is drawn in red
    """
    import clip
    from src.encode_with_pseudo_tokens import encode_with_pseudo_tokens
    target = engine.feats[engine.id2idx[target_id]].to(engine.device)
    x = engine.preprocess(ref.convert("RGB"))
    _, h, w = x.shape
    grey = ((0.5 - CLIP_MEAN) / CLIP_STD).to(x.dtype)
    ys, xs = list(range(0, h - patch + 1, stride)), list(range(0, w - patch + 1, stride))
    batch = [x]
    for y in ys:
        for xx in xs:
            o = x.clone()
            o[:, y:y + patch, xx:xx + patch] = grey
            batch.append(o)
    batch = torch.stack(batch)
    prompt = f"a photo of $ that {caption.replace('$', ' ').strip()}" if caption.strip() else "a photo of $"
    tokens = clip.tokenize([prompt], context_length=77, truncate=True).to(engine.device)
    scores = []
    with engine.lock:
        for i in range(0, len(batch), 64):
            b = batch[i:i + 64].to(engine.device)
            pseudo = engine.phi(engine.clip.encode_image(b).float())
            text = encode_with_pseudo_tokens(engine.clip, tokens.expand(len(b), -1), pseudo).float()
            scores.append(F.normalize(text, dim=-1) @ target)
    scores = torch.cat(scores).cpu().numpy()
    base, drops = float(scores[0]), scores[0] - scores[1:]
    heat = np.zeros((h, w), np.float64)
    count = np.zeros((h, w), np.float64)
    k = 0
    for y in ys:
        for xx in xs:
            heat[y:y + patch, xx:xx + patch] += drops[k]
            count[y:y + patch, xx:xx + patch] += 1
            k += 1
    heat = heat / np.maximum(count, 1)
    heat = np.clip(heat / max(heat.max(), 1e-8), 0, 1)
    shown = (x * CLIP_STD + CLIP_MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
    color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1] / 255.0
    over = (0.55 * shown + 0.45 * color) * 255
    return {"overlay": data_url(over, 448, "PNG"), "base_score": round(base, 4),
            "max_drop": round(float(drops.max()), 4), "patch": patch, "stride": stride}
