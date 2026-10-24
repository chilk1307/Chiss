"""
SEARLE web demo - Zero-shot Composed Image Retrieval (ICCV 2023).

Backend Flask: nạp CLIP ViT-B/32 + mạng Phi bạn đã huấn luyện (data/phi_models/phi_b32),
cho phép dựng thư viện ảnh (gallery) rồi truy vấn "ảnh tham chiếu + mô tả thay đổi".

Chạy:  python demo/app.py      ->  http://127.0.0.1:5050
"""
import io
import json
import pickle
import re
import sys
import threading
import time
import uuid
from pathlib import Path

import clip
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEMO_DIR = Path(__file__).resolve().parent
REPO_DIR = DEMO_DIR.parent
OUTPUTS_DIR = REPO_DIR.parent / "outputs"
sys.path.insert(0, str(REPO_DIR))

from src.data_utils import targetpad_transform  # noqa: E402
from src.encode_with_pseudo_tokens import encode_with_pseudo_tokens  # noqa: E402
from src.phi import Phi  # noqa: E402
import extras  # noqa: E402  (C17: attribute re-ranking, degradation/restoration, CBIR, occlusion map)

CLIP_NAME = "ViT-B/32"
# Ưu tiên bản tái lập đầy đủ (OTI 100k ảnh) trong outputs/, nếu không có thì dùng data/ của repo
_PHI_CANDIDATES = [OUTPUTS_DIR / "phi_models" / "phi_B32", REPO_DIR / "data" / "phi_models" / "phi_b32"]
PHI_DIR = next((p for p in _PHI_CANDIDATES if (p / "checkpoints").is_dir()), _PHI_CANDIDATES[-1])
CKPT_DIR = PHI_DIR / "checkpoints"
OTI_NAMES = next((p for p in [
    OUTPUTS_DIR / "oti_pseudo_tokens/imagenet/test/oti_B32/image_names.pkl",
    REPO_DIR / "data/oti_pseudo_tokens/imagenet/test/oti_b32/image_names.pkl"] if p.exists()), None)
UPLOAD_DIR = DEMO_DIR / "gallery"
CACHE_FILE = DEMO_DIR / "cache" / "gallery.pt"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# Kết quả gốc của paper (SEARLE, ViT-B/32) - arXiv 2303.15247v2, Table 9 (CIRR val), Table 1 (FashionIQ val).
# Đã đối chiếu từng số với arXiv v2 ngày 17/10/2026 (việc C7). Cùng bài: CIRCO val ở Table 10
# (mAP@5/10/25/50 = 6.82 / 7.83 / 9.15 / 9.77); CIRR test ở Table 2 (R@1 24.00) - không trộn với số val.
PAPER_CIRR_VAL = {
    "cirr_recall_at1": 25.09, "cirr_recall_at5": 55.18, "cirr_recall_at10": 68.79, "cirr_recall_at50": 90.82,
    "cirr_group_recall_at1": 54.84, "cirr_group_recall_at2": 76.63, "cirr_group_recall_at3": 87.95,
}
PAPER_FIQ_VAL = {  # (R@10, R@50)
    "shirt": (24.44, 41.61), "dress": (18.54, 39.51), "toptee": (25.70, 46.46),
}
PAPER_URL = "https://arxiv.org/abs/2303.15247"
METRIC_LABELS = {
    "cirr_recall_at1": "R@1", "cirr_recall_at5": "R@5", "cirr_recall_at10": "R@10", "cirr_recall_at50": "R@50",
    "cirr_group_recall_at1": "Rsubset@1", "cirr_group_recall_at2": "Rsubset@2", "cirr_group_recall_at3": "Rsubset@3",
}

app = Flask(__name__, static_folder=str(DEMO_DIR / "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024


# ----------------------------------------------------------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------------------------------------------------------
class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state = "loading"  # loading | ready | error
        self.error = ""
        self.clip = None
        self.preprocess = None
        self.phi = None
        self.ckpt_name = None
        self.tok_emb = None  # normalized CLIP token embeddings (for pseudo-word neighbours)
        self.tokenizer = None
        # gallery
        self.items: list[dict] = []  # {id, path, name}
        self.feats: torch.Tensor | None = None
        self.id2idx: dict[str, int] = {}
        self.job = {"running": False, "done": 0, "total": 0, "message": "", "failed": 0}
        self.thumbs: dict[tuple, bytes] = {}

    # -- model --------------------------------------------------------------------------------------------------------
    def load(self):
        try:
            t0 = time.time()
            model, _ = clip.load(CLIP_NAME, device=self.device, jit=False)
            model.eval().float()
            self.clip = model
            self.preprocess = targetpad_transform(1.25, model.visual.input_resolution)
            with torch.no_grad():
                emb = model.token_embedding.weight.float().cpu()
            self.tok_emb = F.normalize(emb, dim=-1)
            from clip.simple_tokenizer import SimpleTokenizer
            self.tokenizer = SimpleTokenizer()
            self.load_phi(self.default_ckpt())
            self.load_gallery_cache()
            self.state = "ready"
            print(f"[demo] model ready in {time.time() - t0:.1f}s on {self.device}")
        except Exception as e:  # noqa: BLE001
            self.state, self.error = "error", f"{type(e).__name__}: {e}"
            print("[demo] load error:", self.error)

    @staticmethod
    def list_ckpts() -> list[str]:
        def key(p: str):
            m = re.search(r"phi_(\d+)\.pt", p)
            return (0, int(m.group(1))) if m else (1, 0)
        return sorted([p.name for p in CKPT_DIR.glob("phi_*.pt")], key=key)

    def default_ckpt(self) -> str:
        return "phi_last.pt" if (CKPT_DIR / "phi_last.pt").exists() else self.list_ckpts()[-1]

    def load_phi(self, name: str):
        path = CKPT_DIR / Path(name).name
        if not path.exists():
            raise FileNotFoundError(path)
        phi = Phi(input_dim=512, hidden_dim=2048, output_dim=512, dropout=0.5)
        ck = torch.load(path, map_location="cpu")
        phi.load_state_dict(ck["Phi"])
        self.phi = phi.to(self.device).eval()
        self.ckpt_name = path.name
        self.ckpt_epoch = int(ck.get("epoch", -1)) if isinstance(ck, dict) else -1

    # -- gallery ------------------------------------------------------------------------------------------------------
    @staticmethod
    def make_id(path: Path) -> str:
        import hashlib
        return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]

    def load_gallery_cache(self):
        if not CACHE_FILE.exists():
            return
        try:
            data = torch.load(CACHE_FILE, map_location="cpu", weights_only=False)
            keep = [i for i, it in enumerate(data["items"]) if Path(it["path"]).exists()]
            self.items = [data["items"][i] for i in keep]
            self.feats = data["feats"][keep] if keep else None
            self.id2idx = {it["id"]: i for i, it in enumerate(self.items)}
        except Exception as e:  # noqa: BLE001
            print("[demo] gallery cache ignored:", e)

    def save_gallery_cache(self):
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"items": self.items, "feats": self.feats}, CACHE_FILE)

    def clear_gallery(self):
        with self.lock:
            self.items, self.feats, self.id2idx, self.thumbs = [], None, {}, {}
            if CACHE_FILE.exists():
                CACHE_FILE.unlink()
            for p in UPLOAD_DIR.glob("*"):
                if p.is_file():
                    p.unlink()

    def add_paths(self, paths: list[Path]):
        """Chạy trong thread nền: trích đặc trưng CLIP cho ảnh mới và thêm vào gallery."""
        self.job.update(running=True, done=0, total=len(paths), message="Đang trích đặc trưng CLIP…", failed=0)
        try:
            known = set(self.id2idx)
            todo = []
            for p in paths:
                pid = self.make_id(p)
                if pid not in known:
                    known.add(pid)
                    todo.append((pid, p))
            self.job["total"] = len(todo)
            new_items, new_feats = [], []
            batch_imgs, batch_meta = [], []

            def flush():
                if not batch_imgs:
                    return
                with self.lock, torch.no_grad():
                    f = self.clip.encode_image(torch.stack(batch_imgs).to(self.device)).float()
                new_feats.append(F.normalize(f, dim=-1).cpu())
                new_items.extend(batch_meta)
                batch_imgs.clear()
                batch_meta.clear()

            for pid, p in todo:
                try:
                    with Image.open(p) as im:
                        batch_imgs.append(self.preprocess(im.convert("RGB")))
                    batch_meta.append({"id": pid, "path": str(p.resolve()), "name": p.name})
                except Exception:  # noqa: BLE001
                    self.job["failed"] += 1
                self.job["done"] += 1
                if len(batch_imgs) >= 32:
                    flush()
            flush()
            if new_items:
                feats = torch.cat(new_feats)
                with self.lock:
                    self.feats = feats if self.feats is None else torch.cat([self.feats, feats])
                    self.items.extend(new_items)
                    self.id2idx = {it["id"]: i for i, it in enumerate(self.items)}
                self.save_gallery_cache()
            self.job["message"] = f"Hoàn tất: thêm {len(new_items)} ảnh."
        except Exception as e:  # noqa: BLE001
            self.job["message"] = f"Lỗi: {type(e).__name__}: {e}"
        finally:
            self.job["running"] = False

    # -- query --------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def search(self, ref: Image.Image, caption: str, mode: str, topk: int, exclude_id: str | None):
        caption = caption.replace("$", " ").strip()
        t0 = time.time()
        with self.lock:
            x = self.preprocess(ref.convert("RGB")).unsqueeze(0).to(self.device)
            img_f = self.clip.encode_image(x).float()
            token = self.phi(img_f)
            if mode == "text":
                prompt = caption
                text_f = self.clip.encode_text(clip.tokenize([prompt], truncate=True).to(self.device)).float()
            else:
                prompt = "a photo of $" if mode == "image" else f"a photo of $ that {caption}"
                tokens = clip.tokenize([prompt], context_length=77, truncate=True).to(self.device)
                text_f = encode_with_pseudo_tokens(self.clip, tokens, token).float()
        q = F.normalize(text_f, dim=-1).cpu()
        sims = (q @ self.feats.T).squeeze(0)
        if exclude_id and exclude_id in self.id2idx:
            sims[self.id2idx[exclude_id]] = -2
        k = min(topk, len(self.items))
        vals, idx = sims.topk(k)
        results = [{"id": self.items[i]["id"], "name": self.items[i]["name"], "score": round(float(v), 4)}
                   for v, i in zip(vals.tolist(), idx.tolist())]
        return {
            "results": results, "prompt": prompt, "mode": mode,
            "neighbors": self.pseudo_word_neighbors(token.cpu()),
            "ms": int((time.time() - t0) * 1000), "gallery_size": len(self.items),
        }

    @torch.no_grad()
    def query_sims(self, ref: Image.Image, caption: str, mode: str):
        """
        Similarity of the query with every gallery image (C17). Modes: composed / image / text (as search), plus
        sum (CLIP image feature + caption feature) and cbir (classical colour/LBP/HOG/Gabor, demo/extras.py).
        Returns (sims, prompt, pseudo token or None).
        """
        caption = caption.replace("$", " ").strip()
        if mode == "cbir":
            return extras.cbir_sims(self, ref), "(CBIR cổ điển: histogram màu, LBP, HOG, Gabor)", None
        with self.lock:
            x = self.preprocess(ref.convert("RGB")).unsqueeze(0).to(self.device)
            img_f = self.clip.encode_image(x).float()
            token = self.phi(img_f)
            if mode in ("text", "sum"):
                prompt = caption
                text_f = self.clip.encode_text(clip.tokenize([prompt], truncate=True).to(self.device)).float()
                if mode == "sum":
                    text_f = F.normalize(img_f, dim=-1) + F.normalize(text_f, dim=-1)
                    prompt = f"ảnh + \"{caption}\" (cộng vector)"
            else:
                prompt = "a photo of $" if mode == "image" else f"a photo of $ that {caption}"
                tokens = clip.tokenize([prompt], context_length=77, truncate=True).to(self.device)
                text_f = encode_with_pseudo_tokens(self.clip, tokens, token).float()
        q = F.normalize(text_f, dim=-1).cpu()
        return (q @ self.feats.T).squeeze(0), prompt, token.cpu()

    def top_results(self, sims: torch.Tensor, topk: int, exclude_id: str | None) -> list[dict]:
        sims = sims.clone()
        if exclude_id and exclude_id in self.id2idx:
            sims[self.id2idx[exclude_id]] = -2
        vals, idx = sims.topk(min(topk, len(self.items)))
        return [{"id": self.items[i]["id"], "name": self.items[i]["name"], "score": round(float(v), 4)}
                for v, i in zip(vals.tolist(), idx.tolist())]

    def pseudo_word_neighbors(self, token: torch.Tensor, n: int = 8) -> list[dict]:
        """Các từ CLIP gần nhất với pseudo-word S* (cho thấy ảnh được 'đảo ngược' thành chữ gì)."""
        sims = (F.normalize(token, dim=-1) @ self.tok_emb[:49406].T).squeeze(0)
        vals, idx = sims.topk(60)
        out = []
        for v, i in zip(vals.tolist(), idx.tolist()):
            w = self.tokenizer.decode([i]).strip()
            if len(w) >= 3 and w.isalpha():
                out.append({"word": w, "sim": round(v, 3)})
            if len(out) >= n:
                break
        return out


engine = Engine()


# ----------------------------------------------------------------------------------------------------------------------
# Đánh giá dự án (đọc từ file thật trong outputs/)
# ----------------------------------------------------------------------------------------------------------------------
def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""


def _md_rows(section: str, label_prefix: str) -> list[float]:
    """Lấy các số trong dòng bảng markdown của REPORT_results.md, vd '| phi_B32 (ours) | 25.47 | ... |'."""
    m = re.search(r"^\|\s*" + re.escape(label_prefix) + r"[^|]*\|(.*)\|\s*$", section, re.M)
    return [float(x) for x in re.findall(r"\d+\.\d+", m.group(1))] if m else []


def parse_results() -> dict:
    """Đọc outputs/REPORT_results.md -> {'cirr': {...}, 'fiq': {...}, 'circo': {...}} cho phi_B32 (ours) và SEARLE official."""
    txt = _read(OUTPUTS_DIR / "REPORT_results.md")
    secs = {name: body for name, body in re.findall(r"^## (CIRR|FashionIQ|CIRCO)[^\n]*\n(.*?)(?=^## |\Z)", txt, re.M | re.S)}
    out = {"cirr": {}, "fiq": {}, "circo": {}}
    for who, prefix in (("ours", "phi_B32"), ("official", "SEARLE")):
        v = _md_rows(secs.get("CIRR", ""), prefix)
        if len(v) == 7:
            out["cirr"][who] = dict(zip(METRIC_LABELS, v))
        v = _md_rows(secs.get("FashionIQ", ""), prefix)  # dress r10 r50 | shirt r10 r50 | toptee r10 r50 | avg r10 r50
        if len(v) == 8:
            out["fiq"][who] = {c: (v[2 * i], v[2 * i + 1]) for i, c in enumerate(("dress", "shirt", "toptee"))}
        v = _md_rows(secs.get("CIRCO", ""), prefix)  # mAP@5..50, Recall@5..50
        if len(v) == 8:
            out["circo"][who] = {"map": v[:4], "recall": v[4:]}
    return out


SUBMISSION_DIRS = [OUTPUTS_DIR / "submissions", REPO_DIR / "data" / "test_submissions"]
SUBMISSION_KINDS = {"cirr_recall": "CIRR", "cirr_recall_subset": "CIRR subset", "circo": "CIRCO"}
_submission_cache: dict[tuple, dict] = {}


def scan_submissions() -> list[dict]:
    """Tìm file nộp test do generate_test_submission.py sinh ra (CIRR có trường 'metric', CIRCO thì không)."""
    found = []
    for root in SUBMISSION_DIRS:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*.json")):
            key = (str(p), p.stat().st_mtime)
            if key not in _submission_cache:
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    continue
                if not isinstance(data, dict):
                    continue
                metric = data.get("metric")
                kind = f"cirr_{metric}" if metric else "circo"
                n = sum(isinstance(v, list) for v in data.values())
                _submission_cache[key] = {"file": p.name, "kind": kind, "queries": n}
            if _submission_cache[key]["kind"] in SUBMISSION_KINDS:
                found.append(_submission_cache[key])
    return found


def _ratio(a, b):
    return round(a / b * 100, 1) if a is not None and b else None


def build_assessment() -> dict:
    res = parse_results()
    cirr, cirr_off = res["cirr"].get("ours", {}), res["cirr"].get("official", {})

    metrics = []
    for k, label in METRIC_LABELS.items():
        mine, paper, off = cirr.get(k), PAPER_CIRR_VAL[k], cirr_off.get(k)
        metrics.append({"key": k, "label": label, "mine": mine, "paper": paper, "official": off,
                        "ratio": _ratio(mine, paper), "ratio_official": _ratio(mine, off)})
    ratios = [m["ratio"] for m in metrics if m["ratio"] is not None]
    rat_off = [m["ratio_official"] for m in metrics if m["ratio_official"] is not None]
    cirr_pct = round(sum(ratios) / len(ratios), 1) if ratios else None
    cirr_pct_off = round(sum(rat_off) / len(rat_off), 1) if rat_off else None
    avg = lambda d: (d["cirr_recall_at5"] + d["cirr_group_recall_at1"]) / 2 if d else None  # noqa: E731
    avg_mine, avg_off = avg(cirr), avg(cirr_off)
    avg_paper = (PAPER_CIRR_VAL["cirr_recall_at5"] + PAPER_CIRR_VAL["cirr_group_recall_at1"]) / 2

    n_oti = len(pickle.load(open(OTI_NAMES, "rb"))) if OTI_NAMES else 0
    vm = PHI_DIR / "validation_metrics.csv"
    vm_df = pd.read_csv(vm) if vm.exists() else pd.DataFrame()
    n_epochs = len(vm_df)
    gpt_pkl = REPO_DIR / "data/GPT_phrases/GPTNeo27B/concept_to_phrases.pkl"
    if not gpt_pkl.exists():
        gpt_pkl = OUTPUTS_DIR / "GPT_phrases/GPTNeo27B/concept_to_phrases.pkl"
    assoc_csv = OUTPUTS_DIR / "similar_concepts/imagenet/test/ViT-B32_top_250_ensembling_image_texts.csv"
    fiq, circo = res["fiq"].get("ours", {}), res["circo"].get("ours")
    full_oti = n_oti >= 100000
    subs = scan_submissions()
    sub_kinds = {s["kind"] for s in subs}
    sub_status = "done" if sub_kinds >= set(SUBMISSION_KINDS) else ("partial" if sub_kinds else "missing")
    sub_missing = [SUBMISSION_KINDS[k] for k in SUBMISSION_KINDS if k not in sub_kinds]
    sub_evidence = "; ".join(f"{s['file']} ({SUBMISSION_KINDS[s['kind']]}, {s['queries']:,} truy vấn)" for s in subs)
    if subs and sub_missing:
        sub_evidence += " · thiếu: " + ", ".join(sub_missing)

    items = [
        {"id": "gpt", "step": "0 · Sinh cụm từ GPT cho các khái niệm", "group": "core", "credit": 1.0,
         "status": "done" if gpt_pkl.exists() else "missing",
         "evidence": f"concept_to_phrases.pkl ({gpt_pkl.stat().st_size / 1e6:.0f} MB, bản phát hành chính thức)" if gpt_pkl.exists() else "chưa có"},
        {"id": "assoc", "step": "1 · Gắn khái niệm cho ảnh ImageNet", "group": "core",
         "credit": 1.0 if assoc_csv.exists() else 0.5,
         "status": "done" if assoc_csv.exists() else "partial",
         "evidence": f"ViT-B32_top_250_ensembling_image_texts.csv ({assoc_csv.stat().st_size / 1e6:.0f} MB) - toàn bộ ảnh test"
         if assoc_csv.exists() else "chỉ có bản nhỏ trong data/"},
        {"id": "oti", "step": "2 · Optimization-based Textual Inversion (OTI)", "group": "core",
         "credit": 1.0 if full_oti else 0.5, "status": "done" if full_oti else "partial",
         "evidence": f"{n_oti:,}/100,000 pseudo-token; chạy 3 shard (seed 42/43/44) rồi gộp"},
        {"id": "phi", "step": "3 · Huấn luyện mạng Phi (distillation)", "group": "core",
         "credit": 1.0 if n_epochs >= 100 else 0.5, "status": "done" if n_epochs >= 100 else "partial",
         "evidence": f"{n_epochs}/100 epoch; siêu tham số khớp mặc định README (lr 1e-4, batch 256, λgpt 0.75, T 0.25)"},
        {"id": "cirr", "step": "4a · Đánh giá CIRR val", "group": "core", "credit": 1.0 if cirr else 0.0,
         "status": "done" if cirr else "missing",
         "evidence": f"đủ 7 chỉ số; {cirr_pct_off}% so với SEARLE official chạy cùng pipeline" if cirr else "chưa có kết quả"},
        {"id": "fiq", "step": "4a · Đánh giá FashionIQ val", "group": "core",
         "credit": 0.5 if fiq else 0.0, "status": "partial" if fiq else "missing",
         "evidence": "đủ 3 loại nhưng ~2.431 ảnh là link chết đã bị lọc - không so trực tiếp được với paper" if fiq else "chưa có"},
        {"id": "circo", "step": "4a · Đánh giá CIRCO val", "group": "core",
         "credit": 1.0 if circo else 0.0, "status": "done" if circo else "missing",
         "evidence": f"mAP@5 = {circo['map'][0]:.2f}, Recall@50 = {circo['recall'][3]:.2f}" if circo else "chưa có"},
        {"id": "test", "step": "4b · Sinh file nộp test (CIRR / CIRCO server)", "group": "extra",
         "credit": {"done": 1.0, "partial": 0.5, "missing": 0.0}[sub_status], "status": sub_status,
         "evidence": sub_evidence or "chưa chạy generate_test_submission.py"},
        {"id": "xl", "step": "SEARLE-XL (CLIP ViT-L/14)", "group": "extra",
         "credit": 0.0, "status": "missing", "evidence": "chỉ có phi_B32"},
        {"id": "otib", "step": "SEARLE-OTI trực tiếp trên benchmark", "group": "extra",
         "credit": 0.0, "status": "missing", "evidence": "chưa chạy"},
    ]
    core = [i for i in items if i["group"] == "core"]
    core_pct = round(sum(i["credit"] for i in core) / len(core) * 100)
    all_pct = round(sum(i["credit"] for i in items) / len(items) * 100)

    curve = []
    if len(vm_df):
        vm_df["avg"] = (vm_df.cirr_recall_at5 + vm_df.cirr_group_recall_at1) / 2
        curve = [{"epoch": int(r.epoch), "r1": round(r.cirr_recall_at1, 2), "avg": round(r.avg, 2)}
                 for r in vm_df.itertuples()]
    best = max(curve, key=lambda c: c["avg"]) if curve else None

    fiq_off = res["fiq"].get("official", {})
    circo_off = res["circo"].get("official")
    return {
        "paper_url": PAPER_URL,
        "headline": {"core_pct": core_pct, "all_pct": all_pct, "cirr_pct": cirr_pct, "cirr_pct_official": cirr_pct_off},
        "cirr": {"metrics": metrics, "avg_mine": round(avg_mine, 2) if avg_mine else None,
                 "avg_paper": avg_paper, "avg_official": round(avg_off, 2) if avg_off else None,
                 "avg_ratio": _ratio(avg_mine, avg_paper)},
        "fiq": [{"cat": c, "r10": fiq.get(c, (None, None))[0], "r50": fiq.get(c, (None, None))[1],
                 "o10": fiq_off.get(c, (None, None))[0], "o50": fiq_off.get(c, (None, None))[1],
                 "p10": PAPER_FIQ_VAL[c][0], "p50": PAPER_FIQ_VAL[c][1]} for c in ("dress", "shirt", "toptee")],
        "circo": {"ours": circo, "official": circo_off},
        "items": items,
        "curve": curve,
        "best_epoch": best,
        "paper_curve_refs": {"r1": PAPER_CIRR_VAL["cirr_recall_at1"], "avg": avg_paper},
        "oti": {"have": n_oti, "full": 100000},
        "notes": [
            "Thước đo đáng tin nhất là so với SEARLE official (torch.hub) chạy trên cùng dữ liệu, cùng code đánh giá: "
            "chênh lệch CIRR val chỉ ≤ 0.2 điểm ở mọi chỉ số.",
            "Cột 'Paper' là số val do demo gốc ghi lại, chưa được kiểm chứng lại với bảng trong bài báo; "
            "dùng cột 'Official' để so sánh công bằng.",
            "FashionIQ thiếu ~2.431 ảnh (link chết) nên chạy trên tập con: số tuyệt đối (shirt/toptee cao hơn paper) "
            "KHÔNG so sánh được với paper; chỉ so với 'Official' trên cùng tập con.",
            "OTI chạy 3 shard với seed 42/43/44 thay vì một lần seed 42 - lệch nhỏ so với quy trình gốc.",
            "Mã nguồn src/ khác bản gốc ở datasets.py (lọc ảnh thiếu, LOAD_TRUNCATED_IMAGES, nuốt ngoại lệ thay vì in) "
            "và data_utils.py (bỏ batch rỗng); các thay đổi này không ảnh hưởng CIRR/CIRCO.",
        ],
    }


# ----------------------------------------------------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(DEMO_DIR / "static", "index.html")


@app.get("/api/status")
def status():
    return jsonify({
        "state": engine.state, "error": engine.error, "device": str(engine.device),
        "clip": CLIP_NAME, "ckpt": engine.ckpt_name, "ckpts": engine.list_ckpts(),
        "gallery_size": len(engine.items), "job": engine.job,
    })


@app.post("/api/checkpoint")
def set_checkpoint():
    name = (request.get_json(silent=True) or {}).get("name", "")
    if name not in engine.list_ckpts():
        return jsonify({"error": "Checkpoint không tồn tại"}), 400
    with engine.lock:
        engine.load_phi(name)
    return jsonify({"ckpt": engine.ckpt_name})


@app.get("/api/assessment")
def assessment():
    return jsonify(build_assessment())


@app.get("/api/gallery")
def gallery_list():
    limit = min(int(request.args.get("limit", 120)), 500)
    return jsonify({"total": len(engine.items),
                    "items": [{"id": i["id"], "name": i["name"]} for i in engine.items[:limit]]})


@app.post("/api/gallery/upload")
def gallery_upload():
    if engine.state != "ready":
        return jsonify({"error": "Mô hình chưa sẵn sàng"}), 503
    if engine.job["running"]:
        return jsonify({"error": "Đang xử lý một tác vụ khác"}), 409
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in request.files.getlist("files"):
        ext = Path(f.filename or "").suffix.lower()
        if ext not in IMG_EXT:
            continue
        stem = re.sub(r"[^\w\-.]", "_", Path(f.filename).stem)[:60]
        p = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{stem}{ext}"
        f.save(p)
        paths.append(p)
    if not paths:
        return jsonify({"error": "Không có file ảnh hợp lệ"}), 400
    engine.job.update(running=True, total=len(paths), done=0, message="Bắt đầu…")
    threading.Thread(target=engine.add_paths, args=(paths,), daemon=True).start()
    return jsonify({"queued": len(paths)})


@app.post("/api/gallery/folder")
def gallery_folder():
    if engine.state != "ready":
        return jsonify({"error": "Mô hình chưa sẵn sàng"}), 503
    if engine.job["running"]:
        return jsonify({"error": "Đang xử lý một tác vụ khác"}), 409
    body = request.get_json(silent=True) or {}
    folder = Path(str(body.get("path", "")).strip().strip('"'))
    limit = max(1, min(int(body.get("limit", 2000)), 20000))
    if not folder.is_dir():
        return jsonify({"error": f"Không tìm thấy thư mục: {folder}"}), 400
    paths = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            paths.append(p)
            if len(paths) >= limit:
                break
    if not paths:
        return jsonify({"error": "Thư mục không có ảnh (jpg/png/webp/bmp/gif)"}), 400
    engine.job.update(running=True, total=len(paths), done=0, message="Bắt đầu…")
    threading.Thread(target=engine.add_paths, args=(paths,), daemon=True).start()
    return jsonify({"queued": len(paths)})


@app.post("/api/gallery/clear")
def gallery_clear():
    if engine.job["running"]:
        return jsonify({"error": "Đang xử lý một tác vụ khác"}), 409
    engine.clear_gallery()
    return jsonify({"ok": True})


@app.get("/api/img/<img_id>")
def serve_img(img_id: str):
    idx = engine.id2idx.get(img_id)
    if idx is None:
        abort(404)
    path = Path(engine.items[idx]["path"])
    w = request.args.get("w", type=int)
    if not w:
        return send_file(path)
    key = (img_id, w)
    if key not in engine.thumbs:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((w, w))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=85)
        if len(engine.thumbs) > 3000:
            engine.thumbs.clear()
        engine.thumbs[key] = buf.getvalue()
    return send_file(io.BytesIO(engine.thumbs[key]), mimetype="image/jpeg", max_age=3600)


@app.post("/api/search")
def search():
    if engine.state != "ready":
        return jsonify({"error": "Mô hình chưa sẵn sàng"}), 503
    if not engine.items:
        return jsonify({"error": "Thư viện ảnh đang trống - hãy thêm ảnh trước."}), 400
    mode = request.form.get("mode", "composed")
    if mode not in ("composed", "image", "text"):
        return jsonify({"error": "mode không hợp lệ"}), 400
    caption = request.form.get("caption", "").strip()
    if mode != "image" and not caption:
        return jsonify({"error": "Hãy nhập mô tả thay đổi (caption)."}), 400
    topk = max(1, min(int(request.form.get("topk", 12)), 60))

    ref_id = request.form.get("ref_id")
    exclude = None
    try:
        if "ref" in request.files and request.files["ref"].filename:
            ref = Image.open(request.files["ref"].stream)
            ref.load()
        elif ref_id and ref_id in engine.id2idx:
            ref = Image.open(engine.items[engine.id2idx[ref_id]]["path"])
            ref.load()
            exclude = ref_id
        else:
            return jsonify({"error": "Hãy chọn ảnh tham chiếu."}), 400
    except Exception:  # noqa: BLE001
        return jsonify({"error": "Không đọc được ảnh tham chiếu."}), 400
    return jsonify(engine.search(ref, caption, mode, topk, exclude))


# ----------------------------------------------------------------------------------------------------------------------
# C17: extended search (attributes, degraded query -> restoration, more methods), comparison, "Why?" map
# ----------------------------------------------------------------------------------------------------------------------
MODES_EXT = ("composed", "image", "text", "sum", "cbir")


def _reference_from_request():
    """(PIL image, id to exclude) from the form fields ref (file) or ref_id (gallery image)"""
    ref_id = request.form.get("ref_id")
    if "ref" in request.files and request.files["ref"].filename:
        ref = Image.open(request.files["ref"].stream)
        ref.load()
        return ref.convert("RGB"), None
    if ref_id and ref_id in engine.id2idx:
        ref = Image.open(engine.items[engine.id2idx[ref_id]]["path"])
        ref.load()
        return ref.convert("RGB"), ref_id
    raise ValueError("Hãy chọn ảnh tham chiếu.")


def _check_ready():
    if engine.state != "ready":
        return jsonify({"error": "Mô hình chưa sẵn sàng"}), 503
    if not engine.items:
        return jsonify({"error": "Thư viện ảnh đang trống - hãy thêm ảnh trước."}), 400
    return None


@app.post("/api/search2")
def search2():
    bad = _check_ready()
    if bad:
        return bad
    f = request.form
    mode = f.get("mode", "composed")
    if mode not in MODES_EXT:
        return jsonify({"error": "mode không hợp lệ"}), 400
    caption = f.get("caption", "").strip()
    if mode not in ("image", "cbir") and not caption:
        return jsonify({"error": "Hãy nhập mô tả thay đổi (caption)."}), 400
    topk = max(1, min(int(f.get("topk", 12)), 60))
    try:
        ref, exclude = _reference_from_request()
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e) or "Không đọc được ảnh tham chiếu."}), 400
    t0 = time.time()
    out = {"mode": mode, "gallery_size": len(engine.items)}
    rgb = np.asarray(ref)

    # 2. degraded query -> restoration
    degrade_spec, restore_mode = f.get("degrade", "none"), f.get("restore", "none")
    query_rgb = rgb
    if degrade_spec != "none" or restore_mode != "none":
        try:
            dr = extras.degrade_and_restore(rgb, degrade_spec, restore_mode)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        sims_d, _, _ = engine.query_sims(Image.fromarray(dr["degraded"]), caption, mode)
        out["restoration"] = {
            "degradation": degrade_spec, "mode": restore_mode, "method": dr["method"], "features": dr["features"],
            "original": extras.data_url(rgb), "degraded": extras.data_url(dr["degraded"]),
            "restored": extras.data_url(dr["restored"]),
            "psnr_degraded": round(extras.psnr(rgb, dr["degraded"]), 2),
            "psnr_restored": round(extras.psnr(rgb, dr["restored"]), 2),
            "results_degraded": engine.top_results(sims_d, topk, exclude),
        }
        query_rgb = dr["restored"]
        sims_c, _, _ = engine.query_sims(ref, caption, mode)
        out["restoration"]["results_clean"] = engine.top_results(sims_c, topk, exclude)

    sims, prompt, token = engine.query_sims(Image.fromarray(query_rgb), caption, mode)
    out["prompt"] = prompt
    out["neighbors"] = engine.pseudo_word_neighbors(token) if token is not None and mode != "text" else []

    # 1. attribute re-ranking
    if f.get("attr") == "1" and caption:
        weight = float(f.get("attr_weight", 0.3))
        rerank_k = max(topk, min(int(f.get("attr_k", 50)), 200))
        s = sims.clone()
        if exclude and exclude in engine.id2idx:
            s[engine.id2idx[exclude]] = -2
        out["attributes"] = extras.attribute_rerank(engine, s, query_rgb, caption, topk, weight, rerank_k)
        out["results"] = out["attributes"]["results"]
    else:
        out["results"] = engine.top_results(sims, topk, exclude)
    out["ms"] = int((time.time() - t0) * 1000)
    return jsonify(out)


@app.post("/api/compare")
def compare():
    bad = _check_ready()
    if bad:
        return bad
    caption = request.form.get("caption", "").strip()
    topk = max(1, min(int(request.form.get("topk", 6)), 24))
    try:
        ref, exclude = _reference_from_request()
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 400
    rows = []
    for mode in MODES_EXT:
        if mode in ("text", "sum", "composed") and not caption:
            continue
        t0 = time.time()
        sims, prompt, _ = engine.query_sims(ref, caption, mode)
        rows.append({"mode": mode, "prompt": prompt, "ms": int((time.time() - t0) * 1000),
                     "results": engine.top_results(sims, topk, exclude)})
    return jsonify({"rows": rows})


@app.post("/api/explain")
def explain_result():
    bad = _check_ready()
    if bad:
        return bad
    target = request.form.get("target_id", "")
    if target not in engine.id2idx:
        return jsonify({"error": "Ảnh kết quả không có trong thư viện"}), 400
    try:
        ref, _ = _reference_from_request()
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 400
    out = extras.occlusion_map(engine, ref, request.form.get("caption", ""), target)
    out["mask"], out["mask_kind"] = extras.mask_overlay(engine, target)
    return jsonify(out)


if __name__ == "__main__":
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=engine.load, daemon=True).start()
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)
