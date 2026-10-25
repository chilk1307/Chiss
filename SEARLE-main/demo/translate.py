"""
Vietnamese -> English for the caption box of the demo. CLIP ViT-B/32 and the attribute parser only read English, so a
caption written in Vietnamese is translated before it reaches SEARLE; nothing is retrained.

Model: Helsinki-NLP/opus-mt-vi-en (MarianMT, CC-BY-4.0, ~300 MB, downloaded on first use; needs transformers and
sentencepiece). Another seq2seq model can be chosen with the environment variable SEARLE_TRANSLATE_MODEL, and
SEARLE_TRANSLATE=0 turns the translation off. If the model cannot be loaded the caption is used as typed and the
reason is returned, so the demo keeps working.
"""
import os
import re
import threading
import unicodedata
from functools import lru_cache

MODEL_NAME = os.environ.get("SEARLE_TRANSLATE_MODEL", "Helsinki-NLP/opus-mt-vi-en")
ENABLED = os.environ.get("SEARLE_TRANSLATE", "1") != "0"

_lock = threading.Lock()
_model = None
_tokenizer = None
_error = None

# letters that only Vietnamese (among the languages a user of this demo would type) writes: đ and the vowels with a
# breve, circumflex or horn, plus any vowel with one of the five tone marks
_VI_BASE = set("đăâêôơư")
_TONES = {"̀", "́", "̃", "̉", "̣"}  # grave, acute, tilde, hook above, dot below


def looks_vietnamese(text: str) -> bool:
    for ch in text.lower():
        if ch in _VI_BASE:
            return True
        decomposed = unicodedata.normalize("NFD", ch)
        if len(decomposed) > 1 and any(m in _TONES for m in decomposed[1:]):
            return True
    return False


def _load():
    global _model, _tokenizer, _error
    if _model is not None or _error is not None:
        return
    with _lock:
        if _model is not None or _error is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            model.eval()
            if torch.cuda.is_available():
                model = model.to("cuda")
            _model = model
        except Exception as e:  # noqa: BLE001  (missing package, no network, ...)
            _error = f"{type(e).__name__}: {e}"[:300]


@lru_cache(maxsize=2048)
def _translate(text: str) -> str:
    import torch
    batch = _tokenizer([text], return_tensors="pt", truncation=True, max_length=128).to(_model.device)
    with torch.no_grad():
        out = _model.generate(**batch, max_new_tokens=96, num_beams=4)
    en = _tokenizer.decode(out[0], skip_special_tokens=True).strip()
    return re.sub(r"\s+", " ", en)


def to_english(caption: str) -> dict:
    """
    {"text": caption for SEARLE, "source": caption as typed, "translated": bool, "note": str | None}
    """
    caption = (caption or "").strip()
    result = {"text": caption, "source": caption, "translated": False, "note": None}
    if not caption or not ENABLED or not looks_vietnamese(caption):
        return result
    _load()
    if _model is None:
        result["note"] = f"Không dịch được ({MODEL_NAME} chưa tải được: {_error}); dùng nguyên câu đã nhập."
        return result
    en = _translate(caption)
    # CIR captions describe the change ("is red", "has two dogs"); the translation of a Vietnamese sentence without a
    # subject often starts with "It is ..." / "There are ...", which reads badly after "a photo of $ that"
    en = re.sub(r"^(it|this|that)\s+(is|has|was)\b", r"\2", en, flags=re.IGNORECASE)
    en = re.sub(r"^there\s+(are|is)\s+", "has ", en, flags=re.IGNORECASE)  # "There are two dogs" -> "has two dogs"
    if len(en) > 1 and en[0].isupper() and en[1].islower():
        en = en[0].lower() + en[1:]
    en = en.rstrip(".")
    result.update(text=en, translated=True)
    return result
