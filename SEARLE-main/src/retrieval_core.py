"""
Shared retrieval pipeline for validate.py (val metrics) and generate_test_submission.py (test submission files),
so that every change to how queries are built or ranked reaches both (plan 01, section 1.3).

With the default options the results are the same as the original SEARLE code: same CLIP weights, same
preprocessing, same batches of 32, same query template and the same ranking code.

Methods (--eval-type):
    oti, phi, searle, searle-xl   pseudo-token methods: "a photo of $ that {caption}"
    image                         reference image feature only
    text                          caption only (no pseudo token)
    sum                           normalize(normalize(image) + normalize(text))
    cbir                          classical hand-crafted features of the reference image (see cbir.py),
                                  optionally plus --cbir-text-weight * CLIP text score
"""
import hashlib
import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import clip
import numpy as np
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_utils import PROJECT_ROOT, collate_fn, build_preprocess, preprocess_key, INTERPOLATIONS
from datasets import CIRRDataset, CIRCODataset, FashionIQDataset, ImageListDataset
from encode_with_pseudo_tokens import encode_with_pseudo_tokens
from phi import Phi
from utils import device

TOKEN_EVAL_TYPES = ['oti', 'phi', 'searle', 'searle-xl']
CLIP_EVAL_TYPES = ['image', 'text', 'sum']
EVAL_TYPES = TOKEN_EVAL_TYPES + CLIP_EVAL_TYPES + ['cbir']
TTA_CHOICES = ['none', 'flip-ref', 'flip-gallery', 'flip-both']
FIQ_DRESS_TYPES = ['shirt', 'dress', 'toptee']  # order of the per-category metrics, as in the original code
FIQ_REFERENCE_DRESS_TYPES = ['dress', 'toptee', 'shirt']  # order of the reference images, as in the original code
QUERY_TEMPLATE = "a photo of $ that {}"
BATCH_SIZE = 32
DUMP_TOP_K = 100
CACHE_VERSION = 1


# ----------------------------------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------------------------------

def add_eval_args(parser, datasets: List[str]) -> None:
    """
    Arguments shared by validate.py and generate_test_submission.py. Names follow the command contract of
    plans/01_nen_tang_danh_gia.md; every new option defaults to the original behaviour.
    """
    parser.add_argument("--exp-name", type=str, help="Experiment to evaluate")
    parser.add_argument("--eval-type", type=str, choices=EVAL_TYPES, required=True,
                        help="'oti': inverted OTI pseudo tokens; 'phi': pseudo tokens predicted by a trained phi; "
                             "'searle' / 'searle-xl': pre-trained SEARLE / SEARLE-XL; "
                             "'image', 'text', 'sum': CLIP baselines; 'cbir': classical hand-crafted features")
    parser.add_argument("--dataset", type=str, required=True, choices=datasets, help="Dataset to use")
    parser.add_argument("--dataset-path", type=str, help="Path to the dataset", required=True)
    parser.add_argument("--preprocess-type", default="targetpad", type=str, choices=['clip', 'targetpad'],
                        help="Preprocess pipeline to use")
    parser.add_argument("--phi-checkpoint-name", type=str,
                        help="Phi checkpoint to use, needed when using phi, e.g. 'phi_20.pt'")
    parser.add_argument("--clip-model-name", type=str, default='ViT-B/32',
                        help="CLIP backbone of the image/text/sum/cbir baselines (other methods read it from "
                             "their experiment)")

    parser.add_argument("--output-dir", type=str, default=None,
                        help="Write metrics.json, dump.jsonl, log.txt there and append a line to "
                             "<parent>/runs_<RUNNER>.csv")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache of image features")
    parser.add_argument("--note", type=str, default='', help="Free text stored with the run")

    parser.add_argument("--exclude-reference", action='store_true',
                        help="Remove the reference image from the ranking (CIRR always does it)")
    parser.add_argument("--targetpad-ratio", type=float, default=1.25, help="Target ratio of the targetpad pipeline")
    parser.add_argument("--interp", type=str, default='bicubic', choices=list(INTERPOLATIONS),
                        help="Interpolation used to resize the images")
    parser.add_argument("--tta", type=str, default='none', choices=TTA_CHOICES,
                        help="Average the features of each image and of its horizontal flip")
    parser.add_argument("--cbir-features", type=str, default='all',
                        help="'all' or a comma separated subset of color,lbp,hog,gabor (only for --eval-type cbir)")
    parser.add_argument("--cbir-text-weight", type=float, default=0.0,
                        help="Weight of the CLIP caption score added to the CBIR score")
    parser.add_argument("--num-workers", type=int, default=10, help="DataLoader workers")


def check_args(args) -> None:
    if args.eval_type in ['phi', 'oti'] and args.exp_name is None:
        raise ValueError("Experiment name is required when using phi or oti evaluation type")
    if args.eval_type == 'phi' and args.phi_checkpoint_name is None:
        raise ValueError("Phi checkpoint name is required when using phi evaluation type")
    if args.eval_type == 'cbir':
        from cbir import parse_cbir_features
        parse_cbir_features(args.cbir_features)
    elif args.cbir_text_weight != 0:
        raise ValueError("--cbir-text-weight is only used with --eval-type cbir")


def nondefault_args(parser, args) -> dict:
    ignore = {'dataset_path', 'output_dir', 'cache_dir', 'note', 'num_workers'}
    return {k: v for k, v in vars(args).items() if k not in ignore and v != parser.get_default(k)}


# ----------------------------------------------------------------------------------------------------------------------
# Queries
# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class Query:
    qid: str  # CIRR pair id, CIRCO query id, FashionIQ "<category>:<index>"
    reference: str
    captions: List[str]  # one caption (CIRR, CIRCO) or two (FashionIQ)
    target: Optional[str] = None
    gt: Optional[List[str]] = None  # CIRCO val: all ground truths, the target first
    group: Optional[List[str]] = None  # CIRR: members of the image set
    category: Optional[str] = None  # FashionIQ category


def cirr_queries(dataset: CIRRDataset) -> List[Query]:
    with_target = dataset.split in ['train', 'val']
    return [Query(qid=str(t['pairid']), reference=t['reference'], captions=[t['caption']],
                  target=t['target_hard'] if with_target else None, group=list(t['img_set']['members']))
            for t in dataset.triplets]


def circo_queries(dataset: CIRCODataset) -> List[Query]:
    with_target = dataset.split == 'val'
    return [Query(qid=str(a['id']), reference=str(a['reference_img_id']), captions=[a['relative_caption']],
                  target=str(a['target_img_id']) if with_target else None,
                  gt=[str(x) for x in a['gt_img_ids']] if with_target else None)
            for a in dataset.annotations]


def fiq_queries(dataset: FashionIQDataset, category: str) -> List[Query]:
    return [Query(qid=f"{category}:{i}", reference=t['candidate'], captions=list(t['captions']),
                  target=t.get('target'), category=category)
            for i, t in enumerate(dataset.triplets)]


# ----------------------------------------------------------------------------------------------------------------------
# Image features and cache
# ----------------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def encode_images(dataset: Dataset, encoder: Callable[[torch.Tensor], torch.Tensor], num_workers: int,
                  desc: str) -> Tuple[torch.Tensor, List[str], int]:
    """
    Run `encoder` on every image of `dataset` (batches of 32). Returns the features (on CPU), the image names and
    the number of images that could not be loaded (they are skipped, never silently).
    """
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=True,
                        collate_fn=collate_fn)
    features, names = [], []
    print(f"extracting features: {desc}")
    for batch in tqdm(loader, desc=desc):
        if batch is None:  # every image of the batch failed to load
            continue
        images = batch.get('image')
        batch_names = batch.get('image_name')
        if images is None:
            images = batch.get('reference_image')
        if batch_names is None:
            batch_names = batch.get('reference_name')
        features.append(encoder(images).cpu())
        names.extend(batch_names)
    n_missing = len(dataset) - len(names)
    if n_missing:
        print(f"WARNING: {n_missing}/{len(dataset)} images of {desc} could not be loaded and were skipped")
    if not features:
        raise RuntimeError(f"No image of {desc} could be loaded")
    return torch.vstack(features), names, n_missing


def clip_image_encoder(clip_model: CLIP, flip: bool) -> Callable[[torch.Tensor], torch.Tensor]:
    def encode(images: torch.Tensor) -> torch.Tensor:
        images = images.to(device)
        features = clip_model.encode_image(images)
        if flip:
            features = (features + clip_model.encode_image(torch.flip(images, dims=[-1]))) / 2
        return features

    return encode


class FeatureCache:
    """
    Stores image features in --cache-dir. The key holds everything the features depend on (dataset, split, image
    set, backbone, preprocessing, flip...), so a different configuration never reuses stale features.
    """

    def __init__(self, cache_dir: Optional[str]):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits, self.misses = [], []

    def path(self, key: dict) -> Path:
        digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
        readable = '-'.join(str(key[k]) for k in ['kind', 'dataset', 'split', 'subset', 'model', 'preprocess']
                            if key.get(k) is not None)
        readable = readable.replace('/', '').replace(' ', '')
        return self.cache_dir / f"{readable}{'-flip' if key.get('flip') else ''}-{digest}.pt"

    def get_or_compute(self, key: dict, compute: Callable[[], Tuple[torch.Tensor, List[str], int]]) \
            -> Tuple[torch.Tensor, List[str], int]:
        if self.cache_dir is None:
            return compute()
        path = self.path(key)
        if path.exists():
            data = torch.load(path, map_location='cpu')
            if data.get('key') == key:
                print(f"[cache] loaded {path.name}")
                self.hits.append(path.name)
                return data['features'], data['names'], data['n_missing']
        features, names, n_missing = compute()
        tmp = path.with_suffix('.tmp')
        torch.save({'key': key, 'features': features, 'names': names, 'n_missing': n_missing}, tmp)
        os.replace(tmp, path)
        print(f"[cache] saved {path.name}")
        self.misses.append(path.name)
        return features, names, n_missing


# ----------------------------------------------------------------------------------------------------------------------
# Query features
# ----------------------------------------------------------------------------------------------------------------------

def token_dict(names: List[str], tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Name -> pseudo token; with duplicated names the first one wins, like list.index in the original code
    """
    result = {}
    for name, token in zip(names, tokens):
        result.setdefault(name, token)
    return result


def _fiq_caption_pairs(query: Query) -> Tuple[str, str]:
    c1, c2 = query.captions[0].strip('.?, '), query.captions[1].strip('.?, ')
    return f"{c1} and {c2}", f"{c2} and {c1}"


@torch.no_grad()
def encode_token_queries(clip_model: CLIP, queries: List[Query], tokens: Dict[str, torch.Tensor],
                         fiq: bool = False) -> torch.Tensor:
    """
    Normalized features of "a photo of $ that {caption}" with $ replaced by the pseudo token of the reference image.
    FashionIQ joins its two captions in both orders and averages the two normalized features.
    """
    predicted = []
    for i in tqdm(range(0, len(queries), BATCH_SIZE), desc='encoding queries'):
        chunk = queries[i:i + BATCH_SIZE]
        batch_tokens = torch.vstack([tokens[q.reference].unsqueeze(0) for q in chunk]).to(device)
        if fiq:
            pairs = [_fiq_caption_pairs(q) for q in chunk]
            texts = [QUERY_TEMPLATE.format(a) for a, _ in pairs]
            texts_reversed = [QUERY_TEMPLATE.format(b) for _, b in pairs]
            features = encode_with_pseudo_tokens(clip_model, clip.tokenize(texts, context_length=77).to(device),
                                                 batch_tokens)
            features_reversed = encode_with_pseudo_tokens(
                clip_model, clip.tokenize(texts_reversed, context_length=77).to(device), batch_tokens)
            predicted.append(F.normalize((F.normalize(features) + F.normalize(features_reversed)) / 2))
        else:
            texts = [QUERY_TEMPLATE.format(q.captions[0]) for q in chunk]
            features = encode_with_pseudo_tokens(clip_model, clip.tokenize(texts, context_length=77).to(device),
                                                 batch_tokens)
            predicted.append(F.normalize(features))
    return torch.vstack(predicted)


@torch.no_grad()
def encode_text_queries(clip_model: CLIP, queries: List[Query], fiq: bool = False) -> torch.Tensor:
    """
    Normalized CLIP features of the captions alone (no pseudo token)
    """
    predicted = []
    for i in range(0, len(queries), BATCH_SIZE):
        chunk = queries[i:i + BATCH_SIZE]
        if fiq:
            pairs = [_fiq_caption_pairs(q) for q in chunk]
            a = clip_model.encode_text(clip.tokenize([p[0] for p in pairs], truncate=True).to(device))
            b = clip_model.encode_text(clip.tokenize([p[1] for p in pairs], truncate=True).to(device))
            predicted.append(F.normalize((F.normalize(a) + F.normalize(b)) / 2))
        else:
            texts = [q.captions[0] for q in chunk]
            predicted.append(F.normalize(clip_model.encode_text(clip.tokenize(texts, truncate=True).to(device))))
    return torch.vstack(predicted)


# ----------------------------------------------------------------------------------------------------------------------
# Ranking and metrics
# ----------------------------------------------------------------------------------------------------------------------

def _indices(names: List[str], gallery_index: Dict[str, int]) -> np.ndarray:
    return np.array([gallery_index.get(n, -1) for n in names], dtype=np.int64)


def _top(names: List[str], indices: np.ndarray, scores: np.ndarray, k: int = DUMP_TOP_K) -> List[list]:
    return [[names[i], round(float(s), 5)] for i, s in zip(indices[:k], scores[:k])]


def rank_cirr(similarity: torch.Tensor, gallery_names: List[str], queries: List[Query]) -> dict:
    """
    Full ranking for CIRR (val or test) with the reference image removed, plus the ranking inside each image set.
    Same operations as the original code (argsort of 1 - similarity), on integer indices instead of strings.
    """
    gallery_index = {n: i for i, n in enumerate(gallery_names)}
    distances = 1 - similarity
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_scores = torch.gather(similarity.cpu(), 1, sorted_indices).numpy()
    sorted_indices = sorted_indices.numpy()
    n_query, n_gallery = sorted_indices.shape

    # Delete the reference image from the results (if a reference is missing from the gallery, drop the last result
    # instead so that every row keeps n_gallery - 1 results)
    reference_idx = _indices([q.reference for q in queries], gallery_index)
    keep = sorted_indices != reference_idx[:, None]
    keep[keep.all(axis=1), -1] = False
    ranked = sorted_indices[keep].reshape(n_query, n_gallery - 1)
    ranked_scores = sorted_scores[keep].reshape(n_query, n_gallery - 1)

    # Ranking restricted to the image set of each query
    group_idx = np.array([_indices(q.group, gallery_index) for q in queries])
    group_mask = (ranked[..., None] == group_idx[:, None, :]).any(-1)
    group_ranked = [ranked[i][group_mask[i]] for i in range(n_query)]
    return {'ranked': ranked, 'scores': ranked_scores, 'group_ranked': group_ranked}


def cirr_metrics(ranking: dict, gallery_names: List[str], queries: List[Query]) -> Tuple[dict, List[dict]]:
    gallery_index = {n: i for i, n in enumerate(gallery_names)}
    target_idx = _indices([q.target for q in queries], gallery_index)
    labels = ranking['ranked'] == target_idx[:, None]
    group_labels = [g == t for g, t in zip(ranking['group_ranked'], target_idx)]
    assert labels.sum(1).tolist() == [1] * len(queries), "every CIRR target must appear exactly once in the ranking"
    assert all(g.sum() == 1 for g in group_labels), "every CIRR target must belong to its image set"

    per_query = {f'cirr_recall_at{k}': labels[:, :k].any(1).astype(np.float64) for k in [1, 5, 10, 50]}
    per_query.update({f'cirr_group_recall_at{k}': np.array([g[:k].any() for g in group_labels], dtype=np.float64)
                      for k in [1, 2, 3]})
    metrics = {k: float(v.mean() * 100) for k, v in per_query.items()}

    records = []
    for i, q in enumerate(queries):
        records.append({
            'qid': q.qid, 'reference': q.reference, 'caption': q.captions[0], 'target': q.target,
            'group': q.group, 'target_rank': int(labels[i].argmax()) + 1,
            'group_target_rank': int(group_labels[i].argmax()) + 1,
            'top': _top(gallery_names, ranking['ranked'][i], ranking['scores'][i]),
            'group_top': [gallery_names[j] for j in ranking['group_ranked'][i]],
            'per_query': {k: float(v[i]) for k, v in per_query.items()},
        })
    return metrics, records


def fiq_metrics(similarity: torch.Tensor, gallery_names: List[str], queries: List[Query],
                exclude_reference: bool) -> Tuple[dict, List[dict]]:
    gallery_index = {n: i for i, n in enumerate(gallery_names)}
    distances = 1 - similarity
    if exclude_reference:
        reference_idx = torch.as_tensor(_indices([q.reference for q in queries], gallery_index))
        rows = torch.nonzero(reference_idx >= 0).flatten()
        distances[rows.to(distances.device), reference_idx[rows].to(distances.device)] = float('inf')
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_scores = torch.gather(similarity.cpu(), 1, sorted_indices).numpy()
    sorted_indices = sorted_indices.numpy()

    target_idx = _indices([q.target for q in queries], gallery_index)
    labels = sorted_indices == target_idx[:, None]
    assert labels.sum(1).tolist() == [1] * len(queries), "every FashionIQ target must appear exactly once"
    if exclude_reference:  # a target equal to its reference is ranked last: it cannot be retrieved
        for i, q in enumerate(queries):
            if q.target == q.reference:
                labels[i] = False

    per_query = {f'fiq_recall_at{k}': labels[:, :k].any(1).astype(np.float64) for k in [10, 50]}
    metrics = {k: float(v.mean() * 100) for k, v in per_query.items()}

    records = []
    for i, q in enumerate(queries):
        order = sorted_indices[i]
        if exclude_reference:
            order_mask = order != gallery_index.get(q.reference, -1)
            order, scores = order[order_mask], sorted_scores[i][order_mask]
        else:
            scores = sorted_scores[i]
        rank = int(labels[i].argmax()) + 1 if labels[i].any() else None
        records.append({
            'qid': q.qid, 'category': q.category, 'reference': q.reference, 'captions': q.captions,
            'target': q.target, 'target_rank': rank, 'top': _top(gallery_names, order, scores),
            'per_query': {k: float(v[i]) for k, v in per_query.items()},
        })
    return metrics, records


def rank_topk(similarity: torch.Tensor, gallery_names: List[str], queries: List[Query], exclude_reference: bool,
              k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Top-k gallery indices and scores per query (CIRCO)
    """
    if exclude_reference:
        gallery_index = {n: i for i, n in enumerate(gallery_names)}
        reference_idx = torch.as_tensor(_indices([q.reference for q in queries], gallery_index))
        rows = torch.nonzero(reference_idx >= 0).flatten()
        similarity[rows.to(similarity.device), reference_idx[rows].to(similarity.device)] = float('-inf')
    top = torch.topk(similarity, dim=-1, k=min(k, similarity.shape[1]))
    return top.indices.cpu().numpy(), top.values.cpu().numpy()


def circo_metrics(top_indices: np.ndarray, top_scores: np.ndarray, gallery_names: List[str],
                  queries: List[Query]) -> Tuple[dict, List[dict]]:
    """
    mAP@K and recall@K exactly as in the original code (computed on the top 50)
    """
    per_query = {f'circo_map_at{k}': [] for k in [5, 10, 25, 50]}
    per_query.update({f'circo_recall_at{k}': [] for k in [5, 10, 25, 50]})
    records = []
    for i, q in enumerate(queries):
        names = np.array([gallery_names[j] for j in top_indices[i][:50]])
        gt_img_ids = np.array(q.gt)
        map_labels = torch.tensor(np.isin(names, gt_img_ids), dtype=torch.uint8)
        precisions = torch.cumsum(map_labels, dim=0) * map_labels  # Consider only positions corresponding to GTs
        precisions = precisions / torch.arange(1, map_labels.shape[0] + 1)  # Compute precision for each position
        for k in [5, 10, 25, 50]:
            per_query[f'circo_map_at{k}'].append(float(torch.sum(precisions[:k]) / min(len(gt_img_ids), k)))

        assert q.target == gt_img_ids[0], f"Target name not in GTs {q.target} {gt_img_ids}"
        single_gt_labels = torch.tensor(names == q.target)
        for k in [5, 10, 25, 50]:
            per_query[f'circo_recall_at{k}'].append(float(torch.sum(single_gt_labels[:k])))

        hits = np.nonzero(names == q.target)[0]
        records.append({
            'qid': q.qid, 'reference': q.reference, 'caption': q.captions[0], 'target': q.target, 'gt': q.gt,
            'target_rank': int(hits[0]) + 1 if len(hits) else None,
            'top': _top(gallery_names, top_indices[i], top_scores[i]),
            'per_query': {k: v[-1] for k, v in per_query.items()},
        })
    metrics = {k: float(np.mean(v) * 100) for k, v in per_query.items()}
    return metrics, records


# ----------------------------------------------------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------------------------------------------------

class RetrievalPipeline:
    """
    Loads the model given by the command line, extracts (or loads from cache) the features and scores the queries of
    one dataset split. Used by validate.py (split 'val') and generate_test_submission.py (split 'test').
    """

    def __init__(self, args, split: str):
        check_args(args)
        self.args = args
        self.split = split
        self.dataset = args.dataset.lower()
        self.cache = FeatureCache(args.cache_dir)
        self.load_errors: Dict[str, int] = {}
        self.dropped_queries = 0
        self._memo = {}  # reference features/tokens are shared by the three FashionIQ categories
        self.flip_reference = args.tta in ['flip-ref', 'flip-both']
        self.flip_gallery = args.tta in ['flip-gallery', 'flip-both']

        self.phi = None
        self.oti_tokens = None
        self.clip_model = None
        self.cbir_features = None
        if args.eval_type == 'cbir':
            from cbir import parse_cbir_features
            self.cbir_features = parse_cbir_features(args.cbir_features)
        if args.eval_type != 'cbir' or args.cbir_text_weight != 0:
            self._load_models()

    # Models -----------------------------------------------------------------------------------------------------------

    def _load_models(self) -> None:
        args = self.args
        if args.eval_type == 'oti':
            experiment_path = PROJECT_ROOT / 'data' / "oti_pseudo_tokens" / self.dataset / self.split / args.exp_name
            if not experiment_path.exists():
                raise ValueError(f"Experiment {args.exp_name} not found")
            with open(experiment_path / 'hyperparameters.json') as f:
                hyperparameters = json.load(f)
            clip_model_name = hyperparameters['clip_model_name']
            pseudo_tokens = torch.load(experiment_path / 'ema_oti_pseudo_tokens.pt', map_location=device)
            with open(experiment_path / 'image_names.pkl', 'rb') as f:
                self.oti_tokens = token_dict(pickle.load(f), pseudo_tokens)
        elif args.eval_type == 'phi':
            phi_path = PROJECT_ROOT / 'data' / "phi_models" / args.exp_name
            if not phi_path.exists():
                raise ValueError(f"Experiment {args.exp_name} not found")
            hyperparameters = json.load(open(phi_path / "hyperparameters.json"))
            clip_model_name = hyperparameters['clip_model_name']
        elif args.eval_type in ['searle', 'searle-xl']:
            clip_model_name = 'ViT-B/32' if args.eval_type == 'searle' else 'ViT-L/14'
            phi, _ = torch.hub.load(repo_or_dir='miccunifi/SEARLE', model='searle', source='github',
                                    backbone=clip_model_name)
            self.phi = phi.to(device).eval()
        else:
            clip_model_name = args.clip_model_name

        clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)
        self.clip_model = clip_model.float().eval().requires_grad_(False)
        self.clip_model_name = clip_model_name

        if args.eval_type == 'phi':
            phi = Phi(input_dim=clip_model.visual.output_dim, hidden_dim=clip_model.visual.output_dim * 4,
                      output_dim=clip_model.token_embedding.embedding_dim, dropout=hyperparameters['phi_dropout'])
            phi = phi.to(device)
            phi.load_state_dict(
                torch.load(phi_path / 'checkpoints' / args.phi_checkpoint_name, map_location=device)[
                    phi.__class__.__name__])
            self.phi = phi.eval()

        self.preprocess = build_preprocess(args.preprocess_type, clip_model.visual.input_resolution, clip_preprocess,
                                           args.targetpad_ratio, args.interp)
        self.preprocess_key = preprocess_key(args.preprocess_type, args.targetpad_ratio, args.interp)

    # Datasets -------------------------------------------------------------------------------------------------------

    def _relative_dataset(self, category: Optional[str] = None, no_duplicates: bool = False, preprocess=None):
        path = self.args.dataset_path
        if self.dataset == 'cirr':
            return CIRRDataset(path, self.split, 'relative', preprocess, no_duplicates=no_duplicates)
        if self.dataset == 'circo':
            return CIRCODataset(path, self.split, 'relative', preprocess)
        categories = [category] if category else FIQ_REFERENCE_DRESS_TYPES
        return FashionIQDataset(path, self.split, categories, 'relative', preprocess, no_duplicates=no_duplicates)

    def _gallery_dataset(self, preprocess, category: Optional[str] = None):
        path = self.args.dataset_path
        if self.dataset == 'cirr':
            return CIRRDataset(path, self.split, 'classic', preprocess)
        if self.dataset == 'circo':
            return CIRCODataset(path, self.split, 'classic', preprocess)
        return FashionIQDataset(path, self.split, [category], 'classic', preprocess)

    def _gallery_split(self) -> str:
        # CIRCO val and test share the same gallery (COCO 2017 unlabeled), so they share the cache
        return 'coco-unlabeled2017' if self.dataset == 'circo' else self.split

    def queries(self, category: Optional[str] = None) -> List[Query]:
        dataset = self._relative_dataset(category)
        if self.dataset == 'cirr':
            return cirr_queries(dataset)
        if self.dataset == 'circo':
            return circo_queries(dataset)
        return fiq_queries(dataset, category)

    def _reference_images(self, preprocess) -> ImageListDataset:
        """
        The unique reference images of the split, in the order used by the original code
        """
        dataset = self._relative_dataset(no_duplicates=True)
        if self.dataset == 'cirr':
            names = [t['reference'] for t in dataset.triplets]
        elif self.dataset == 'circo':
            names = list(dict.fromkeys(str(a['reference_img_id']) for a in dataset.annotations))
        else:
            names = [t['candidate'] for t in dataset.triplets]
        return ImageListDataset(names, [dataset.image_path(n) for n in names], preprocess, split=self.split)

    # Features -------------------------------------------------------------------------------------------------------

    def _cache_key(self, kind: str, subset: str, n_items: int, flip: bool = False) -> dict:
        key = {'version': CACHE_VERSION, 'kind': kind, 'dataset': self.dataset,
               'split': self._gallery_split() if subset.startswith('gallery') else self.split,
               'subset': subset, 'n_items': n_items, 'flip': flip}
        if kind == 'clip':
            key.update({'model': self.clip_model_name, 'preprocess': self.preprocess_key})
        else:
            from cbir import CBIR_VERSION, IMAGE_SIZE
            key.update({'model': f'cbir-v{CBIR_VERSION}', 'preprocess': f'resize{IMAGE_SIZE}'})
        return key

    def _features(self, kind: str, dataset: Dataset, subset: str, flip: bool = False) \
            -> Tuple[torch.Tensor, List[str]]:
        desc = f"{self.dataset} {self.split} {subset}" + (' (+flip)' if flip else '')
        if kind == 'clip':
            encoder = clip_image_encoder(self.clip_model, flip)
        else:
            encoder = lambda x: x  # cbir features are computed by the dataset 'preprocess'
        key = self._cache_key(kind, subset, len(dataset), flip)
        features, names, n_missing = self.cache.get_or_compute(
            key, lambda: encode_images(dataset, encoder, self.args.num_workers, desc))
        if n_missing:
            self.load_errors[f"{kind}:{subset}"] = n_missing
        return features, names

    def gallery_features(self, category: Optional[str] = None) -> Tuple[torch.Tensor, List[str]]:
        dataset = self._gallery_dataset(self.preprocess, category)
        subset = f"gallery-{category}" if category else 'gallery'
        features, names = self._features('clip', dataset, subset, self.flip_gallery)
        return F.normalize(features.to(device), dim=-1).float(), names

    def _memoized(self, name: str, compute: Callable):
        if name not in self._memo:
            self._memo[name] = compute()
        return self._memo[name]

    def reference_features(self) -> Tuple[torch.Tensor, List[str]]:
        return self._memoized('clip_references', lambda: self._features(
            'clip', self._reference_images(self.preprocess), 'references', self.flip_reference))

    def cbir_gallery_features(self, category: Optional[str] = None) -> Tuple[torch.Tensor, List[str]]:
        from cbir import CBIRFeatureExtractor
        dataset = self._gallery_dataset(CBIRFeatureExtractor(), category)
        features, names = self._features('cbir', dataset, f"gallery-{category}" if category else 'gallery')
        return features.to(device), names

    def cbir_reference_features(self) -> Tuple[torch.Tensor, List[str]]:
        from cbir import CBIRFeatureExtractor
        return self._memoized('cbir_references', lambda: self._features(
            'cbir', self._reference_images(CBIRFeatureExtractor()), 'references'))

    @torch.no_grad()
    def reference_tokens(self) -> Dict[str, torch.Tensor]:
        if self.args.eval_type == 'oti':
            return self.oti_tokens

        def compute():
            features, names = self.reference_features()
            print("Extracting tokens using phi model")
            tokens = [self.phi(features[i:i + BATCH_SIZE].to(device)).cpu()
                      for i in range(0, len(names), BATCH_SIZE)]
            return token_dict(names, torch.vstack(tokens))

        return self._memoized('tokens', compute)

    # Scoring --------------------------------------------------------------------------------------------------------

    def _keep_answerable(self, queries: List[Query], available_references, gallery_names: List[str]) -> List[Query]:
        """
        Drop (with a warning) the queries whose reference has no feature/token or whose target is not in the gallery,
        e.g. because the image could not be loaded. The original code dropped them without notice.
        """
        gallery = set(gallery_names)
        kept = [q for q in queries if (available_references is None or q.reference in available_references)
                and (q.target is None or q.target in gallery)]
        dropped = len(queries) - len(kept)
        if dropped:
            print(f"WARNING: {dropped}/{len(queries)} queries dropped (missing reference token/feature or target "
                  f"image not in the gallery)")
            self.dropped_queries += dropped
        return kept

    def score(self, category: Optional[str] = None) -> Tuple[torch.Tensor, List[str], List[Query]]:
        """
        Similarity between every query of the split (or FashionIQ category) and every gallery image
        """
        eval_type = self.args.eval_type
        fiq = self.dataset == 'fashioniq'
        queries = self.queries(category)

        if eval_type == 'cbir':
            gallery, gallery_names = self.cbir_gallery_features(category)
            ref_features, ref_names = self.cbir_reference_features()
            queries = self._keep_answerable(queries, set(ref_names), gallery_names)
            ref_index = {n: i for i, n in enumerate(ref_names)}
            query_features = ref_features[[ref_index[q.reference] for q in queries]].to(device)
            from cbir import cbir_similarity
            similarity = cbir_similarity(query_features, gallery, self.cbir_features)
            if self.args.cbir_text_weight != 0:
                clip_gallery, clip_names = self.gallery_features(category)
                clip_index = {n: i for i, n in enumerate(clip_names)}
                columns = torch.as_tensor([clip_index.get(n, -1) for n in gallery_names], device=device)
                text_scores = encode_text_queries(self.clip_model, queries, fiq) @ clip_gallery.T
                # reorder the CLIP scores to the CBIR gallery order (0 for an image missing from the CLIP gallery)
                text_scores = torch.where(columns[None] >= 0, text_scores[:, columns.clamp_min(0)],
                                          torch.zeros_like(text_scores[:, :1]))
                similarity = similarity + self.args.cbir_text_weight * text_scores
            return similarity, gallery_names, queries

        gallery, gallery_names = self.gallery_features(category)
        if eval_type in TOKEN_EVAL_TYPES:
            tokens = self.reference_tokens()
            queries = self._keep_answerable(queries, tokens, gallery_names)
            predicted = encode_token_queries(self.clip_model, queries, tokens, fiq)
        else:
            if eval_type == 'text':
                queries = self._keep_answerable(queries, None, gallery_names)
            else:
                ref_features, ref_names = self.reference_features()
                queries = self._keep_answerable(queries, set(ref_names), gallery_names)
                ref_index = {n: i for i, n in enumerate(ref_names)}
                image_features = F.normalize(ref_features[[ref_index[q.reference] for q in queries]].to(device))
            if eval_type == 'image':
                predicted = image_features
            elif eval_type == 'text':
                predicted = encode_text_queries(self.clip_model, queries, fiq)
            else:  # sum
                predicted = F.normalize(image_features + encode_text_queries(self.clip_model, queries, fiq))
        return predicted @ gallery.T, gallery_names, queries

    # Validation -----------------------------------------------------------------------------------------------------

    def evaluate(self) -> Tuple[dict, List[dict]]:
        """
        Validation metrics (same keys as the original code) and one dump record per query
        """
        if self.dataset == 'cirr':
            similarity, gallery_names, queries = self.score()
            ranking = rank_cirr(similarity, gallery_names, queries)
            return cirr_metrics(ranking, gallery_names, queries)

        if self.dataset == 'circo':
            similarity, gallery_names, queries = self.score()
            top_indices, top_scores = rank_topk(similarity, gallery_names, queries, self.args.exclude_reference,
                                                max(50, DUMP_TOP_K))
            return circo_metrics(top_indices, top_scores, gallery_names, queries)

        metrics, records = {}, []
        for category in FIQ_DRESS_TYPES:
            similarity, gallery_names, queries = self.score(category)
            category_metrics, category_records = fiq_metrics(similarity, gallery_names, queries,
                                                             self.args.exclude_reference)
            metrics.update({f"{category}_{k}": v for k, v in category_metrics.items()})
            records.extend(category_records)
        for k in ['fiq_recall_at10', 'fiq_recall_at50']:
            metrics[f"average_{k}"] = float(np.mean([metrics[f"{c}_{k}"] for c in FIQ_DRESS_TYPES]))
        return metrics, records

    # Test submissions -----------------------------------------------------------------------------------------------

    def cirr_submission(self) -> Tuple[dict, dict, List[dict]]:
        similarity, gallery_names, queries = self.score()
        ranking = rank_cirr(similarity, gallery_names, queries)
        submission = {'version': 'rc2', 'metric': 'recall'}
        group_submission = {'version': 'rc2', 'metric': 'recall_subset'}
        records = []
        for i, q in enumerate(queries):
            pair_id = str(int(q.qid))
            submission[pair_id] = [gallery_names[j] for j in ranking['ranked'][i][:50]]
            group_submission[pair_id] = [gallery_names[j] for j in ranking['group_ranked'][i][:3]]
            records.append({'qid': q.qid, 'reference': q.reference, 'caption': q.captions[0], 'group': q.group,
                            'top': _top(gallery_names, ranking['ranked'][i], ranking['scores'][i]),
                            'group_top': [gallery_names[j] for j in ranking['group_ranked'][i]]})
        return submission, group_submission, records

    def circo_submission(self) -> Tuple[dict, List[dict]]:
        similarity, gallery_names, queries = self.score()
        top_indices, top_scores = rank_topk(similarity, gallery_names, queries, self.args.exclude_reference,
                                            max(50, DUMP_TOP_K))
        submission, records = {}, []
        for i, q in enumerate(queries):
            submission[q.qid] = [gallery_names[j] for j in top_indices[i][:50]]
            records.append({'qid': q.qid, 'reference': q.reference, 'caption': q.captions[0],
                            'top': _top(gallery_names, top_indices[i], top_scores[i])})
        return submission, records

    def info(self) -> dict:
        return {
            'clip_model': getattr(self, 'clip_model_name', None),
            'preprocess': getattr(self, 'preprocess_key', None),
            'tta': self.args.tta,
            'load_errors': self.load_errors,
            'dropped_queries': self.dropped_queries,
            'cache_hits': self.cache.hits,
            'cache_writes': self.cache.misses,
        }
