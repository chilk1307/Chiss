"""
Re-ranking of the top-K candidates with the attribute meters (plan 02, section 2.3).

For a query whose caption contains constraints (parser.py) in the selected families:
    1. take the top-K candidates of the CLIP-based method (the reference image excluded for CIRR, or with
       --exclude-reference);
    2. z-normalise their CLIP scores inside the top-K;
    3. new score = z + w * mean_constraints(c), with c in [0, 1] the satisfaction of each constraint (meters.py);
    4. the top-K is re-ordered by the new score and stays above the other images.
Queries without constraints keep their ranking, so --attr-rerank off and on give the same results for them.

Measurements are computed once per image (only the reference images and the candidates that appear in a top-K, not
the whole CIRCO gallery) and cached in --cache-dir.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import PIL.Image
import torch

from attributes.meters import METERS_VERSION, VECTOR_SIZE, measure, satisfaction_batch, unpack, explain
from attributes.parser import Constraint, parse_families, parse_query

EXPLAIN_TOP = 10  # candidates described in dump.jsonl


class AttributeMeasurer:
    """
    Callable usable as a dataset 'preprocess': PIL image -> tensor of measurements
    """

    def __init__(self, mask_mode: str):
        self.mask_mode = mask_mode

    def __call__(self, image: PIL.Image.Image) -> torch.Tensor:
        return torch.from_numpy(measure(np.asarray(image.convert('RGB')), self.mask_mode))


class AttributeStore:
    """
    Measurements of images by name, computed on demand (in parallel DataLoader workers) and cached in
    <cache_dir>/attributes-<dataset>-<mask>-<processing>-v<version>.pt
    """

    def __init__(self, cache_dir: Optional[str], dataset: str, mask_mode: str, num_workers: int,
                 processor=None):
        self.mask_mode = mask_mode
        self.num_workers = num_workers
        self.processor = processor
        self.data: Dict[str, np.ndarray] = {}
        self.n_failed = 0
        self.path = None
        if cache_dir:
            processing = 'clean' if processor is None else \
                'processed-' + hashlib.sha1(processor.key.encode()).hexdigest()[:8]
            self.path = Path(cache_dir) / f"attributes-{dataset}-{mask_mode}-{processing}-v{METERS_VERSION}.pt"
            if self.path.exists():
                saved = torch.load(self.path, map_location='cpu')
                self.data = dict(zip(saved['names'], saved['vectors'].numpy()))
                print(f"[cache] loaded {len(self.data)} attribute measurements from {self.path.name}")

    def get(self, names: List[str], path_of: Callable[[str], Path]) -> Dict[str, np.ndarray]:
        missing = [n for n in dict.fromkeys(names) if n not in self.data]
        if missing:
            from datasets import ImageListDataset
            from retrieval_core import encode_images
            dataset = ImageListDataset(missing, [path_of(n) for n in missing], AttributeMeasurer(self.mask_mode),
                                       transform=self.processor)
            vectors, measured, n_missing, _ = encode_images(dataset, lambda x: x, self.num_workers,
                                                            f"attributes ({self.mask_mode}) of {len(missing)} images")
            self.n_failed += n_missing
            for name, vector in zip(measured, vectors.numpy()):
                self.data[name] = vector
            self.save()
        return {n: self.data[n] for n in names if n in self.data}

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        names = list(self.data)
        vectors = np.stack([self.data[n] for n in names]) if names else np.zeros((0, VECTOR_SIZE), np.float32)
        tmp = self.path.with_suffix('.tmp')
        # tensors (not numpy arrays) so that recent torch versions can load the file with weights_only=True
        torch.save({'names': names, 'vectors': torch.from_numpy(np.ascontiguousarray(vectors, dtype=np.float32)),
                    'mask': self.mask_mode, 'version': METERS_VERSION}, tmp)
        os.replace(tmp, self.path)


class AttributeReranker:
    def __init__(self, args, dataset: str, split: str, reference_processor=None):
        self.families = parse_families(args.attr_families)
        self.negation = args.attr_negation == 'on'
        self.weight = args.attr_weight
        self.topk = args.attr_topk
        self.gallery_store = AttributeStore(args.cache_dir, dataset, args.attr_mask, args.num_workers)
        # the reference images may be degraded / restored (plan 03): measured on what the user gives
        self.reference_store = self.gallery_store if reference_processor is None else \
            AttributeStore(args.cache_dir, dataset, args.attr_mask, args.num_workers, reference_processor)
        self.n_queries = 0
        self.n_with_constraints = 0
        self.family_counts: Dict[str, int] = {}
        self.n_unmeasured = 0

    def constraints(self, captions: List[str]) -> List[Constraint]:
        return parse_query(captions, self.negation, self.families)

    def rerank(self, similarity: torch.Tensor, gallery_names: List[str], queries, gallery_path: Callable,
               reference_paths: Dict[str, Path], exclude_reference: bool, category: Optional[str] = None) \
            -> Tuple[torch.Tensor, Dict[str, dict]]:
        self.n_queries += len(queries)
        parsed = [self.constraints(q.captions) for q in queries]
        active = [i for i, cs in enumerate(parsed) if cs]
        self.n_with_constraints += len(active)
        for i in active:
            for family in {c.family for c in parsed[i]}:
                self.family_counts[family] = self.family_counts.get(family, 0) + 1
        if not active:
            return similarity, {}

        gallery_index = {n: j for j, n in enumerate(gallery_names)}
        k = min(self.topk + 1, similarity.shape[1])
        top_indices = torch.topk(similarity[active], k, dim=1).indices.cpu().numpy()
        candidates = {}
        for row, i in enumerate(active):
            reference = gallery_index.get(queries[i].reference, -1) if exclude_reference else -1
            candidates[i] = [j for j in top_indices[row] if j != reference][:self.topk]

        needed = sorted({gallery_names[j] for js in candidates.values() for j in js})
        gallery_measures = self.gallery_store.get(needed, gallery_path)
        needs_reference = [queries[i].reference for i in active if any(c.relative for c in parsed[i])]
        reference_measures = self.reference_store.get(needs_reference, reference_paths.__getitem__) \
            if needs_reference else {}

        new_similarity = similarity.clone()
        offset = float(similarity.max()) + 10.0
        info = {}
        for i in active:
            q, cs, js = queries[i], parsed[i], candidates[i]
            names = [gallery_names[j] for j in js]
            measured = np.array([n in gallery_measures for n in names])
            self.n_unmeasured += int((~measured).sum())
            vectors = np.stack([gallery_measures.get(n, np.zeros(VECTOR_SIZE, np.float32)) for n in names])
            ref_vector = reference_measures.get(q.reference)
            per_constraint = []
            for c in cs:
                values = satisfaction_batch(c, vectors, ref_vector)
                if values is not None:
                    per_constraint.append((c, np.where(measured, values, 0.5)))  # unmeasured image: neutral
            if not per_constraint:
                continue
            satisfaction = np.mean([v for _, v in per_constraint], axis=0)
            clip_scores = similarity[i, js].double().cpu().numpy()
            z = (clip_scores - clip_scores.mean()) / (clip_scores.std() + 1e-6)
            new_scores = z + self.weight * satisfaction
            new_similarity[i, js] = torch.as_tensor(offset + new_scores, dtype=similarity.dtype,
                                                    device=similarity.device)
            order = np.argsort(-new_scores)[:EXPLAIN_TOP]
            info[q.qid] = {
                'constraints': [c.describe() for c, _ in per_constraint],
                'reranked_top': [{
                    'name': names[o], 'clip_rank': int(o) + 1, 'clip_z': round(float(z[o]), 3),
                    'satisfaction': round(float(satisfaction[o]), 3),
                    'per_constraint': [round(float(v[o]), 3) for _, v in per_constraint],
                    'why': [explain(c, unpack(vectors[o]), None if ref_vector is None else unpack(ref_vector))
                            for c, _ in per_constraint] if measured[o] else [],
                } for o in order],
            }
        return new_similarity, info

    def summary(self) -> dict:
        return {'families': self.families, 'weight': self.weight, 'topk': self.topk,
                'negation': self.negation, 'n_queries': self.n_queries,
                'n_with_constraints': self.n_with_constraints, 'family_counts': self.family_counts,
                'unmeasured_candidates': self.n_unmeasured,
                'failed_images': self.gallery_store.n_failed + (self.reference_store.n_failed
                                                                if self.reference_store is not self.gallery_store
                                                                else 0)}
