"""
Helpers shared by the visualisation tools (task C16): read a run folder written by validate.py --output-dir, find the
image file of a dataset image name, and rebuild the retrieval pipeline of a run from its metrics.json.
"""
import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import PIL.Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import retrieval_core as core  # noqa: E402

KEEP_FROM_COMMAND = {'dataset_path', 'output_dir', 'cache_dir', 'note', 'num_workers'}


def load_run(run_dir) -> Tuple[dict, List[dict]]:
    run_dir = Path(run_dir)
    metrics = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))
    with open(run_dir / 'dump.jsonl', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]
    return metrics, records


def captions_of(record: dict) -> List[str]:
    return record['captions'] if 'captions' in record else [record['caption']]


def correct_names(record: dict) -> set:
    return ({record.get('target')} | set(record.get('gt') or [])) - {None, ''}


def image_locator(dataset: str, dataset_path: str, split: str = 'val') -> Callable[[str], Path]:
    """
    Function giving the file of an image name (reference, target or gallery image) of the dataset
    """
    from datasets import CIRRDataset, CIRCODataset
    if dataset == 'cirr':
        return CIRRDataset(dataset_path, split, 'classic', None).image_path
    if dataset == 'circo':
        return CIRCODataset(dataset_path, split, 'classic', None).image_path
    return lambda name: Path(dataset_path) / 'images' / f"{name}.jpg"


def load_image(path) -> PIL.Image.Image:
    from datasets import open_image
    return open_image(path).convert('RGB')


def pipeline_from_run(run_dir, dataset_path: str, cache_dir: Optional[str] = None, num_workers: int = 4,
                      **overrides) -> core.RetrievalPipeline:
    """
    The RetrievalPipeline of a run (same method, checkpoint and options), with a dataset path for this machine
    """
    metrics, _ = load_run(run_dir)
    saved = metrics['args']
    parser = ArgumentParser()
    core.add_eval_args(parser, datasets=['cirr', 'fashioniq', 'circo'])
    args = parser.parse_args(['--eval-type', saved['eval_type'], '--dataset', saved['dataset'],
                              '--dataset-path', dataset_path])
    for key, value in saved.items():
        if hasattr(args, key) and key not in KEEP_FROM_COMMAND:
            setattr(args, key, value)
    args.cache_dir, args.num_workers, args.output_dir = cache_dir, num_workers, None
    for key, value in overrides.items():
        setattr(args, key, value)
    split = 'test' if metrics.get('script') == 'generate_test_submission' else 'val'
    return core.RetrievalPipeline(args, split)


def query_from_record(record: dict, reference: Optional[str] = None) -> 'core.Query':
    return core.Query(qid=record['qid'], reference=reference or record['reference'], captions=captions_of(record),
                      target=record.get('target'), gt=record.get('gt'), category=record.get('category'))


def score_images(pipeline: core.RetrievalPipeline, images, queries, gallery) -> 'torch.Tensor':
    """
    Similarity with the gallery of the run's method for preprocessed reference images (one image per query; the
    queries must have distinct reference names). Supports phi, searle, searle-xl, image and sum.
    """
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        clip_model = pipeline.clip_model
        features = clip_model.encode_image(images.to(core.device)).float()
        eval_type = pipeline.args.eval_type
        fiq = pipeline.dataset == 'fashioniq'
        if eval_type in ('phi', 'searle', 'searle-xl'):
            tokens = {q.reference: t for q, t in zip(queries, pipeline.phi(features))}
            predicted = core.encode_token_queries(clip_model, queries, tokens, fiq, pipeline.templates)
        elif eval_type == 'image':
            predicted = F.normalize(features)
        elif eval_type == 'sum':
            predicted = F.normalize(F.normalize(features) + core.encode_text_queries(clip_model, queries, fiq))
        else:
            raise ValueError(f"--eval-type {eval_type} cannot be recomputed from images")
        similarity = predicted @ gallery.T
        if pipeline.args.fusion_weight:
            similarity = similarity + pipeline.args.fusion_weight * (F.normalize(features) @ gallery.T)
        return similarity
