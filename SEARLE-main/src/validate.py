from argparse import ArgumentParser
from typing import List, Dict

import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import Dataset

import retrieval_core as core
from run_recorder import RunRecorder
from utils import device

torch.multiprocessing.set_sharing_strategy('file_system')


@torch.no_grad()
def cirr_compute_val_metrics(relative_val_dataset: Dataset, clip_model: CLIP, index_features: torch.Tensor,
                             index_names: List[str], ref_names_list: List[str], pseudo_tokens: torch.Tensor) \
        -> Dict[str, float]:
    """
    Compute the retrieval metrics on the CIRR validation set given the dataset, pseudo tokens and the reference names.
    Kept for train_phi.py; the computation is the one of retrieval_core.
    """
    # The original implementation iterated a DataLoader here, which draws one number from the global random generator
    # (its base seed). Draw it too, so that a training run keeps exactly the random stream of the original code.
    torch.empty((), dtype=torch.int64).random_()
    queries = core.cirr_queries(relative_val_dataset)
    predicted_features = core.encode_token_queries(clip_model, queries,
                                                   core.token_dict(ref_names_list, pseudo_tokens))
    index_features = F.normalize(index_features.to(device), dim=-1).float()
    similarity = predicted_features @ index_features.T
    ranking = core.rank_cirr(similarity, list(index_names), queries)
    metrics, _ = core.cirr_metrics(ranking, list(index_names), queries)
    return metrics


def print_metrics(dataset: str, metrics: Dict[str, float]) -> None:
    if dataset == 'fashioniq':
        for dress_type in core.FIQ_DRESS_TYPES:
            for k in ['fiq_recall_at10', 'fiq_recall_at50']:
                print(f"{dress_type}_{k} = {metrics[f'{dress_type}_{k}']:.2f}")
            print("\n")
        print(f"average_fiq_recall_at10 = {metrics['average_fiq_recall_at10']:.2f}")
        print(f"average_fiq_recall_at50 = {metrics['average_fiq_recall_at50']:.2f}")
    else:
        for k, v in metrics.items():
            print(f"{k} = {v:.2f}")


def main():
    parser = ArgumentParser()
    core.add_eval_args(parser, datasets=['cirr', 'fashioniq', 'circo'])
    args = parser.parse_args()

    recorder = RunRecorder(args.output_dir, args, 'validate', core.nondefault_args(parser, args))
    pipeline = core.RetrievalPipeline(args, split='val')

    print(f"Eval type = {args.eval_type} \t exp name = {args.exp_name} \t")
    metrics, records = pipeline.evaluate()
    print_metrics(args.dataset.lower(), metrics)

    info = pipeline.info()
    info['n_queries'] = len(records)
    if info['load_errors'] or info['dropped_queries']:
        print(f"WARNING: images not loaded: {info['load_errors']}; queries dropped: {info['dropped_queries']}")
    recorder.write_dump(records)
    recorder.finish(metrics, info)


if __name__ == '__main__':
    main()
