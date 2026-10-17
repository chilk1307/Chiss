"""
Merge OTI token shards into one experiment (plan 08, section 8.0; generalises outputs/merge_oti.py).

    python tools/merge_oti.py --shards oti_B32_s0,oti_B32_s1 --out oti_B32_s01

Reads data/oti_pseudo_tokens/<dataset>/<split>/<shard>/{image_names.pkl, oti_pseudo_tokens.pt,
ema_oti_pseudo_tokens.pt} for every shard and writes the concatenation (in the given order) to <out>. Image names
must be unique across the shards.
"""
import json
import pickle
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data_utils import PROJECT_ROOT  # noqa: E402

FILES = ['oti_pseudo_tokens.pt', 'ema_oti_pseudo_tokens.pt']


def main():
    parser = ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--shards', required=True, help='Comma separated OTI experiment names')
    parser.add_argument('--out', required=True, help='Name of the merged experiment')
    parser.add_argument('--dataset', default='imagenet')
    parser.add_argument('--split', default='test')
    args = parser.parse_args()

    root = PROJECT_ROOT / 'data' / 'oti_pseudo_tokens' / args.dataset / args.split
    shards = [s.strip() for s in args.shards.split(',') if s.strip()]
    names, tokens = [], {f: [] for f in FILES}
    for shard in shards:
        directory = root / shard
        with open(directory / 'image_names.pkl', 'rb') as f:
            shard_names = pickle.load(f)
        names += shard_names
        for file in FILES:
            t = torch.load(directory / file, map_location='cpu')
            if len(t) != len(shard_names):
                raise SystemExit(f"{directory / file}: {len(t)} tokens for {len(shard_names)} names")
            tokens[file].append(t)
        print(f"{shard}: {len(shard_names)} images")
    if len(names) != len(set(names)):
        raise SystemExit(f"duplicated image names across the shards ({len(names) - len(set(names))})")

    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'image_names.pkl', 'wb') as f:
        pickle.dump(names, f)
    for file in FILES:
        torch.save(torch.vstack(tokens[file]), out / file)
    hyperparameters = {}
    first = root / shards[0] / 'hyperparameters.json'
    if first.exists():
        hyperparameters = json.load(open(first))
    hyperparameters.update({'exp_name': args.out, 'merged_from': shards})
    with open(out / 'hyperparameters.json', 'w') as f:
        json.dump(hyperparameters, f, sort_keys=True, indent=4)
    print(f"Merged {len(names)} images into {out}")


if __name__ == '__main__':
    main()
