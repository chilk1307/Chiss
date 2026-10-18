import json
import os
import pickle
import random
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

import clip
import numpy as np
import pandas as pd
import torch
from PIL import ImageFile
from clip.model import CLIP
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import PROJECT_ROOT, targetpad_transform, collate_fn
from datasets import FashionIQDataset, CIRRDataset, CIRCODataset, ImageNetDataset
from gpt_phrases_utils import CONCEPT_FILTER, PHRASE_MATCH, prepare_phrases, replace_concept
from phi import Phi
from utils import device
from encode_with_pseudo_tokens import encode_with_pseudo_tokens
from utils import get_templates

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy('file_system')


def load_init_phi(args, clip_model: CLIP) -> nn.Module:
    """
    phi used to initialise the tokens (plan 07, section 7.1): a phi trained by train_phi.py or the released SEARLE
    """
    if args.init == 'phi':
        phi_path = PROJECT_ROOT / 'data' / 'phi_models' / args.init_phi_exp_name
        with open(phi_path / 'hyperparameters.json') as f:
            hyperparameters = json.load(f)
        phi = Phi(input_dim=clip_model.visual.output_dim, hidden_dim=clip_model.visual.output_dim * 4,
                  output_dim=clip_model.token_embedding.embedding_dim, dropout=hyperparameters['phi_dropout'])
        phi.load_state_dict(torch.load(phi_path / 'checkpoints' / args.init_phi_checkpoint,
                                       map_location=device)[phi.__class__.__name__])
        return phi.to(device).eval()
    phi, _ = torch.hub.load(repo_or_dir='miccunifi/SEARLE', model='searle', source='github',
                            backbone=args.clip_model_name)
    return phi.to(device).eval()


def oti_inversion(args):
    """
    Perform Optimization-based Textual Inversion (OTI) using the CLIP model
    """

    # load the CLIP model
    clip_model, clip_preprocess = clip.load(args.clip_model_name, device=device)
    clip_model: CLIP = clip_model.float()
    clip_model.visual.requires_grad_(False)
    embedding_dim = clip_model.token_embedding.embedding_dim

    use_gpt = args.gpt_exp_name != 'none'
    if not use_gpt and args.lambda_gpt != 0:
        raise ValueError("--gpt-exp-name none (no GPT phrases) requires --lambda_gpt 0")
    snapshot_steps = sorted({int(s) for s in args.snapshot_steps.split(',') if s.strip()})
    if any(s < 0 or s > args.oti_steps for s in snapshot_steps):
        raise ValueError(f"--snapshot-steps must be between 0 and --oti-steps ({args.oti_steps})")
    init_phi = load_init_phi(args, clip_model) if args.init != 'random' else None

    # Initialize the pseudo tokens tensor
    names_list = []
    ema_global_oti_pseudo_tokens = torch.empty((0, embedding_dim))
    global_oti_pseudo_tokens = torch.empty((0, embedding_dim))
    snapshots = {s: {'raw': torch.empty((0, embedding_dim)), 'ema': torch.empty((0, embedding_dim))}
                 for s in snapshot_steps}
    batch_seconds = []

    # Set the experiment path
    experiment_path = (
            PROJECT_ROOT / 'data' / "oti_pseudo_tokens" / args.dataset.lower() / args.split / args.exp_name)

    # Resume training from a saved experiment
    if args.resume_experiment:
        print("Resuming training from a saved experiment", flush=True)

        # Load names and pseudo tokens
        with open(experiment_path / f"image_names.pkl", 'rb') as f:
            names_list = pickle.load(f)

        global_oti_pseudo_tokens = torch.load(experiment_path / f'oti_pseudo_tokens.pt')
        ema_global_oti_pseudo_tokens = torch.load(experiment_path / f'ema_oti_pseudo_tokens.pt')
        for s in snapshot_steps:
            raw_file, ema_file = experiment_path / f'oti_pseudo_tokens_step{s}.pt', \
                experiment_path / f'ema_oti_pseudo_tokens_step{s}.pt'
            if raw_file.exists() and ema_file.exists():
                snapshots[s] = {'raw': torch.load(raw_file), 'ema': torch.load(ema_file)}
            elif len(names_list):
                raise ValueError(f"snapshot of step {s} missing in {experiment_path}: resume with the same "
                                 f"--snapshot-steps as the first run")
        if (experiment_path / 'timing.json').exists():
            with open(experiment_path / 'timing.json') as f:
                batch_seconds = json.load(f).get('batch_seconds', [])

        # Load the saved hyperparameters
        with open(experiment_path / 'hyperparameters.json') as f:
            old_hyperparamters = json.load(f)

        # Check if the hyperparameters are the same
        for k, v in old_hyperparamters.items():
            if k in args:
                if v != vars(args)[k]:
                    print(f"Warning: {k} is different from the saved experiment")
                    print(f"saved parameter: {v} \t new_parameter: {vars(args)[k]}")

    if experiment_path.exists() and not args.resume_experiment:
        print("BE CAREFUL: training path already exists, you are about to overwrite it", flush=True)

    name_to_concepts, concept_to_phrases, concepts_to_filter_out = {}, {}, set()
    if use_gpt:
        # Load the similar concepts into a dictionary
        similar_concepts_df = pd.read_csv(
            PROJECT_ROOT / "data" / "similar_concepts" / args.dataset.lower() / args.split /
            f"{args.clip_model_name.replace('/', '')}_top_250_ensembling_image_texts.csv")
        name_to_concepts = {str(el[0]): el[1:] for el in similar_concepts_df.values.tolist()}

        # Filter out problematic concepts from the GPT phrases
        with open(PROJECT_ROOT / 'data' / 'GPT_phrases' / args.gpt_exp_name / 'concept_to_phrases.pkl', 'rb') as f:
            concept_to_phrases = pickle.load(f)
        concept_to_phrases, concepts_to_filter_out = prepare_phrases(concept_to_phrases, args.concept_filter,
                                                                     args.phrase_match)

    # Load source code for logging
    with open(Path(__file__).absolute()) as f:
        source_code = f.read()

    # Set preprocessing function
    if args.preprocess_type == 'clip':
        preprocess = clip_preprocess
    elif args.preprocess_type == 'targetpad':
        preprocess = targetpad_transform(1.25, clip_model.visual.input_resolution)
    else:
        raise ValueError("preprocess should be in ['targetpad', 'clip']")

    # Define the dataset
    if args.dataset.lower() == 'fashioniq':
        dataset = FashionIQDataset(args.dataset_path, args.split, ['dress', 'shirt', 'toptee'], 'relative',
                                   preprocess, no_duplicates=True)
    elif args.dataset.lower() == 'cirr':
        dataset = CIRRDataset(args.dataset_path, args.split, 'relative', preprocess, no_duplicates=True)
    elif args.dataset.lower() == 'circo':
        dataset = CIRCODataset(args.dataset_path, args.split, 'relative', preprocess)
    elif args.dataset.lower() == 'imagenet':
        dataset = ImageNetDataset(args.dataset_path, args.split, preprocess)
    else:
        raise ValueError("dataset should be in ['fashioniq', 'cirr', 'circo', 'imagenet']")

    # Define the dataloader and the criterions
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers)
    criterion = nn.CosineEmbeddingLoss()
    criterion_target = torch.as_tensor([1], device=device)

    templates = get_templates()

    # Start the loop
    for batch_idx, batch in enumerate(tqdm(loader)):
        if batch is None:  # every image of the batch failed to load
            continue
        # Load the batch
        images = batch.get('image')
        names = batch.get('image_name')
        if images is None:
            images = batch.get('reference_image')
        if names is None:
            names = batch.get('reference_name')

        # Check if the batch has already been processed
        if set(names) <= set(names_list):
            continue

        start_time = time.time()
        bs = len(images)

        # Extract the image features (before the initialisation, which may need them; this does not change the
        # random stream: encode_image draws no random number)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                batch_im_features = clip_model.encode_image(images.to(device))

        # Initialize the oti pseudo tokens: randomly (original OTI) or with the prediction of a phi (warm start)
        if init_phi is None:
            oti_pseudo_tokens = torch.empty((bs, embedding_dim), device=device)
            nn.init.normal_(oti_pseudo_tokens, std=0.02)
        else:
            with torch.no_grad():
                oti_pseudo_tokens = init_phi(batch_im_features.float()).detach().clone().float()

        # Get the GPT phrases associated with the similar concepts in the batch
        batch_gpt_phrases = []
        batch_concepts = []
        if use_gpt:
            for idx, name in enumerate(names):
                l_phrases = []
                l_concept_phrases = []
                l_similar_concepts = name_to_concepts[name]
                stripped_concepts = [str(w).strip() for w in l_similar_concepts]
                l_similar_concepts = list(dict.fromkeys(stripped_concepts))[:args.top_k]

                for w in l_similar_concepts:
                    if w in concepts_to_filter_out:
                        continue
                    for p in concept_to_phrases.get(w, []):
                        l_phrases.append(p)
                        l_concept_phrases.append(w)

                batch_gpt_phrases.append(l_phrases)
                batch_concepts.append(l_concept_phrases)

        # Copy the oti pseudo tokens for the EMA
        oti_pseudo_tokens = nn.Parameter(oti_pseudo_tokens)
        ema_oti_pseudo_tokens = oti_pseudo_tokens.clone().detach()

        # Define the optimizer and the scaler
        optimizer = optim.AdamW([oti_pseudo_tokens], lr=args.learning_rate, weight_decay=args.weight_decay)
        scaler = torch.cuda.amp.GradScaler()

        batch_snapshots = {}
        if 0 in snapshots:
            batch_snapshots[0] = (oti_pseudo_tokens.detach().cpu().clone(), ema_oti_pseudo_tokens.detach().cpu().clone())

        # Start the oti optimization loop
        for step in range(args.oti_steps):
            optimizer.zero_grad()

            # Sample the templates sentences
            template_indexes = random.choices(range(len(templates)), k=bs)
            template_oti_texts = [templates[i].format(" $ ") for i in template_indexes]
            tokenized_template_oti_texts = clip.tokenize(template_oti_texts, truncate=True).to(device)

            if use_gpt:
                # Sample the GPT phrases
                gpt_phrases_indexes = [random.choice(range(len(batch_gpt_phrases[i]))) for i in
                                       range(len(batch_gpt_phrases))]
                reg_gpt_phrases = [batch_gpt_phrases[i][rand_idx] for i, rand_idx in enumerate(gpt_phrases_indexes)]
                concepts_to_replace = [batch_concepts[i][rand_idx] for i, rand_idx in enumerate(gpt_phrases_indexes)]
                # Replace the concepts in the GPT phrases with the oti pseudo tokens
                oti_gpt_phrases = [replace_concept(reg_text, w, args.phrase_match, first_only=True) for w, reg_text in
                                   zip(concepts_to_replace, reg_gpt_phrases)]

                # Tokenize the sentences
                tokenized_oti_gpt_phrases = clip.tokenize(oti_gpt_phrases, truncate=True).to(device)
                tokenized_reg_gpt_phrases = clip.tokenize(reg_gpt_phrases, truncate=True).to(device)

            with torch.cuda.amp.autocast():
                # Extract the features
                template_oti_features = encode_with_pseudo_tokens(clip_model, tokenized_template_oti_texts,
                                                                  oti_pseudo_tokens)
                # Compute the loss
                cosine_loss = criterion(template_oti_features, batch_im_features, criterion_target)
                if use_gpt:
                    reg_gpt_features = clip_model.encode_text(tokenized_reg_gpt_phrases)
                    reg_oti_gpt_features = encode_with_pseudo_tokens(clip_model, tokenized_oti_gpt_phrases,
                                                                     oti_pseudo_tokens)
                    gpt_loss = criterion(reg_oti_gpt_features, reg_gpt_features, criterion_target)
                    oti_loss = (args.lambda_cos * cosine_loss + args.lambda_gpt * gpt_loss) / \
                               (args.lambda_cos + args.lambda_gpt)
                else:  # cosine loss only (no GPT phrases needed)
                    oti_loss = cosine_loss

            # Backpropagate the loss
            scaler.scale(oti_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update the EMA
            ema_oti_pseudo_tokens = args.ema_decay * ema_oti_pseudo_tokens + (1 - args.ema_decay) * oti_pseudo_tokens

            if step + 1 in snapshots:
                batch_snapshots[step + 1] = (oti_pseudo_tokens.detach().cpu().clone(),
                                             ema_oti_pseudo_tokens.detach().cpu().clone())

        # Update the global oti pseudo tokens and the names list
        names_list.extend(names)
        ema_global_oti_pseudo_tokens = torch.vstack(
            (ema_global_oti_pseudo_tokens, ema_oti_pseudo_tokens.detach().cpu()))
        global_oti_pseudo_tokens = torch.vstack((global_oti_pseudo_tokens, oti_pseudo_tokens.detach().cpu()))
        for s, (raw, ema) in batch_snapshots.items():
            snapshots[s]['raw'] = torch.vstack((snapshots[s]['raw'], raw))
            snapshots[s]['ema'] = torch.vstack((snapshots[s]['ema'], ema))
        batch_seconds.append(time.time() - start_time)

        # Save the experiment
        if batch_idx % args.save_frequency == 0 and batch_idx > 0:
            save_experiment(args, ema_global_oti_pseudo_tokens, experiment_path, global_oti_pseudo_tokens, names_list,
                            source_code, snapshots, batch_seconds)

    # Before finishing save the experiment
    save_experiment(args, ema_global_oti_pseudo_tokens, experiment_path, global_oti_pseudo_tokens, names_list,
                    source_code, snapshots, batch_seconds)


def save_experiment(args, ema_global_oti_pseudo_tokens: torch.tensor, experiment_path: Path,
                    global_oti_pseudo_tokens: torch.tensor, names_list: List, source_code: str,
                    snapshots: Dict[int, dict] = None, batch_seconds: List[float] = None) -> None:
    """
    Saves the pseudo tokens, the names list, the source code and the hyperparameters of the experiment, plus the
    tokens of every snapshot step (oti_pseudo_tokens_step<N>.pt, ema_oti_pseudo_tokens_step<N>.pt) and timing.json
    """

    experiment_path.mkdir(exist_ok=True, parents=True)
    with open(experiment_path / f'image_names.pkl', 'wb+') as f:
        pickle.dump(names_list, f)
    torch.save(global_oti_pseudo_tokens, experiment_path / f'oti_pseudo_tokens.pt')
    torch.save(ema_global_oti_pseudo_tokens, experiment_path / f'ema_oti_pseudo_tokens.pt')
    for s, tokens in (snapshots or {}).items():
        torch.save(tokens['raw'], experiment_path / f'oti_pseudo_tokens_step{s}.pt')
        torch.save(tokens['ema'], experiment_path / f'ema_oti_pseudo_tokens_step{s}.pt')
    if batch_seconds is not None:
        with open(experiment_path / 'timing.json', 'w') as f:
            json.dump({'batch_seconds': batch_seconds, 'mean_batch_seconds': float(np.mean(batch_seconds))
                       if batch_seconds else None, 'batch_size': args.batch_size, 'oti_steps': args.oti_steps,
                       'init': args.init, 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available()
                       else 'cpu'}, f, indent=2)
    with open(experiment_path / 'source_code.py', 'w+') as f:
        f.write(source_code)
    with open(experiment_path / 'hyperparameters.json', 'w+') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)


def main():
    parser = ArgumentParser()
    parser.add_argument("--exp-name", required=True, type=str, help="Experiment name")
    parser.add_argument("--clip-model-name", required=True, type=str,
                        help="CLIP model to use, e.g 'ViT-B/32', 'ViT-L/14'")
    parser.add_argument("--dataset", type=str, required=True, choices=['cirr', 'fashioniq', 'circo', 'imagenet'],
                        help="Dataset to use")
    parser.add_argument("--dataset-path", type=str, help="Path to the dataset", required=True)
    parser.add_argument("--split", type=str, required=True, choices=['train', 'val', 'test'],
                        help="Dataset split to use")

    parser.add_argument("--gpt-exp-name", type=str, default="GPTNeo27B",
                        help="Name of the GPT generation phrases experiment ('none': no GPT phrases, cosine loss "
                             "only, requires --lambda_gpt 0)")
    parser.add_argument("--learning-rate", default=2e-2, type=float, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--batch-size", default=32, type=int, help='batch size for each optimization iteration')
    parser.add_argument("--preprocess-type", default="targetpad", type=str, choices=['clip', 'targetpad'],
                        help="Preprocess pipeline to use")
    parser.add_argument("--top-k", type=int, default=15, help="Number of concepts associated to each image")
    parser.add_argument("--oti-steps", default=350, type=int, help="Number of optimization steps for each image")
    parser.add_argument("--lambda_gpt", type=float, default=0.5, help="Weight of the gpt loss")
    parser.add_argument("--lambda_cos", type=float, default=1, help="Weight of the cosine loss")
    parser.add_argument("--ema-decay", type=float, default=0.99, help="Decay for the exponential moving average")
    parser.add_argument("--save-frequency", default=5, type=int, help="Saving frequency expressed in batches")
    parser.add_argument("--resume-experiment", action='store_true', help="Resume the experiment if it exists",
                        default=False)
    parser.add_argument("--seed", type=int, default=42, help="seed for reproducibility")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")

    # Plan 07 (warm start) and plan 06 (GPT phrases); the defaults keep the original behaviour
    parser.add_argument("--init", type=str, default='random', choices=['random', 'phi', 'searle'],
                        help="Token initialisation: random N(0, 0.02^2) (original), a phi trained by train_phi.py, "
                             "or the released SEARLE phi (torch.hub)")
    parser.add_argument("--init-phi-exp-name", type=str, default=None, help="phi experiment for --init phi")
    parser.add_argument("--init-phi-checkpoint", type=str, default='phi_last.pt', help="checkpoint for --init phi")
    parser.add_argument("--snapshot-steps", type=str, default='',
                        help="Comma separated steps (e.g. 0,10,25,50,100,200,350) at which the raw and EMA tokens are "
                             "also saved, as oti_pseudo_tokens_step<N>.pt / ema_oti_pseudo_tokens_step<N>.pt")
    parser.add_argument("--phrase-match", type=str, default='legacy', choices=PHRASE_MATCH,
                        help="How the concept is replaced by $ in the GPT phrases (gpt_phrases_utils.py)")
    parser.add_argument("--concept-filter", type=str, default='legacy', choices=CONCEPT_FILTER,
                        help="Drop whole concepts (legacy) or only the unusable phrases (per-phrase)")

    args = parser.parse_args()
    if args.init == 'phi' and not args.init_phi_exp_name:
        parser.error("--init phi needs --init-phi-exp-name")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    oti_inversion(args)


if __name__ == '__main__':
    main()
