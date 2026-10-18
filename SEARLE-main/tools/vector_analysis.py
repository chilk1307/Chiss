"""
Image vectors and word vectors (plan 01, section 1.15; plan 03, experiment E3.9).

SEARLE turns the reference image into a chain of vectors: image -> CLIP image feature -> phi -> pseudo token (a "word"
placed among the CLIP word embeddings) -> "a photo of $ that {caption}" -> query feature. The metrics only look at the
end of the chain; this tool looks at the vectors in between.

    tokens  pseudo tokens of one or more methods next to the CLIP word embeddings: norms, nearest words, how much the
            tokens of different images look alike, agreement between methods (e.g. phi vs OTI); with --snap-eval also
            the metric when every token is replaced by its nearest real word, or rescaled to the norm of a real word
            (experiment E1.15a)

        python tools/vector_analysis.py tokens --dataset cirr --dataset-path $DATA/CIRR --cache-dir $DRIVE/cache \
            --sources searle phi:phi_B32 --snap-eval --out R/E1.15a-tokens-cirr

    drift   how far the image feature, the pseudo token and the query feature move away from those of the clean image
            when the reference image is degraded / restored / filtered, compared with PSNR / SSIM as predictors of the
            damage (loss of the target's score and rank, loss of the metric) (experiment E3.9). Configurations are
            '<degrade>[+<restore>[+<freq filter>]]' (--configs) and/or the settings of existing runs (--runs: only
            their --degrade / --restore / --freq-filter are read, the method is --method)

        python tools/vector_analysis.py drift --dataset cirr --dataset-path $DATA/CIRR --cache-dir $DRIVE/cache \
            --method phi:phi_B32 --runs R/E3.1-*-cirr R/E3.2-* --out R/E3.9-drift-cirr

    gap     modality gap: distance between the centroid of the gallery image features and the centroid of each kind
            of query feature (reference image, caption, sum, pseudo-token queries), score scale of each kind (mean and
            std over the gallery, score and z-score of the target) and a PCA plot (experiment E1.15b)

        python tools/vector_analysis.py gap --dataset cirr --dataset-path $DATA/CIRR --cache-dir $DRIVE/cache \
            --sources phi:phi_B32 --out R/E1.15b-gap-cirr

Methods (--sources, --method): searle, searle-xl, phi:<exp>[:<checkpoint>] (default phi_last.pt),
oti:<exp>[:<token file>] (tokens of data/oti_pseudo_tokens/<dataset>/val/<exp>; tokens and gap only), image and sum
(drift and gap only). <exp> may also be the absolute path of the experiment folder. Validation split only.
Every subcommand writes <out>/report.md, a .json summary, a .csv and plots.
"""
import csv
import json
import math
import re
import sys
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_utils import query_from_record  # noqa: E402  (also puts src/ on the path)
from quality_vs_retrieval import MAIN, spearman  # noqa: E402
import retrieval_core as core  # noqa: E402
from reference_processing import ReferenceProcessor  # noqa: E402

METHOD = re.compile(r'^(?P<kind>searle-xl|searle|phi|oti|image|sum)(?::(?P<exp>.+?))?(?::(?P<file>[^:/\\]+\.pt))?$')
SPECIAL_TOKENS = 2  # <|startoftext|> and <|endoftext|>, the last two rows of the CLIP vocabulary
CLEAN = ('none', 'none', 'none')
CIRCO_RANK_CAP = 51  # CIRCO records give the target rank inside the top 50 only


# ----------------------------------------------------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------------------------------------------------

def parse_method(spec: str) -> dict:
    m = METHOD.match(spec)
    if m is None or (m['kind'] in ('phi', 'oti')) != (m['exp'] is not None):
        raise ValueError(f"unknown method '{spec}': searle, searle-xl, phi:<exp>[:<checkpoint>], "
                         f"oti:<exp>[:<token file>], image, sum")
    label = m['kind'] if m['exp'] is None else f"{m['kind']}:{Path(m['exp']).name}"
    if m['file']:
        label += f":{m['file']}"
    return {'kind': m['kind'], 'exp': m['exp'], 'file': m['file'], 'label': label}


def make_pipeline(spec: str, args) -> Tuple[core.RetrievalPipeline, str]:
    """
    The validation pipeline of one method, with the defaults of validate.py
    """
    method = parse_method(spec)
    parser = ArgumentParser()
    core.add_eval_args(parser, datasets=['cirr', 'fashioniq', 'circo'])
    argv = ['--eval-type', method['kind'], '--dataset', args.dataset, '--dataset-path', args.dataset_path,
            '--clip-model-name', args.clip_model_name, '--num-workers', str(args.num_workers)]
    if args.cache_dir:
        argv += ['--cache-dir', args.cache_dir]
    if method['kind'] == 'phi':
        argv += ['--exp-name', method['exp'], '--phi-checkpoint-name', method['file'] or 'phi_last.pt']
    elif method['kind'] == 'oti':
        argv += ['--exp-name', method['exp']] + (['--token-file', method['file']] if method['file'] else [])
    return core.RetrievalPipeline(parser.parse_args(argv), 'val'), method['label']


def is_token_method(pipeline: core.RetrievalPipeline) -> bool:
    return pipeline.args.eval_type in core.TOKEN_EVAL_TYPES


def describe(values) -> dict:
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if values.size == 0:
        return {}
    return {'mean': float(values.mean()), 'p5': float(np.percentile(values, 5)), 'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)), 'n': int(values.size)}


def mean(values) -> Optional[float]:
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def fmt(value, digits: int = 3, sign: bool = False) -> str:
    if value is None:
        return 'n/a'
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def rho_text(result) -> str:
    rho, p = result
    return 'n/a' if rho is None else f"{rho:+.2f} (p={p:.2g})"


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


@torch.no_grad()
def image_query_features(pipeline: core.RetrievalPipeline, queries: List[core.Query]) -> torch.Tensor:
    features, names = pipeline.reference_features()
    index = {n: i for i, n in enumerate(names)}
    return F.normalize(features[[index[q.reference] for q in queries]].float().to(core.device), dim=-1)


@torch.no_grad()
def query_features(pipeline: core.RetrievalPipeline, queries: List[core.Query]) -> torch.Tensor:
    """
    Normalized query features of the pipeline's method (no score fusion)
    """
    fiq = pipeline.dataset == 'fashioniq'
    if is_token_method(pipeline):
        return core.encode_token_queries(pipeline.clip_model, queries, pipeline.reference_tokens(), fiq,
                                         pipeline.templates).float()
    image = image_query_features(pipeline, queries)
    if pipeline.args.eval_type == 'image':
        return image
    if pipeline.args.eval_type == 'sum':
        return F.normalize(image + core.encode_text_queries(pipeline.clip_model, queries, fiq).float(), dim=-1)
    raise ValueError(f"--eval-type {pipeline.args.eval_type} is not supported here")


# ----------------------------------------------------------------------------------------------------------------------
# tokens: pseudo tokens vs CLIP word embeddings
# ----------------------------------------------------------------------------------------------------------------------

def vocabulary(clip_model) -> Tuple[torch.Tensor, List[str]]:
    from clip.simple_tokenizer import SimpleTokenizer
    weight = clip_model.token_embedding.weight.detach().float().cpu()[:-SPECIAL_TOKENS]
    decoder = SimpleTokenizer().decoder
    return weight, [decoder[i].replace('</w>', '') for i in range(len(weight))]


@torch.no_grad()
def topk_cosine(x: torch.Tensor, vocab: torch.Tensor, k: int, chunk: int = 2048) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Top-k cosine similarity of the rows of x with the vocabulary embeddings
    """
    vocab_n = F.normalize(vocab.to(core.device), dim=-1)
    x = F.normalize(x.float(), dim=-1)
    values, indices = [], []
    for i in range(0, len(x), chunk):
        top = (x[i:i + chunk].to(core.device) @ vocab_n.T).topk(k, dim=-1)
        values.append(top.values.cpu())
        indices.append(top.indices.cpu())
    return torch.cat(values), torch.cat(indices)


def mean_pairwise_cosine(x: torch.Tensor, max_n: int, rng) -> Optional[float]:
    """
    Mean cosine between two different rows (1 = every vector points the same way)
    """
    if len(x) > max_n:
        x = x[torch.as_tensor(rng.choice(len(x), max_n, replace=False))]
    n = len(x)
    if n < 2:
        return None
    x = F.normalize(x.float(), dim=-1)
    return float(((x @ x.T).sum() - n) / (n * (n - 1)))


def evaluate_with_tokens(pipeline: core.RetrievalPipeline, tokens: Dict[str, torch.Tensor]) -> float:
    pipeline.reference_tokens = lambda: tokens  # the instance attribute shadows the method for this evaluation
    try:
        metrics, _ = pipeline.evaluate()
    finally:
        del pipeline.reference_tokens
    return metrics[MAIN[pipeline.dataset]]


def run_tokens(args) -> None:
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sources, vocabularies = {}, {}
    for spec in args.sources:
        pipeline, label = make_pipeline(spec, args)
        if not is_token_method(pipeline):
            raise ValueError(f"{spec}: 'tokens' needs a pseudo-token method (searle, searle-xl, phi:..., oti:...)")
        tokens = pipeline.reference_tokens()
        names = list(tokens)
        if pipeline.clip_model_name not in vocabularies:
            vocabularies[pipeline.clip_model_name] = vocabulary(pipeline.clip_model)
        sources[label] = {'pipeline': pipeline, 'backbone': pipeline.clip_model_name, 'names': names,
                          'tokens': torch.stack([tokens[n].detach().float().cpu() for n in names])}

    # The real words of each backbone
    words_summary = {}
    for backbone, (vocab, _) in vocabularies.items():
        sample = torch.as_tensor(rng.choice(len(vocab), min(args.vocab_sample, len(vocab)), replace=False))
        nn_values, _ = topk_cosine(vocab[sample], vocab, 2)  # the first neighbour of a word is the word itself
        norms = vocab.norm(dim=-1).numpy()
        words_summary[backbone] = {'n_words': len(vocab), 'norm': describe(norms),
                                   'norm_p99': float(np.percentile(norms, 99)),
                                   'nearest_word_cos': describe(nn_values[:, 1].numpy()),
                                   'pairwise_cos': mean_pairwise_cosine(vocab, args.pairwise_sample, rng),
                                   '_norms': norms, '_nn': nn_values[:, 1].numpy()}

    # Each method: norms, nearest words, collapse
    summary, rows = {}, []
    for label, s in sources.items():
        vocab, words = vocabularies[s['backbone']]
        w = words_summary[s['backbone']]
        norms = s['tokens'].norm(dim=-1).numpy()
        values, indices = topk_cosine(s['tokens'], vocab, args.top_words)
        s['nearest'] = indices
        top1 = Counter(indices[:, 0].tolist())
        summary[label] = {
            'backbone': s['backbone'], 'n_tokens': len(norms), 'norm': describe(norms),
            'share_above_word_p99': float((norms > w['norm_p99']).mean()),
            'nearest_word_cos': describe(values[:, 0].numpy()),
            'pairwise_cos': mean_pairwise_cosine(s['tokens'], args.pairwise_sample, rng),
            'frequent_nearest_words': [[words[i], c] for i, c in top1.most_common(args.frequent_words)],
            'distinct_nearest_words': len(top1),
        }
        s['_norms'], s['_nn'] = norms, values[:, 0].numpy()
        for name, norm, v, idx in zip(s['names'], norms, values, indices):
            rows.append({'source': label, 'reference': name, 'norm': round(float(norm), 4),
                         'nearest_cos': round(float(v[0]), 4),
                         'nearest_words': ' '.join(words[i] for i in idx.tolist())})

    # Agreement between methods of the same backbone (same image: cosine of the two tokens; "other image": the same
    # with the names shuffled, i.e. the level expected by chance)
    agreement = []
    labels = list(sources)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            sa, sb = sources[a], sources[b]
            if sa['backbone'] != sb['backbone']:
                continue
            ia, ib = {n: k for k, n in enumerate(sa['names'])}, {n: k for k, n in enumerate(sb['names'])}
            common = [n for n in sa['names'] if n in ib]
            if len(common) < 2:
                continue
            ta = sa['tokens'][[ia[n] for n in common]]
            tb = sb['tokens'][[ib[n] for n in common]]
            order = rng.permutation(len(common))
            other = np.empty_like(order)
            other[order] = np.roll(order, 1)  # a random pairing where no image is paired with itself
            shuffled = tb[torch.as_tensor(other)]
            same = F.cosine_similarity(ta, tb, dim=-1).numpy()
            agreement.append({'a': a, 'b': b, 'n_common': len(common), 'cos_same_image': describe(same),
                              'cos_other_image': float(F.cosine_similarity(ta, shuffled, dim=-1).mean()),
                              'norm_ratio_b_over_a': describe((tb.norm(dim=-1) / ta.norm(dim=-1)).numpy()),
                              'same_nearest_word': float((sa['nearest'][[ia[n] for n in common], 0]
                                                          == sb['nearest'][[ib[n] for n in common], 0])
                                                         .float().mean())})

    # Replace every token by its nearest real word, or keep its direction with the median norm of the words
    snap = {}
    if args.snap_eval:
        for label, s in sources.items():
            vocab, _ = vocabularies[s['backbone']]
            word_norm = float(np.median(words_summary[s['backbone']]['_norms']))
            nearest_word = dict(zip(s['names'], vocab[s['nearest'][:, 0]]))
            rescaled = dict(zip(s['names'], F.normalize(s['tokens'], dim=-1) * word_norm))
            pipeline = s['pipeline']
            snap[label] = {'original': evaluate_with_tokens(pipeline, dict(zip(s['names'], s['tokens']))),
                           'nearest_word': evaluate_with_tokens(pipeline, nearest_word),
                           'word_norm': evaluate_with_tokens(pipeline, rescaled)}

    examples = []
    name_sets = [set(s['names']) for s in sources.values()]
    common = [n for n in sources[labels[0]]['names'] if all(n in names for names in name_sets)]
    chosen = sorted(rng.choice(len(common), min(args.n_examples, len(common)), replace=False)) if common else []
    for name in [common[i] for i in chosen]:
        row = {'reference': name}
        for label, s in sources.items():
            _, words = vocabularies[s['backbone']]
            k = s['names'].index(name)
            row[label] = ', '.join(words[i] for i in s['nearest'][k, :5].tolist())
        examples.append(row)

    write_csv(out / 'nearest_words.csv', rows)
    public_words = {b: {k: v for k, v in w.items() if not k.startswith('_')} for b, w in words_summary.items()}
    (out / 'tokens.json').write_text(json.dumps({'dataset': args.dataset, 'words': public_words, 'sources': summary,
                                                 'agreement': agreement, 'snap_eval': snap, 'examples': examples},
                                                indent=2, ensure_ascii=False), encoding='utf-8')
    report = tokens_report(args, public_words, summary, agreement, snap, examples)
    (out / 'report.md').write_text(report, encoding='utf-8')
    plot_tokens(words_summary, sources, out)
    print(report)


def tokens_report(args, words, summary, agreement, snap, examples) -> str:
    lines = [f"# E1.15a Pseudo tokens vs CLIP words ({args.dataset} val)", "",
             "Norms of the vectors, cosine with the nearest real word (for a word: with the nearest *other* word), "
             "mean cosine between two different vectors of the set (1 = all identical).", "",
             "| Set | Vectors | Norm (p5 / median / p95) | Above the p99 of word norms | Nearest-word cosine (median) "
             "| Pairwise cosine |", "|---|---|---|---|---|---|"]
    for backbone, w in words.items():
        n = w['norm']
        lines.append(f"| CLIP words ({backbone}) | {w['n_words']} | {fmt(n['p5'])} / {fmt(n['median'])} / "
                     f"{fmt(n['p95'])} | 1% | {fmt(w['nearest_word_cos']['median'])} | {fmt(w['pairwise_cos'])} |")
    for label, s in summary.items():
        n = s['norm']
        lines.append(f"| {label} | {s['n_tokens']} | {fmt(n['p5'])} / {fmt(n['median'])} / {fmt(n['p95'])} | "
                     f"{100 * s['share_above_word_p99']:.1f}% | {fmt(s['nearest_word_cos']['median'])} | "
                     f"{fmt(s['pairwise_cos'])} |")
    lines += ["", "## Most frequent nearest words", ""]
    for label, s in summary.items():
        words_text = ', '.join(f"{w} ({c})" for w, c in s['frequent_nearest_words'])
        lines.append(f"- **{label}** ({s['distinct_nearest_words']} distinct words for {s['n_tokens']} tokens): "
                     f"{words_text}")
    if agreement:
        lines += ["", "## Agreement between methods", "",
                  "Cosine between the tokens of the same image, and of two different images (chance level).", "",
                  "| A | B | Common images | Cosine, same image (median) | Cosine, other image | Norm B / A (median) "
                  "| Same nearest word |", "|---|---|---|---|---|---|---|"]
        for a in agreement:
            lines.append(f"| {a['a']} | {a['b']} | {a['n_common']} | {fmt(a['cos_same_image']['median'])} | "
                         f"{fmt(a['cos_other_image'])} | {fmt(a['norm_ratio_b_over_a']['median'])} | "
                         f"{100 * a['same_nearest_word']:.1f}% |")
    if snap:
        metric = MAIN[args.dataset]
        lines += ["", f"## Is the pseudo token a word? ({metric})", "",
                  "Same evaluation with every token replaced by the embedding of its nearest real word, or rescaled "
                  "to the median norm of the words (direction kept).", "",
                  "| Method | Pseudo token | Nearest real word | Word norm |", "|---|---|---|---|"]
        for label, s in snap.items():
            lines.append(f"| {label} | {s['original']:.2f} | {s['nearest_word']:.2f} | {s['word_norm']:.2f} |")
    if examples:
        labels = [k for k in examples[0] if k != 'reference']
        lines += ["", "## Examples: 5 nearest words", "", "| Reference | " + " | ".join(labels) + " |",
                  "|---|" + "---|" * len(labels)]
        for e in examples:
            lines.append(f"| {e['reference']} | " + " | ".join(e[k] for k in labels) + " |")
    return '\n'.join(lines) + '\n'


def plot_tokens(words_summary, sources, out: Path) -> None:
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for backbone, w in words_summary.items():
        axes[0].hist(w['_norms'], bins=80, density=True, alpha=0.5, label=f"words {backbone}")
        axes[1].hist(w['_nn'], bins=80, density=True, alpha=0.5, label=f"words {backbone}")
    for label, s in sources.items():
        axes[0].hist(s['_norms'], bins=80, density=True, alpha=0.5, label=label)
        axes[1].hist(s['_nn'], bins=80, density=True, alpha=0.5, label=label)
    axes[0].set_xlabel('norm')
    axes[1].set_xlabel('cosine with the nearest real word')
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'tokens.png', dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------
# drift: vectors of processed reference images vs clean ones
# ----------------------------------------------------------------------------------------------------------------------

def parse_config(text: str) -> Tuple[str, str, str]:
    parts = text.split('+')
    if not 1 <= len(parts) <= 3 or not all(parts):
        raise ValueError(f"bad configuration '{text}': '<degrade>[+<restore>[+<freq filter>]]'")
    return tuple(parts + ['none'] * (3 - len(parts)))


def config_label(config: Tuple[str, str, str]) -> str:
    degrade, restore, freq = config
    if freq != 'none':
        return f"{degrade}+{restore}+{freq}"
    return degrade if restore == 'none' else f"{degrade}+{restore}"


def configs_of_runs(paths: List[str], dataset: str) -> List[Tuple[str, str, str]]:
    configs = []
    for run in map(Path, paths):
        if not (run / 'metrics.json').exists():
            print(f"skipped {run} (no metrics.json)")
            continue
        metrics = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
        if metrics.get('dataset') != dataset:
            print(f"skipped {run.name} (dataset {metrics.get('dataset')})")
            continue
        a = metrics.get('args', {})
        configs.append((a.get('degrade', 'none'), a.get('restore', 'none'), a.get('freq_filter', 'none')))
    return configs


@torch.no_grad()
def capture(pipeline: core.RetrievalPipeline, config, galleries: dict) -> dict:
    """
    Metric, per-reference vectors and per-query vectors / target score / target rank under one processing
    """
    processor = None if config == CLEAN else ReferenceProcessor(*config, save_quality=True)
    pipeline.set_reference_processor(processor)
    metrics, records = pipeline.evaluate()
    features, names = pipeline.reference_features()
    image = core.token_dict(names, F.normalize(features.float(), dim=-1).cpu())
    tokens = {}
    if is_token_method(pipeline):
        tokens = {n: t.detach().float().cpu() for n, t in pipeline.reference_tokens().items()}
    queries = [query_from_record(r) for r in records]
    features_q = query_features(pipeline, queries)

    scores = torch.empty(len(queries))
    rows_of = defaultdict(list)
    for i, q in enumerate(queries):
        rows_of[q.category].append(i)
    for category, rows in rows_of.items():
        if category not in galleries:
            gallery, gallery_names = pipeline.gallery_features(category)
            galleries[category] = (gallery, {n: i for i, n in enumerate(gallery_names)})
        gallery, index = galleries[category]
        columns = torch.as_tensor([index[queries[i].target] for i in rows], device=gallery.device)
        scores[rows] = (features_q[rows].to(gallery.device) * gallery[columns]).sum(-1).cpu()

    cap = CIRCO_RANK_CAP if pipeline.dataset == 'circo' else None
    return {'metric': metrics[MAIN[pipeline.dataset]], 'image': image, 'tokens': tokens,
            'queries': {q.qid: (q, f, float(s), r.get('target_rank') or cap)
                        for q, f, s, r in zip(queries, features_q.cpu(), scores, records)},
            'info': dict(pipeline.reference_info)}


def compare(clean: dict, state: dict, config) -> Tuple[dict, List[dict], dict]:
    """
    Drift of one processing with respect to the clean images: summary, per-query rows and per-reference values
    """
    refs = [n for n in state['image'] if n in clean['image']]
    d_img = {n: 1 - float(state['image'][n] @ clean['image'][n]) for n in refs}
    d_tok, ratio = {}, {}
    for n in refs:
        if n in state['tokens'] and n in clean['tokens']:
            a, b = clean['tokens'][n], state['tokens'][n]
            d_tok[n] = 1 - float(F.cosine_similarity(a, b, dim=0))
            ratio[n] = float(b.norm() / a.norm())
    info = state['info']
    psnr = {n: info[n]['psnr'] for n in refs if 'psnr' in info.get(n, {})}
    ssim = {n: info[n]['ssim'] for n in refs if 'ssim' in info.get(n, {})}

    label = config_label(config)
    rows = []
    for qid, (q, f, s, rank) in state['queries'].items():
        if qid not in clean['queries']:
            continue
        _, f0, s0, rank0 = clean['queries'][qid]
        ref_info = info.get(q.reference, {})
        rows.append({'config': label, 'qid': qid, 'reference': q.reference,
                     'degradation': ref_info.get('degradation', config[0]),
                     'restoration': ref_info.get('restoration', config[1]),
                     'psnr': psnr.get(q.reference), 'ssim': ssim.get(q.reference),
                     'd_img': d_img.get(q.reference), 'd_tok': d_tok.get(q.reference),
                     'norm_ratio': ratio.get(q.reference), 'd_query': 1 - float(f @ f0),
                     'score_loss': s0 - s, 'rank0': rank0, 'rank': rank,
                     'rank_loss': math.log(rank) - math.log(rank0) if rank and rank0 else None})
    summary = {'config': label, 'degrade': config[0], 'restore': config[1], 'freq_filter': config[2],
               'metric': state['metric'], 'metric_loss': clean['metric'] - state['metric'],
               'psnr': mean(psnr.values()), 'ssim': mean(ssim.values()),
               'd_img': mean(d_img.values()), 'd_tok': mean(d_tok.values()),
               'norm_ratio': describe(ratio.values()).get('median'),
               'd_query': mean(r['d_query'] for r in rows), 'score_loss': mean(r['score_loss'] for r in rows),
               'rank_loss': mean(r['rank_loss'] for r in rows), 'n_references': len(refs), 'n_queries': len(rows)}
    labels = Counter(i.get('restoration') for i in info.values() if 'restoration' in i)
    if len(labels) > 1:
        summary['restoration_counts'] = dict(labels)
    return summary, rows, {'d_img': d_img, 'd_tok': d_tok, 'psnr': psnr, 'ssim': ssim, 'info': info}


def mixed_breakdown(rows: List[dict], per_ref: dict) -> List[dict]:
    """
    Per degradation type, for a configuration whose images carry several degradations ('mixed')
    """
    labels = defaultdict(list)
    for name, i in per_ref['info'].items():
        if 'degradation' in i:
            labels[i['degradation']].append(name)
    if len(labels) < 2:
        return []
    result = []
    for label, names in sorted(labels.items()):
        names_set = set(names)
        group = [r for r in rows if r['reference'] in names_set]
        result.append({'degradation': label, 'n_references': len(names),
                       'psnr': mean(per_ref['psnr'].get(n) for n in names),
                       'd_img': mean(per_ref['d_img'].get(n) for n in names),
                       'd_tok': mean(per_ref['d_tok'].get(n) for n in names),
                       'score_loss': mean(r['score_loss'] for r in group),
                       'rank_loss': mean(r['rank_loss'] for r in group)})
    return result


def restoration_effects(summaries: Dict[tuple, dict], per_ref: Dict[tuple, dict]) -> List[dict]:
    """
    Restored vs only degraded: does the restoration bring the image back for the eye (PSNR, SSIM) and for CLIP
    (image drift)?
    """
    effects = []
    for config, s in summaries.items():
        degrade, restore, freq = config
        base_config = (degrade, 'none', freq)
        if restore == 'none' or degrade == 'none' or base_config not in summaries:
            continue
        base = summaries[base_config]
        before, after = per_ref[base_config]['d_img'], per_ref[config]['d_img']
        common = [n for n in after if n in before]
        diff = lambda k: s[k] - base[k] if s[k] is not None and base[k] is not None else None  # noqa: E731
        effects.append({'degradation': degrade, 'restoration': restore, 'freq_filter': freq,
                        'psnr_gain': diff('psnr'), 'ssim_gain': diff('ssim'), 'd_img_change': diff('d_img'),
                        'd_tok_change': diff('d_tok'), 'metric_gain': s['metric'] - base['metric'],
                        'share_closer': float(np.mean([after[n] < before[n] for n in common])) if common else None})
    return effects


def run_drift(args) -> None:
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    configs = [parse_config(c) for c in args.configs or []] + configs_of_runs(args.runs or [], args.dataset)
    configs = [c for c in dict.fromkeys(configs) if c != CLEAN]
    if not configs:
        raise ValueError("no configuration: give --configs and/or --runs")
    for config in configs:  # check every specification before the long computations
        ReferenceProcessor(*config)
    pipeline, label = make_pipeline(args.method, args)
    if pipeline.args.eval_type == 'oti':
        raise ValueError("drift: OTI tokens are computed on the clean images, use phi / searle / image / sum")

    galleries = {}
    clean = capture(pipeline, CLEAN, galleries)
    summaries, per_ref, rows, breakdowns = {}, {}, [], {}
    for config in configs:
        print(f"== {config_label(config)}")
        summary, config_rows, refs = compare(clean, capture(pipeline, config, galleries), config)
        summaries[config], per_ref[config] = summary, refs
        rows.extend(config_rows)
        breakdown = mixed_breakdown(config_rows, refs)
        if breakdown:
            breakdowns[summary['config']] = breakdown

    measures = ['d_img', 'd_tok', 'd_query', 'psnr', 'ssim']
    correlations = {}
    for m in measures:
        q_score = [(r[m], r['score_loss']) for r in rows if r[m] is not None]
        q_rank = [(r[m], r['rank_loss']) for r in rows if r[m] is not None and r['rank_loss'] is not None]
        c_metric = [(s[m], s['metric_loss']) for s in summaries.values() if s[m] is not None]
        correlations[m] = {key: spearman(*zip(*pairs)) if len(pairs) >= 3 else (None, None)
                           for key, pairs in [('query_score_loss', q_score), ('query_rank_loss', q_rank),
                                              ('config_metric_loss', c_metric)]}
    effects = restoration_effects(summaries, per_ref)

    write_csv(out / 'per_query.csv', rows)
    (out / 'drift.json').write_text(json.dumps(
        {'dataset': args.dataset, 'method': label, 'clean_metric': clean['metric'], 'metric': MAIN[args.dataset],
         'configs': list(summaries.values()), 'correlations': correlations, 'restoration_effects': effects,
         'mixed_breakdown': breakdowns}, indent=2), encoding='utf-8')
    report = drift_report(args, label, clean['metric'], list(summaries.values()), correlations, effects, breakdowns,
                          is_token_method(pipeline))
    (out / 'report.md').write_text(report, encoding='utf-8')
    plot_drift(list(summaries.values()), rows, correlations, out, rng, args.max_points)
    print(report)


def drift_report(args, label, clean_metric, summaries, correlations, effects, breakdowns, tokens: bool) -> str:
    metric = MAIN[args.dataset]
    lines = [f"# E3.9 Vector drift of the processed reference images ({args.dataset} val, {label})", "",
             f"Clean images: {metric} = {clean_metric:.2f}. Drift = 1 - cos(vector of the processed image, vector of "
             f"the clean image): mean over the reference images (query drift: over the queries). Losses are clean "
             f"minus processed (positive = damage); target score = cos(query, target image).", "",
             "| Configuration | PSNR | SSIM | Image drift | Token drift | Token norm ratio | Query drift | Metric | "
             "Metric loss | Target score loss | Log rank loss |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        lines.append(f"| {s['config']} | {fmt(s['psnr'], 1)} | {fmt(s['ssim'])} | {fmt(s['d_img'], 4)} | "
                     f"{fmt(s['d_tok'], 4)} | {fmt(s['norm_ratio'])} | {fmt(s['d_query'], 4)} | {s['metric']:.2f} | "
                     f"{s['metric_loss']:+.2f} | {fmt(s['score_loss'], 4, True)} | {fmt(s['rank_loss'], 2, True)} |")
    names = {'d_img': 'Image drift', 'd_tok': 'Token drift', 'd_query': 'Query drift', 'psnr': 'PSNR',
             'ssim': 'SSIM'}
    lines += ["", "## Which measure goes with the damage", "",
              "Spearman rho. A good predictor has a large positive rho for the drifts and a large negative rho for "
              "PSNR / SSIM. Per query: pooled over every configuration; per configuration: means of the table above.",
              "", "| Measure | Per query: target score loss | Per query: log rank loss | Per configuration: metric "
                  "loss |", "|---|---|---|---|"]
    for m, c in correlations.items():
        if m == 'd_tok' and not tokens:
            continue
        lines.append(f"| {names[m]} | {rho_text(c['query_score_loss'])} | {rho_text(c['query_rank_loss'])} | "
                     f"{rho_text(c['config_metric_loss'])} |")
    if effects:
        lines += ["", "## Restoration: back to the clean image for the eye, for CLIP, for the retrieval?", "",
                  "Restored minus only degraded. A restoration helps CLIP when the image drift goes down; 'closer' "
                  "is the share of images whose CLIP feature got closer to the clean one.", "",
                  "| Degradation | Restoration | Δ PSNR | Δ SSIM | Δ image drift | Closer | Δ token drift | "
                  "Δ metric |", "|---|---|---|---|---|---|---|---|"]
        for e in effects:
            closer = 'n/a' if e['share_closer'] is None else f"{100 * e['share_closer']:.0f}%"
            lines.append(f"| {e['degradation']} | {e['restoration']} | {fmt(e['psnr_gain'], 1, True)} | "
                         f"{fmt(e['ssim_gain'], 3, True)} | {fmt(e['d_img_change'], 4, True)} | {closer} | "
                         f"{fmt(e['d_tok_change'], 4, True)} | {e['metric_gain']:+.2f} |")
    for config, rows in breakdowns.items():
        lines += ["", f"## {config}: per degradation type", "",
                  "| Degradation | Images | PSNR | Image drift | Token drift | Target score loss | Log rank loss |",
                  "|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['degradation']} | {r['n_references']} | {fmt(r['psnr'], 1)} | {fmt(r['d_img'], 4)} | "
                         f"{fmt(r['d_tok'], 4)} | {fmt(r['score_loss'], 4, True)} | {fmt(r['rank_loss'], 2, True)} |")
    return '\n'.join(lines) + '\n'


def plot_drift(summaries, rows, correlations, out: Path, rng, max_points: int) -> None:
    plt = pyplot()
    panels = [('d_img', 'image drift'), ('d_query', 'query drift'), ('psnr', 'PSNR (dB)')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (key, name) in zip(axes, panels):
        points = [s for s in summaries if s[key] is not None]
        ax.scatter([s[key] for s in points], [s['metric_loss'] for s in points], s=18)
        for s in points:
            ax.annotate(s['config'], (s[key], s['metric_loss']), fontsize=6)
        rho = correlations[key]['config_metric_loss'][0]
        ax.set_title(f"rho = {'n/a' if rho is None else f'{rho:+.2f}'}", fontsize=9)
        ax.set_xlabel(f"{name} (mean per configuration)")
        ax.set_ylabel('metric loss')
        ax.axhline(0, color='gray', lw=0.8)
    fig.tight_layout()
    fig.savefig(out / 'drift_vs_metric.png', dpi=120)
    plt.close(fig)

    if rows:
        sample = [rows[i] for i in rng.choice(len(rows), min(max_points, len(rows)), replace=False)]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for ax, (key, name) in zip(axes, panels[:2]):
            points = [r for r in sample if r[key] is not None]
            ax.scatter([r[key] for r in points], [r['score_loss'] for r in points], s=4, alpha=0.4)
            ax.set_xlabel(f"{name} of the query")
            ax.set_ylabel('target score loss')
            ax.axhline(0, color='gray', lw=0.8)
        fig.tight_layout()
        fig.savefig(out / 'drift_per_query.png', dpi=120)
        plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------
# gap: where the queries sit between the image and text sides of CLIP
# ----------------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def run_gap(args) -> None:
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    built = [make_pipeline(spec, args) for spec in (args.sources or ['image'])]
    base = built[0][0]
    if len({p.clip_model_name for p, _ in built}) > 1:
        raise ValueError("gap compares the vectors of one CLIP backbone: run it once per backbone")
    token_sources = {label: (p, p.reference_tokens()) for p, label in built if is_token_method(p)}

    features, names = base.reference_features()
    reference = core.token_dict(names, F.normalize(features.float(), dim=-1))
    available = set(reference)
    for _, tokens in token_sources.values():
        available &= set(tokens)
    fiq = base.dataset == 'fashioniq'
    kinds = ['image', 'text', 'sum'] + [f"query {label}" for label in token_sources]

    values = {k: defaultdict(list) for k in kinds}
    sums = {k: 0 for k in kinds}
    gallery_sum, gallery_count = 0, 0
    pca_points = {k: [] for k in ['gallery'] + kinds}
    rows = []
    for category in (core.FIQ_DRESS_TYPES if fiq else [None]):
        gallery, gallery_names = base.gallery_features(category)
        index = {n: i for i, n in enumerate(gallery_names)}
        queries = [q for q in base.queries(category) if q.reference in available and q.target in index]
        if not queries:
            continue
        gallery_sum = gallery_sum + gallery.sum(0)
        gallery_count += len(gallery)
        pca_points['gallery'].append(gallery[torch.as_tensor(rng.choice(len(gallery), min(args.sample, len(gallery)),
                                                                         replace=False))].cpu())
        image = torch.stack([reference[q.reference] for q in queries]).to(gallery.device)
        text = core.encode_text_queries(base.clip_model, queries, fiq).float()
        query = {'image': image, 'text': text, 'sum': F.normalize(image + text, dim=-1)}
        for label, (p, tokens) in token_sources.items():
            query[f"query {label}"] = core.encode_token_queries(p.clip_model, queries, tokens, fiq,
                                                                p.templates).float()
        targets = torch.as_tensor([index[q.target] for q in queries], device=gallery.device)
        for kind, x in query.items():
            sums[kind] = sums[kind] + x.sum(0)
            pca_points[kind].append(x[torch.as_tensor(rng.choice(len(x), min(args.sample, len(x)),
                                                                 replace=False))].cpu())
            for i in range(0, len(x), 1024):
                s = x[i:i + 1024] @ gallery.T
                target = s.gather(1, targets[i:i + 1024, None]).squeeze(1)
                mu, sd = s.mean(1), s.std(1)
                v = values[kind]
                v['target'] += target.tolist()
                v['mean'] += mu.tolist()
                v['std'] += sd.tolist()
                v['z'] += ((target - mu) / sd).tolist()
                v['to_reference'] += (x[i:i + 1024] * image[i:i + 1024]).sum(-1).tolist()
                columns = torch.as_tensor(rng.integers(0, len(gallery), size=(len(s), 8)), device=s.device)
                v['background'] += s.gather(1, columns).flatten().tolist()
        for i, q in enumerate(queries):
            row = {'qid': q.qid, 'reference': q.reference, 'target': q.target}
            offset = len(values['image']['target']) - len(queries) + i
            for kind in kinds:
                row[f"{kind}: target"] = round(values[kind]['target'][offset], 4)
                row[f"{kind}: z"] = round(values[kind]['z'][offset], 3)
            rows.append(row)
    if gallery_count == 0:
        raise ValueError("no query with a reference feature/token and a target in the gallery")

    gallery_centroid = gallery_sum / gallery_count
    n_queries = len(values['image']['target'])
    std_image = float(np.mean(values['image']['std']))
    summary = {}
    for kind in kinds:
        centroid = sums[kind] / n_queries
        v = values[kind]
        summary[kind] = {'centroid_distance': float((gallery_centroid - centroid).norm()),
                         'centroid_cos': float(F.cosine_similarity(gallery_centroid, centroid, dim=0)),
                         'target': float(np.mean(v['target'])), 'gallery_mean': float(np.mean(v['mean'])),
                         'gallery_std': float(np.mean(v['std'])), 'z_target': float(np.mean(v['z'])),
                         'to_reference': float(np.mean(v['to_reference'])),
                         'std_image_over_std': std_image / float(np.mean(v['std']))}

    write_csv(out / 'per_query.csv', rows)
    (out / 'gap.json').write_text(json.dumps({'dataset': args.dataset, 'backbone': base.clip_model_name,
                                              'n_queries': n_queries, 'kinds': summary}, indent=2), encoding='utf-8')
    report = gap_report(args, base.clip_model_name, n_queries, summary)
    (out / 'report.md').write_text(report, encoding='utf-8')
    plot_gap(pca_points, values, kinds, out)
    print(report)


def gap_report(args, backbone: str, n_queries: int, summary: dict) -> str:
    lines = [f"# E1.15b Image side and text side of CLIP ({args.dataset} val, {backbone}, {n_queries} queries)", "",
             "Centroid distance = ||mean(gallery image features) - mean(query features)|| (normalized features; "
             "about 0 for image queries, the 'modality gap' for text queries). Scores are cosines with the gallery: "
             "target, mean and std over the gallery, z = (target - mean) / std. 'std image / std' tells how much "
             "more the image score cos(reference, g) spreads than the query score: with --fusion-weight alpha the "
             "image term weighs about alpha times this ratio.", "",
             "| Query | Centroid distance | Centroid cosine | Target score | Gallery mean | Gallery std | z of the "
             "target | Cosine with the reference image | std image / std |", "|---|---|---|---|---|---|---|---|---|"]
    for kind, s in summary.items():
        lines.append(f"| {kind} | {s['centroid_distance']:.3f} | {s['centroid_cos']:.3f} | {s['target']:.3f} | "
                     f"{s['gallery_mean']:.3f} | {s['gallery_std']:.4f} | {s['z_target']:.2f} | "
                     f"{s['to_reference']:.3f} | {s['std_image_over_std']:.2f} |")
    return '\n'.join(lines) + '\n'


def plot_gap(pca_points, values, kinds, out: Path) -> None:
    plt = pyplot()
    groups = {k: torch.cat(v).float().numpy() for k, v in pca_points.items() if v}
    stacked = np.concatenate(list(groups.values()))
    centre = stacked.mean(0)
    _, singular, vt = np.linalg.svd(stacked - centre, full_matrices=False)
    explained = (singular[:2] ** 2).sum() / (singular ** 2).sum()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for kind, points in groups.items():
        xy = (points - centre) @ vt[:2].T
        axes[0].scatter(xy[:, 0], xy[:, 1], s=4, alpha=0.4, label=kind)
    axes[0].set_title(f"PCA of normalized features ({100 * explained:.0f}% of the variance)", fontsize=9)
    axes[0].legend(fontsize=7, markerscale=3)
    for i, kind in enumerate(kinds):
        color = f"C{i + 1}"
        axes[1].hist(values[kind]['background'], bins=80, density=True, histtype='step', color=color,
                     label=f"{kind}: gallery")
        axes[1].hist(values[kind]['target'], bins=40, density=True, alpha=0.3, color=color, label=f"{kind}: target")
    axes[1].set_xlabel('cosine with gallery images')
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / 'gap.png', dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------

def main():
    parser = ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    commands = {'tokens': run_tokens, 'drift': run_drift, 'gap': run_gap}
    for name in commands:
        p = sub.add_parser(name)
        p.add_argument('--dataset', required=True, choices=['cirr', 'fashioniq', 'circo'])
        p.add_argument('--dataset-path', required=True)
        p.add_argument('--out', required=True)
        p.add_argument('--cache-dir', default=None)
        p.add_argument('--clip-model-name', default='ViT-B/32', help="Backbone of the image and sum methods")
        p.add_argument('--num-workers', type=int, default=8)
        p.add_argument('--seed', type=int, default=0)
        if name in ('tokens', 'gap'):
            p.add_argument('--sources', nargs='+', required=name == 'tokens', help="Methods (see the module help)")
        if name == 'tokens':
            p.add_argument('--snap-eval', action='store_true',
                           help="Also evaluate with the tokens replaced by their nearest word / rescaled to a word norm")
            p.add_argument('--top-words', type=int, default=10, help="Nearest words kept per token in the csv")
            p.add_argument('--frequent-words', type=int, default=15)
            p.add_argument('--n-examples', type=int, default=12)
            p.add_argument('--vocab-sample', type=int, default=5000)
            p.add_argument('--pairwise-sample', type=int, default=2000)
        if name == 'drift':
            p.add_argument('--method', required=True, help="searle, searle-xl, phi:<exp>[:<checkpoint>], image, sum")
            p.add_argument('--configs', nargs='+', help="'<degrade>[+<restore>[+<freq filter>]]', e.g. jpeg:2+deblock")
            p.add_argument('--runs', nargs='+', help="Runs whose --degrade / --restore / --freq-filter are used")
            p.add_argument('--max-points', type=int, default=5000, help="Queries drawn in the per-query plot")
        if name == 'gap':
            p.add_argument('--sample', type=int, default=1000, help="Points per kind in the PCA plot")
    args = parser.parse_args()
    commands[args.command](args)


if __name__ == '__main__':
    main()
