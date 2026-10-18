"""
Generate a new GPT phrase set with a recent open language model (plan 06, section 6.3; task C23, experiment D16).
Run it in a separate Colab runtime with a recent Python and transformers (not the Python 3.8 project venv); the
output has the format of data/GPT_phrases/<name>/concept_to_phrases.pkl (pickle protocol 4, readable by Python 3.8).

    # P2: base model, same prompt as the original set ("a photo of {concept} that" ...)
    python tools/generate_phrases_llm.py --mode continue --model Qwen/Qwen2.5-1.5B --out-dir $DRIVE/outputs/GPT_phrases/P2_qwen15_s0 \
        --similar-concepts $DRIVE/outputs/similar_concepts/imagenet/test/ViT-B32_top_250_ensembling_image_texts.csv \
        --image-names $DRIVE/outputs/oti_pseudo_tokens/imagenet/test/oti_B32_s0/image_names.pkl --n-phrases 64
    # P3: instruct model, visual sentences in the style of the CIR modifications (prompt in tools/prompts/p3.txt)
    python tools/generate_phrases_llm.py --mode instruct --model microsoft/Phi-3.5-mini-instruct ...

Small-scale mode (--similar-concepts + --image-names): only the concepts used for those images (top-15 per image for
OTI, top-150 for phi). Without it every concept of data/oidv7-class-descriptions.csv is generated.
Post-processing: the concept must appear as a whole word (it is rewritten with the exact concept string), lines like
"Here are ..." and mostly non-ASCII lines are dropped, duplicates are removed, at most --max-words words. Progress is
saved every --save-every concepts and a rerun continues where it stopped. Check the licence of the model before use
and write it in hyperparameters.json (--license).
"""
import json
import pickle
import re
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gpt_phrases_utils import normalise_concept  # noqa: E402

PREAMBLE = re.compile(r"^\s*(here (are|is)|sure|certainly|of course|note)\b", re.IGNORECASE)
SENTENCE_END = re.compile(r"[.!?](?:\s|$)|\n")


def wanted_concepts(args):
    if args.similar_concepts:
        import csv
        names = None
        if args.image_names:
            with open(args.image_names, 'rb') as f:
                names = set(pickle.load(f))
        concepts = {}
        with open(args.similar_concepts, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if names is not None and row[0] not in names:
                    continue
                top_oti = list(dict.fromkeys(w.strip() for w in row[1:]))[:args.top_k_oti]
                for c in top_oti + [w for w in row[1:args.top_k_phi + 1]]:
                    concepts[c.strip()] = None
        return list(concepts)
    import csv
    with open(Path(__file__).resolve().parents[1] / 'data' / 'oidv7-class-descriptions.csv', newline='',
              encoding='utf-8') as f:
        return list(dict.fromkeys(row['DisplayName'].strip() for row in csv.DictReader(f)))


PHOTO_PREFIX = re.compile(r"^\W*(a (?:photo|picture|image) of\b.*)$", re.IGNORECASE | re.DOTALL)


def clean(texts, concept, max_words):
    kept, seen = [], set()
    for text in texts:
        text = re.sub(r"\s+", ' ', text.replace('\n', ' ')).strip().strip('"').strip()
        if not text or PREAMBLE.match(text):
            continue
        prefix = PHOTO_PREFIX.match(text)  # the prompt asks for sentences starting with "a photo of"
        if prefix is None:
            continue
        text = prefix.group(1)
        text = 'a' + text[1:]
        if sum(ch.isascii() for ch in text) < 0.9 * len(text):
            continue
        text = normalise_concept(text, concept)
        if text is None:
            continue
        words = text.split(' ')[:max_words]
        text = ' '.join(words).rstrip(',;:')
        key = re.sub(r"[^\w\s]", '', text.lower())
        if key not in seen:
            seen.add(key)
            kept.append(text)
    return kept


def parse_json_list(text: str):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            items = json.loads(match.group(0))
            if isinstance(items, list):
                return [str(x) for x in items]
        except json.JSONDecodeError:
            pass
    return [line.strip(' -*"0123456789.') for line in text.splitlines() if line.strip()]


def main():
    parser = ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--mode', required=True, choices=['continue', 'instruct'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--similar-concepts', default=None)
    parser.add_argument('--image-names', default=None, help='image_names.pkl of an OTI shard (small-scale mode)')
    parser.add_argument('--top-k-oti', type=int, default=15)
    parser.add_argument('--top-k-phi', type=int, default=150)
    parser.add_argument('--n-phrases', type=int, default=64)
    parser.add_argument('--batch-concepts', type=int, default=8, help='Concepts generated together (continue mode)')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--max-new-tokens', type=int, default=40)
    parser.add_argument('--max-words', type=int, default=40)
    parser.add_argument('--rounds', type=int, default=3, help='Instruct mode: requests per concept at most')
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--limit', type=int, default=None, help='Only the first N concepts (quick test)')
    parser.add_argument('--license', default='', help='Licence of the model, written to hyperparameters.json')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    set_seed(args.seed)
    concepts = wanted_concepts(args)[:args.limit]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result_file = out / 'concept_to_phrases.pkl'
    result = pickle.load(open(result_file, 'rb')) if result_file.exists() else {}
    todo = [c for c in concepts if c not in result]
    print(f"{len(concepts)} concepts, {len(result)} already done, {len(todo)} to generate")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype='auto',
                                                 device_map='auto' if torch.cuda.is_available() else None)
    model.eval()
    template = (Path(__file__).resolve().parent / 'prompts' / 'p3.txt').read_text(encoding='utf-8').strip()
    sample = dict(do_sample=True, temperature=args.temperature, top_p=args.top_p,
                  pad_token_id=tokenizer.pad_token_id)

    def save():
        tmp = result_file.with_suffix('.tmp')
        with open(tmp, 'wb') as f:
            pickle.dump(result, f, protocol=4)
        tmp.replace(result_file)

    start, generated = time.time(), 0
    step = args.batch_concepts if args.mode == 'continue' else 1
    for i in range(0, len(todo), step):
        chunk = todo[i:i + step]
        with torch.no_grad():
            if args.mode == 'continue':
                prompts = [f"a photo of {c} that" for c in chunk]
                inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(model.device)
                output = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                        num_return_sequences=args.n_phrases, **sample)
                texts = tokenizer.batch_decode(output[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                for j, c in enumerate(chunk):
                    continuations = texts[j * args.n_phrases:(j + 1) * args.n_phrases]
                    # keep the first sentence of each continuation
                    first_sentences = [SENTENCE_END.split(t)[0].rstrip('.') for t in continuations]
                    result[c] = clean([f"a photo of {c} that{s if s[:1].isspace() else ' ' + s}"
                                       for s in first_sentences], c, args.max_words)
            else:
                c = chunk[0]
                phrases = []
                for _ in range(args.rounds):
                    messages = [{'role': 'user', 'content': template.format(n=args.n_phrases, concept=c)}]
                    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                            return_tensors='pt', return_dict=True)
                    ids = encoded['input_ids'].to(model.device)  # recent transformers return a dict here
                    mask = encoded.get('attention_mask')
                    output = model.generate(input_ids=ids, attention_mask=None if mask is None else
                                            mask.to(model.device), max_new_tokens=args.n_phrases * 30, **sample)
                    phrases = clean(phrases + parse_json_list(
                        tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)), c, args.max_words)
                    if len(phrases) >= args.n_phrases:
                        break
                result[c] = phrases[:args.n_phrases]
        generated += len(chunk)
        if generated % args.save_every < step or i + step >= len(todo):
            save()
            rate = generated / max(time.time() - start, 1e-6)
            print(f"{len(result)}/{len(concepts)} concepts, {rate:.2f} concepts/s, "
                  f"{(len(todo) - generated) / max(rate, 1e-6) / 3600:.2f} h left", flush=True)
    save()
    elapsed = time.time() - start
    sizes = [len(v) for v in result.values()]
    hyper = {'mode': args.mode, 'model': args.model, 'license': args.license, 'n_phrases': args.n_phrases,
             'temperature': args.temperature, 'top_p': args.top_p, 'max_new_tokens': args.max_new_tokens,
             'prompt': 'a photo of {concept} that' if args.mode == 'continue' else template,
             'small_scale': bool(args.similar_concepts), 'image_names': args.image_names,
             'concepts': len(result), 'phrases_per_concept_mean': sum(sizes) / max(len(sizes), 1),
             'concepts_without_phrases': sum(1 for s in sizes if s == 0), 'seconds_this_run': round(elapsed),
             'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}
    (out / 'hyperparameters.json').write_text(json.dumps(hyper, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(hyper, indent=2, ensure_ascii=False))
    for c in list(result)[:3]:
        print(c, '->', result[c][:3])


if __name__ == '__main__':
    main()
