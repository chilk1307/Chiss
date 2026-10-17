"""
Dataset check (plan 01, section 1.2; experiment E1.2): counts the declared images and queries of a dataset and how many
of them have their files on disk.

    python tools/check_datasets.py --dataset cirr --dataset-path $DATA/CIRR --out $DRIVE/results/E1.2-dataset-check

Writes <out>/<dataset>.json and <out>/<dataset>.md, then rebuilds <out>/report.md from every <dataset>.md in <out>
(run it once per dataset into the same folder). For FashionIQ it also writes <out>/fiq_val_subset.json, the list of
validation triplets whose images all exist, to freeze the subset used by the experiments.
"""
import json
import os
from argparse import ArgumentParser
from pathlib import Path


def existing_files(folder: Path) -> set:
    return set(os.listdir(folder)) if folder.is_dir() else set()


def check_cirr(path: Path) -> dict:
    result = {}
    for split in ['train', 'val', 'test1']:
        split_file = path / 'cirr' / 'image_splits' / f'split.rc2.{split}.json'
        captions_file = path / 'cirr' / 'captions' / f'cap.rc2.{split}.json'
        if not split_file.exists():
            continue
        name_to_relpath = json.load(open(split_file))
        present = {n for n, rel in name_to_relpath.items() if (path / rel).exists()}
        entry = {'images_declared': len(name_to_relpath), 'images_present': len(present),
                 'images_missing_examples': sorted(set(name_to_relpath) - present)[:10]}
        if captions_file.exists():
            triplets = json.load(open(captions_file))
            refs_missing = sum(t['reference'] not in present for t in triplets)
            targets_missing = sum(t.get('target_hard', t['reference']) not in present for t in triplets)
            members_missing = sum(any(m not in present for m in t['img_set']['members']) for t in triplets)
            entry.update({'queries': len(triplets), 'queries_reference_missing': refs_missing,
                          'queries_target_missing': targets_missing if split != 'test1' else None,
                          'queries_with_group_member_missing': members_missing})
        result[split] = entry
    return result


def check_circo(path: Path) -> dict:
    info = json.load(open(path / 'COCO2017_unlabeled' / 'annotations' / 'image_info_unlabeled2017.json'))
    on_disk = existing_files(path / 'COCO2017_unlabeled' / 'unlabeled2017')
    id_to_file = {str(im['id']): im['file_name'] for im in info['images']}
    present_ids = {i for i, f in id_to_file.items() if f in on_disk}
    result = {'gallery': {'images_declared': len(id_to_file), 'images_present': len(present_ids),
                          'images_missing': len(id_to_file) - len(present_ids),
                          'images_missing_examples': sorted(set(id_to_file) - present_ids)[:10]}}
    for split in ['val', 'test']:
        ann_file = path / 'annotations' / f'{split}.json'
        if not ann_file.exists():
            continue
        annotations = json.load(open(ann_file))
        entry = {'queries': len(annotations),
                 'queries_reference_missing': sum(str(a['reference_img_id']) not in present_ids for a in annotations)}
        if split == 'val':
            entry['queries_target_missing'] = sum(str(a['target_img_id']) not in present_ids for a in annotations)
            all_gts = [str(g) for a in annotations for g in a['gt_img_ids']]
            entry['ground_truths'] = len(all_gts)
            entry['ground_truths_missing'] = sum(g not in present_ids for g in all_gts)
        result[split] = entry
    return result


def check_fiq(path: Path, out: Path) -> dict:
    on_disk = {f[:-4] for f in existing_files(path / 'images') if f.endswith('.jpg')}
    result, subset = {}, {}
    declared_all = set()
    for split in ['train', 'val', 'test']:
        for category in ['dress', 'shirt', 'toptee']:
            captions_file = path / 'captions' / f'cap.{category}.{split}.json'
            split_file = path / 'image_splits' / f'split.{category}.{split}.json'
            if not captions_file.exists() or not split_file.exists():
                continue
            triplets = json.load(open(captions_file))
            names = json.load(open(split_file))
            declared_all.update(names)
            valid = [i for i, t in enumerate(triplets) if t['candidate'] in on_disk
                     and (split == 'test' or t['target'] in on_disk)]
            result[f'{split}/{category}'] = {
                'queries': len(triplets), 'queries_with_all_images': len(valid),
                'gallery_declared': len(names), 'gallery_present': sum(n in on_disk for n in names)}
            if split == 'val':
                subset[category] = [{'index': i, 'candidate': triplets[i]['candidate'],
                                     'target': triplets[i]['target']} for i in valid]
    result['images'] = {'declared_in_splits': len(declared_all), 'present': len(declared_all & on_disk),
                        'missing': len(declared_all - on_disk)}
    if subset:
        with open(out / 'fiq_val_subset.json', 'w') as f:
            json.dump(subset, f)
        result['val_subset_file'] = str(out / 'fiq_val_subset.json')
    return result


def to_markdown(dataset: str, path: str, result: dict) -> str:
    lines = [f"## {dataset}", "", f"Dataset path: `{path}`", ""]
    for section, entry in result.items():
        if not isinstance(entry, dict):
            lines.append(f"- {section}: {entry}")
            continue
        lines.append(f"### {section}")
        lines.append("")
        lines.append("| | |")
        lines.append("|---|---|")
        for k, v in entry.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")
    return '\n'.join(lines) + '\n'


def main():
    parser = ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--dataset', required=True, choices=['cirr', 'circo', 'fashioniq'])
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--out', required=True, help='Output folder, e.g. $DRIVE/results/E1.2-dataset-check')
    args = parser.parse_args()

    path, out = Path(args.dataset_path), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.dataset == 'cirr':
        result = check_cirr(path)
    elif args.dataset == 'circo':
        result = check_circo(path)
    else:
        result = check_fiq(path, out)

    with open(out / f'{args.dataset}.json', 'w') as f:
        json.dump(result, f, indent=2)
    markdown = to_markdown(args.dataset, args.dataset_path, result)
    (out / f'{args.dataset}.md').write_text(markdown, encoding='utf-8')
    parts = [(out / f'{d}.md').read_text(encoding='utf-8') for d in ['cirr', 'circo', 'fashioniq']
             if (out / f'{d}.md').exists()]
    (out / 'report.md').write_text('# E1.2 Dataset check\n\n' + '\n'.join(parts), encoding='utf-8')
    print(markdown)
    print(f"Written {out / 'report.md'}")


if __name__ == '__main__':
    main()
