import json
import shutil
from argparse import ArgumentParser
from pathlib import Path

import torch

import retrieval_core as core
from data_utils import PROJECT_ROOT
from run_recorder import RunRecorder

torch.multiprocessing.set_sharing_strategy('file_system')


def write_json(content: dict, path: Path) -> None:
    with open(path, 'w+') as file:
        json.dump(content, file, sort_keys=True)
    print(f"Submission file written: {path} ({len(content)} entries)")


def main():
    parser = ArgumentParser()
    parser.add_argument("--submission-name", type=str, required=True, help="Filename of the generated submission file")
    core.add_eval_args(parser, datasets=['cirr', 'circo'])
    args = parser.parse_args()

    recorder = RunRecorder(args.output_dir, args, 'generate_test_submission', core.nondefault_args(parser, args))
    pipeline = core.RetrievalPipeline(args, split='test')
    print(f"Eval type = {args.eval_type} \t exp name = {args.exp_name} \t")

    submissions_folder_path = PROJECT_ROOT / 'data' / "test_submissions" / args.dataset
    submissions_folder_path.mkdir(exist_ok=True, parents=True)
    if args.dataset == 'cirr':
        # CIRR always needs both files: the main ranking (recall) and the ranking inside each image set (recall_subset)
        submission, group_submission, records = pipeline.cirr_submission()
        files = [submissions_folder_path / f"{args.submission_name}.json",
                 submissions_folder_path / f"subset_{args.submission_name}.json"]
        write_json(submission, files[0])
        write_json(group_submission, files[1])
    elif args.dataset == 'circo':
        submission, records = pipeline.circo_submission()
        files = [submissions_folder_path / f"{args.submission_name}.json"]
        write_json(submission, files[0])
    else:
        raise ValueError("Dataset not supported")

    if recorder.enabled:
        for file in files:
            shutil.copy(file, recorder.output_dir / file.name)
    info = pipeline.info()
    info['n_queries'] = len(records)
    info['submission_files'] = [str(f) for f in files]
    recorder.write_dump(records)
    recorder.finish({}, info)


if __name__ == '__main__':
    main()
