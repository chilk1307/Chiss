"""
Records one evaluation run when --output-dir is given:
    <output-dir>/metrics.json   all metrics, command line arguments, git commit, runtime, GPU
    <output-dir>/dump.jsonl     one line per query (written by the caller through write_dump)
    <output-dir>/log.txt        everything printed on stdout/stderr during the run
    <parent of output-dir>/runs_<RUNNER>.csv   one line appended per run (RUNNER is an environment variable)
Without --output-dir nothing is written and the scripts behave as before.
"""
import csv
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import torch

REPO_DIR = Path(__file__).absolute().parents[2]

# Metrics shown in the runs_<RUNNER>.csv line, per dataset
MAIN_METRICS = {
    'cirr': ['cirr_recall_at1', 'cirr_recall_at5', 'cirr_recall_at10', 'cirr_recall_at50', 'cirr_group_recall_at1'],
    'circo': ['circo_map_at5', 'circo_map_at10', 'circo_map_at25', 'circo_map_at50'],
    'fashioniq': ['average_fiq_recall_at10', 'average_fiq_recall_at50'],
}
RUNS_CSV_FIELDS = ['timestamp', 'exp_code', 'runner', 'script', 'dataset', 'eval_type', 'main_metrics', 'runtime_s',
                   'gpu', 'git_commit', 'note', 'command']


class _Tee:
    """
    Copies a text stream to a log file. Progress-bar refreshes (text with carriage returns) only reach the file once
    the line is finished, so the log keeps the final state of each bar instead of every refresh.
    """

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file
        self.pending = ''

    def write(self, text):
        self.stream.write(text)
        self.pending += text
        if '\n' in self.pending:
            *lines, self.pending = self.pending.split('\n')
            for line in lines:
                self.log_file.write(line.rsplit('\r', 1)[-1] + '\n')
            self.log_file.flush()
        return len(text)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def git_commit() -> str:
    try:
        commit = subprocess.run(['git', '-C', str(REPO_DIR), 'rev-parse', '--short', 'HEAD'], capture_output=True,
                                text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(REPO_DIR), 'status', '--porcelain', '--untracked-files=no'],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return f"{commit}-dirty" if commit and dirty else (commit or 'unknown')
    except Exception:
        return 'unknown'


def gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'


class RunRecorder:
    def __init__(self, output_dir: Optional[str], args, script: str, nondefault_args: Optional[dict] = None):
        self.output_dir = Path(output_dir) if output_dir else None
        self.args = args
        self.script = script
        self.nondefault_args = nondefault_args or {}
        self.start_time = time.time()
        self.runner = os.environ.get('RUNNER', 'unknown')
        self.command = ' '.join([Path(sys.executable).name, *sys.argv])
        self._log_file = None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._log_file = open(self.output_dir / 'log.txt', 'w', encoding='utf-8')
            sys.stdout = _Tee(sys.stdout, self._log_file)
            sys.stderr = _Tee(sys.stderr, self._log_file)
            print(f"[run] {self.command}")
            print(f"[run] output dir {self.output_dir}, runner {self.runner}, commit {git_commit()}, gpu {gpu_name()}")

    @property
    def enabled(self) -> bool:
        return self.output_dir is not None

    def write_dump(self, records: Iterable[dict]) -> None:
        if not self.enabled:
            return
        with open(self.output_dir / 'dump.jsonl', 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def finish(self, metrics: dict, info: Optional[dict] = None) -> None:
        if not self.enabled:
            return
        runtime = time.time() - self.start_time
        timestamp = datetime.datetime.now().isoformat(timespec='seconds')
        dataset = getattr(self.args, 'dataset', '')
        content = {
            'exp_code': self.output_dir.name,
            'script': self.script,
            'dataset': dataset,
            'eval_type': getattr(self.args, 'eval_type', ''),
            'metrics': metrics,
            'info': info or {},
            'args': vars(self.args),
            'nondefault_args': self.nondefault_args,
            'command': self.command,
            'runner': self.runner,
            'note': getattr(self.args, 'note', '') or '',
            'git_commit': git_commit(),
            'gpu': gpu_name(),
            'runtime_s': round(runtime, 1),
            'timestamp': timestamp,
        }
        with open(self.output_dir / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2, ensure_ascii=False)

        main = [f"{k}={metrics[k]:.2f}" for k in MAIN_METRICS.get(dataset, []) if k in metrics]
        runs_csv = self.output_dir.parent / f"runs_{self.runner}.csv"
        new_file = not runs_csv.exists()
        with open(runs_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=RUNS_CSV_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow({
                'timestamp': timestamp, 'exp_code': content['exp_code'], 'runner': self.runner,
                'script': self.script, 'dataset': dataset, 'eval_type': content['eval_type'],
                'main_metrics': '; '.join(main), 'runtime_s': content['runtime_s'], 'gpu': content['gpu'],
                'git_commit': content['git_commit'], 'note': content['note'], 'command': self.command,
            })
        print(f"[run] wrote {self.output_dir / 'metrics.json'} and a line in {runs_csv} ({runtime:.0f} s)")
        self.close()

    def close(self) -> None:
        if self._log_file is not None:
            sys.stdout.flush()
            sys.stderr.flush()
            if isinstance(sys.stdout, _Tee):
                sys.stdout = sys.stdout.stream
            if isinstance(sys.stderr, _Tee):
                sys.stderr = sys.stderr.stream
            self._log_file.close()
            self._log_file = None
