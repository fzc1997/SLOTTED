#!/usr/bin/env python3
"""Assign long reads to TE loci with the complete SLOTTED Joint workflow."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bam', 'reads', 'rmsk', 'genome'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--out', type=Path, help='new output directory; generated automatically if omitted')
    parser.add_argument('--flank', type=int, default=500)
    parser.add_argument('--ckpt', type=Path, default=ROOT / 'assets/ckpt_step30000.pt')
    parser.add_argument('--pool-params', type=Path, default=ROOT / 'assets/pool_frozen_params_v2.json')
    args = parser.parse_args(argv)
    if args.flank < 0:
        parser.error('--flank must be non-negative')
    inputs = {key: getattr(args, key).resolve() for key in ('bam', 'reads', 'rmsk', 'genome', 'ckpt', 'pool_params')}
    for name, path in inputs.items():
        if not path.is_file():
            parser.error(f'{name}: file not found: {path}')
    output = (args.out or Path('results') / (
        'SLOTTED_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + secrets.token_hex(3))).resolve()
    # Exclusive directory creation also protects concurrent runs.
    output.mkdir(parents=True, exist_ok=False)
    work = output / 'intermediates'
    work.mkdir()
    print(f'[SLOTTED] output: {output}', flush=True)
    final = output / 'SLOTTED_joint_predictions.tsv'
    stages = [
        ('prepare', ['--bam', inputs['bam'], '--reads', inputs['reads'], '--rmsk', inputs['rmsk'],
                     '--genome', inputs['genome'], '--flank', args.flank, '--out', work / 'input.pkl']),
        ('score', ['--d', work / 'input.pkl', '--ckpt', inputs['ckpt'], '--pool-params', inputs['pool_params'],
                   '--out', work / 'scores.parquet']),
        ('merge', ['--d', work / 'input.pkl', '--shards', work / 'scores.parquet', '--out', work / 'channels.parquet']),
        ('fit', ['--d', work / 'input.pkl', '--channels', work / 'channels.parquet',
                 '--out', final, '--receipt', work / 'joint_fit.json']),
    ]
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'NUMBA_NUM_THREADS': '1', 'PYTHONNOUSERSITE': '1'}
    receipt = {'status': 'RUNNING', 'inputs': {key: str(path) for key, path in inputs.items()},
               'flank': args.flank, 'stages': {}}
    start = time.monotonic()
    (work / 'README.md').write_text('Intermediate inputs, scores and logs from SLOTTED. The final table is ../SLOTTED_joint_predictions.tsv. run.json records inputs and stage status; joint_fit.json records EM convergence.\n')
    try:
        for name, options in stages:
            print(f'[SLOTTED] {name}', flush=True)
            before = time.monotonic()
            command = [sys.executable, str(ROOT / 'scripts/joint_pipeline.py'), name, *map(str, options)]
            with (work / f'{name}.log').open('x') as log:
                proc = subprocess.run(command, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT)
            receipt['stages'][name] = {'returncode': proc.returncode, 'seconds': round(time.monotonic() - before, 3)}
            if proc.returncode:
                raise RuntimeError(f'{name} failed; see {work / (name + ".log")}')
        fit = json.loads((work / 'joint_fit.json').read_text())
        receipt['status'] = 'COMPLETED'
        receipt['fit_status'] = fit['fit_status']
        print(f'[SLOTTED] {fit["fit_status"]}: {final}', flush=True)
    except Exception as error:
        receipt['status'] = 'FAILED'
        receipt['error'] = str(error)
        raise
    finally:
        receipt['seconds'] = round(time.monotonic() - start, 3)
        (work / 'run.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
