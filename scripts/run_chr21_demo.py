#!/usr/bin/env python3
"""Run and verify the bundled real-data chr21 example, entirely on CPU."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import secrets
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def verify_checksums(directory: Path, required: set[str] | None = None) -> int:
    """Verify only relative files listed in this directory's checksum manifest."""
    names = set()
    for line in (directory / 'SHA256SUMS').read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.lstrip('* ')
        if name in names:
            raise ValueError(f'duplicate checksum entry: {name}')
        names.add(name)
        path = directory / name
        if (Path(name).is_absolute() or '..' in Path(name).parts
                or not path.resolve().is_relative_to(directory.resolve())):
            raise ValueError('checksum manifest contains an unsafe path')
        if not path.is_file():
            raise FileNotFoundError(f'bundled input missing: {path}')
        actual = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b''):
                actual.update(chunk)
        if actual.hexdigest() != digest:
            raise ValueError(f'bundled input checksum mismatch: {path}')
    if not names or (required and not required <= names):
        raise ValueError(f'checksum manifest is empty or lacks required files: {sorted((required or set()) - names)}')
    return len(names)


def compare_tsv(actual: Path, expected: Path, numeric_columns: set[str], key_columns: tuple[str, ...]):
    """Compare observed output with a frozen software expectation, not truth."""
    def read(path):
        with path.open() as handle:
            reader = csv.DictReader(handle, delimiter='\t')
            fields = reader.fieldnames
            if not fields or len(fields) != len(set(fields)):
                raise ValueError(f'missing or duplicate TSV column names: {path}')
            rows = {}
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f'malformed TSV row in {path}: wrong number of fields')
                key = tuple(row[column] for column in key_columns)
                if key in rows:
                    raise ValueError(f'duplicate output row in {path}: {key}')
                rows[key] = row
        return fields, rows
    fields, got = read(actual)
    expected_fields, want = read(expected)
    if fields != expected_fields or set(got) != set(want):
        raise ValueError(f'output schema or row population differs: {actual.name}')
    for key in got:
        for field in fields:
            a, b = got[key][field], want[key][field]
            if a == b:
                continue
            if field in numeric_columns and a and b:
                x, y = float(a), float(b)
                if (math.isnan(x) and math.isnan(y)) or math.isclose(x, y, rel_tol=1e-5, abs_tol=1e-5):
                    continue
            raise ValueError(f'expected-output mismatch in {actual.name}, {key}, {field}: {a} != {b}')
    return len(got)


def run(root: Path, output: Path, *, skip_expected_check: bool = False):
    data = root / 'examples/chr21'
    expected = data / 'expected'
    if not skip_expected_check and not (expected / 'lm_assignments.tsv').is_file():
        raise FileNotFoundError('the complete example, including expected outputs, is required')
    if output.exists():
        raise FileExistsError(f'output already exists; choose a new --out directory: {output}')
    checked = verify_checksums(data, {'chr21.fa', 'chr21.fa.fai', 'chr21.rmsk.txt', 'reads.fa',
                                     'alignments.bam', 'alignments.bam.bai', 'selected_reads.tsv', 'provenance.json'})
    checked += verify_checksums(root / 'assets', {'ckpt_step30000.pt', 'pool_frozen_params_v2.json'})
    if not skip_expected_check:
        checked += verify_checksums(expected, {'lm_assignments.tsv', 'joint_predictions.tsv'})
    output.mkdir(parents=True)
    work = output / 'intermediates'
    work.mkdir()
    print(f'[chr21 demo] output: {output}', flush=True)
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'NUMBA_NUM_THREADS': '1', 'PYTHONNOUSERSITE': '1'}
    inputs = ['--bam', data / 'alignments.bam', '--reads', data / 'reads.fa',
              '--rmsk', data / 'chr21.rmsk.txt', '--genome', data / 'chr21.fa', '--flank', '500']
    stages = [
        ('lm', 'assign_reads_m2d.py', [*inputs, '--ckpt', root / 'assets/ckpt_step30000.pt',
                                     '--out', output / 'SLOTTED_lm_assignments.tsv', '--receipt', work / 'lm_receipt.json',
                                     '--device', 'cpu', '--band-backend', 'python', '--verify-exact', '--selective']),
        ('prepare', 'joint_pipeline.py', ['prepare', *inputs, '--out', work / 'input.pkl']),
        ('score', 'joint_pipeline.py', ['score', '--d', work / 'input.pkl', '--out', work / 'scores.parquet']),
        ('merge', 'joint_pipeline.py', ['merge', '--d', work / 'input.pkl', '--shards', work / 'scores.parquet',
                                      '--out', work / 'channels.parquet']),
        ('fit', 'joint_pipeline.py', ['fit', '--d', work / 'input.pkl', '--channels', work / 'channels.parquet',
                                    '--out', output / 'SLOTTED_joint_predictions.tsv', '--receipt', work / 'joint_fit.json']),
    ]
    receipt = {'schema': 'slotted.chr21_demo_run.v1', 'status': 'RUNNING',
               'verified_input_files': checked, 'python_version': platform.python_version(),
               'platform': platform.platform(), 'device': 'cpu', 'threads': 1, 'stages': {}}
    started = time.monotonic()
    try:
        for name, script, args in stages:
            command = [sys.executable, str(root / 'scripts' / script), *map(str, args)]
            print(f'[chr21 demo] {name}', flush=True)
            before = time.monotonic()
            with (work / f'{name}.log').open('w') as log:
                completed = subprocess.run(command, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT)
            receipt['stages'][name] = {'seconds': round(time.monotonic() - before, 3),
                                      'returncode': completed.returncode}
            if completed.returncode:
                raise RuntimeError(f'{name} failed; see {work / (name + ".log")}')
        import importlib.metadata as metadata
        if not skip_expected_check:
            receipt['verified_lm_rows'] = compare_tsv(output / 'SLOTTED_lm_assignments.tsv', expected / 'lm_assignments.tsv',
                                                     {'winner_logit', 'margin', 'prob_margin'}, ('read_id',))
            receipt['verified_joint_rows'] = compare_tsv(output / 'SLOTTED_joint_predictions.tsv', expected / 'joint_predictions.tsv',
                                                        {'posterior'}, ('read_id', 'method'))
        receipt['packages'] = {name: metadata.version(name) for name in ('numpy', 'pysam', 'torch', 'pyarrow', 'pandas')}
        receipt['population'] = json.loads((work / 'lm_receipt.json').read_text())['population']
        receipt['joint_fit'] = json.loads((work / 'joint_fit.json').read_text())
        receipt['status'] = 'EXPECTED_OUTPUTS_MATCH' if not skip_expected_check else 'GENERATED_WITHOUT_EXPECTED_COMPARISON'
        receipt['seconds'] = round(time.monotonic() - started, 3)
        rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        receipt['largest_child_peak_rss_mib'] = round(rss / (1024 * 1024 if sys.platform == 'darwin' else 1024), 2)
        (work / 'README.md').write_text('Intermediate files and logs from run_chr21_demo.py. Final tables are ../SLOTTED_lm_assignments.tsv and ../SLOTTED_joint_predictions.tsv. Input preparation and candidate scoring files support troubleshooting; demo_receipt.json records parameters, versions and expected-output comparison.\n')
        print(json.dumps({'status': receipt['status'], 'seconds': receipt['seconds'],
                          'joint_fit_converged': receipt['joint_fit']['em_converged'],
                          'reads': receipt['population']['reads_total'], 'output': str(output)}, indent=2))
    except Exception as error:
        receipt['status'] = 'FAILED'
        receipt['error'] = str(error)
        raise
    finally:
        (work / 'demo_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, help='new output directory; default: timestamped directory under demo_output/')
    parser.add_argument('--skip-expected-check', action='store_true',
                        help='maintainer bootstrap only: generate outputs without claiming verification')
    args = parser.parse_args(argv)
    output = args.out or Path('demo_output') / (
        'chr21_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + secrets.token_hex(3))
    run(ROOT, output.resolve(), skip_expected_check=args.skip_expected_check)


if __name__ == '__main__':
    main()
