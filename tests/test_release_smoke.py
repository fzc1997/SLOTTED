"""Run the distributed synthetic fixture through the public commands."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'assets/ckpt_step30000.pt'


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason='model checkpoint is distributed with the repository, not the sdist')
def test_public_commands_end_to_end(tmp_path):
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}

    def run(script, *args):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts' / script), *map(str, args)],
                                cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    inputs = ['--bam', ROOT / 'examples/aln.bam', '--reads', ROOT / 'examples/reads.fa',
              '--rmsk', ROOT / 'examples/repeats.rmsk.txt', '--genome', ROOT / 'examples/reference.fa',
              '--flank', '20']
    winner_path, receipt_path = tmp_path / 'winners.tsv', tmp_path / 'receipt.json'
    run('assign_reads_m2d.py', *inputs, '--ckpt', CHECKPOINT, '--out', winner_path,
        '--receipt', receipt_path, '--device', 'cpu', '--band-backend', 'python', '--verify-exact')
    assert winner_path.read_text() == (ROOT / 'examples/expected_winners.tsv').read_text()
    receipt = json.loads(receipt_path.read_text())
    assert receipt['population']['reads_total'] == 3
    assert receipt['population']['reads_with_candidates'] == 2
    assert receipt['population']['no_candidate_reads'] == 1
    assert receipt['verification']['still_bad'] == []

    data, score, channels = (tmp_path / name for name in ('data.pkl', 'score.parquet', 'channels.parquet'))
    predictions = tmp_path / 'predictions.parquet'
    run('joint_pipeline.py', 'prepare', *inputs, '--out', data)
    run('joint_pipeline.py', 'score', '--d', data, '--out', score)
    run('joint_pipeline.py', 'merge', '--d', data, '--shards', score, '--out', channels)
    run('joint_pipeline.py', 'fit', '--d', data, '--channels', channels, '--out', predictions)
    frame = pd.read_parquet(predictions)
    assert len(frame) == 6
    assert set(frame['read_id']) == {'read_single', 'read_multi', 'read_outside'}
    assert set(frame['method']) == {'CF_LM_v1_EM', 'CF_LM_v1_EM_guard'}
    outside = frame[frame['read_id'] == 'read_outside']
    assert not outside['called'].any()
    assert set(outside['report_reason']) == {'NO_CANDIDATE'}
    base = frame[frame['method'] == 'CF_LM_v1_EM']
    assert base['called'].sum() == 2
    guard = frame[frame['method'] == 'CF_LM_v1_EM_guard']
    assert guard.set_index('read_id').loc['read_single', 'called']
    assert not guard.set_index('read_id').loc['read_multi', 'called']
    meta = json.loads(Path(str(predictions) + '.json').read_text())
    assert meta['em_converged'] is True
    assert meta['n_fit'] == 2 and meta['n_unique_guard'] == 1


@pytest.mark.parametrize('probabilities,seeds', [
    ([float('nan'), 0.2], [10, 0]), ([float('inf')], [5]),
    ([1.1], [5]), ([-0.1, 1.1], [1, 10]), ([1.0], [-1]), ([1.0], [5.5]),
])
def test_guard_rejects_invalid_evidence(probabilities, seeds):
    from tracete_lm.selective import decide_selective
    with pytest.raises(ValueError):
        decide_selective(probabilities, seeds)


@pytest.mark.parametrize('margin', [float('nan'), float('inf'), -0.1, 1.1])
def test_guard_rejects_invalid_margin(margin):
    from tracete_lm.selective import decide_selective
    with pytest.raises(ValueError):
        decide_selective([1.0], [5], margin=margin)
