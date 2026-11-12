"""Invalid user calibration must fail before it changes candidate ranking."""
from copy import deepcopy
import json
from pathlib import Path
import pickle
import importlib.util

import pytest

from tracete_lm.pipeline_core import score

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('field,value', [
    ('rho_star', 0), ('rho_star', -1), ('rho_star', float('nan')),
    ('rho_star', float('inf')), ('rho_star', True),
    ('beta0_closed_form', float('nan')), ('beta0_closed_form', float('inf')),
    ('temperature', 0), ('temperature', -1), ('temperature', float('nan')),
    ('temperature', float('inf')), ('temperature', '1.0'),
])
def test_invalid_pool_parameters_fail_before_scoring(tmp_path, field, value):
    pool = json.loads((ROOT / 'assets/pool_frozen_params_v2.json').read_text())
    if field == 'temperature':
        pool['temperatures']['m2d'] = value
    else:
        pool[field] = value
    path = tmp_path / 'pool.json'
    path.write_text(json.dumps(pool))
    empty = {'fit_ids': [], 'cands': {}, 'windows': {}, 'seqs': {}}
    with pytest.raises(ValueError, match='finite'):
        score(ROOT, empty, ['joint_v1_pool_v2_em'], pool_params=path)


def test_frozen_pool_accepts_an_empty_population():
    result = score(ROOT, {'fit_ids': [], 'cands': {}, 'windows': {}, 'seqs': {}}, ['joint_v1_pool_v2_em'])
    assert result['joint'] == {}


@pytest.mark.parametrize('field,value', [('seqs', 'acgt'), ('seqs', 'ACGR'),
                                         ('windows', 'acgt'), ('windows', 'ACGR')])
def test_primitive_bundles_cannot_bypass_dna_validation(tmp_path, field, value):
    spec = importlib.util.spec_from_file_location('joint_bundle_validation', ROOT / 'scripts/joint_pipeline.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = {'ids': ['r'], 'fit_ids': ['r'], 'seqs': {'r': 'ACGT'},
            'cands': {'r': ['L']}, 'windows': {'L': 'ACGT'}}
    data[field]['r' if field == 'seqs' else 'L'] = value
    path = tmp_path / 'input.pkl'
    path.write_bytes(pickle.dumps(data))
    with pytest.raises(ValueError, match='uppercase ACGTN'):
        module.load_bundle(path)
