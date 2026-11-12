"""User entrypoint: automatic orchestration, safe output and actionable failure."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def app():
    path = Path(__file__).resolve().parents[1] / 'scripts/slotted.py'
    spec = importlib.util.spec_from_file_location('slotted_entry_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(tmp_path):
    args = []
    for name in ('bam', 'reads', 'rmsk', 'genome', 'ckpt', 'pool-params'):
        path = tmp_path / (name + ' with spaces')
        path.write_text('software test fixture')
        args += ['--' + name, str(path)]
    return args


def test_one_command_creates_structure_and_runs_all_stages(app, tmp_path, monkeypatch):
    args = inputs(tmp_path)
    out = tmp_path / 'new results'
    stages = []
    def subprocess_run(command, **kwargs):
        stage = command[2]
        stages.append(stage)
        target = Path(command[command.index('--out') + 1])
        assert target.parent.exists()
        target.write_text('test output')
        if stage == 'fit':
            Path(command[command.index('--receipt') + 1]).write_text(json.dumps({'fit_status': 'CONVERGED'}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(app.subprocess, 'run', subprocess_run)
    assert app.main(args + ['--out', str(out)]) == 0
    assert stages == ['prepare', 'score', 'merge', 'fit']
    assert {p.name for p in out.iterdir()} == {'SLOTTED_joint_predictions.tsv', 'intermediates'}
    assert json.loads((out/'intermediates/run.json').read_text())['status'] == 'COMPLETED'
    with pytest.raises(FileExistsError):
        app.main(args + ['--out', str(out)])


def test_stage_failure_stops_workflow_and_keeps_log(app, tmp_path, monkeypatch):
    args = inputs(tmp_path)
    out = tmp_path / 'failed'
    stages = []
    def subprocess_run(command, **kwargs):
        stages.append(command[2])
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(app.subprocess, 'run', subprocess_run)
    with pytest.raises(RuntimeError, match='prepare failed'):
        app.main(args + ['--out', str(out)])
    assert stages == ['prepare']
    assert json.loads((out/'intermediates/run.json').read_text())['status'] == 'FAILED'
    assert not (out/'SLOTTED_joint_predictions.tsv').exists()
