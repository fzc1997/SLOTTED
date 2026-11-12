"""The download-and-run entrypoint must detect incomplete or altered bundles."""
import hashlib
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def demo():
    path = Path(__file__).resolve().parents[1] / 'scripts/run_chr21_demo.py'
    spec = importlib.util.spec_from_file_location('slotted_demo_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checksum_verification_detects_modified_or_missing_input(demo, tmp_path):
    data = tmp_path / 'input.txt'
    data.write_text('original\n')
    (tmp_path / 'SHA256SUMS').write_text(hashlib.sha256(data.read_bytes()).hexdigest() + '  input.txt\n')
    assert demo.verify_checksums(tmp_path) == 1
    data.write_text('modified\n')
    with pytest.raises(ValueError, match='checksum mismatch'):
        demo.verify_checksums(tmp_path)
    data.unlink()
    with pytest.raises(FileNotFoundError, match='missing'):
        demo.verify_checksums(tmp_path)


@pytest.mark.parametrize('name', ['../outside.txt', '/outside.txt'])
def test_checksum_paths_stay_inside_bundle(demo, tmp_path, name):
    (tmp_path / 'SHA256SUMS').write_text('0' * 64 + '  ' + name + '\n')
    with pytest.raises(ValueError, match='unsafe path'):
        demo.verify_checksums(tmp_path)


def test_checksum_manifest_must_cover_consumed_files(demo, tmp_path):
    (tmp_path / 'SHA256SUMS').write_text('')
    with pytest.raises(ValueError, match='empty or lacks'):
        demo.verify_checksums(tmp_path)
    data = tmp_path / 'one.txt'
    data.write_text('one')
    line = hashlib.sha256(data.read_bytes()).hexdigest() + '  one.txt\n'
    (tmp_path / 'SHA256SUMS').write_text(line)
    with pytest.raises(ValueError, match='empty or lacks'):
        demo.verify_checksums(tmp_path, {'one.txt', 'missing.txt'})
    (tmp_path / 'SHA256SUMS').write_text(line + line)
    with pytest.raises(ValueError, match='duplicate checksum'):
        demo.verify_checksums(tmp_path)


def test_comparison_allows_roundoff_but_rejects_changed_call(demo, tmp_path):
    expected, actual = tmp_path / 'expected.tsv', tmp_path / 'actual.tsv'
    expected.write_text('read_id\tposterior\tcall\nr1\t0.75\tUNIQUE\n')
    actual.write_text('read_id\tposterior\tcall\nr1\t0.75000001\tUNIQUE\n')
    assert demo.compare_tsv(actual, expected, {'posterior'}, ('read_id',)) == 1
    actual.write_text('read_id\tposterior\tcall\nr1\t0.75\tABSTAIN\n')
    with pytest.raises(ValueError, match='mismatch'):
        demo.compare_tsv(actual, expected, {'posterior'}, ('read_id',))


def test_comparison_rejects_missing_or_duplicate_rows(demo, tmp_path):
    expected, actual = tmp_path / 'expected.tsv', tmp_path / 'actual.tsv'
    expected.write_text('read_id\tvalue\nr1\t1\n')
    actual.write_text('read_id\tvalue\n')
    with pytest.raises(ValueError, match='population'):
        demo.compare_tsv(actual, expected, {'value'}, ('read_id',))
    actual.write_text('read_id\tvalue\nr1\t1\nr1\t1\n')
    with pytest.raises(ValueError, match='duplicate'):
        demo.compare_tsv(actual, expected, {'value'}, ('read_id',))


@pytest.mark.parametrize('row', ['r1\t1\tUNEXPECTED\n', 'r1\n'])
def test_comparison_rejects_malformed_row_width(demo, tmp_path, row):
    expected, actual = tmp_path / 'expected.tsv', tmp_path / 'actual.tsv'
    expected.write_text('read_id\tvalue\nr1\t1\n')
    actual.write_text('read_id\tvalue\n' + row)
    with pytest.raises(ValueError, match='malformed TSV'):
        demo.compare_tsv(actual, expected, {'value'}, ('read_id',))


def test_existing_demo_output_is_never_overwritten(demo, tmp_path):
    output = tmp_path / 'existing'
    output.mkdir()
    sentinel = output / 'result.txt'
    sentinel.write_text('keep')
    with pytest.raises(FileExistsError):
        demo.run(tmp_path, output, skip_expected_check=True)
    assert sentinel.read_text() == 'keep'


def test_default_runs_receive_distinct_timestamped_directories(demo, tmp_path, monkeypatch):
    import re
    monkeypatch.chdir(tmp_path)
    tokens = iter(['a1b2c3', 'd4e5f6'])
    monkeypatch.setattr(demo.secrets, 'token_hex', lambda n: next(tokens))
    destinations = []
    monkeypatch.setattr(demo, 'run', lambda root, output, **kw: destinations.append(output))
    demo.main([])
    demo.main([])
    assert destinations[0] != destinations[1]
    for path in destinations:
        assert path.parent == tmp_path / 'demo_output'
        assert re.fullmatch(r'chr21_\d{8}_\d{6}_[0-9a-f]{6}', path.name)
