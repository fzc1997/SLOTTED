"""Joint stage contracts on toy populations, without loading a checkpoint."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["read_id", "locus_id", "score_joint", "seed_count_guard_4096cap"]
RESULT_FIELDS = ["read_id", "n_candidates", "method", "fit_converged", "fit_status", "winner",
                 "posterior", "called", "report_reason"]


@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location(
        "slotted_joint_cli_test", ROOT / "scripts/joint_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bundle(tmp_path):
    data = {
        "ids": ["read_a", "read_b", "read_none"],
        "fit_ids": ["read_a", "read_b"],
        "seqs": {q: "ACGT" * 12 for q in ("read_a", "read_b", "read_none")},
        "cands": {"read_a": ["L1", "L2"], "read_b": ["L2"], "read_none": []},
        "windows": {"L1": "ACGT" * 12, "L2": "TGCA" * 12},
    }
    path = tmp_path / "prepared.pkl"
    path.write_bytes(pickle.dumps(data, protocol=4))
    return path, data


@pytest.fixture
def rows():
    return [dict(zip(FIELDS, values)) for values in [
        ("read_a", "L1", -0.1, 8),
        ("read_a", "L2", -3.0, 1),
        ("read_b", "L2", -0.1, 7),
    ]]


def provenance(bundle_path):
    return {"bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
            "checkpoint_sha256": "a" * 64, "pool_sha256": "b" * 64,
            "device": "cpu", "precision": "float32", "stream_batch_size": 4,
            "configuration": "joint_v1_pool_v2_em"}


def write_channels(path, rows, metadata, *, seed_type=None):
    # Write the external format independently of the CLI's serialization helper.
    schema = pa.schema([
        ("read_id", pa.string()), ("locus_id", pa.string()),
        ("score_joint", pa.float64()),
        ("seed_count_guard_4096cap", seed_type if seed_type is not None else pa.int64()),
    ])
    table = pa.Table.from_pylist(rows, schema=schema)
    if metadata is not None:
        table = table.replace_schema_metadata({
            b"slotted.scoring.v1": json.dumps(metadata, sort_keys=True).encode(),
        })
    pq.write_table(table, path)


def run_fit(cli, bundle_path, channels, output):
    cli.main(["fit", "--d", str(bundle_path), "--channels", str(channels), "--out", str(output)])


def test_fit_preserves_candidate_and_no_candidate_populations(cli, bundle, rows, tmp_path):
    bundle_path, _ = bundle
    channels, output = tmp_path / "channels.parquet", tmp_path / "predictions.parquet"
    metadata = provenance(bundle_path)
    write_channels(channels, rows, metadata)
    run_fit(cli, bundle_path, channels, output)
    predictions = pq.read_table(output).to_pylist()
    assert len(predictions) == 6
    assigned = {r["read_id"]: r for r in predictions if r["method"] == "CF_LM_v1_EM"}
    assert assigned["read_a"]["winner"] == "L1"
    assert assigned["read_b"]["winner"] == "L2"
    for row in predictions:
        if row["read_id"] == "read_none":
            assert row["n_candidates"] == 0
            assert row["called"] is False
            assert row["winner"] == ""
            assert row["report_reason"] == "NO_CANDIDATE"
    receipt = json.loads(Path(str(output) + ".json").read_text())
    assert receipt["n_ids"] == 3 and receipt["n_fit"] == 2
    assert receipt["n_winner"] == 2 and receipt["n_unique_guard"] == 2
    assert receipt["scoring_provenance"] == metadata


@pytest.mark.parametrize("indices", [[0, 2], [0, 1], [0, 0, 1, 2], [1, 0, 2]])
def test_fit_rejects_missing_duplicate_or_reordered_candidate_channels(
        cli, bundle, rows, tmp_path, indices):
    bundle_path, _ = bundle
    channels, output = tmp_path / "bad.parquet", tmp_path / "predictions.parquet"
    write_channels(channels, [rows[i] for i in indices], provenance(bundle_path))
    with pytest.raises(ValueError, match="candidate channels"):
        run_fit(cli, bundle_path, channels, output)
    assert not output.exists()


@pytest.mark.parametrize("field,value", [("read_id", "unknown_read"), ("locus_id", "unknown_locus")])
def test_fit_rejects_unknown_read_or_candidate(cli, bundle, rows, tmp_path, field, value):
    bundle_path, _ = bundle
    rows[0][field] = value
    channels, output = tmp_path / "bad.parquet", tmp_path / "predictions.parquet"
    write_channels(channels, rows, provenance(bundle_path))
    with pytest.raises(ValueError, match="unknown read/candidate"):
        run_fit(cli, bundle_path, channels, output)
    assert not output.exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), None])
def test_fit_rejects_nonfinite_or_missing_emissions(cli, bundle, rows, tmp_path, value):
    bundle_path, _ = bundle
    rows[0]["score_joint"] = value
    channels = tmp_path / "bad.parquet"
    write_channels(channels, rows, provenance(bundle_path))
    with pytest.raises(ValueError, match="channel score"):
        run_fit(cli, bundle_path, channels, tmp_path / "predictions.parquet")


@pytest.mark.parametrize("value,seed_type", [
    (-1, pa.int64()), (0.5, pa.float64()), (None, pa.int64()), (True, pa.bool_()),
])
def test_fit_rejects_invalid_seed_counts(cli, bundle, rows, tmp_path, value, seed_type):
    bundle_path, _ = bundle
    if pa.types.is_boolean(seed_type):
        for row in rows:
            row["seed_count_guard_4096cap"] = False
    rows[0]["seed_count_guard_4096cap"] = value
    channels = tmp_path / "bad.parquet"
    write_channels(channels, rows, provenance(bundle_path), seed_type=seed_type)
    with pytest.raises(ValueError, match="seed count"):
        run_fit(cli, bundle_path, channels, tmp_path / "predictions.parquet")


def test_fit_rejects_channels_from_a_different_bundle(cli, bundle, rows, tmp_path):
    bundle_path, _ = bundle
    metadata = provenance(bundle_path)
    metadata["bundle_sha256"] = "0" * 64
    channels = tmp_path / "bad.parquet"
    write_channels(channels, rows, metadata)
    with pytest.raises(ValueError, match="different prepared bundle"):
        run_fit(cli, bundle_path, channels, tmp_path / "predictions.parquet")


def test_merge_accepts_absolute_globs_and_identical_overlapping_shards(cli, bundle, rows, tmp_path):
    bundle_path, _ = bundle
    metadata = provenance(bundle_path)
    write_channels(tmp_path / "shard1.parquet", rows[:2], metadata)
    write_channels(tmp_path / "shard2.parquet", rows[1:], metadata)
    output = tmp_path / "merged.parquet"
    cli.main(["merge", "--d", str(bundle_path), "--shards", str(tmp_path / "shard*.parquet"),
              "--out", str(output)])
    table = pq.read_table(output)
    assert table.to_pylist() == rows
    assert json.loads(table.schema.metadata[b"slotted.scoring.v1"]) == metadata


def test_merge_rejects_no_match_glob(cli, bundle, tmp_path):
    bundle_path, _ = bundle
    with pytest.raises(ValueError, match="no shard files match"):
        cli.main(["merge", "--d", str(bundle_path), "--shards", str(tmp_path / "missing*.parquet"),
                  "--out", str(tmp_path / "merged.parquet")])


@pytest.mark.parametrize("field,value", [("checkpoint_sha256", "c" * 64),
                                         ("pool_sha256", "d" * 64), ("precision", "bfloat16")])
def test_merge_rejects_incompatible_scoring_metadata(cli, bundle, rows, tmp_path, field, value):
    bundle_path, _ = bundle
    metadata = provenance(bundle_path)
    write_channels(tmp_path / "shard1.parquet", rows[:2], metadata)
    write_channels(tmp_path / "shard2.parquet", rows[2:], {**metadata, field: value})
    with pytest.raises(ValueError, match="different scoring provenance"):
        cli.main(["merge", "--d", str(bundle_path), "--shards", str(tmp_path / "shard*.parquet"),
                  "--out", str(tmp_path / "merged.parquet")])


def test_merge_rejects_conflicting_duplicate_scores(cli, bundle, rows, tmp_path):
    bundle_path, _ = bundle
    metadata = provenance(bundle_path)
    write_channels(tmp_path / "shard1.parquet", rows, metadata)
    conflicting = {**rows[0], "score_joint": rows[0]["score_joint"] + 0.01}
    write_channels(tmp_path / "shard2.parquet", [conflicting], metadata)
    with pytest.raises(ValueError, match="redundant shards disagree"):
        cli.main(["merge", "--d", str(bundle_path), "--shards", str(tmp_path / "shard*.parquet"),
                  "--out", str(tmp_path / "merged.parquet")])


@pytest.mark.parametrize("text,message", [("unknown_read\n", "unknown read IDs"),
                                         ("read_a\nread_a\n", "duplicate read IDs")])
def test_score_rejects_invalid_shard_ids_before_model_loading(
        cli, bundle, tmp_path, monkeypatch, text, message):
    from tracete_lm import pipeline_core
    bundle_path, _ = bundle
    ids_path = tmp_path / "ids.txt"
    ids_path.write_text(text)
    monkeypatch.setattr(pipeline_core, "score", lambda *a, **kw: pytest.fail("invalid shard reached scorer"))
    with pytest.raises(ValueError, match=message):
        cli.main(["score", "--d", str(bundle_path), "--ids", str(ids_path),
                  "--out", str(tmp_path / "scores.parquet")])


@pytest.mark.parametrize("ids", [[], ["no_candidate"]])
def test_empty_and_all_no_candidate_fits_produce_complete_outputs(cli, tmp_path, monkeypatch, ids):
    from tracete_lm import research_em
    monkeypatch.setattr(research_em, "em_locus_assignment",
                        lambda *a, **kw: pytest.fail("empty population attempted an EM fit"))
    data = {"ids": ids, "fit_ids": [], "seqs": {q: "ACGT" for q in ids},
            "cands": {q: [] for q in ids}, "windows": {}}
    bundle_path = tmp_path / "empty.pkl"
    bundle_path.write_bytes(pickle.dumps(data, protocol=4))
    channels, output = tmp_path / "empty.parquet", tmp_path / "predictions.parquet"
    write_channels(channels, [], provenance(bundle_path))
    run_fit(cli, bundle_path, channels, output)
    table = pq.read_table(output)
    assert set(table.column_names) == set(RESULT_FIELDS)
    assert table.num_rows == 2 * len(ids)
    for row in table.to_pylist():
        assert row["read_id"] == ids[0]
        assert row["called"] is False and row["winner"] == ""
        assert row["report_reason"] == "NO_CANDIDATE"
        assert row["fit_status"] == "NO_CANDIDATE_READS"
        assert row["fit_converged"] is None
    receipt = json.loads(Path(str(output) + ".json").read_text())
    assert receipt["n_ids"] == len(ids)
    assert receipt["n_fit"] == receipt["n_winner"] == receipt["n_unique_guard"] == 0
    assert receipt["em_iterations"] == 0 and receipt["em_converged"] is None


def test_pickle_global_objects_are_rejected(cli, bundle, tmp_path):
    _, data = bundle
    data["unexpected_object"] = datetime(2000, 1, 1)
    path = tmp_path / "object.pkl"
    path.write_bytes(pickle.dumps(data, protocol=4))
    with pytest.raises(pickle.UnpicklingError, match="primitive data only"):
        cli.load_bundle(path)


@pytest.mark.parametrize("mutation", ["duplicate_ids", "duplicate_candidates", "missing_window", "wrong_fit_ids"])
def test_invalid_bundle_population_is_rejected(cli, bundle, tmp_path, mutation):
    _, original = bundle
    data = deepcopy(original)
    if mutation == "duplicate_ids":
        data["ids"].append("read_a")
    elif mutation == "duplicate_candidates":
        data["cands"]["read_a"].append("L1")
    elif mutation == "missing_window":
        del data["windows"]["L1"]
    else:
        data["fit_ids"] = ["read_a"]
    path = tmp_path / "bad.pkl"
    path.write_bytes(pickle.dumps(data, protocol=4))
    with pytest.raises(ValueError):
        cli.load_bundle(path)


def test_fit_writes_top_level_tsv_with_separate_diagnostics(cli, bundle, rows, tmp_path):
    import csv
    bundle_path, _ = bundle
    work = tmp_path / 'intermediates'
    work.mkdir()
    channels = work / 'channels.parquet'
    write_channels(channels, rows, provenance(bundle_path))
    output = tmp_path / 'SLOTTED_joint_predictions.tsv'
    receipt = work / 'joint_fit.json'
    cli.main(['fit', '--d', str(bundle_path), '--channels', str(channels),
              '--out', str(output), '--receipt', str(receipt)])
    with output.open() as handle:
        predictions = list(csv.DictReader(handle, delimiter='\t'))
    assert len(predictions) == 6
    assert {r['read_id'] for r in predictions} == {'read_a', 'read_b', 'read_none'}
    assert json.loads(receipt.read_text())['n_fit'] == 2
    assert not Path(str(output) + '.json').exists()
    assert sorted(p.name for p in tmp_path.iterdir() if p.suffix == '.tsv') == ['SLOTTED_joint_predictions.tsv']


def test_fit_never_overwrites_an_existing_result(cli, tmp_path):
    output = tmp_path / 'SLOTTED_joint_predictions.tsv'
    output.write_text('preserve result')
    with pytest.raises(FileExistsError):
        cli.main(['fit', '--d', 'missing.pkl', '--channels', 'missing.parquet', '--out', str(output)])
    assert output.read_text() == 'preserve result'


def test_channel_write_is_exclusive_even_without_cli_preflight(cli, tmp_path):
    output = tmp_path / 'scores.parquet'
    output.write_bytes(b'preserve scores')
    with pytest.raises(FileExistsError):
        cli._write_channels([], output)
    assert output.read_bytes() == b'preserve scores'
