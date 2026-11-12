"""Corpus edge loading, identity, and empty-fit regression coverage."""
from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location("slotted_em_cli_test", ROOT / "scripts/fit_em_corpus.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_shard(tmp_path, name="a", *, loci=("A", "B"), arrays=None, legacy=False):
    path = tmp_path / f"{name}.npz"
    values = {"edge_scores": np.array([0.0, 0.0], dtype=np.float32),
              "edge_locus": np.array([0, 1], dtype=np.int32),
              "read_start": np.array([0, 2], dtype=np.int64),
              "read_ids": np.array([name], dtype=np.str_)}
    values.update(arrays or {})
    if legacy:
        del values["read_ids"]
    np.savez(path, **values)
    Path(str(path) + ".loci.tsv").write_text("".join(f"{i}\t{locus}\n" for i, locus in enumerate(loci)))
    return path


def test_shard_boundaries_are_not_zero_candidate_reads(cli, tmp_path):
    a = write_shard(tmp_path, "a", loci=("A",), arrays={
        "edge_scores": np.array([0.0]), "edge_locus": np.array([0]), "read_start": np.array([0, 1]),
    })
    b = write_shard(tmp_path, "b", loci=("B",), arrays={
        "edge_scores": np.array([0.0]), "edge_locus": np.array([0]),
        "read_start": np.array([0, 0, 1]), "read_ids": np.array(["zero", "b"]),
    })
    c = write_shard(tmp_path, "c", loci=(), arrays={
        "edge_scores": np.empty(0), "edge_locus": np.empty(0, dtype=np.int64),
        "read_start": np.array([0]), "read_ids": np.array([], dtype=np.str_),
    })
    corpus = cli.assemble_corpus([a, b, c])
    assert corpus["read_start"].tolist() == [0, 1, 2]
    assert corpus["edge_locus"].tolist() == [0, 1]
    assert corpus["read_ids"].tolist() == ["a", "b"]
    assert corpus["n_input_reads"] == 3
    assert corpus["n_zero_candidate_reads"] == 1
    assert corpus["read_id_check"] == "complete"


def test_overlapping_shards_are_rejected(cli, tmp_path):
    a = write_shard(tmp_path, "a", arrays={"read_ids": np.array(["same"])} )
    b = write_shard(tmp_path, "b", arrays={"read_ids": np.array(["same"])} )
    with pytest.raises(ValueError, match="occurs in multiple shards"):
        cli.assemble_corpus([a, b])


def test_duplicate_file_is_rejected_even_for_legacy_dump(cli, tmp_path):
    path = write_shard(tmp_path, legacy=True)
    with pytest.raises(ValueError, match="same shard file"):
        cli.assemble_corpus([path, path])


@pytest.mark.parametrize("all_legacy,expected", [(False, "partial"), (True, "unavailable")])
def test_legacy_shards_warn_and_record_identity_limitation(cli, tmp_path, all_legacy, expected):
    a = write_shard(tmp_path, "a", legacy=True)
    b = write_shard(tmp_path, "b", legacy=all_legacy)
    with pytest.warns(RuntimeWarning, match="disjointness cannot be checked"):
        corpus = cli.assemble_corpus([a, b])
    assert corpus["read_ids"] is None
    assert corpus["read_id_check"] == expected
    assert corpus["n_zero_candidate_reads"] == 0
    assert corpus["read_start"].tolist() == [0, 2, 4]


@pytest.mark.parametrize("arrays", [
    {"edge_scores": np.array([np.nan, 0.0])},
    {"edge_scores": np.array([np.inf, 0.0])},
    {"edge_scores": np.array([[0.0, 0.0]])},
    {"edge_scores": np.array([False, True])},
    {"edge_scores": np.array([0j, 0j])},
    {"edge_locus": np.array([0.0, 1.0])},
    {"edge_locus": np.array([-1, 1])},
    {"edge_locus": np.array([0, 2])},
    {"edge_locus": np.array([0, 2**64 - 1], dtype=np.uint64)},
    {"edge_locus": np.array([0])},
    {"edge_locus": np.array([0, 0])},
    {"read_start": np.array([0.0, 2.0])},
    {"read_start": np.array([], dtype=np.int64)},
    {"read_start": np.array([1, 2])},
    {"read_start": np.array([0, 1])},
    {"read_start": np.array([0, 2, 1, 2], dtype=np.uint64)},
    {"read_ids": np.array(["r1", "r2"])},
    {"read_ids": np.array([""])},
    {"read_ids": np.array([b"r1"])},
    {"read_ids": np.array(["same", "same"]), "read_start": np.array([0, 1, 2])},
])
def test_malformed_arrays_are_rejected(cli, tmp_path, arrays):
    path = write_shard(tmp_path, arrays=arrays)
    with pytest.raises(ValueError):
        cli.load_shard(path)


@pytest.mark.parametrize("content", ["0\tA\n2\tB\n", "0\tA\n1\tA\n", "0\t\n", "0.0\tA\n"])
def test_malformed_locus_table_is_rejected(cli, tmp_path, content):
    path = write_shard(tmp_path)
    Path(str(path) + ".loci.tsv").write_text(content)
    with pytest.raises(ValueError):
        cli.load_shard(path)


def test_missing_required_array_is_rejected(cli, tmp_path):
    path = write_shard(tmp_path)
    np.savez(path, edge_scores=np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="missing arrays"):
        cli.load_shard(path)


def test_empty_corpus_has_explicit_no_fit_outputs(cli, monkeypatch, tmp_path):
    write_shard(tmp_path, "empty", loci=(), arrays={
        "edge_scores": np.empty(0), "edge_locus": np.empty(0, dtype=np.int64),
        "read_start": np.array([0, 0]), "read_ids": np.array(["zero"]),
    })
    prefix = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["fit_em_corpus", "--edges-glob", str(tmp_path / "empty.npz"),
                                      "--out-prefix", str(prefix), "--save-gamma"])
    monkeypatch.setattr(cli, "em_locus_assignment", lambda *a, **kw: pytest.fail("empty corpus fitted EM"))
    assert cli.main() == 0
    diag = json.loads(Path(str(prefix) + ".em.json").read_text())
    assert diag["fit_status"] == "NO_CANDIDATE_READS"
    assert diag["converged"] is None
    assert diag["n_iter"] == 0
    assert diag["n_reads"] == 0
    assert diag["n_input_reads"] == diag["n_zero_candidate_reads"] == 1
    with gzip.open(str(prefix) + ".pi.tsv.gz", "rt") as handle:
        assert handle.read() == "locus_id\tpi\n"
    with np.load(str(prefix) + ".gamma.npz", allow_pickle=False) as gamma:
        assert gamma["gamma"].size == gamma["edge_locus"].size == gamma["read_ids"].size == 0
        assert gamma["read_start"].tolist() == [0]


def test_uniform_prior_output_retains_read_identity_and_csr(cli, monkeypatch, tmp_path):
    write_shard(tmp_path, "a")
    write_shard(tmp_path, "b", loci=("B",), arrays={
        "edge_scores": np.array([0.0]), "edge_locus": np.array([0]), "read_start": np.array([0, 1]),
    })
    prefix = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["fit_em_corpus", "--edges-glob", str(tmp_path / "*.npz"),
                                      "--out-prefix", str(prefix), "--save-gamma", "--no-em"])
    assert cli.main() == 0
    diag = json.loads(Path(str(prefix) + ".em.json").read_text())
    assert diag["n_reads"] == diag["n_input_reads"] == 2
    assert diag["n_zero_candidate_reads"] == 0
    assert diag["read_id_check"] == "complete"
    with np.load(str(prefix) + ".gamma.npz", allow_pickle=False) as gamma:
        np.testing.assert_allclose(gamma["gamma"], [0.5, 0.5, 1.0])
        assert gamma["edge_locus"].tolist() == [0, 1, 1]
        assert gamma["read_start"].tolist() == [0, 2, 3]
        assert gamma["read_ids"].tolist() == ["a", "b"]


@pytest.mark.parametrize("options", [
    ["--temperature", "0"], ["--temperature", "nan"], ["--max-iter", "0"],
    ["--tol", "-1"], ["--tol", "inf"],
])
def test_invalid_fit_options_fail_before_loading(cli, monkeypatch, tmp_path, options):
    monkeypatch.setattr(sys, "argv", ["fit_em_corpus", "--edges-glob", str(tmp_path / "*.npz"),
                                      "--out-prefix", str(tmp_path / "output"), *options])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_corpus_fit_refuses_existing_outputs(cli, tmp_path, monkeypatch):
    prefix = str(tmp_path / 'result')
    target = Path(prefix + '.em.json')
    target.write_text('preserve diagnostics')
    monkeypatch.setattr(sys, 'argv', ['fit_em_corpus', '--edges-glob', 'missing.npz', '--out-prefix', prefix])
    with pytest.raises(FileExistsError):
        cli.main()
    assert target.read_text() == 'preserve diagnostics'
