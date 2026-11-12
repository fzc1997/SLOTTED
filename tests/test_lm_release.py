"""Release regressions for CLI receipts and inference verification contracts."""
from __future__ import annotations

from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location("slotted_lm_cli_test", ROOT / "scripts/assign_reads_m2d.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_args(**overrides):
    return SimpleNamespace(**({
        "verify_exact": False, "verify_exact_lazy": False,
        "batch_size": 3, "band_backend": "python",
        "max_candidates": 32, "max_rows": 2000, "lazy_margin": 0.05,
    } | overrides))


def command_args(tmp_path):
    return ["assign_reads_m2d", "--bam", str(ROOT / "examples/aln.bam"),
            "--reads", str(tmp_path / "reads.fa"),
            "--rmsk", str(ROOT / "examples/repeats.rmsk.txt"),
            "--genome", str(ROOT / "examples/reference.fa"),
            "--ckpt", str(tmp_path / "checkpoint.pt"),
            "--out", str(tmp_path / "winners.tsv"), "--device", "cpu",
            "--band-backend", "python"]


@pytest.mark.parametrize("options", [
    ["--max-candidates", "0"], ["--max-rows", "-1"], ["--batch-size", "0"],
    ["--flank", "-1"], ["--min-mapq", "256"], ["--lazy-margin", "nan"],
    ["--selective-margin", "inf"], ["--selective-margin", "1.1"],
    ["--verify-exact", "--verify-exact-lazy"],
])
def test_invalid_options_fail_before_input_access(cli, monkeypatch, tmp_path, options):
    monkeypatch.setattr(sys, "argv", command_args(tmp_path) + options)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert not (tmp_path / "winners.tsv").exists()


def test_no_candidate_run_writes_receipt_and_empty_edges(cli, monkeypatch, tmp_path):
    import torch

    (tmp_path / "reads.fa").write_text(">not_in_the_example_bam\n" + "acgt" * 12 + "\n")
    (tmp_path / "checkpoint.pt").write_bytes(b"not loaded when there are no candidates")
    receipt_path = tmp_path / "receipt.json"
    edge_path = tmp_path / "edges_without_extension"
    monkeypatch.setattr(sys, "argv", command_args(tmp_path) + [
        "--receipt", str(receipt_path), "--dump-edges", str(edge_path), "--selective",
    ])
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("no-candidate run loaded model"))
    assert cli.main() == 0
    assert len((tmp_path / "winners.tsv").read_text().splitlines()) == 1
    receipt = json.loads(receipt_path.read_text())
    assert receipt["population"] == {
        "reads_total": 1, "reads_with_candidates": 0, "no_candidate_reads": 1, "loci_used": 0,
    }
    assert receipt["selective"]["n"] == 0
    with np.load(edge_path) as edges:
        assert edges["edge_scores"].size == edges["edge_locus"].size == 0
        assert edges["read_start"].tolist() == [0]
    assert Path(str(edge_path) + ".loci.tsv").read_text() == ""


def test_lazy_flip_counter_counts_winner_changes(cli, monkeypatch):
    import tracete_lm.m2d_model as model_module

    groups = [("ACGT", ["ACGT", "TGCA"]), ("TGCA", ["ACGT", "TGCA"])]
    initial = [[0.10, 0.09], [1.0, 0.99]]
    replacements = iter([[0.11, 0.09], [0.9, 1.0]])
    monkeypatch.setattr(model_module, "batched_read_group_logits", lambda *a, **kw: initial)
    monkeypatch.setattr(model_module, "matrix2d_scores", lambda *a, **kw: next(replacements))
    scores, stats = cli._score_groups(object(), groups, score_args(verify_exact_lazy=True), "cpu")
    assert scores == [[0.11, 0.09], [0.9, 1.0]]
    assert stats["n_replaced"] == 2
    assert stats["n_flips"] == 1


def test_exact_reference_shares_cuda_dtype_and_batch_protocol(cli, monkeypatch):
    import torch
    import tracete_lm.m2d_model as model_module

    contexts = []

    def autocast(**kwargs):
        contexts.append(kwargs)
        return nullcontext()

    def reference(model, read, candidates, **kwargs):
        assert kwargs["batch_size"] == 3
        assert kwargs["band_backend"] == "python"
        return [1.0, 0.0]

    def verified(model, groups, references, **kwargs):
        assert references == [[1.0, 0.0]]
        assert kwargs["reference_batch_size"] == 3
        return references, {"n_flips": 0, "fallback_fixed": 0, "still_bad": []}

    monkeypatch.setattr(torch, "autocast", autocast)
    monkeypatch.setattr(model_module, "matrix2d_scores", reference)
    monkeypatch.setattr(model_module, "batched_group_logits_verified", verified)
    cli._score_groups(object(), [("ACGT", ["ACGT", "TGCA"])],
                      score_args(verify_exact=True), "cuda:0")
    assert contexts == [{"device_type": "cuda", "dtype": torch.bfloat16, "enabled": True}]


def test_exact_verification_fails_closed(cli, monkeypatch):
    import tracete_lm.m2d_model as model_module

    monkeypatch.setattr(model_module, "matrix2d_scores", lambda *a, **kw: [1.0, 0.0])
    monkeypatch.setattr(model_module, "batched_group_logits_verified", lambda *a, **kw: (
        [[0.0, 1.0]], {"n_flips": 1, "fallback_fixed": 0, "still_bad": [0]},
    ))
    with pytest.raises(RuntimeError, match="Exact verification failed"):
        cli._score_groups(object(), [("ACGT", ["ACGT", "TGCA"])],
                          score_args(verify_exact=True), "cpu")


@pytest.mark.parametrize("scores", [[[float("nan"), 0.0]], [[1.0]], []])
def test_invalid_scorer_output_is_rejected(cli, monkeypatch, scores):
    import tracete_lm.m2d_model as model_module

    monkeypatch.setattr(model_module, "batched_read_group_logits", lambda *a, **kw: scores)
    with pytest.raises(ValueError):
        cli._score_groups(object(), [("ACGT", ["ACGT", "TGCA"])], score_args(), "cpu")


def test_verified_fallback_uses_requested_chunk_size(monkeypatch):
    import torch
    import tracete_lm.m2d_model as model_module

    groups = [("ACGT", ["ACGT"] * 5)]
    references = [[0.0, 1.0, 0.0, 1.0, 0.0]]
    chunk_sizes = []
    monkeypatch.setattr(model_module, "batched_read_group_logits",
                        lambda *a, **kw: [[2.0, 1.0, 0.0, 1.0, 0.0]])
    monkeypatch.setattr(model_module, "candidate_group_bands", lambda *a, **kw: [None] * 5)

    def collate(bands):
        chunk_sizes.append(len(bands))
        return torch.arange(len(bands), dtype=torch.float32), torch.ones(len(bands), dtype=torch.bool)

    monkeypatch.setattr(model_module, "collate_bands", collate)
    scores, stats = model_module.batched_group_logits_verified(
        lambda x, mask: x, groups, references, device="cpu", reference_batch_size=2)
    assert chunk_sizes == [2, 2, 1]
    assert scores == references
    assert stats["fallback_fixed"] == 1
    assert stats["still_bad"] == []


@pytest.mark.parametrize("references", [[[]], [[float("nan")]]])
def test_verifier_rejects_missing_or_nonfinite_reference(references):
    from tracete_lm.m2d_model import batched_group_logits_verified

    with pytest.raises(ValueError, match="one finite score per candidate"):
        batched_group_logits_verified(object(), [("ACGT", ["ACGT"])], references, device="cpu")


def test_existing_assignment_is_rejected_before_input_loading(cli, tmp_path, monkeypatch):
    args = command_args(tmp_path)
    target = Path(args[args.index('--out') + 1])
    target.write_text('preserve original result')
    monkeypatch.setattr(sys, 'argv', args)
    with pytest.raises(FileExistsError):
        cli.main()
    assert target.read_text() == 'preserve original result'
