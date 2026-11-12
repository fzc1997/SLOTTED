"""Public FASTA parsing must preserve the input population without guessing."""
from __future__ import annotations

import gzip

import pytest

from tracete_lm.inputs import read_fasta


@pytest.mark.parametrize("compressed", [False, True])
def test_plain_and_gzip_fasta_normalize_case_and_join_lines(tmp_path, compressed):
    path = tmp_path / ("reads.fa.gz" if compressed else "reads.fa")
    opener = gzip.open if compressed else open
    with opener(path, "wt") as handle:
        handle.write("\n>read1 description\nacgt\nnN\n\n>read2\ntgca\n")
    assert read_fasta(path) == {"read1": "ACGTNN", "read2": "TGCA"}


@pytest.mark.parametrize("compressed", [False, True])
def test_empty_fasta_is_an_empty_population(tmp_path, compressed):
    path = tmp_path / ("empty.fa.gz" if compressed else "empty.fa")
    opener = gzip.open if compressed else open
    with opener(path, "wt") as handle:
        handle.write("\n \n")
    assert read_fasta(path) == {}


@pytest.mark.parametrize("text, message", [
    ("ACGT\n>r\nACGT\n", "before FASTA header"),
    (">\nACGT\n", "empty FASTA identifier"),
    (">r\n", "empty FASTA sequence"),
    (">r\nACGT\n>empty\n", "empty FASTA sequence"),
    (">empty\n>r\nACGT\n", "empty FASTA sequence"),
    (">r first description\nACGT\n>r second description\nTGCA\n", "duplicate FASTA identifier"),
    (">r\nAC GT\n", "invalid DNA sequence"),
    (">r\nACGU\n", "invalid DNA sequence"),
    (">r\nACG-\n", "invalid DNA sequence"),
])
def test_malformed_records_are_rejected(tmp_path, text, message):
    path = tmp_path / "invalid.fa"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        read_fasta(path)


@pytest.mark.parametrize("ambiguity", list("RYSWKMBDHV"))
def test_non_n_iupac_ambiguity_is_rejected(tmp_path, ambiguity):
    # Frozen CF/seed channels understand ACGTN, not ambiguity-code evidence.
    path = tmp_path / "ambiguous.fa"
    path.write_text(f">r\nACG{ambiguity.lower()}\n")
    with pytest.raises(ValueError, match="invalid DNA sequence"):
        read_fasta(path)
