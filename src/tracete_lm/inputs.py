"""Validated FASTA input shared by the public assignment commands."""
from __future__ import annotations

import gzip
from pathlib import Path


def read_fasta(path: str | Path) -> dict[str, str]:
    """Read plain/gzipped FASTA, preserving unique IDs and normalizing case.

    Empty files are valid empty populations; duplicate IDs and empty records
    are errors because silently replacing either would change the population.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    sequences: dict[str, list[str]] = {}
    name = None
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                fields = line[1:].split()
                if not fields:
                    raise ValueError(f"empty FASTA identifier at line {line_number}")
                name = fields[0]
                if name in sequences:
                    raise ValueError(f"duplicate FASTA identifier: {name}")
                sequences[name] = []
            elif name is None:
                raise ValueError(f"sequence before FASTA header at line {line_number}")
            else:
                sequence = line.upper()
                if not set(sequence) <= set("ACGTN"):
                    raise ValueError(f"invalid DNA sequence for {name} at line {line_number}: use ACGTN")
                sequences[name].append(sequence)
    result = {key: "".join(parts) for key, parts in sequences.items()}
    empty = [key for key, sequence in result.items() if not sequence]
    if empty:
        raise ValueError(f"empty FASTA sequence: {empty[0]}")
    return result
