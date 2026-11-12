#!/usr/bin/env python3
"""Global EM fit over dumped SLOTTED edge scores (the EM arm, corpus-scale).

Inputs are the per-shard npz dumps written by `assign_reads_m2d.py --dump-edges`
(each with edge_scores f32 / edge_locus int32 into a shard-local index /
read_start int64 / read_ids Unicode) plus their `<npz>.loci.tsv` index tables. Shards partition
reads, so concatenation is exact; locus ids are remapped to one global index,
then a single `em_locus_assignment` fit runs over the whole corpus.

Outputs (prefix): <prefix>.pi.tsv.gz (locus_id, pi), <prefix>.em.json
(diagnostics), and with --save-gamma <prefix>.gamma.npz (f32 responsibilities,
read-major CSR order matching the concatenated edge arrays).
Zero-candidate reads are omitted from the fit and counted explicitly. If all
shards are empty, no fit is performed and the pi table contains only its header.
Legacy dumps without read_ids are accepted with a warning: their disjointness
cannot be checked. Gamma output includes read_ids when all input shards do.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent.parent / "src"
for p in (str(here),):
    if p not in sys.path:
        sys.path.insert(0, p)
from slotted_em import em_locus_assignment  # noqa: E402


def load_shard(path: str | Path) -> dict:
    """Read one edge dump, rejecting lossy index casts and malformed CSR."""
    path = Path(path)
    loci_path = Path(str(path) + ".loci.tsv")
    loci = []
    seen_loci = set()
    with loci_path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 2 or not fields[1].strip():
                raise ValueError(f"{loci_path}:{line_number}: expected index and nonempty locus ID")
            try:
                index = int(fields[0])
            except ValueError as error:
                raise ValueError(f"{loci_path}:{line_number}: locus index must be an integer") from error
            if index != len(loci) or fields[1] in seen_loci:
                raise ValueError(f"{loci_path}:{line_number}: locus indices must be contiguous and IDs unique")
            loci.append(fields[1])
            seen_loci.add(fields[1])

    with np.load(path, allow_pickle=False) as arrays:
        required = {"edge_scores", "edge_locus", "read_start"}
        if not required <= set(arrays.files):
            raise ValueError(f"{path}: missing arrays {sorted(required - set(arrays.files))}")
        scores = arrays["edge_scores"]
        edge_locus = arrays["edge_locus"]
        starts = arrays["read_start"]
        read_ids = arrays["read_ids"] if "read_ids" in arrays.files else None

    if scores.ndim != 1 or scores.dtype.kind not in "fiu" or not np.isfinite(scores).all():
        raise ValueError(f"{path}: edge_scores must be a one-dimensional finite numeric array")
    if edge_locus.ndim != 1 or edge_locus.dtype.kind not in "iu":
        raise ValueError(f"{path}: edge_locus must be a one-dimensional integer array")
    if len(edge_locus) != len(scores) or np.any(edge_locus < 0) or np.any(edge_locus >= len(loci)):
        raise ValueError(f"{path}: edge_locus must align with scores and index the locus table")
    if starts.ndim != 1 or starts.dtype.kind not in "iu" or len(starts) == 0:
        raise ValueError(f"{path}: read_start must be a nonempty one-dimensional integer array")
    if starts[0] != 0 or starts[-1] != len(scores) or np.any(starts[1:] < starts[:-1]):
        raise ValueError(f"{path}: read_start must start at zero, be nondecreasing, and end at the edge count")
    for start, end in zip(starts[:-1], starts[1:]):
        group = edge_locus[int(start):int(end)]
        if len(set(group.tolist())) != len(group):
            raise ValueError(f"{path}: a read has repeated candidate loci")
    if read_ids is not None:
        if read_ids.ndim != 1 or read_ids.dtype.kind != "U" or len(read_ids) != len(starts) - 1:
            raise ValueError(f"{path}: read_ids must contain one Unicode ID per CSR read")
        names = read_ids.tolist()
        if any(not name.strip() for name in names) or len(set(names)) != len(names):
            raise ValueError(f"{path}: read_ids must be nonempty and unique within the shard")

    with np.errstate(over="ignore", invalid="ignore"):
        scores = scores.astype(np.float64, copy=False)
    if not np.isfinite(scores).all():
        raise ValueError(f"{path}: edge_scores cannot be represented as finite float64 values")
    return {"scores": scores,
            "edge_locus": edge_locus.astype(np.int64, copy=False),
            "starts": starts.astype(np.int64, copy=False),
            "loci": loci, "read_ids": read_ids}


def assemble_corpus(files: list[str | Path]) -> dict:
    """Concatenate read counts, avoiding artificial reads at shard boundaries."""
    if not files:
        raise ValueError("at least one shard is required")
    if len({Path(path).resolve() for path in files}) != len(files):
        raise ValueError("the same shard file was supplied more than once")
    global_locus = {}
    score_chunks, locus_chunks, count_chunks, id_chunks = [], [], [], []
    seen_reads = set()
    n_input_reads = n_zero = legacy_shards = 0
    for path in files:
        shard = load_shard(path)
        remap = []
        for locus in shard["loci"]:
            if locus not in global_locus:
                global_locus[locus] = len(global_locus)
            remap.append(global_locus[locus])
        counts = np.diff(shard["starts"])
        keep = counts > 0
        read_ids = shard["read_ids"]
        if read_ids is None:
            legacy_shards += 1
        else:
            for read_id in read_ids.tolist():
                if read_id in seen_reads:
                    raise ValueError(f"read ID {read_id!r} occurs in multiple shards")
                seen_reads.add(read_id)
            id_chunks.append(read_ids[keep])
        score_chunks.append(shard["scores"])
        locus_chunks.append(np.asarray(remap, dtype=np.int64)[shard["edge_locus"]])
        count_chunks.append(counts[keep])
        n_input_reads += len(counts)
        n_zero += int((~keep).sum())
        print(f"[em] {Path(path).name}: reads={len(counts):,} edges={len(shard['scores']):,} "
              f"loci={len(shard['loci']):,}", flush=True)
    if legacy_shards:
        warnings.warn(
            f"{legacy_shards} legacy shard(s) lack read_ids; read disjointness cannot be checked "
            "for these shards. Supply non-overlapping input shards or regenerate the dumps.",
            RuntimeWarning, stacklevel=2)
    counts = np.concatenate(count_chunks)
    return {
        "edge_scores": np.concatenate(score_chunks),
        "edge_locus": np.concatenate(locus_chunks),
        "read_start": np.concatenate(([0], np.cumsum(counts, dtype=np.int64))),
        "read_ids": np.concatenate(id_chunks) if legacy_shards == 0 else None,
        "loci": list(global_locus), "n_input_reads": n_input_reads,
        "n_zero_candidate_reads": n_zero, "n_shards_without_read_ids": legacy_shards,
        "read_id_check": "complete" if legacy_shards == 0 else
                         ("unavailable" if legacy_shards == len(files) else "partial"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges-glob", required=True,
                    help="glob of per-shard dump npz files")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--no-em", action="store_true",
                    help="uniform-prior ablation (pi frozen at init)")
    ap.add_argument("--save-gamma", action="store_true",
                    help="also write f32 responsibilities (large)")
    args = ap.parse_args()
    suffixes = ['.pi.tsv.gz', '.em.json'] + (['.gamma.npz'] if args.save_gamma else [])
    for suffix in suffixes:
        path = Path(args.out_prefix + suffix)
        if path.exists():
            raise FileExistsError(f'output already exists: {path}')

    if not math.isfinite(args.temperature) or args.temperature <= 0:
        ap.error("--temperature must be finite and positive")
    if args.max_iter <= 0:
        ap.error("--max-iter must be positive")
    if not math.isfinite(args.tol) or args.tol <= 0:
        ap.error("--tol must be finite and positive")

    files = sorted(glob.glob(args.edges_glob))
    if not files:
        raise SystemExit(f"no files match {args.edges_glob}")
    print(f"[em] {len(files)} shard dumps", flush=True)

    corpus = assemble_corpus(files)
    edge_scores, edge_locus, read_start = (corpus[key] for key in
                                          ("edge_scores", "edge_locus", "read_start"))
    global_ids = corpus["loci"]
    total_reads = len(read_start) - 1
    n_zero = corpus["n_zero_candidate_reads"]
    if n_zero:
        print(f"[em] dropped {n_zero:,} zero-candidate reads", flush=True)
    print(f"[em] corpus: reads={total_reads:,} edges={len(edge_scores):,} "
          f"unique loci={len(global_ids):,}", flush=True)

    if total_reads:
        with np.errstate(over="ignore", invalid="ignore"):
            scaled_scores = edge_scores / args.temperature
        if not np.isfinite(scaled_scores).all():
            raise ValueError("temperature-scaled edge scores are not finite")
        del scaled_scores
        res = em_locus_assignment(
            edge_scores, edge_locus, read_start, len(global_ids),
            temperature=args.temperature, max_iter=args.max_iter, tol=args.tol,
            em=not args.no_em)
        if not all(np.isfinite(values).all() for values in (res.pi, res.gamma, res.loglik_trace)):
            raise ValueError("EM returned non-finite fit values; no output was written")
        pi, gamma = res.pi, res.gamma
        fit_diag = {"fit_status": "FITTED", "converged": res.converged,
                    "n_iter": res.n_iter, "loglik_trace": res.loglik_trace[-5:]}
        print(f"[em] converged={res.converged} n_iter={res.n_iter} "
              f"loglik_final={res.loglik_trace[-1]:.1f}", flush=True)
    else:
        pi = gamma = np.empty(0, dtype=np.float64)
        fit_diag = {"fit_status": "NO_CANDIDATE_READS", "converged": None,
                    "n_iter": 0, "loglik_trace": []}
        print("[em] no candidate-bearing reads; no fit performed", flush=True)

    with gzip.open(args.out_prefix + ".pi.tsv.gz", "xt") as fh:
        fh.write("locus_id\tpi\n")
        for lid, proportion in zip(global_ids, pi):
            fh.write(f"{lid}\t{proportion:.8g}\n")
    diag = {
        "n_shards": len(files), "n_reads": total_reads,
        "n_edges": int(len(edge_scores)), "n_loci": len(global_ids),
        "n_input_reads": corpus["n_input_reads"], "n_zero_candidate_reads": n_zero,
        "read_id_check": corpus["read_id_check"],
        "n_shards_without_read_ids": corpus["n_shards_without_read_ids"],
        **fit_diag,
        "temperature": args.temperature, "em": not args.no_em,
    }
    with Path(args.out_prefix + ".em.json").open("x") as handle:
        handle.write(json.dumps(diag, indent=1, allow_nan=False) + "\n")
    if args.save_gamma:
        identities = {} if corpus["read_ids"] is None else {"read_ids": corpus["read_ids"]}
        with open(args.out_prefix + ".gamma.npz", "xb") as handle:
            np.savez_compressed(handle,
                            gamma=gamma.astype(np.float32),
                            edge_locus=edge_locus, read_start=read_start, **identities)
    print(f"[em] outputs -> {args.out_prefix}.pi.tsv.gz / .em.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
