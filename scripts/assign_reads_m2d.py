#!/usr/bin/env python
"""SLOTTED-LM (m2d): assign long reads to TE loci.

Input:  a minimap2 BAM of long reads against the reference genome, the reads
        FASTA, a UCSC RepeatMasker annotation, the reference genome FASTA and
        the frozen m2d checkpoint.
Output: per-read winner table (TSV) + a JSON receipt with the run environment.
With --selective the table gains prob_margin / winner_seeds / max_other_seeds /
call columns (UNIQUE | ABSTAIN) from the guarded abstention rule:

  UNIQUE iff (K==1 and seeds>=5) or (prob margin >= --selective-margin AND
  winner seeds>=5 AND winner seeds strictly greater than every other candidate)

where seeds are exact k=31 word hits of the read against the same +/-flank
strand-oriented candidate windows the model sees, and the margin is the
top1-top2 difference of the per-group softmax probabilities.  On the frozen
NanoSim constructive-truth benchmark (margin 0.10) the rule calls 3,271/6,046
reads at P=0.9991 (45.9% abstention) on the bench's truth-augmented candidate
universe; with this CLI's mapper-only universe the same rule calls 3,293 of
the 5,852 mapper-candidate reads at P=0.9332, and reproduces the bench
decisions 5456/5456 on the reads whose candidate sets are identical (see
README).

Protocol (identical to the published benchmark recipe):
  * candidates   = TE loci overlapped by ANY alignment record of the read
                   (RepeatIndex interval query over the CIGAR blocks)
  * windows      = locus +/- --flank, strand-oriented, upper-cased
  * scoring      = BandCNN per (read, candidate) diagonal-band image; the
                   disc-word channel needs the union of the OTHER candidates
                   (window_lut + group_other_unions, O(K) not O(K^2))
  * batching     = cross-read candidate batching with one device sync per
                   batch (batched_read_group_logits)
  * verification = --verify-exact recomputes every group with the reference
                   per-read recipe and applies the argmax fallback, so the
                   emitted winners are guaranteed to equal the per-read
                   recipe (near-tie argmax flips from batch composition are
                   expected at ~0.1% and are repaired, never hidden)
  * winner       = argmax logit, first-index tie-break

Requires: pip install pysam torch (+ the m2d_bands Rust wheel for the fast
band backend; without it the pure-python backend is used automatically, or
force it with --band-backend python).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _score_groups(model, groups, args, device):
    """Score and verify using one dtype and per-read chunking protocol."""
    import torch
    from tracete_lm.m2d_model import (
        batched_group_logits_verified, batched_read_group_logits, matrix2d_scores,
    )

    def exact_scores(group):
        # The batched scorer and its fallback use CUDA bf16. A fp32 reference
        # here would compare different numerical recipes and leave false flips.
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                            enabled=device.startswith("cuda")):
            return matrix2d_scores(model, group[0], group[1], device=device,
                                   batch_size=args.batch_size, band_backend=args.band_backend)

    if args.verify_exact:
        reference = [exact_scores(group) for group in groups]
        logits, stats = batched_group_logits_verified(
            model, groups, reference, device=device,
            max_candidates=args.max_candidates, max_rows=args.max_rows,
            band_backend=args.band_backend, progress_every=100,
            reference_batch_size=args.batch_size)
        print(f"[slotted] verify-exact: flips={stats['n_flips']} "
              f"fallback_fixed={stats['fallback_fixed']} still_bad={len(stats['still_bad'])}",
              flush=True)
        if stats["still_bad"]:
            raise RuntimeError(
                f"Exact verification failed for {len(stats['still_bad'])} groups "
                f"(indices {stats['still_bad'][:10]}); no winner output was written")
    else:
        logits = batched_read_group_logits(
            model, groups, device=device,
            max_candidates=args.max_candidates, max_rows=args.max_rows,
            band_backend=args.band_backend, progress_every=100)
        stats = None
        if len(logits) != len(groups):
            raise ValueError("scorer returned a different number of read groups")
        if args.verify_exact_lazy:
            t_v = time.time()
            flagged = []
            for gi, scores in enumerate(logits):
                if not np.isfinite(scores).all():
                    raise ValueError(f"non-finite logits in read group {gi}")
                if len(scores) != len(groups[gi][1]):
                    flagged.append(gi)
                elif len(scores) > 1:
                    ordered = sorted(scores, reverse=True)
                    if ordered[0] - ordered[1] < args.lazy_margin:
                        flagged.append(gi)
            n_flips = 0
            for gi in flagged:
                exact = exact_scores(groups[gi])
                previous = logits[gi]
                old_winner = max(range(len(previous)), key=previous.__getitem__) if previous else None
                new_winner = max(range(len(exact)), key=exact.__getitem__) if exact else None
                n_flips += old_winner != new_winner
                logits[gi] = exact
            stats = {"n_groups": len(groups), "n_flagged": len(flagged),
                     "n_replaced": len(flagged), "n_flips": n_flips,
                     "lazy_margin": args.lazy_margin, "mode": "lazy-exact"}
            print(f"[slotted] verify-exact-lazy: flagged={len(flagged)} "
                  f"replaced={len(flagged)} flips={n_flips} ({time.time()-t_v:.0f}s)", flush=True)
    if len(logits) != len(groups):
        raise ValueError("scorer returned a different number of read groups")
    for gi, (scores, (_, candidates)) in enumerate(zip(logits, groups)):
        if len(scores) != len(candidates) or not np.isfinite(scores).all():
            raise ValueError(f"read group {gi} must have one finite logit per candidate")
    return logits, stats


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="assign_reads_m2d",
        description="SLOTTED-LM: assign long reads to RepeatMasker TE loci")
    ap.add_argument("--bam", required=True, help="minimap2 alignment BAM")
    ap.add_argument("--reads", required=True, help="reads FASTA")
    ap.add_argument("--rmsk", required=True, help="UCSC RepeatMasker rmsk.txt (or repeat GTF)")
    ap.add_argument("--genome", required=True, help="reference genome FASTA (bgzip+faidx or plain)")
    ap.add_argument("--ckpt", required=True, help="m2d checkpoint (ckpt_step30000.pt)")
    ap.add_argument("--out", required=True, help="winner TSV output path")
    ap.add_argument("--receipt", default=None, help="optional JSON receipt output path")
    ap.add_argument("--flank", type=int, default=500, help="window flank (default 500)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    ap.add_argument("--band-backend", default="auto", choices=["auto", "rust", "python"])
    ap.add_argument("--max-candidates", type=int, default=512)
    ap.add_argument("--max-rows", type=int, default=200000)
    ap.add_argument("--batch-size", type=int, default=128,
                    help="chunk size for the exact per-read recipe (--verify-exact)")
    verification = ap.add_mutually_exclusive_group()
    verification.add_argument("--verify-exact-lazy", action="store_true",
                    help="batched pass first, then the exact per-read recipe ONLY for "
                         "near-tie or incomplete groups; a heuristic, not full verification")
    ap.add_argument("--lazy-margin", type=float, default=0.05,
                    help="raw-logit top1-top2 margin below which a group is re-scored exactly")
    verification.add_argument("--verify-exact", action="store_true",
                    help="verify batched winners against the exact per-read recipe "
                         "and repair flips via fallback (recommended for recorded runs)")
    ap.add_argument("--min-mapq", type=int, default=0, help="drop alignments below this MAPQ")
    ap.add_argument("--selective", action="store_true",
                    help="guarded selective reporting: adds prob_margin, winner_seeds, "
                         "max_other_seeds, call (UNIQUE|ABSTAIN) columns and a summary line; "
                         "UNIQUE requires prob margin >= --selective-margin, winner exact k=31 "
                         "seeds >= 5 and strict seed superiority over every other candidate "
                         "(K==1 groups need only the seed floor)")
    ap.add_argument("--selective-margin", type=float, default=0.10,
                    help="probability-margin threshold for --selective (default 0.10)")
    ap.add_argument("--dump-edges", default=None,
                    help="optional path: write all per-candidate edge scores as npz "
                         "(edge_scores f32, edge_locus int32, read_start i64) plus a "
                         "<path>.loci.tsv index table — input for the EM arm")
    args = ap.parse_args()
    outputs = [Path(args.out)]
    if args.receipt:
        outputs.append(Path(args.receipt))
    if args.dump_edges:
        outputs.extend([Path(args.dump_edges), Path(str(args.dump_edges) + '.loci.tsv')])
    if len({path.resolve() for path in outputs}) != len(outputs):
        ap.error('output files must have distinct paths')
    for path in outputs:
        if path.exists():
            raise FileExistsError(f'output already exists: {path}')

    for name in ("max_candidates", "max_rows", "batch_size"):
        if getattr(args, name) <= 0:
            ap.error(f"--{name.replace('_', '-')} must be positive")
    if args.flank < 0:
        ap.error("--flank must be non-negative")
    if not 0 <= args.min_mapq <= 255:
        ap.error("--min-mapq must be between 0 and 255")
    if not math.isfinite(args.lazy_margin) or args.lazy_margin < 0:
        ap.error("--lazy-margin must be finite and non-negative")
    if not math.isfinite(args.selective_margin) or not 0 <= args.selective_margin <= 1:
        ap.error("--selective-margin must be finite and between 0 and 1")

    t0 = time.time()
    import pysam

    # --- packages (release layout: src/ on sys.path or pip-installed)
    here = Path(__file__).resolve().parent.parent / "src"
    for p in (str(here),):
        if p not in sys.path:
            sys.path.insert(0, p)
    from tracete.extract.annotations import RepeatIndex, iter_repeat_annotation
    from tracete_lm.inputs import read_fasta
    from tracete_lm.d6_likelihood import reverse_complement
    from tracete_lm.m2d_matrix import READ_CAP, resolve_band_backend
    from tracete_lm.m2d_model import BandCNN, M2DConfig, group_softmax_probs
    from tracete_lm.selective import (
        MIN_SEED, SEED_K, decide_selective, group_seed_counts, summarize_calls, window_kmers,
    )
    import torch

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    backend = resolve_band_backend(args.band_backend)
    print(f"[slotted] device={device} band_backend={backend}", flush=True)

    # --- reads
    seqs = read_fasta(args.reads)
    print(f"[slotted] reads={len(seqs)}", flush=True)

    # --- repeat universe + windows
    idx = RepeatIndex(iter_repeat_annotation(args.rmsk))
    fa = pysam.FastaFile(args.genome)
    lengths = dict(zip(fa.references, fa.lengths))

    # --- BAM scan: candidates = loci overlapped by any alignment block
    def blocks_from_cigar(ref_start: int, cigartuples) -> list[list[int]]:
        pos = ref_start
        out = []
        for op, ln in cigartuples or ():
            if op in (0, 7, 8):
                out.append([pos, pos + ln])
                pos += ln
            elif op in (2, 3):
                pos += ln
        return out

    cands: dict[str, list[str]] = {}
    with pysam.AlignmentFile(args.bam, "rb") as bam:
        for aln in bam:
            q = aln.query_name
            if q not in seqs or aln.reference_name is None:
                continue
            if aln.is_unmapped or (aln.mapping_quality or 0) < args.min_mapq:
                continue
            hits = set()
            for s, e in blocks_from_cigar(aln.reference_start, aln.cigartuples):
                for it in idx.query(aln.reference_name, s, e):
                    if it.overlap_bp(s, e) > 0:
                        hits.add(it.locus_id)
            if hits:
                cands.setdefault(q, [])
                for lid in sorted(hits):
                    if lid not in cands[q]:
                        cands[q].append(lid)
    header = "read_id\tn_candidates\twinner_locus\twinner_logit\tmargin"
    if args.selective:
        header += "\tprob_margin\twinner_seeds\tmax_other_seeds\tcall"
    ids = sorted(q for q in seqs if cands.get(q))
    skipped = len(seqs) - len(ids)
    print(f"[slotted] reads with TE candidates={len(ids)} (no-candidate reads: {skipped})", flush=True)
    if not ids:
        print("[slotted] nothing to assign", flush=True)

    needed = {lid for q in ids for lid in cands[q]}
    windows: dict[str, str] = {}
    for lid in needed:
        iv = idx.get(lid)
        if iv.chrom not in lengths or not 0 <= iv.start < iv.end <= lengths[iv.chrom]:
            raise ValueError(f"candidate {lid} has coordinates outside the reference genome")
        s = max(0, iv.start - args.flank)
        e = min(iv.end + args.flank, lengths[iv.chrom])
        seq = fa.fetch(iv.chrom, s, e).upper()
        if set(seq) - set("ACGTN"):
            raise ValueError(f"candidate window {lid} contains symbols outside ACGTN")
        windows[lid] = seq if iv.strand != "-" else reverse_complement(seq)
    print(f"[slotted] loci={len(needed)} windows built (flank={args.flank})", flush=True)

    fa.close()
    logits, stats = [], None
    if ids:
        blob = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        model = BandCNN(M2DConfig(**blob["config"])).to(device)
        model.load_state_dict(blob["model"])
        model.eval()
        print(f"[slotted] ckpt loaded (step={blob.get('step')})", flush=True)
        groups = [(seqs[q][:READ_CAP], [windows[lid] for lid in cands[q]]) for q in ids]
        logits, stats = _score_groups(model, groups, args, device)

    # --- selective guard: exact k=31 seeds against the same windows
    decisions: dict = {}
    if args.selective:
        t_seed = time.time()
        kmer_sets = {lid: window_kmers(windows[lid], SEED_K) for lid in needed}
        for q, scores in zip(ids, logits):
            if not scores:
                continue
            seed_counts = group_seed_counts(
                seqs[q][:READ_CAP], [kmer_sets[lid] for lid in cands[q]], k=SEED_K)
            decisions[q] = decide_selective(
                group_softmax_probs(scores), seed_counts, margin=args.selective_margin)
        print(f"[slotted] selective guard computed ({time.time() - t_seed:.0f}s)", flush=True)

    # --- winners (argmax, first-index tie-break)
    with open(args.out, "x") as out:
        out.write(header + "\n")
        for q, scores in zip(ids, logits):
            if not scores:
                out.write(f"{q}\t0\t.\t\tnan"
                          + ("\tnan\t\t\tABSTAIN" if args.selective else "") + "\n")
                continue
            best = max(scores)
            wi = scores.index(best)
            top2 = max((v for j, v in enumerate(scores) if j != wi), default=float("nan"))
            row = f"{q}\t{len(scores)}\t{cands[q][wi]}\t{best:.6f}\t{best - top2:.6f}"
            if args.selective:
                d = decisions[q]
                row += f"\t{d.prob_margin:.6f}\t{d.winner_seeds}\t{d.max_other_seeds}\t{d.call}"
            out.write(row + "\n")

    if args.selective:
        summary = summarize_calls(d.call for d in decisions.values())
        print(f"[slotted] selective: margin={args.selective_margin} min_seed={MIN_SEED} "
              f"n_called={summary['n_called']}/{summary['n']} "
              f"abstention_rate={summary['abstention_rate']}", flush=True)

    if args.dump_edges:
        import numpy as np
        t_edge = time.time()
        locus_ids = sorted(needed)
        locus_idx = {lid: i for i, lid in enumerate(locus_ids)}
        score_chunks: list = []
        locus_flat: list = []
        starts = [0]
        for q, sc in zip(ids, logits):
            if sc:
                locus_flat.extend(locus_idx[lid] for lid in cands[q])
                score_chunks.append(np.asarray(sc, dtype=np.float32))
                starts.append(starts[-1] + len(sc))
            else:
                starts.append(starts[-1])
        edge_scores = (np.concatenate(score_chunks) if score_chunks
                       else np.empty(0, dtype=np.float32))
        edge_locus = np.asarray(locus_flat, dtype=np.int32)
        read_start = np.asarray(starts, dtype=np.int64)
        with Path(args.dump_edges).open("xb") as edge_file:
            np.savez(edge_file, edge_scores=edge_scores,
                     edge_locus=edge_locus, read_start=read_start,
                     read_ids=np.asarray(ids, dtype=np.str_))
        loci_path = str(args.dump_edges) + ".loci.tsv"
        with open(loci_path, "x") as fh:
            for i, lid in enumerate(locus_ids):
                fh.write(f"{i}\t{lid}\n")
        print(f"[slotted] edges -> {args.dump_edges} "
              f"(reads={len(ids)} edges={len(edge_locus)} loci={len(locus_ids)}; "
              f"{time.time() - t_edge:.0f}s)", flush=True)

    if args.receipt:
        receipt = {
            "schema": "slotted.assign_reads_m2d.v1",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "inputs": {"bam": str(args.bam), "reads": str(args.reads), "rmsk": str(args.rmsk),
                       "genome": str(args.genome),
                       "ckpt_sha256": sha256_file(Path(args.ckpt))},
            "config": {"flank": args.flank, "device": device, "band_backend": backend,
                       "max_candidates": args.max_candidates, "max_rows": args.max_rows,
                       "batch_size": args.batch_size, "verify_exact": args.verify_exact,
                       "inference_dtype": "bfloat16" if device.startswith("cuda") else "float32",
                       "verify_exact_lazy": args.verify_exact_lazy,
                       "lazy_margin": args.lazy_margin,
                       "min_mapq": args.min_mapq, "selective": args.selective,
                       "selective_margin": args.selective_margin,
                       "dump_edges": args.dump_edges},
            "population": {"reads_total": len(seqs), "reads_with_candidates": len(ids),
                           "no_candidate_reads": skipped, "loci_used": len(needed)},
            "verification": stats,
            "python": sys.executable, "platform": platform.platform(),
            "seconds": round(time.time() - t0, 1),
        }
        if args.selective:
            receipt["selective"] = {
                "rule": "UNIQUE iff (K==1 and seeds>=5) or (prob margin>=m AND winner "
                        "seeds>=5 AND winner seeds strictly greater than every other "
                        "candidate); exact top-probability ties abstain",
                "margin": args.selective_margin, "min_seed": MIN_SEED, "seed_k": SEED_K,
                **summary,
            }
        with Path(args.receipt).open("x") as handle:
            handle.write(json.dumps(receipt, indent=1, default=str) + "\n")
        print(f"[slotted] receipt -> {args.receipt}", flush=True)
    print(f"[slotted] winners -> {args.out}  ({time.time() - t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
