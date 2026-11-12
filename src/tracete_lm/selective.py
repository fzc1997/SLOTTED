"""Selective (guarded-abstention) locus calling for m2d assignment.

Guard v5 — the decision layer validated on the NanoSim constructive-truth
benchmark (N=6,046; receipt results/upgrade_v2/phaseD7/nanosim_bench/
m2d_debug/fixed_benchmark.json): at the headline margin 0.10 it calls 3,271
of 6,046 reads with 3 errors (P=0.9991 / R=0.5405 / F1=0.7015, 45.9%
abstention).  Note the bench's candidate universe is truth-augmented (the
true locus is unioned into every read's candidates); on a mapper-only
universe the same rule applies verbatim but its precision is lower (the
release CLI reproduces the bench decisions 5456/5456 on identical candidate
sets — see release/slotted/README.md).  The guard exists because
nested/adjacent RepeatMasker annotations make exact-seed TIES on ~50% of
mapper-candidate groups, and unguarded softmax argmax wins only ~54% of
those ties.

Rule (per read group with K candidates, probability margin m, seed floor s):

    UNIQUE  iff (K == 1 and winner seeds >= s)
            or  (prob margin >= m AND winner seeds >= s AND winner seeds
                 strictly greater than every other candidate's seeds)
    ABSTAIN otherwise; an exact top-probability tie abstains at any margin.

prob margin is the top1 - top2 difference of the per-group softmax
PROBABILITIES (tracete_lm.m2d_model.group_softmax_probs), not of raw
logits; a single-candidate group has no runner-up and reports margin 1.0.
Seeds are exact k=31 word hits: the read's forward k-mers (N-containing
words excluded) intersected with the candidate window's two-strand k-mer
set — the emission of the em_k31 benchmark arm.  Windows are the same
+/-flank strand-oriented sequences the band scorer sees, so the guard is
computed on exactly the model's candidate evidence.

Deliberately stdlib-only (the d7_recert posture) so the decision layer is
unit-testable without the model stack; the two-strand k-mer helper
re-implements d6_likelihood.two_strand_kmers to avoid its pyarrow import
chain (equivalence locked by tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

SEED_K = 31
MIN_SEED = 5
SELECTIVE_MARGIN = 0.10

CALL_UNIQUE = "UNIQUE"
CALL_ABSTAIN = "ABSTAIN"

_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def read_kmers(sequence: str, k: int = SEED_K) -> set[str]:
    """Forward k-mers of the read; words containing N carry no evidence."""
    return {sequence[i : i + k] for i in range(len(sequence) - k + 1)
            if "N" not in sequence[i : i + k]}


def window_kmers(window: str, k: int = SEED_K) -> set[str]:
    """Two-strand k-mer set of a candidate window (covers both read orientations)."""
    forward = {window[i : i + k] for i in range(len(window) - k + 1)}
    return forward | {w.translate(_COMPLEMENT)[::-1] for w in forward}


def group_seed_counts(
    read_seq: str,
    window_kmer_sets: Sequence[set[str]],
    *,
    k: int = SEED_K,
) -> list[int]:
    """Exact seed count of one read against each candidate window, in order."""
    reads = read_kmers(read_seq, k)
    return [len(reads & window) for window in window_kmer_sets]


@dataclass(frozen=True, slots=True)
class SelectiveDecision:
    """One read group's guard verdict.

    call:           "UNIQUE" (report the argmax winner) or "ABSTAIN".
    reason:         pass | k1 | seed_floor | margin | seed_parity | tie | empty.
    argmax:         first-index argmax of probs (None when the group is empty).
    prob_margin:    top1 - top2 softmax-probability difference; 1.0 for K == 1
                    (no runner-up), None for K == 0.
    winner_seeds:   seed count of the argmax candidate.
    max_other_seeds: best seed count among the other candidates (-1 for K == 1).
    """

    call: str
    reason: str
    argmax: int | None
    prob_margin: float | None
    winner_seeds: int | None
    max_other_seeds: int | None


def decide_selective(
    probs: Sequence[float],
    seed_counts: Sequence[int],
    *,
    margin: float = SELECTIVE_MARGIN,
    min_seed: int = MIN_SEED,
) -> SelectiveDecision:
    """Guard v5 on one read group's softmax probabilities and seed counts."""
    probs = list(probs)
    seed_counts = list(seed_counts)
    n = len(probs)
    if n != len(seed_counts):
        raise ValueError(f"probs and seed_counts must align: {n} vs {len(seed_counts)}")
    import math
    import numbers
    if not math.isfinite(margin) or not 0.0 <= margin <= 1.0:
        raise ValueError("margin must be finite and between zero and one")
    if not isinstance(min_seed, numbers.Integral) or min_seed < 0:
        raise ValueError("min_seed must be a non-negative integer")
    if any(not math.isfinite(p) or not 0.0 <= p <= 1.0 for p in probs):
        raise ValueError("probabilities must be finite and between zero and one")
    if any(not isinstance(s, numbers.Integral) or s < 0 for s in seed_counts):
        raise ValueError("seed counts must be non-negative integers")
    if n == 0:
        return SelectiveDecision(CALL_ABSTAIN, "empty", None, None, None, None)

    top1 = max(probs)
    argmax = probs.index(top1)
    ordered = sorted(probs, reverse=True)
    prob_margin = 1.0 if n == 1 else ordered[0] - ordered[1]
    winner_seeds = seed_counts[argmax]
    others = [s for i, s in enumerate(seed_counts) if i != argmax]
    max_other = max(others) if others else -1

    if n > 1 and sum(1 for v in probs if v == top1) >= 2:
        return SelectiveDecision(CALL_ABSTAIN, "tie", argmax, prob_margin,
                                 winner_seeds, max_other)
    if n == 1:
        if winner_seeds >= min_seed:
            return SelectiveDecision(CALL_UNIQUE, "k1", argmax, prob_margin,
                                     winner_seeds, max_other)
        return SelectiveDecision(CALL_ABSTAIN, "seed_floor", argmax, prob_margin,
                                 winner_seeds, max_other)
    if winner_seeds < min_seed:
        return SelectiveDecision(CALL_ABSTAIN, "seed_floor", argmax, prob_margin,
                                 winner_seeds, max_other)
    if prob_margin < margin:
        return SelectiveDecision(CALL_ABSTAIN, "margin", argmax, prob_margin,
                                 winner_seeds, max_other)
    if winner_seeds <= max_other:
        return SelectiveDecision(CALL_ABSTAIN, "seed_parity", argmax, prob_margin,
                                 winner_seeds, max_other)
    return SelectiveDecision(CALL_UNIQUE, "pass", argmax, prob_margin,
                             winner_seeds, max_other)


def summarize_calls(calls: Iterable[str]) -> dict:
    """Population summary of per-read calls (the receipt/stdout line)."""
    calls = list(calls)
    n = len(calls)
    n_called = sum(1 for c in calls if c == CALL_UNIQUE)
    return {
        "n": n,
        "n_called": n_called,
        "n_abstained": n - n_called,
        "abstention_rate": round((n - n_called) / n, 4) if n else None,
    }
