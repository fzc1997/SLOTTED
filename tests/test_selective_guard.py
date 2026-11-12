"""Unit tests for the selective (guarded-abstention) decision layer.

Release mirror of the research-repo tests; toy data only, no torch: the
rule operates on per-group softmax probability lists and integer seed
counts.  The fuzz case pins decide_selective to a verbatim copy of the
benchmark's reference implementation (run_ns3_m2d_fixed.py fixed_decision),
whose NanoSim headline numbers the CLI reproduces.
"""

from __future__ import annotations

import math
import random
import unittest

from tracete_lm.selective import (
    CALL_ABSTAIN,
    CALL_UNIQUE,
    MIN_SEED,
    SELECTIVE_MARGIN,
    decide_selective,
    group_seed_counts,
    read_kmers,
    summarize_calls,
    window_kmers,
)

try:  # d6_likelihood pulls pyarrow via bench.py; optional in minimal installs
    from tracete_lm.d6_likelihood import two_strand_kmers
except ImportError:  # pragma: no cover
    two_strand_kmers = None

BASES = "ACGT"


def _reference_fixed_decision(probs, seed_counts, *, margin, min_seed=MIN_SEED):
    """Verbatim copy of the NanoSim bench v5 rule (run_ns3_m2d_fixed.py)."""
    n = len(probs)
    if n == 0:
        return "EMPTY", None
    top1 = max(probs)
    argmax = probs.index(top1)
    if sum(1 for v in probs if v == top1) >= 2:
        return "TIE", argmax
    if n == 1:
        return ("UNIQUE" if seed_counts[0] >= min_seed else "WEAK"), 0
    rest = [v for v in probs if v != top1]
    top2 = max(rest)
    ws = seed_counts[argmax]
    others = [s for i, s in enumerate(seed_counts) if i != argmax]
    if ws >= min_seed and (top1 - top2) >= margin and ws > max(others):
        return "UNIQUE", argmax
    return "WEAK", argmax


def _softmax(logits):
    peak = max(logits)
    exps = [math.exp(v - peak) for v in logits]
    total = math.fsum(exps)
    return [v / total for v in exps]


class TestSeedHelpers(unittest.TestCase):
    def test_read_kmers_excludes_n_words(self):
        seq = "A" * 31 + "N" + "C" * 31
        kmers = read_kmers(seq, 31)
        self.assertIn("A" * 31, kmers)
        self.assertIn("C" * 31, kmers)
        self.assertEqual(len(kmers), 2)
        self.assertTrue(all("N" not in w for w in kmers))

    def test_read_kmers_short_read_is_empty(self):
        self.assertEqual(read_kmers("ACGT" * 7, 31), set())

    def test_window_kmers_are_two_strand(self):
        window = "ACGT" * 20
        kmers = window_kmers(window, 31)
        probe = window[3:34]
        revcomp = probe.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]
        self.assertIn(probe, kmers)
        self.assertIn(revcomp, kmers)

    @unittest.skipIf(two_strand_kmers is None, "d6_likelihood needs pyarrow")
    def test_window_kmers_match_d6_likelihood(self):
        rng = random.Random(7)
        for k in (7, 31):
            for _ in range(50):
                seq = "".join(rng.choice(BASES + "N") for _ in range(rng.randint(k, 3 * k)))
                self.assertEqual(window_kmers(seq, k), two_strand_kmers(seq, k))

    def test_group_seed_counts_manual(self):
        k = 5
        read = "AACCGGTT"
        windows = [
            window_kmers("AACCGGTT", k),   # identical: all 4 read kmers
            window_kmers("TTTTTCCCC", k),  # none
            window_kmers("AACCG", k),      # AACCG plus its revcomp CGGTT, both in the read
        ]
        self.assertEqual(group_seed_counts(read, windows, k=k), [4, 0, 2])


class TestDecideSelective(unittest.TestCase):
    def test_empty_group_abstains(self):
        d = decide_selective([], [])
        self.assertEqual(d.call, CALL_ABSTAIN)
        self.assertEqual(d.reason, "empty")
        self.assertIsNone(d.argmax)

    def test_single_candidate_seed_floor(self):
        d = decide_selective([1.0], [MIN_SEED])
        self.assertEqual((d.call, d.reason), (CALL_UNIQUE, "k1"))
        self.assertEqual(d.prob_margin, 1.0)
        self.assertEqual(d.max_other_seeds, -1)
        d = decide_selective([1.0], [MIN_SEED - 1])
        self.assertEqual((d.call, d.reason), (CALL_ABSTAIN, "seed_floor"))

    def test_pass_requires_margin_and_strict_seed_superiority(self):
        d = decide_selective([0.80, 0.20], [9, 3], margin=0.10)
        self.assertEqual((d.call, d.reason), (CALL_UNIQUE, "pass"))
        self.assertAlmostEqual(d.prob_margin, 0.60)
        self.assertEqual(d.winner_seeds, 9)
        self.assertEqual(d.max_other_seeds, 3)

    def test_margin_boundary_is_inclusive(self):
        d = decide_selective([0.55, 0.45], [9, 3], margin=0.10)
        self.assertEqual(d.call, CALL_UNIQUE)
        d = decide_selective([0.55, 0.45], [9, 3], margin=0.1000000001)
        self.assertEqual((d.call, d.reason), (CALL_ABSTAIN, "margin"))

    def test_seed_tie_abstains_even_at_high_margin(self):
        d = decide_selective([0.90, 0.10], [7, 7], margin=0.10)
        self.assertEqual((d.call, d.reason), (CALL_ABSTAIN, "seed_parity"))

    def test_seed_inferiority_abstains(self):
        d = decide_selective([0.90, 0.10], [7, 8], margin=0.10)
        self.assertEqual((d.call, d.reason), (CALL_ABSTAIN, "seed_parity"))

    def test_seed_floor_checked_before_margin(self):
        d = decide_selective([0.90, 0.10], [2, 0], margin=0.10)
        self.assertEqual((d.call, d.reason), (CALL_ABSTAIN, "seed_floor"))

    def test_exact_probability_tie_abstains_at_zero_margin(self):
        d = decide_selective([0.50, 0.50], [9, 1], margin=0.0)
        self.assertEqual((d.call, d.reason), (CALL_ABSTAIN, "tie"))
        self.assertEqual(d.prob_margin, 0.0)

    def test_winner_need_not_be_first_candidate(self):
        d = decide_selective([0.10, 0.85, 0.05], [3, 12, 4], margin=0.10)
        self.assertEqual((d.call, d.argmax), (CALL_UNIQUE, 1))
        self.assertEqual(d.max_other_seeds, 4)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            decide_selective([0.5, 0.5], [5])

    def test_fuzz_matches_benchmark_reference(self):
        rng = random.Random(31)
        for _ in range(4000):
            n = rng.choice([0, 1, 2, 3, 5, 9])
            logits = [rng.uniform(-4, 4) for _ in range(n)]
            if n >= 2 and rng.random() < 0.3:  # force exact near/ties
                logits[1] = logits[0]
            probs = _softmax(logits) if n else []
            seeds = [rng.choice([0, 1, 4, 5, 7, 7, 12, 40]) for _ in range(n)]
            margin = rng.choice([0.0, 0.05, 0.10, 0.30])
            got = decide_selective(probs, seeds, margin=margin)
            want, want_argmax = _reference_fixed_decision(probs, seeds, margin=margin)
            self.assertEqual(got.call, CALL_UNIQUE if want == "UNIQUE" else CALL_ABSTAIN)
            self.assertEqual(got.argmax, want_argmax)

    def test_summarize_calls(self):
        s = summarize_calls([CALL_UNIQUE, CALL_ABSTAIN, CALL_ABSTAIN, CALL_UNIQUE])
        self.assertEqual(
            s, {"n": 4, "n_called": 2, "n_abstained": 2, "abstention_rate": 0.5})
        self.assertEqual(
            summarize_calls([]),
            {"n": 0, "n_called": 0, "n_abstained": 0, "abstention_rate": None})


class TestHeadlineConstants(unittest.TestCase):
    def test_constants_match_the_frozen_receipt(self):
        self.assertEqual(SELECTIVE_MARGIN, 0.10)
        self.assertEqual(MIN_SEED, 5)


if __name__ == "__main__":
    unittest.main()
