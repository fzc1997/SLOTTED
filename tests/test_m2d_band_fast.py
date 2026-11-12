"""Bit-identity tests for the fast band-construction path.

The optimized path must be array-identical to the reference composition:
  reference: per candidate k, other_union_lookup over the OTHER windows
             (O(K^2) window encodes) + build_pair_band
  fast:      window_lut once per window (O(K)) + group_other_unions
             (prefix/suffix ORs) + build_pair_band
and the numba greedy-walk kernel must emit the same events as the pure-python
_path_events on every input.
"""
from __future__ import annotations

import random
import unittest

import numpy as np

from tracete_lm.m2d_matrix import (
    BAND,
    CAND_CAP,
    READ_CAP,
    _path_events,
    build_pair_band,
    group_other_unions,
    other_union_lookup,
    window_lut,
)
from tracete_lm.m2d_model import candidate_group_bands

try:
    from tracete_lm.m2d_matrix import _path_events_nb
except Exception:  # pragma: no cover
    _path_events_nb = None

BASES = "ACGT"


def _rand_seq(rng: random.Random, n: int, n_prob: float = 0.0) -> str:
    letters = BASES + ("N" if rng.random() < n_prob else "")
    return "".join(rng.choice(letters) for _ in range(n))


def _mutate(rng: random.Random, seq: str) -> str:
    """Indel/substitution-mutated copy (mimics read-vs-genome divergence)."""
    out = []
    for b in seq:
        r = rng.random()
        if r < 0.02:
            continue  # deletion
        if r < 0.04:
            out.append(rng.choice(BASES))  # substitution
            continue
        if r < 0.05:
            out.append(rng.choice(BASES))  # insertion
        out.append(b)
    return "".join(out)


def _reference_group_bands(read: str, windows: list[str], *, read_cap: int, cand_cap: int) -> list[np.ndarray]:
    out = []
    for k, cand in enumerate(windows):
        others = [w for j, w in enumerate(windows) if j != k]
        union = other_union_lookup(others, cand_cap=cand_cap) if others else None
        out.append(build_pair_band(read, cand, union, read_cap=read_cap, cand_cap=cand_cap))
    return out


class TestGroupOtherUnions(unittest.TestCase):
    def test_matches_reference_lookup(self):
        rng = random.Random(11)
        for trial in range(30):
            k = rng.randint(1, 12)
            windows = [_rand_seq(rng, rng.randint(40, 900), n_prob=0.05) for _ in range(k)]
            if k >= 2 and trial % 3 == 0:
                windows[1] = windows[0]  # identical members
            luts = [window_lut(w) for w in windows]
            unions = group_other_unions(luts)
            for j, w in enumerate(windows):
                others = [x for i, x in enumerate(windows) if i != j]
                ref = other_union_lookup(others) if others else None
                self.assertEqual(ref is None, unions[j] is None)
                if ref is not None:
                    np.testing.assert_array_equal(ref, unions[j], err_msg=f"trial={trial} k={j}")

    def test_single_member_returns_none(self):
        self.assertEqual(group_other_unions([window_lut("ACGT" * 20)]), [None])
        self.assertEqual(group_other_unions([]), [])


class TestPathEventsKernel(unittest.TestCase):
    def test_numba_matches_python(self):
        if _path_events_nb is None:
            self.skipTest("numba unavailable")
        rng = random.Random(7)
        cases = 0
        for trial in range(120):
            h = rng.randint(0, 400)
            w = rng.randint(0, 400)
            if trial % 4 == 0 and h > 8:
                read = _rand_seq(rng, h)
                cand = _mutate(rng, read[: min(w, h)])
            else:
                read = _rand_seq(rng, h, n_prob=0.08)
                cand = _rand_seq(rng, w, n_prob=0.08)
            if trial % 10 == 0 and h > 4:  # homopolymer stress
                read = ("A" * (h // 2 + 1) + _rand_seq(rng, h // 2))[:h]
                cand = ("A" * (w // 2 + 1) + _rand_seq(rng, w // 2))[:w]
            diag = rng.randint(-60, 60)
            band = rng.choice([BAND, 4, 0])
            ref = _path_events(np.frombuffer(read.encode(), np.uint8),
                               np.frombuffer(cand.encode(), np.uint8), diag, band)
            ch, ii, oo = _path_events_nb(
                np.frombuffer(read.encode(), np.uint8),
                np.frombuffer(cand.encode(), np.uint8), diag, band)
            self.assertEqual(len(ref), len(ch), f"trial={trial}")
            for (rc, ri, ro), nc, ni, no in zip(ref, ch, ii, oo):
                self.assertEqual((rc, ri, ro), (int(nc), int(ni), int(no)), f"trial={trial}")
            cases += 1
        self.assertGreater(cases, 100)


class TestCandidateGroupBandsBitIdentity(unittest.TestCase):
    def test_matches_reference_composition(self):
        rng = random.Random(20261011)
        checked = 0
        for trial in range(12):
            k = rng.choice([1, 2, 3, 5, 9, 17])
            windows = [_rand_seq(rng, rng.randint(60, 700), n_prob=0.04) for _ in range(k)]
            if k >= 3 and trial % 4 == 0:
                windows[2] = windows[0]
            read = _mutate(rng, windows[0][: min(len(windows[0]), rng.randint(80, 1200))])
            if trial % 5 == 0:  # reverse-oriented read exercises the strand vote
                comp = str.maketrans("ACGTN", "TGCAN")
                read = read.translate(comp)[::-1]
            fast = candidate_group_bands(read, windows, read_cap=READ_CAP, cand_cap=CAND_CAP)
            ref = _reference_group_bands(read, windows, read_cap=READ_CAP, cand_cap=CAND_CAP)
            self.assertEqual(len(fast), len(ref))
            for j, (a, b) in enumerate(zip(ref, fast)):
                np.testing.assert_array_equal(a, b, err_msg=f"trial={trial} k={j}")
                checked += 1
        self.assertGreater(checked, 20)

    def test_band_width_guard(self):
        with self.assertRaises(ValueError):
            candidate_group_bands("ACGT", ["ACGT"], band_width=32)


class TestRustBackend(unittest.TestCase):
    """m2d_bands Rust kernel must be bit-identical to the reference."""

    def _rust_available(self) -> bool:
        try:
            import m2d_bands  # noqa: F401
            return True
        except ImportError:
            return False

    def test_rust_group_bands_match_reference(self):
        if not self._rust_available():
            self.skipTest("m2d_bands Rust kernel not installed")
        import m2d_bands
        from tracete_lm.m2d_model import candidate_group_bands
        rng = random.Random(777)
        checked = 0
        for trial in range(10):
            k = rng.choice([1, 2, 3, 5, 9, 17])
            windows = [_rand_seq(rng, rng.randint(60, 700), n_prob=0.04) for _ in range(k)]
            if k >= 3 and trial % 4 == 0:
                windows[2] = windows[0]
            read = _mutate(rng, windows[0][: min(len(windows[0]), rng.randint(80, 1200))])
            if trial % 5 == 0:
                comp = str.maketrans("ACGTN", "TGCAN")
                read = read.translate(comp)[::-1]
            rust = candidate_group_bands(read, windows, backend="rust", read_cap=READ_CAP, cand_cap=CAND_CAP)
            ref = _reference_group_bands(read, windows, read_cap=READ_CAP, cand_cap=CAND_CAP)
            self.assertEqual(len(rust), len(ref))
            for j, (a, b) in enumerate(zip(ref, rust)):
                np.testing.assert_array_equal(a, b, err_msg=f"trial={trial} k={j}")
                checked += 1
        self.assertGreater(checked, 15)

    def test_backend_resolution(self):
        from tracete_lm.m2d_matrix import resolve_band_backend
        import os
        from unittest import mock
        if self._rust_available():
            self.assertEqual(resolve_band_backend("auto"), "rust")
            with mock.patch.dict(os.environ, {"TRACE_LM_BAND_BACKEND": "python"}):
                self.assertEqual(resolve_band_backend("auto"), "python")
        else:
            self.assertEqual(resolve_band_backend("auto"), "python")
            with self.assertRaises(RuntimeError):
                resolve_band_backend("rust")
        with mock.patch.dict(os.environ, {"TRACE_LM_BAND_BACKEND": "auto"}):
            with self.assertRaises(ValueError):
                resolve_band_backend("warp")

    def test_rust_group_unions_match_python(self):
        if not self._rust_available():
            self.skipTest("m2d_bands Rust kernel not installed")
        import m2d_bands
        rng = random.Random(555)
        for k in (1, 2, 5, 12):
            windows = [_rand_seq(rng, rng.randint(50, 600), n_prob=0.05) for _ in range(k)]
            luts = [window_lut(w) for w in windows]
            py = group_other_unions(luts)
            rs = m2d_bands.group_other_unions([np.asarray(x) for x in luts])
            for a, b in zip(py, rs):
                self.assertEqual(a is None, b is None)
                if a is not None:
                    np.testing.assert_array_equal(a, np.asarray(b))


class TestBatchedReadGroupLogits(unittest.TestCase):
    """Cross-read batched scoring: same argmax as the per-read recipe on CPU,
    bounded logit deviation, candidate-order-preserving results."""

    def test_batched_matches_per_read(self):
        import torch
        from tracete_lm.m2d_model import BandCNN, M2DConfig, batched_read_group_logits, matrix2d_scores
        torch.manual_seed(0)
        rng = random.Random(99)
        config = M2DConfig(width=8, blocks=(1, 1, 1, 1), head_dim=16)
        model = BandCNN(config).eval()
        groups = []
        for _ in range(12):
            k = rng.choice([1, 2, 4])
            base = _rand_seq(rng, rng.randint(120, 500))
            cands = [base if j == 0 else _mutate(rng, base) for j in range(k)]
            read = _mutate(rng, base[: rng.randint(80, 300)])
            if rng.random() < 0.4:
                comp = str.maketrans("ACGTN", "TGCAN")
                read = read.translate(comp)[::-1]
            groups.append((read, cands))
        slow = [matrix2d_scores(model, s, c, device="cpu", batch_size=8) for s, c in groups]
        fast = batched_read_group_logits(model, groups, device="cpu")
        self.assertEqual(len(fast), len(slow))
        max_diff = 0.0
        for a, b in zip(slow, fast):
            self.assertEqual(len(a), len(b))
            self.assertEqual(max(range(len(a)), key=lambda i: a[i]),
                             max(range(len(b)), key=lambda i: b[i]))
            max_diff = max(max_diff, max(abs(x - y) for x, y in zip(a, b)))
        self.assertLess(max_diff, 0.1)

    def test_group_order_and_giant_chunk(self):
        import torch
        from tracete_lm.m2d_model import BandCNN, M2DConfig, batched_read_group_logits
        torch.manual_seed(1)
        model = BandCNN(M2DConfig(width=8, blocks=(1, 1, 1, 1), head_dim=16)).eval()
        rng = random.Random(5)
        groups = [(_rand_seq(rng, 150), [_rand_seq(rng, 200) for _ in range(3)])
                  for _ in range(4)]
        out = batched_read_group_logits(model, groups, device="cpu",
                                        max_candidates=2, max_rows=10_000)
        self.assertEqual([len(g) for g in out], [3, 3, 3, 3])  # giant-chunk path keeps order


class TestRobustness(unittest.TestCase):
    """Edge cases and misuse guards for the new band/batching paths."""

    def _tiny_model(self):
        import torch
        from tracete_lm.m2d_model import BandCNN, M2DConfig
        torch.manual_seed(3)
        return BandCNN(M2DConfig(width=8, blocks=(1, 1, 1, 1), head_dim=16)).eval()

    def test_rust_signature_keyword_only(self):
        try:
            import m2d_bands
        except ImportError:
            self.skipTest("m2d_bands not installed")
        with self.assertRaises(TypeError):
            m2d_bands.build_pair_band("ACGT" * 50, "ACGT" * 60, None, 4096, 4096)
        with self.assertRaises(TypeError):
            m2d_bands.candidate_group_bands("ACGT" * 50, ["ACGT" * 60], 4096)

    def test_group_unions_readonly(self):
        luts = [window_lut(_rand_seq(random.Random(i), 300)) for i in range(4)]
        unions = group_other_unions(luts)
        for u in unions:
            self.assertIsNotNone(u)
            with self.assertRaises(ValueError):
                u[0] = True

    def test_batched_empty_groups_and_k0(self):
        model = self._tiny_model()
        from tracete_lm.m2d_model import batched_read_group_logits
        rng = random.Random(21)
        groups = [(_rand_seq(rng, 150), [])]                     # K=0 (CELLO has 441)
        groups.append((_rand_seq(rng, 200), [_rand_seq(rng, 250) for _ in range(3)]))
        groups.append((_rand_seq(rng, 120), []))                 # another K=0
        out = batched_read_group_logits(model, groups, device="cpu")
        self.assertEqual([len(g) for g in out], [0, 3, 0])
        self.assertEqual(batched_read_group_logits(model, [], device="cpu"), [])

    def test_batched_order_and_truncation(self):
        model = self._tiny_model()
        from tracete_lm.m2d_model import batched_read_group_logits, matrix2d_scores
        rng = random.Random(31)
        groups = []
        for length in (150, 600, 90, 350):                       # shuffled lengths
            base = _rand_seq(rng, length)
            groups.append((base[: length // 2], [_mutate(rng, base) for _ in range(2)]))
        # read far beyond READ_CAP must be truncated, not crash (cap 4096; use
        # small band_backend groups to keep CPU time sane)
        long_read = _rand_seq(rng, 4300)
        groups.append((long_read, [_rand_seq(rng, 300)]))
        fast = batched_read_group_logits(model, groups, device="cpu")
        slow = [matrix2d_scores(model, s, c, device="cpu", batch_size=8) for s, c in groups]
        self.assertEqual([len(f) for f in fast], [len(s) for s in slow])
        for a, b in zip(slow, fast):
            self.assertEqual(max(range(len(a)), key=lambda i: a[i]),
                             max(range(len(b)), key=lambda i: b[i]))

    def test_batched_empty_read(self):
        model = self._tiny_model()
        from tracete_lm.m2d_model import batched_read_group_logits
        # zero-height bands cannot be forwarded (the per-read recipe crashes
        # just as hard, only deeper in the conv stack); the batched path must
        # fail fast with a clear error instead
        with self.assertRaises(ValueError):
            batched_read_group_logits(model, [("", ["ACGT" * 40])], device="cpu")

    def test_verified_wrapper_fallback(self):
        import tracete_lm.m2d_model as mm
        model = self._tiny_model()
        from tracete_lm.m2d_model import batched_group_logits_verified, matrix2d_scores
        rng = random.Random(41)
        groups = []
        for _ in range(6):
            base = _rand_seq(rng, rng.randint(120, 300))
            groups.append((_mutate(rng, base[:150]), [_mutate(rng, base) for _ in range(rng.choice([1, 2, 4]))]))
        groups.insert(3, (_rand_seq(rng, 100), []))              # K=0 in the mix
        exact = [matrix2d_scores(model, s, c, device="cpu", batch_size=8) for s, c in groups]

        # production path: batched numerics flip one near-tie (simulated by
        # doctoring the batched result), the fallback must restore the exact
        # recipe and match the (correct) reference
        orig_batched = mm.batched_read_group_logits

        def doctored(model_, groups_, **kw):
            out = orig_batched(model_, groups_, **kw)
            out[4] = list(out[4])
            out[4][0] += 5.0  # force a batched-vs-reference argmax flip
            return out

        mm.batched_read_group_logits = doctored
        try:
            logits, stats = batched_group_logits_verified(model, groups, exact, device="cpu")
        finally:
            mm.batched_read_group_logits = orig_batched
        self.assertEqual(stats["n_groups"], len(groups))
        self.assertEqual(stats["n_flips"], 1)
        self.assertEqual(stats["fallback_fixed"], 1)
        self.assertEqual(stats["still_bad"], [])
        self.assertEqual(logits[4], exact[4])
        for gi in (0, 1, 2, 5):
            self.assertEqual(max(range(len(logits[gi])), key=lambda i: logits[gi][i]),
                             max(range(len(exact[gi])), key=lambda i: exact[gi][i]))

    def test_verified_wrapper_surfaces_bad_reference(self):
        model = self._tiny_model()
        from tracete_lm.m2d_model import batched_group_logits_verified, matrix2d_scores
        rng = random.Random(61)
        groups = [(_mutate(rng, (b := _rand_seq(rng, 200))[:120]), [_mutate(rng, b) for _ in range(2)])
                  for _ in range(3)]
        exact = [matrix2d_scores(model, s, c, device="cpu", batch_size=8) for s, c in groups]
        corrupted = [list(r) for r in exact]
        corrupted[1][0] += 5.0  # reference itself disagrees with any recompute
        logits, stats = batched_group_logits_verified(model, groups, corrupted, device="cpu")
        self.assertEqual(stats["n_flips"], 1)
        self.assertEqual(stats["fallback_fixed"], 0)
        self.assertEqual(stats["still_bad"], [1])  # surfaced, never hidden

    def test_verified_wrapper_length_mismatch(self):
        model = self._tiny_model()
        from tracete_lm.m2d_model import batched_group_logits_verified
        groups = [("ACGT" * 30, ["ACGT" * 40])]
        with self.assertRaises(ValueError):
            batched_group_logits_verified(model, groups, [], device="cpu")

    def test_batched_rust_backend_matches_python(self):
        model = self._tiny_model()
        from tracete_lm.m2d_model import batched_read_group_logits
        try:
            import m2d_bands  # noqa: F401
        except ImportError:
            self.skipTest("m2d_bands not installed")
        rng = random.Random(51)
        groups = [(_mutate(rng, (b := _rand_seq(rng, 250))[:120]), [_mutate(rng, b) for _ in range(3)])
                  for _ in range(4)]
        rust = batched_read_group_logits(model, groups, device="cpu", band_backend="rust")
        py = batched_read_group_logits(model, groups, device="cpu", band_backend="python")
        for a, b in zip(rust, py):
            self.assertEqual(a, b)  # bit-identical bands -> identical CPU logits


if __name__ == "__main__":
    unittest.main()
