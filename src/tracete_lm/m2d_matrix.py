"""Variant 2 (matrix2d): sparse 2D alignment-matrix representation of
(read, candidate) pairs.

The alignment pattern of a read against a candidate window is naturally
2D: rows = read positions, columns = candidate positions.  A 1D sequence
input (window <SEP> read) flattens that pattern; this module makes it
explicit.

Geometry.  A dense 4096 x 4096 grid is never materialised.  Exact 8-mer
anchor matches vote for a main alignment diagonal d* = j - i (votes
smoothed over +/-1 diagonal); everything outside the band
|(j - i) - d*| <= BAND is dropped (sparse storage), and the band is
densified to a (C, H, 2*BAND+1) image whose column axis is the DIAGONAL
OFFSET o = (j - i) - d*.  Diagonal kinks from indels appear as horizontal
shifts of the pattern — exactly the 2D structure a conv stack can read and
a 1D token stream cannot.

Channels (9, >= the 4 preregistered groups):
    0 anchor     read 8-mer at i == candidate 8-mer at j (exact, band only)
                 — hash-anchored match dots ("k-mer 锚定通道")
    1 match      DENSE diagonalized per-base equality read[i] == cand[j]
                 over the whole band ("逐碱基互补匹配通道", one-hot pair
                 with channel 2; off-path cells keep the random ~25% ACGT
                 equality floor, the path itself is a clean ridge)
    2 mismatch   one-hot complement of channel 1
    3 sub        sparse events on the greedy local-alignment path: mismatch
                 with no shift evidence — substitution prior
    4 ins        path event: read-side insertion (r[i+1..] == c[j..],
                 2-base confirmed) — the path shifts by -1 offset
    5 del        path event: read-side deletion (r[i] == c[j+1..],
                 2-base confirmed) — the path shifts by +1 offset
                 3-5 form the error-type position priors "从锚定对角线局部
                 比对得出"
    6 disc_word  candidate 8-mer STARTING at column j absent from the
                 two-strand 8-mer union of the OTHER candidates of the
                 group — the candidate-side distinguishing-word channel
                 ("区分词通道"; columns with no complete k-mer are 0)
    7 row_pos    i / H  — absolute read-position prior
    8 col_pos    j / W  — absolute candidate-position prior

Strand.  Both read orientations are voted (in-band anchor support after
each orientation's own diagonal vote); the winner is used and recorded.
Bench-style molecules are already candidate-oriented; raw recert reads
are not — training augmentation flips the read 50% so the representation
is strand-robust.

Pure-function contract: build_pair_matrix(read, candidate, other_union)
-> PairMatrix, whose COO view (channel, row, col, value) with shape
(C, H, W) is the sparse tensor; densify_band() reconstitutes the band
image losslessly.  build_pair_band() is the fast dense-only path used by
training/inference and is unit-tested to equal the sparse round trip.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:  # optional accelerator for the greedy path walk (bit-identical port below)
    from numba import njit as _njit
except ImportError:  # pragma: no cover
    _njit = None

try:  # optional Rust kernel, bit-identical port of this module (rust/m2d_bands)
    import m2d_bands as _rust_bands
except ImportError:  # pragma: no cover
    _rust_bands = None

K_ANCHOR = 8
BAND = 32
BAND_WIDTH = 2 * BAND + 1
READ_CAP = 4096
CAND_CAP = 4096
N_CHANNELS = 9
(
    CH_ANCHOR,
    CH_MATCH,
    CH_MISMATCH,
    CH_SUB,
    CH_INS,
    CH_DEL,
    CH_DISC,
    CH_ROWPOS,
    CH_COLPOS,
) = range(N_CHANNELS)
DENSE_CHANNELS = (CH_MATCH, CH_MISMATCH, CH_ROWPOS, CH_COLPOS)
PER_CODE_CAP = 32  # max candidate positions paired per read k-mer (repeat blowup guard)

_ASCII_LUT = np.full(256, 4, dtype=np.uint8)
for _b, _c in zip(b"ACGT", range(4)):
    _ASCII_LUT[_b] = _c

_COMPLEMENT_LUT = np.array([3, 2, 1, 0, 4, 255], dtype=np.uint8)


def encode_u8(sequence: str, cap: int) -> np.ndarray:
    """ASCII sequence -> uint8 base codes (A0 C1 G2 T3, other 4), capped."""
    arr = np.frombuffer(sequence.upper().encode("ascii"), dtype=np.uint8)[:cap]
    return _ASCII_LUT[arr].copy()


def reverse_complement_u8(u8: np.ndarray) -> np.ndarray:
    return _COMPLEMENT_LUT[u8[::-1]]


def kmer_codes(u8: np.ndarray, k: int = K_ANCHOR) -> tuple[np.ndarray, np.ndarray]:
    """(codes, has_n): rolling 2-bit codes of the k-mers; N-containing flagged."""
    n = len(u8) - k + 1
    if n <= 0:
        return np.zeros(0, np.uint16), np.zeros(0, bool)
    sw = np.lib.stride_tricks.sliding_window_view(u8, k)
    has_n = (sw >= 4).any(axis=1)
    weights = (1 << (2 * np.arange(k, dtype=np.uint32))).astype(np.uint32)
    codes = (sw.astype(np.uint32) * weights).sum(axis=1)
    return codes.astype(np.uint16), has_n


@dataclass(frozen=True, slots=True)
class _CodeIndex:
    """Candidate-side inverse index: code -> sorted original positions."""

    positions_sorted: np.ndarray  # int32 original j's ordered by (code, j)
    uniq: np.ndarray              # uint16 unique sorted codes
    start: np.ndarray             # int64 first slot of each code
    counts: np.ndarray            # int64 positions per code

    @classmethod
    def build(cls, codes: np.ndarray, has_n: np.ndarray) -> "_CodeIndex":
        keep = ~has_n
        kept_codes = codes[keep]
        positions = np.flatnonzero(keep).astype(np.int32)
        order = np.argsort(kept_codes, kind="stable")
        positions_sorted = positions[order]
        sorted_codes = kept_codes[order]
        uniq, start = np.unique(sorted_codes, return_index=True)
        counts = np.diff(np.append(start, len(sorted_codes))).astype(np.int64)
        return cls(positions_sorted=positions_sorted, uniq=uniq, start=start, counts=counts)

    def empty(self) -> bool:
        return len(self.uniq) == 0


def _match_pairs(
    read_codes: np.ndarray,
    read_has_n: np.ndarray,
    index: _CodeIndex,
    cap_per_code: int = PER_CODE_CAP,
) -> tuple[np.ndarray, np.ndarray]:
    """(i, j) arrays of exact anchor matches; per-code candidate positions capped."""
    keep = ~read_has_n
    rpos = np.flatnonzero(keep).astype(np.int64)
    if index.empty() or len(rpos) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    rcodes = read_codes[rpos]
    loc = np.searchsorted(index.uniq, rcodes)
    loc = np.minimum(loc, len(index.uniq) - 1)
    hit = index.uniq[loc] == rcodes
    if not hit.any():
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    rpos = rpos[hit]
    m = np.minimum(index.counts[loc[hit]], cap_per_code)
    total = int(m.sum())
    if total == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    i_rep = np.repeat(rpos, m)
    starts = np.cumsum(m) - m
    t = np.arange(total, dtype=np.int64) - np.repeat(starts, m)
    j = index.positions_sorted[np.repeat(index.start[loc[hit]], m) + t]
    return i_rep, j.astype(np.int64)


def _vote_diagonal(i: np.ndarray, j: np.ndarray, span: int, h: int | None = None) -> int:
    """argmax diagonal of smoothed j - i anchor votes; ties -> smallest.

    Smoothing weight (1, 2, 1) merges votes split across adjacent diagonals
    by 1-2bp indels while keeping the centre dominant, so a flat plateau
    cannot flip the argmax to a neighbouring diagonal.  The bin range covers
    the full [-(H-1), W-1] diagonal span even when the read outlengths the
    candidate (or vice versa).
    """
    if len(i) == 0:
        return 0
    diag = j - i
    width = max(int(span), int(h if h is not None else 0), int(-diag.min()), int(diag.max())) + 1
    lo = -width
    bins = diag - lo
    votes = np.bincount(bins, minlength=2 * width + 1)
    sm = votes * 2
    sm[1:] += votes[:-1]
    sm[:-1] += votes[1:]
    return int(np.argmax(sm)) + lo


def _path_events(read_u8: np.ndarray, cand_u8: np.ndarray, diag: int, band: int) -> list[tuple[int, int, int]]:
    """Greedy local alignment along the anchored diagonal -> sparse error events.

    Walks the read maintaining the path offset o (start 0); a mismatch is
    classified as read-side deletion (path shifts +1), read-side insertion
    (path shifts -1, consuming a read base without a candidate base) or
    substitution, with 2-base shift confirmation so random coincidences
    rarely masquerade as indels.  Returns (channel, read_row, offset).
    """
    r = read_u8.tobytes()
    c = cand_u8.tobytes()
    h, w = len(r), len(c)
    events: list[tuple[int, int, int]] = []
    i, o = 0, 0
    while i < h:
        j = i + diag + o
        if 0 <= j < w and r[i] == c[j]:
            i += 1
            continue
        if 0 <= j + 1 < w and r[i] == c[j + 1] and (i + 1 >= h or j + 2 >= w or r[i + 1] == c[j + 2]):
            if abs(o) <= band:
                events.append((CH_DEL, i, o))
            o += 1  # read-side deletion: candidate advances without the read
            continue
        if i + 1 < h and 0 <= j < w and r[i + 1] == c[j] and (i + 2 >= h or j + 1 >= w or r[i + 2] == c[j + 1]):
            if abs(o) <= band:
                events.append((CH_INS, i, o))
            i += 1
            o -= 1  # read-side insertion: read advances without the candidate
            continue
        if 0 <= j < w and abs(o) <= band:
            events.append((CH_SUB, i, o))
        i += 1
    return events


if _njit is not None:

    @_njit(cache=True)
    def _path_events_nb(r: np.ndarray, c: np.ndarray, diag: int, band: int):
        """Numba port of _path_events: same greedy walk over uint8 codes,
        returns flat (channel, read_row, offset) arrays instead of tuples."""
        h, w = r.shape[0], c.shape[0]
        cap = h + w + 64  # del runs advance only the offset, ins/match advance i
        ev_ch = np.empty(cap, np.int64)
        ev_i = np.empty(cap, np.int64)
        ev_o = np.empty(cap, np.int64)
        n = 0
        i = np.int64(0)
        o = np.int64(0)
        while i < h:
            j = i + diag + o
            if 0 <= j < w and r[i] == c[j]:
                i += 1
                continue
            if 0 <= j + 1 < w and r[i] == c[j + 1] and (i + 1 >= h or j + 2 >= w or r[i + 1] == c[j + 2]):
                if o <= band and -o <= band:
                    ev_ch[n] = CH_DEL
                    ev_i[n] = i
                    ev_o[n] = o
                    n += 1
                o += 1  # read-side deletion: candidate advances without the read
                continue
            if i + 1 < h and 0 <= j < w and r[i + 1] == c[j] and (i + 2 >= h or j + 1 >= w or r[i + 2] == c[j + 1]):
                if o <= band and -o <= band:
                    ev_ch[n] = CH_INS
                    ev_i[n] = i
                    ev_o[n] = o
                    n += 1
                i += 1
                o -= 1  # read-side insertion: read advances without the candidate
                continue
            if 0 <= j < w and o <= band and -o <= band:
                ev_ch[n] = CH_SUB
                ev_i[n] = i
                ev_o[n] = o
                n += 1
            i += 1
        return ev_ch[:n], ev_i[:n], ev_o[:n]

else:  # pragma: no cover
    _path_events_nb = None


def _band_arrays(
    read_u8: np.ndarray,
    cand_u8: np.ndarray,
    read_codes: np.ndarray,
    read_has_n: np.ndarray,
    cand_index: _CodeIndex,
    diag: int,
    *,
    band: int,
    disc_flag: np.ndarray | None,
) -> np.ndarray:
    """Dense (C, H, 2*band+1) float32 band image around `diag`."""
    h, w = len(read_u8), len(cand_u8)
    out = np.zeros((N_CHANNELS, h, 2 * band + 1), dtype=np.float32)
    if h == 0 or w == 0:
        return out
    offsets = np.arange(-band, band + 1)
    i_axis = np.arange(h)[:, None]
    j_grid = i_axis + diag + offsets[None, :]
    valid = (j_grid >= 0) & (j_grid < w)
    j_clip = np.clip(j_grid, 0, w - 1)

    # channels 1/2: dense diagonalized per-base match / mismatch
    read_b = read_u8[:, None]
    cand_b = cand_u8[j_clip]
    eq = (read_b == cand_b) & valid
    out[CH_MATCH] = eq
    out[CH_MISMATCH] = valid & ~eq

    # channels 3-5: sparse greedy-path error events
    if _path_events_nb is not None:
        ev_ch, ev_i, ev_o = _path_events_nb(read_u8, cand_u8, diag, band)
        out[ev_ch, ev_i, ev_o + band] = 1.0
    else:
        for ch, i, o in _path_events(read_u8, cand_u8, diag, band):
            out[ch, i, o + band] = 1.0

    # channel 0: in-band exact anchor dots
    mi, mj = _match_pairs(read_codes, read_has_n, cand_index)
    if len(mi):
        off = (mj - mi) - diag
        keep = np.abs(off) <= band
        if keep.any():
            out[CH_ANCHOR, mi[keep], off[keep] + band] = 1.0

    # channel 6: candidate-side distinguishing-word flags (per column j,
    # flag of the 8-mer STARTING at j; columns without a full k-mer are 0)
    if disc_flag is not None and len(disc_flag):
        flags = np.zeros(w, dtype=np.float32)
        flags[: len(disc_flag)] = disc_flag
        out[CH_DISC] = np.where(valid, flags[j_clip], 0.0)

    # channels 7/8: absolute position priors (valid cells only)
    out[CH_ROWPOS] = np.where(valid, i_axis / h, 0.0)
    out[CH_COLPOS] = np.where(valid, j_grid / w, 0.0).clip(0.0, 1.0)
    return out


def disc_flags(cand_codes: np.ndarray, cand_has_n: np.ndarray, other_union: np.ndarray) -> np.ndarray:
    """Per-k-mer-position flag: 8-mer at j is intact and absent from others."""
    if len(cand_codes) == 0:
        return np.zeros(0, np.float32)
    return ((~other_union[cand_codes.astype(np.int64)]) & (~cand_has_n)).astype(np.float32)


def other_union_lookup(other_sequences: Sequence[str], *, k: int = K_ANCHOR, cand_cap: int = CAND_CAP) -> np.ndarray:
    """Two-strand 8-mer union of the other candidates as a 65536-bool lookup."""
    lut = np.zeros(1 << (2 * k), dtype=bool)
    for seq in other_sequences:
        u8 = encode_u8(seq, cand_cap)
        for strand in (u8, reverse_complement_u8(u8)):
            codes, has_n = kmer_codes(strand, k)
            if len(codes):
                lut[codes[~has_n].astype(np.int64)] = True
    return lut


def rust_bands_module():
    """The m2d_bands Rust kernel module, or None when not importable."""
    return _rust_bands


def resolve_band_backend(backend: str) -> str:
    """Resolve "auto"/"rust"/"python" to a concrete band backend name.

    TRACE_LM_BAND_BACKEND overrides "auto"; requesting "rust" without the
    kernel raises (fail-closed, no silent fallback in recorded runs).
    """
    if backend == "auto":
        backend = os.environ.get("TRACE_LM_BAND_BACKEND", "auto")
    if backend == "auto":
        return "rust" if _rust_bands is not None else "python"
    if backend == "rust" and _rust_bands is None:
        raise RuntimeError("band backend 'rust' requested but m2d_bands is not importable")
    if backend not in ("rust", "python"):
        raise ValueError(f"unknown band backend: {backend!r}")
    return backend


def window_lut(sequence: str, *, k: int = K_ANCHOR, cand_cap: int = CAND_CAP) -> np.ndarray:
    """Two-strand k-mer presence lookup for ONE candidate window (65536 bools)."""
    lut = np.zeros(1 << (2 * k), dtype=bool)
    u8 = encode_u8(sequence, cand_cap)
    for strand in (u8, reverse_complement_u8(u8)):
        codes, has_n = kmer_codes(strand, k)
        if len(codes):
            lut[codes[~has_n].astype(np.int64)] = True
    return lut


def group_other_unions(luts: Sequence[np.ndarray]) -> list[np.ndarray | None]:
    """Per-member union of the OTHER members' k-mers from per-window LUTs.

    Bit-identical to calling other_union_lookup on each member's complement:
    boolean OR is associative/commutative, so the prefix/suffix ORs reproduce
    the exact 65536-bool lookup while encoding each window ONCE — O(K) window
    builds instead of the O(K^2) of per-candidate rebuilds (the dominant cost
    on large candidate groups; same identity as the CELLO counts-based port).
    Single-member groups return [None] (no others), matching the reference.

    Returned arrays are marked read-only: boundary members share storage with
    the prefix/suffix tables, and downstream (disc_flags) is lookup-only, so
    any in-place mutation of a returned union would silently corrupt the
    shared tables."""
    n = len(luts)
    if n <= 1:
        return [None] * n
    pre = [luts[0]]
    for lut in luts[1:]:
        pre.append(pre[-1] | lut)
    suf: list[np.ndarray | None] = [None] * n
    suf[n - 1] = luts[n - 1]
    for i in range(n - 2, -1, -1):
        suf[i] = suf[i + 1] | luts[i]
    out: list[np.ndarray | None] = []
    for i in range(n):
        union = suf[1] if i == 0 else (pre[n - 2] if i == n - 1 else pre[i - 1] | suf[i + 1])
        union.flags.writeable = False
        out.append(union)
    return out


@dataclass(frozen=True, slots=True)
class PairMatrix:
    """Sparse (C, H, W) alignment matrix, stored as a diagonal band + meta.

    The canonical sparse view is to_coo(): (channel, row, col) indices with
    absolute column j and float values; nnz is confined to the band by
    construction.  densify_band() returns the (C, H, 2*BAND+1) image the
    CNN consumes; to_dense_full() materialises the (C, H, W) grid (tests).
    """

    band_array: np.ndarray   # float32 (C, H, 2*band+1)
    diagonal: int            # d* = j - i of the voted main diagonal
    strand: int              # 0 forward read, 1 reverse-complement read
    anchor_votes: int        # in-band anchor count of the chosen orientation
    band: int
    read_used: int           # H (after cap)
    cand_used: int           # W (after cap)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (N_CHANNELS, self.read_used, self.cand_used)

    def to_coo(self) -> tuple[np.ndarray, np.ndarray]:
        """(3, nnz) int32 [channel, row, col(absolute j)] + (nnz,) float32 values."""
        c_idx: list[np.ndarray] = []
        r_idx: list[np.ndarray] = []
        j_idx: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        rows = np.arange(self.read_used, dtype=np.int32)
        for ch in range(N_CHANNELS):
            plane = self.band_array[ch]
            ii, oo = np.nonzero(plane)
            if len(ii) == 0:
                continue
            val = plane[ii, oo].astype(np.float32)
            c_idx.append(np.full(len(ii), ch, np.int32))
            r_idx.append(ii.astype(np.int32))
            j_idx.append((rows[ii] + self.diagonal + (oo - self.band)).astype(np.int32))
            vals.append(val)
        if not c_idx:
            return np.zeros((3, 0), np.int32), np.zeros(0, np.float32)
        return np.stack([np.concatenate(c_idx), np.concatenate(r_idx), np.concatenate(j_idx)]), np.concatenate(vals)

    def densify_band(self) -> np.ndarray:
        return self.band_array

    def to_dense_full(self) -> np.ndarray:
        dense = np.zeros((N_CHANNELS, self.read_used, self.cand_used), np.float32)
        coords, vals = self.to_coo()
        dense[coords[0], coords[1], coords[2]] = vals
        return dense

    def nnz(self) -> int:
        return int(np.count_nonzero(self.band_array))


def build_pair_matrix(
    read: str,
    candidate: str,
    other_union: np.ndarray | None = None,
    *,
    k: int = K_ANCHOR,
    band: int = BAND,
    read_cap: int = READ_CAP,
    cand_cap: int = CAND_CAP,
    orient: bool = True,
) -> PairMatrix:
    """Pure function: two sequences (+ other-candidate 8-mer union) -> sparse
    (C, H, W) alignment matrix restricted to the voted band.

    `other_union` is a 65536-bool lookup of the other candidates' two-strand
    k-mers (see other_union_lookup); None disables the disc-word channel.
    """
    cand_u8 = encode_u8(candidate, cand_cap)
    cand_codes, cand_has_n = kmer_codes(cand_u8, k)
    cand_index = _CodeIndex.build(cand_codes, cand_has_n)
    read_fwd = encode_u8(read, read_cap)
    read_rc = reverse_complement_u8(read_fwd)

    def orientation(u8: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
        codes, has_n = kmer_codes(u8, k)
        mi, mj = _match_pairs(codes, has_n, cand_index)
        diag = _vote_diagonal(mi, mj, span=len(cand_u8), h=len(u8))
        in_band = np.abs((mj - mi) - diag) <= band
        return codes, has_n, diag, int(np.count_nonzero(in_band))

    f_codes, f_has_n, f_diag, f_votes = orientation(read_fwd)
    if orient:
        r_codes, r_has_n, r_diag, r_votes = orientation(read_rc)
        if r_votes > f_votes:
            read_u8, read_codes, read_has_n, diag, strand, votes = read_rc, r_codes, r_has_n, r_diag, 1, r_votes
        else:
            read_u8, read_codes, read_has_n, diag, strand, votes = read_fwd, f_codes, f_has_n, f_diag, 0, f_votes
    else:
        read_u8, read_codes, read_has_n, diag, strand, votes = read_fwd, f_codes, f_has_n, f_diag, 0, f_votes

    flags = None
    if other_union is not None:
        flags = disc_flags(cand_codes, cand_has_n, other_union)
    band_array = _band_arrays(
        read_u8, cand_u8, read_codes, read_has_n, cand_index, diag,
        band=band, disc_flag=flags,
    )
    return PairMatrix(
        band_array=band_array,
        diagonal=diag,
        strand=strand,
        anchor_votes=votes,
        band=band,
        read_used=len(read_u8),
        cand_used=len(cand_u8),
    )


def build_pair_band(
    read: str,
    candidate: str,
    other_union: np.ndarray | None = None,
    *,
    k: int = K_ANCHOR,
    band: int = BAND,
    read_cap: int = READ_CAP,
    cand_cap: int = CAND_CAP,
    orient: bool = True,
) -> np.ndarray:
    """Fast path: dense (C, H, 2*band+1) float32 band only (training/inference)."""
    return build_pair_matrix(
        read, candidate, other_union, k=k, band=band,
        read_cap=read_cap, cand_cap=cand_cap, orient=orient,
    ).densify_band()


def band_summary(band_array: np.ndarray) -> dict:
    """Per-channel row-density summary for receipts/diagnostics."""
    names = ["anchor", "match", "mismatch", "sub", "ins", "del", "disc_word", "row_pos", "col_pos"]
    h = band_array.shape[1] or 1
    return {name: round(float(band_array[ch].sum() / h), 4) for ch, name in enumerate(names)}
