"""Variant 2 model: small ResNet-18-level CNN over the 2D alignment band.

Input: (B, 9, H<=4096, 65) diagonal-band images (channel semantics in
tracete_lm.m2d_matrix).  H is padded to a multiple of `ALIGN_MULTIPLE`
inside forward; padded rows are excluded from ALL normalisation statistics
(MaskedGroupNorm) and from pooling, so a sample's logit is invariant to
how much padding was appended for batching.  W (65, the diagonal-offset
axis) is fixed by the band.  Output: one scalar logit per (read,
candidate) matrix — the per-candidate score head.

Architecture (width 96 ~= 25M params, "小 ResNet-18 级"):
    stem  conv3x3 stride (2,1)          9 -> 96
    stage1 2 x residual, stride (1,1)   96 -> 96
    stage2 2 x residual, stride (2,2)   96 -> 192
    stage3 2 x residual, stride (2,2)   192 -> 384
    stage4 2 x residual, stride (2,2)   384 -> 768
    masked mean-pool over H -> MLP(768 -> 256 -> 1)
H is reduced 16x, W 65 -> 33 -> 17 -> 9.  MaskedGroupNorm(32) throughout
(BatchNorm statistics are unstable at small per-GPU batch and variable H;
plain GroupNorm would let padding rows shift per-sample statistics).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .m2d_matrix import (
    BAND_WIDTH,
    CAND_CAP,
    N_CHANNELS,
    READ_CAP,
    build_pair_band,
    group_other_unions,
    resolve_band_backend,
    window_lut,
)

try:  # optional Rust kernel, bit-identical port (rust/m2d_bands)
    import m2d_bands as _m2d_bands_rust
except ImportError:  # pragma: no cover
    _m2d_bands_rust = None

ALIGN_MULTIPLE = 16


@dataclass(frozen=True, slots=True)
class M2DConfig:
    in_channels: int = N_CHANNELS
    width: int = 96
    blocks: tuple[int, ...] = (2, 2, 2, 2)
    head_dim: int = 256
    dropout: float = 0.0
    band_width: int = BAND_WIDTH


class MaskedGroupNorm(nn.Module):
    """GroupNorm computed over valid rows only.

    x (B, C, H, W), mask (B, H): statistics per (sample, channel-group)
    over the C_g x H_valid x W cells; padded rows are zeroed on output so
    downstream layers see exact zeros there.
    """

    def __init__(self, channels: int, groups: int = 32) -> None:
        super().__init__()
        if channels % groups:
            groups = 1  # narrow test configs; production widths divide 32
        self.groups = groups
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        g = self.groups
        m = mask[:, None, :, None].to(x.dtype)  # (B,1,H,1)
        count = mask.sum(-1).to(x.dtype) * w  # (B,)
        count = count.clamp(min=1.0)
        xg = x.view(b, g, c // g, h, w)
        mg = m[:, None]  # (B,1,1,H,1)
        denom = (count * (c // g))[:, None]  # (B,1)
        mean = (xg * mg).sum(dim=(2, 3, 4)) / denom  # (B,G)
        var = ((xg * mg) ** 2).sum(dim=(2, 3, 4)) / denom - mean**2
        normed = (xg - mean[:, :, None, None, None]) / torch.sqrt(var + 1e-5)[:, :, None, None, None]
        out = normed.view(b, c, h, w) * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return out * m


class ResidualBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: tuple[int, int], groups: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=tuple(stride), padding=1, bias=False)
        self.norm1 = MaskedGroupNorm(cout, groups)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.norm2 = MaskedGroupNorm(cout, groups)
        self.short: nn.Sequential | None = None
        self.short_stride = (1, 1)
        if cin != cout or tuple(stride) != (1, 1):
            self.short = nn.Conv2d(cin, cout, 1, stride=tuple(stride), bias=False)
            self.short_stride = tuple(stride)

    def _down_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if self.short_stride[0] == 2:
            return mask[:, ::2]
        return mask

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m_in = self._down_mask(mask)  # conv1 stride mask (input rows [2f, 2f+1])
        out = F.gelu(self.norm1(self.conv1(x), m_in))
        out = self.norm2(self.conv2(out), m_in)
        identity = self.short(x) if self.short is not None else x
        out = F.gelu(out + identity * m_in[:, None, :, None].to(out.dtype))
        return out * m_in[:, None, :, None].to(out.dtype)


class BandCNN(nn.Module):
    """Per-candidate score head over diagonal-band alignment images."""

    def __init__(self, config: M2DConfig | None = None) -> None:
        super().__init__()
        self.config = config = config or M2DConfig()
        w = config.width
        self.stem_conv = nn.Conv2d(config.in_channels, w, 3, stride=(2, 1), padding=1, bias=False)
        self.stem_norm = MaskedGroupNorm(w, 32)
        strides = ((1, 1), (2, 2), (2, 2), (2, 2))
        widths = [w, w * 2, w * 4, w * 8]
        stages = []
        in_ch = w
        for blocks, stride, cout in zip(config.blocks, strides, widths):
            stage = [ResidualBlock(in_ch, cout, stride)]
            stage.extend(ResidualBlock(cout, cout, (1, 1)) for _ in range(blocks - 1))
            stages.append(nn.ModuleList(stage))
            in_ch = cout
        self.stages = nn.ModuleList(stages)
        self.head = nn.Sequential(
            nn.Linear(in_ch, config.head_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_dim, 1),
        )

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor, row_valid: torch.Tensor) -> torch.Tensor:
        """x (B, C, H, W); row_valid (B, H) bool -> logits (B,).

        row_valid marks real read rows; padded rows are zeroed at input and
        excluded from normalisation and pooling (logits are invariant to
        appended padding — unit tested).
        """
        bsz, _, height, _ = x.shape
        if height % ALIGN_MULTIPLE:
            pad = ALIGN_MULTIPLE - height % ALIGN_MULTIPLE
            x = F.pad(x, (0, 0, 0, pad))
            row_valid = F.pad(row_valid, (0, pad))
        x = x * row_valid[:, None, :, None].to(x.dtype)
        mask = row_valid[:, ::2]  # stem H-stride 2
        x = F.gelu(self.stem_norm(self.stem_conv(x), mask))
        for stage in self.stages:
            for bi, block in enumerate(stage):
                x = block(x, mask)
                if bi == 0 and block.short_stride[0] == 2:
                    # block0 consumed the stage stride: the remaining blocks of
                    # this stage run at the halved resolution
                    mask = mask[:, ::2]
        feat_mask = mask.to(x.dtype)
        x = x * feat_mask[:, None, :, None]
        pooled = x.sum(dim=(2, 3)) / (feat_mask.sum(-1) * x.shape[3]).clamp(min=1.0)[:, None]
        return self.head(pooled).squeeze(-1)


def collate_bands(bands: list[np.ndarray], *, multiple: int = ALIGN_MULTIPLE) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of (C, H, 65) band arrays to one batch tensor + row mask."""
    hmax = max(b.shape[1] for b in bands)
    hpad = ((hmax + multiple - 1) // multiple) * multiple
    x = torch.zeros(len(bands), bands[0].shape[0], hpad, bands[0].shape[2])
    mask = torch.zeros(len(bands), hpad, dtype=torch.bool)
    for i, b in enumerate(bands):
        x[i, :, : b.shape[1]] = torch.from_numpy(b)
        mask[i, : b.shape[1]] = True
    return x, mask


def candidate_group_bands(
    read: str,
    candidate_windows: list[str],
    *,
    band_width: int = BAND_WIDTH,
    read_cap: int = READ_CAP,
    cand_cap: int = CAND_CAP,
    backend: str = "auto",
) -> list[np.ndarray]:
    """Band images for one read against its full candidate group (the disc-word
    channel needs the union of the OTHER candidates, so build via the group).

    Per-window two-strand k-mer LUTs are built once and combined with
    prefix/suffix ORs (group_other_unions) — array-identical to the reference
    per-candidate other_union_lookup rebuilds, which are O(K^2) in the group
    size and dominate the bench on large candidate groups.

    backend: "auto" uses the m2d_bands Rust kernel when importable (falling
    back to the python/numba path otherwise); "rust" fails closed if the
    kernel is missing; "python" forces the python path.  All backends are
    bit-identical (tests/test_m2d_band_fast.py); TRACE_LM_BAND_BACKEND
    overrides "auto" for recorded runs."""
    if band_width != BAND_WIDTH:
        raise ValueError("band width is fixed by the representation (BAND=32)")
    if resolve_band_backend(backend) == "rust":
        return list(_m2d_bands_rust.candidate_group_bands(read, list(candidate_windows), read_cap=read_cap, cand_cap=cand_cap))
    unions = group_other_unions([window_lut(w, cand_cap=cand_cap) for w in candidate_windows])
    return [
        build_pair_band(read, cand, union, read_cap=read_cap, cand_cap=cand_cap)
        for cand, union in zip(candidate_windows, unions)
    ]


@torch.no_grad()
def matrix2d_scores(
    model: BandCNN,
    read: str,
    candidate_windows: list[str],
    *,
    device: str = "cuda",
    batch_size: int = 8,
    read_cap: int = READ_CAP,
    cand_cap: int = CAND_CAP,
    band_backend: str = "auto",
) -> list[float]:
    """Per-candidate logits for one read group (evaluation scorer)."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model.eval()
    bands = candidate_group_bands(read, candidate_windows, read_cap=read_cap,
                                  cand_cap=cand_cap, backend=band_backend)
    scores: list[float] = []
    for start in range(0, len(bands), batch_size):
        chunk = bands[start : start + batch_size]
        x, mask = collate_bands(chunk)
        logits = model(x.to(device), mask.to(device))
        scores.extend(float(v) for v in logits.float().cpu())
    return scores


@torch.no_grad()
def batched_read_group_logits(
    model: BandCNN,
    groups: Sequence[tuple[str, list[str]]],
    *,
    device: str = "cuda",
    max_candidates: int = 512,
    max_rows: int = 200000,
    read_cap: int = READ_CAP,
    cand_cap: int = CAND_CAP,
    band_backend: str = "auto",
    progress_every: int = 0,
) -> list[list[float]]:
    """Cross-read batched scoring of whole read groups (the fast replay path).

    Candidates from MANY reads share one forward: the per-read loop's
    collate -> forward -> .cpu() per read (one device sync per read) collapses
    to one sync per assembled batch.  Groups are processed in read-length
    order (H bucketing limits padding waste); a group's candidates stay
    contiguous and whole, so per-group logits are returned in candidate order.
    Groups larger than max_candidates fall back to per-group chunks of the
    reference size (128), mirroring the frozen per-read recipe.  Forwards run
    under the same bf16 autocast as the frozen recipe.

    Numerics: logits may differ from per-read forwards in low-order bits
    (cuDNN algorithm choice depends on batch composition).  Pair this with an
    argmax verification against the frozen per-read recipe and an exact
    per-read recompute for any flipped read.
    """
    reads = [(seq[:read_cap], cands) for seq, cands in groups]
    order = sorted(range(len(reads)), key=lambda i: (len(reads[i][0]), i))
    results: list[list[float]] = [[] for _ in reads]
    pend_bands: list[np.ndarray] = []
    pend_owner: list[int] = []
    pend_rows = 0
    autocast_kw = dict(device_type=device.split(":")[0], dtype=torch.bfloat16,
                       enabled=device.startswith("cuda"))
    n_flush = 0

    def _collate_np(bands: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        # numpy-side batch assembly: single-threaded memcpy (a per-band
        # torch.from_numpy assignment wakes the whole OMP pool per band,
        # which pathologically starves the GPU on many-core hosts)
        hpad = max(b.shape[1] for b in bands)
        hpad = ((hpad + ALIGN_MULTIPLE - 1) // ALIGN_MULTIPLE) * ALIGN_MULTIPLE
        x = np.zeros((len(bands), bands[0].shape[0], hpad, bands[0].shape[2]), dtype=np.float32)
        mask = np.zeros((len(bands), hpad), dtype=bool)
        for i, b in enumerate(bands):
            x[i, :, : b.shape[1]] = b
            mask[i, : b.shape[1]] = True
        return torch.from_numpy(x), torch.from_numpy(mask)

    def flush() -> None:
        nonlocal pend_bands, pend_owner, pend_rows, n_flush
        if not pend_bands:
            return
        x, mask = _collate_np(pend_bands)
        with torch.autocast(**autocast_kw):
            out = model(x.to(device), mask.to(device))
        vals = out.float().cpu()  # the single sync for this batch
        for ci, ji in enumerate(pend_owner):
            results[ji].append(float(vals[ci]))
        pend_bands, pend_owner, pend_rows = [], [], 0
        n_flush += 1
        if progress_every and n_flush % progress_every == 0:
            import sys as _sys
            print(f"[batched_read_group_logits] batch {n_flush} "
                  f"({sum(len(r) for r in results)} candidates scored)",
                  file=_sys.stderr, flush=True)

    def forward_chunk(job: int, bands: list[np.ndarray]) -> None:
        x, mask = _collate_np(bands)
        with torch.autocast(**autocast_kw):
            out = model(x.to(device), mask.to(device))
        results[job].extend(float(v) for v in out.float().cpu())

    for ji in order:
        seq, cands = reads[ji]
        if not seq:
            raise ValueError(
                "batched_read_group_logits: empty read sequence (zero-height bands "
                "cannot be forwarded); filter empty reads upstream")
        bands = candidate_group_bands(seq, cands, read_cap=read_cap, cand_cap=cand_cap, backend=band_backend)
        rows = max((b.shape[1] for b in bands), default=0) * len(bands)  # padded-rows upper bound
        if len(bands) > max_candidates or rows > max_rows:
            # giant group (too many candidates or too many padded rows, e.g.
            # full-length bands): reference-size chunks, mirroring the frozen
            # per-read recipe and bounding peak memory
            flush()
            for s in range(0, len(bands), 128):
                forward_chunk(ji, bands[s : s + 128])
            continue
        while pend_bands and (len(pend_bands) + len(bands) > max_candidates or pend_rows + rows > max_rows):
            flush()
        pend_bands.extend(bands)
        pend_owner.extend([ji] * len(bands))
        pend_rows += rows
    flush()
    return results


@torch.no_grad()
def batched_group_logits_verified(
    model: BandCNN,
    groups: Sequence[tuple[str, list[str]]],
    reference_logits: Sequence[Sequence[float]] | dict,
    *,
    device: str = "cuda",
    max_candidates: int = 512,
    max_rows: int = 200000,
    read_cap: int = READ_CAP,
    cand_cap: int = CAND_CAP,
    band_backend: str = "auto",
    progress_every: int = 0,
    reference_batch_size: int = 128,
) -> tuple[list[list[float]], dict]:
    """batched_read_group_logits + the frozen-recipe safety net.

    Any group whose batched argmax disagrees with `reference_logits`' argmax
    (first-index tie-break, empty groups skipped) is recomputed with the
    EXACT per-read recipe (reference_batch_size chunks, default 128), so the returned logits' argmax
    equals the reference argmax for every group that the per-read recipe
    itself reproduces.  This is mandatory for recorded/replay runs: batch
    composition shifts cuDNN algorithm choice and flips near-tie argmaxes at
    a ~0.1% rate (nanosim: 4-5/6046, each fixed by the fallback in ~1s).

    Returns (logits, stats); stats reports flips, fallback fixes and
    still_bad groups (reference argmax not reproduced even by the exact
    recipe — a cross-device artifact to surface, never to hide).
    """
    if reference_batch_size <= 0:
        raise ValueError("reference_batch_size must be positive")
    groups = list(groups)
    if isinstance(reference_logits, dict):
        ref_list = [reference_logits[i] for i in range(len(groups))]
    else:
        ref_list = list(reference_logits)
    if len(ref_list) != len(groups):
        raise ValueError(f"reference_logits has {len(ref_list)} entries, groups has {len(groups)}")
    for gi, (ref, (_, candidates)) in enumerate(zip(ref_list, groups)):
        if len(ref) != len(candidates) or not np.isfinite(ref).all():
            raise ValueError(f"reference_logits group {gi} must have one finite score per candidate")

    logits = batched_read_group_logits(
        model, groups, device=device, max_candidates=max_candidates, max_rows=max_rows,
        read_cap=read_cap, cand_cap=cand_cap, band_backend=band_backend,
        progress_every=progress_every,
    )
    if len(logits) != len(groups):
        raise ValueError("batched scorer returned a different number of read groups")
    for gi, (scores, (_, candidates)) in enumerate(zip(logits, groups)):
        if len(scores) != len(candidates) or not np.isfinite(scores).all():
            raise ValueError(f"batched logits group {gi} must have one finite score per candidate")

    def argmax_first(scores: list[float]) -> int:
        best = max(scores)
        return scores.index(best)

    flips: list[int] = []
    for gi, (scores, ref) in enumerate(zip(logits, ref_list)):
        if not scores or not ref:
            continue  # K=0 groups: nothing to verify
        if argmax_first(scores) != argmax_first(list(ref)):
            flips.append(gi)

    still_bad: list[int] = []
    max_delta = 0.0
    for gi in flips:
        seq, cands = groups[gi][0][:read_cap], groups[gi][1]
        bands = candidate_group_bands(seq, cands, read_cap=read_cap, cand_cap=cand_cap,
                                      backend=band_backend)
        exact: list[float] = []
        for s in range(0, len(bands), reference_batch_size):
            x, mask = collate_bands(bands[s : s + reference_batch_size])
            autocast_kw = dict(device_type=device.split(":")[0], dtype=torch.bfloat16,
                               enabled=device.startswith("cuda"))
            with torch.autocast(**autocast_kw):
                out = model(x.to(device), mask.to(device))
            exact.extend(float(v) for v in out.float().cpu())
        if not np.isfinite(exact).all():
            raise ValueError(f"reference fallback produced non-finite logits for group {gi}")
        logits[gi] = exact
        if argmax_first(exact) != argmax_first(list(ref_list[gi])):
            still_bad.append(gi)
        max_delta = max(max_delta, max(abs(a - b) for a, b in zip(exact, ref_list[gi])))

    stats = {"n_groups": len(groups), "n_flips": len(flips), "fallback_fixed": len(flips) - len(still_bad),
             "still_bad": still_bad, "fallback_max_abs_diff_vs_ref": round(max_delta, 6)}
    return logits, stats


def group_softmax_probs(scores: list[float]) -> list[float]:
    """Softmax over one read group's candidate logits (prob-margin decisions)."""
    t = torch.tensor(scores, dtype=torch.float64)
    p = torch.softmax(t, dim=-1)
    return [float(v) for v in p]
