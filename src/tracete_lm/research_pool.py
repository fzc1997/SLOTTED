"""Calibrated pooling of seed, likelihood and learned-model evidence."""
from __future__ import annotations
import numpy as np


EPS_POOL = 0.02


def cf_innovation(s_cf: list[float], s_k31: list[float], beta0: float,
                  force_zero: bool = False) -> list[float]:
    """h = s_cf - beta0*s_k31; far-word invariant guard: when every read word
    is far for every candidate (seeds all zero and CF flat) h must be the zero
    vector so the innovation channel is uniform (no cap artifacts)."""
    if force_zero:
        return [0.0] * len(s_cf)
    return [cf - beta0 * k for cf, k in zip(s_cf, s_k31)]


def is_cf_flat(s_k31: list[float], s_cf: list[float]) -> bool:
    """Far-word guard condition: all seed counts zero AND s_cf variance ~0."""
    if max(s_k31) > 0:
        return False
    v = np.asarray(s_cf, dtype=np.float64)
    return bool(len(v) <= 1 or v.std() <= 1e-9 * max(1.0, float(np.abs(v).max())))


def softmax_T(x: np.ndarray, t: float) -> np.ndarray:
    z = (x - x.max()) / t
    e = np.exp(z)
    return e / e.sum()


def channel_probs(scores_of: dict, t: float, ids: list,
                  eps: float = EPS_POOL) -> dict:
    """p_j = (1-eps)*softmax(x/T) + eps/K per read."""
    out = {}
    for rid in ids:
        v = np.asarray(scores_of[rid], dtype=np.float64)
        k = len(v)
        if k <= 1:
            out[rid] = [1.0]
            continue
        p = (1.0 - eps) * softmax_T(v, t) + eps / k
        out[rid] = [float(x) for x in p]
    return out


def seed_quality_q(k31: list[float], n_rw: int, rho_star: float) -> float:
    return float(min(1.0, max(0.0, (max(k31) / max(int(n_rw), 1)) / rho_star)))


def pool_log_emission(p_k: list[float], p_m: list[float], p_h: list[float],
                      q: float) -> list[float]:
    wk = 0.50 * q
    wm = 0.35 + 0.25 * (1.0 - q)
    wh = 0.15 + 0.25 * (1.0 - q)
    return [wk * np.log(max(a, 1e-300)) + wm * np.log(max(b, 1e-300))
            + wh * np.log(max(c, 1e-300))
            for a, b, c in zip(p_k, p_m, p_h)]

