"""SLOTTED EM engine, extracted VERBATIM from the published pipeline.

Source: scripts/upgrade_v2/d7/run_d7_em_baseline.py (the Telescope-style
global expression-prior EM used across the frozen benchmark receipts:
TS3 recipe of run_rp_realdata_prior.py, em_as / em_k31 / m2d_emission_em
arms of the nanosim head-to-head).  Only the constants block and the two
functions below were copied; no logic was changed.  The upstream file also
carries corpus/benchmark glue that is intentionally NOT part of the tool
release.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EM_MAX_ITER = 200
EM_TOL = 1e-7
PI_FLOOR = 1e-9

@dataclass
class EMResult:
    gamma: np.ndarray        # per-edge responsibilities, read-major CSR order
    pi: np.ndarray           # locus expression proportions (len = n loci)
    loglik_trace: list[float]
    converged: bool
    n_iter: int


def em_locus_assignment(
    edge_score: np.ndarray,
    edge_locus: np.ndarray,
    read_start: np.ndarray,
    n_loci: int,
    *,
    temperature: float = 1.0,
    pi_init: np.ndarray | None = None,
    max_iter: int = EM_MAX_ITER,
    tol: float = EM_TOL,
    em: bool = True,
) -> EMResult:
    """Mixture EM over locus expression with fixed exp-family emissions.

    Model:  p(read r) = sum_{l in C_r} pi_l * exp(s_rl / T)  (per-read
    constants absorbed by normalization); E-step responsibilities gamma_rl;
    M-step pi_l = mean_r gamma_rl.  All in log space; the log-likelihood trace
    is evaluated at the pi used in each E-step (monotone non-decreasing).
    em=False keeps pi at its initialization (uniform-prior ablation).
    """
    edge_score = np.asarray(edge_score, dtype=np.float64)
    edge_locus = np.asarray(edge_locus, dtype=np.int64)
    read_start = np.asarray(read_start, dtype=np.int64)
    n_reads = len(read_start) - 1
    counts = np.diff(read_start)
    assert (counts > 0).all(), "every read in the fit must have >=1 candidate"
    base = edge_score / float(temperature)

    if pi_init is None:
        pi = np.full(n_loci, 1.0 / n_loci, dtype=np.float64)
    else:
        pi = np.array(pi_init, dtype=np.float64)
        pi = pi / pi.sum()

    def e_step(p: np.ndarray):
        logits = np.log(p)[edge_locus] + base
        lse = np.logaddexp.reduceat(logits, read_start[:-1])
        gamma = np.exp(logits - np.repeat(lse, counts))
        return gamma, float(lse.sum())

    gamma, ll = e_step(pi)
    trace = [ll]
    converged, n_iter = (True, 0) if not em else (False, 0)
    if em:
        for n_iter in range(1, max_iter + 1):
            new_pi = np.bincount(edge_locus, weights=gamma, minlength=n_loci) / n_reads
            new_pi = np.maximum(new_pi, PI_FLOOR)  # guard exp-underflow loci
            gamma, ll = e_step(new_pi)
            trace.append(ll)
            delta = float(np.abs(new_pi - pi).max())
            pi = new_pi
            if delta < tol:
                converged = True
                break
    return EMResult(gamma=gamma, pi=pi, loglik_trace=trace, converged=converged, n_iter=n_iter or 1)


def per_read_top2(gamma: np.ndarray, read_start: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-1 / top-2 value and argmax edge index per read (CSR segments).

    Argmax ties break toward the lowest edge index within the segment (the
    alphabetically-first locus_id in this pipeline's edge ordering).
    """
    starts = read_start[:-1]
    counts = np.diff(read_start)
    top1 = np.maximum.reduceat(gamma, starts)
    seg_id = np.repeat(np.arange(len(starts)), counts)
    hits = np.flatnonzero(gamma >= np.repeat(top1, counts))
    winners = hits[np.unique(seg_id[hits], return_index=True)[1]]
    g2 = gamma.copy()
    g2[winners] = -np.inf
    top2 = np.maximum.reduceat(g2, starts)
    top2[~np.isfinite(top2)] = 0.0  # single-candidate reads
    return top1, top2, winners


