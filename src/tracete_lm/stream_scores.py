"""Memory-bounded scoring with the reference alignment-band representation."""
from __future__ import annotations


def matrix2d_scores_streaming(model, read, candidate_windows, *, device='cuda',
                              batch_size=4, read_cap=4096, cand_cap=4096):
    import torch
    from tracete_lm.m2d_matrix import window_lut, group_other_unions, build_pair_band
    from tracete_lm.m2d_model import collate_bands

    windows = list(candidate_windows)
    luts = [window_lut(w, cand_cap=cand_cap) for w in windows]
    unions = group_other_unions(luts)
    model.eval()
    scores = []
    buf = []

    def flush():
        if not buf:
            return
        x, mask = collate_bands(buf)
        with torch.no_grad():
            logits = model(x.to(device), mask.to(device))
        scores.extend(float(v) for v in logits.float().cpu())
        buf.clear()

    for cand, union in zip(windows, unions):
        buf.append(build_pair_band(read, cand, union, read_cap=read_cap, cand_cap=cand_cap))
        if len(buf) >= batch_size:
            flush()
    flush()
    del luts, unions
    return scores
