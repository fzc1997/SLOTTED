"""Optional newly evaluated report layer; delegates to the existing strict guard."""
from __future__ import annotations
from dataclasses import asdict

from tracete_lm.selective import decide_selective,read_kmers,window_kmers

POST_EM_MARGIN=0.10
POST_EM_SEED_FLOOR=5
GUARD_READ_CAP=4096


def guard_seed_counts(data):
    """The original CLI/B guard vector uses read[:4096], with full candidate windows."""
    windows={l:window_kmers(w,31) for l,w in data['windows'].items()}
    return {q:[len(read_kmers(data['seqs'][q][:GUARD_READ_CAP],31)&windows[l]) for l in data['cands'][q]]
            for q in data['ids']}


def apply_post_em_guard(data,guard_k31_4096,gamma,read_start):
    """Apply the strict rule to EM gamma with the original 4096bp-prefix guard seeds."""
    decisions={}
    for i,q in enumerate(data['fit_ids']):
        start,end=map(int,read_start[i:i+2])
        seeds=guard_k31_4096[q]
        assert len(seeds)==end-start==len(data['cands'][q])
        dec=decide_selective(gamma[start:end].tolist(),seeds,margin=POST_EM_MARGIN,min_seed=POST_EM_SEED_FLOOR)
        decisions[q]=asdict(dec)
    for q in data['ids']:
        if q not in decisions:
            assert not data['cands'][q]
            decisions[q]=asdict(decide_selective([],[],margin=POST_EM_MARGIN,min_seed=POST_EM_SEED_FLOOR))
    return decisions
