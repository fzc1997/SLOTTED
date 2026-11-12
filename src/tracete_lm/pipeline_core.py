"""Joint candidate scoring and whole-population EM assignment."""
from __future__ import annotations
import hashlib
import json
import math
import numbers
from pathlib import Path
import time

import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def score(bundle,d,configurations,*,scorer=None,checkpoint=None,pool_params=None):
    from tracete_lm.selective import read_kmers,window_kmers
    from tracete_lm import research_pool
    ids=d['fit_ids'];cands=d['cands'];wins=d['windows'];seqs=d['seqs']
    pool=json.loads(Path(pool_params or bundle/'assets/pool_frozen_params_v2.json').read_text())
    if not isinstance(pool,dict) or not isinstance(pool.get('temperatures'),dict):
        raise ValueError('pool parameters must include a temperatures dictionary')
    positive={'rho_star':pool.get('rho_star'),
              **{f'temperature {key}':pool['temperatures'].get(key)
                 for key in ('k31','m2d','innovation_h','cf_full')}}
    for key,value in positive.items():
        if isinstance(value,bool) or not isinstance(value,numbers.Real) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
    beta=pool.get('beta0_closed_form')
    if isinstance(beta,bool) or not isinstance(beta,numbers.Real) or not math.isfinite(beta):
        raise ValueError('beta0_closed_form must be finite')
    if research_pool.EPS_POOL != pool['eps_uniform_floor'] or pool['eps_uniform_floor'] != 0.02:
        raise ValueError('pool parameters do not match the frozen uniform floor (0.02)')
    need_cf=bool(set(configurations)&{'cf_em','joint_v1_pool_v2_em'})
    need_lm=bool(set(configurations)&{'lm_v1_em','joint_v1_pool_v2_em'})
    km={lid:window_kmers(w,31) for lid,w in wins.items()}
    nr={q:len(read_kmers(seqs[q],31)) for q in ids}
    k31={q:[len(read_kmers(seqs[q],31)&km[lid]) for lid in cands[q]] for q in ids}
    cf_raw={};cf={};lm={};times={}
    if need_cf:
        from tracete_lm.d6_likelihood import CandidateIndex,D6Weights,score_read_indexed
        t=time.monotonic();weights=D6Weights.from_platform('ONT',31)
        for q in ids:
            cf_raw[q]=score_read_indexed(seqs[q],CandidateIndex.build([wins[l] for l in cands[q]],31),weights,k=31)
            cf[q]=[round(x,6) for x in cf_raw[q]]
        times['reference_cf_seconds']=time.monotonic()-t
    if need_lm and ids:
        import torch
        from tracete_lm.m2d_model import BandCNN,M2DConfig,matrix2d_scores
        from tracete_lm.m2d_matrix import READ_CAP,CAND_CAP
        assert READ_CAP==CAND_CAP==4096
        torch.set_num_threads(1)
        blob=torch.load(checkpoint or bundle/'assets/ckpt_step30000.pt',map_location='cpu',weights_only=True)
        model=BandCNN(M2DConfig(**blob['config'])).to('cpu');model.load_state_dict(blob['model']);model.eval()
        t=time.monotonic()
        with torch.inference_mode():
            for q in ids:lm[q]=(scorer or matrix2d_scores)(model,seqs[q],[wins[l] for l in cands[q]],device='cpu',
                                              batch_size=128,read_cap=4096,cand_cap=4096)
        times['cpu_fp32_lm_seconds']=time.monotonic()-t
    h={};pk={};pl={};ph={};quality={};joint={}
    if 'joint_v1_pool_v2_em' in configurations:
        temp=pool['temperatures']
        h={q:research_pool.cf_innovation(cf[q],k31[q],pool['beta0_closed_form'],
                                        force_zero=research_pool.is_cf_flat(k31[q],cf[q])) for q in ids}
        pk=research_pool.channel_probs(k31,temp['k31'],ids)
        pl=research_pool.channel_probs(lm,temp['m2d'],ids)
        ph=research_pool.channel_probs(h,temp['innovation_h'],ids)
        quality={q:research_pool.seed_quality_q(k31[q],nr[q],pool['rho_star']) for q in ids}
        joint={q:research_pool.pool_log_emission(pk[q],pl[q],ph[q],quality[q]) for q in ids}
    return {'k31':k31,'cf_raw':cf_raw,'cf':cf,'lm':lm,'innovation':h,'p_k31':pk,'p_lm':pl,
            'p_innovation':ph,'q':quality,'joint':joint,'n_read_words':nr,'pool':pool,'timing':times}


def pack_edges(d,emissions):
    loci=[];lp={};scores=[];edge_loci=[];starts=[0]
    if len(d['fit_ids']) != len(set(d['fit_ids'])):
        raise ValueError('fit_ids must be unique')
    for q in d['fit_ids']:
        if not d['cands'][q] or len(d['cands'][q]) != len(emissions[q]):
            raise ValueError(f'candidate/emission count differs for {q}')
        if len(d['cands'][q]) != len(set(d['cands'][q])):
            raise ValueError(f'duplicate candidates for {q}')
        if not np.isfinite(emissions[q]).all():
            raise ValueError(f'non-finite emission for {q}')
        for lid,s in zip(d['cands'][q],emissions[q]):
            if lid not in lp:lp[lid]=len(loci);loci.append(lid)
            scores.append(float(s));edge_loci.append(lp[lid])
        starts.append(len(scores))
    return np.asarray(scores,dtype=np.float64),np.asarray(edge_loci,dtype=np.int64),np.asarray(starts,dtype=np.int64),loci


def fit(d,emissions):
    from tracete_lm import research_em
    es,el,starts,loci=pack_edges(d,emissions)
    if not d['fit_ids']:
        res=research_em.EMResult(gamma=np.empty(0),pi=np.empty(0),loglik_trace=[],converged=True,n_iter=0)
        return {},{},res,(es,el,starts,loci)
    res=research_em.em_locus_assignment(es,el,starts,len(loci),temperature=1.0,pi_init=None,
                                      max_iter=2000,tol=1e-6,em=True)
    winners={};post={}
    for i,q in enumerate(d['fit_ids']):
        start,end=map(int,starts[i:i+2]);j=start+int(np.argmax(res.gamma[start:end]))
        winners[q]=loci[int(el[j])];post[q]=float(res.gamma[j])
        assert np.isclose(res.gamma[start:end].sum(),1.0)
    return winners,post,res,(es,el,starts,loci)
