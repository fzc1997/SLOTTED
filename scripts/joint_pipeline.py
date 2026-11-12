#!/usr/bin/env python3
"""SLOTTED-Joint: prepare, score, merge and whole-population EM assignment.

The frozen Joint scorer uses CPU/FP32, one torch thread and streaming batches
of four candidates. CUDA scoring belongs to the separate LM-only command.
Prepared bundles contain only primitive Python data; loading arbitrary
Python objects from a pickle is deliberately disabled.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
import math
import numbers
import pickle
from pathlib import Path
import sys
import time
import warnings

RELEASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE / 'src'))

from tracete_lm.inputs import read_fasta

COLUMNS = ['read_id', 'locus_id', 'score_joint', 'seed_count_guard_4096cap']
PROVENANCE_KEY = b'slotted.scoring.v1'


class _DataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError('prepared bundles must contain primitive data only')


def load_bundle(path):
    with open(path, 'rb') as handle:
        data = _DataUnpickler(handle).load()
    if not isinstance(data, dict):
        raise ValueError('prepared bundle must be a dictionary')
    ids = data.get('ids')
    if not isinstance(ids, list) or any(not isinstance(q, str) or not q for q in ids):
        raise ValueError('bundle ids must be nonempty strings')
    if len(ids) != len(set(ids)):
        raise ValueError('bundle contains duplicate read identifiers')
    for field in ('seqs', 'cands', 'windows'):
        if not isinstance(data.get(field), dict):
            raise ValueError(f'bundle lacks {field} dictionary')
    if set(data['seqs']) != set(ids) or set(data['cands']) != set(ids):
        raise ValueError('bundle sequences/candidates do not cover exactly the read IDs')
    for q in ids:
        if (not isinstance(data['seqs'][q], str) or not data['seqs'][q]
                or not set(data['seqs'][q]) <= set('ACGTN')):
            raise ValueError(f'prepared sequence must contain uppercase ACGTN bases for {q}')
        cands = data['cands'][q]
        if not isinstance(cands, list) or any(not isinstance(lid, str) or not lid for lid in cands):
            raise ValueError(f'invalid candidate list for {q}')
        if len(cands) != len(set(cands)):
            raise ValueError(f'duplicate candidates for {q}')
        for lid in cands:
            if not isinstance(data['windows'].get(lid), str) or not data['windows'][lid]:
                raise ValueError(f'missing/empty candidate window: {lid}')
            if not set(data['windows'][lid]) <= set('ACGTN'):
                raise ValueError(f'prepared window must contain uppercase ACGTN bases: {lid}')
    expected = [q for q in ids if data['cands'][q]]
    if data.get('fit_ids') != expected:
        raise ValueError('bundle fit_ids differ from the candidate-bearing population')
    return data


def _channel_schema():
    import pyarrow as pa
    return pa.schema([('read_id', pa.string()), ('locus_id', pa.string()),
                      ('score_joint', pa.float64()), ('seed_count_guard_4096cap', pa.int64())])


def _write_channels(rows, path, provenance=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pylist(rows, schema=_channel_schema())
    if provenance is not None:
        table = table.replace_schema_metadata({PROVENANCE_KEY: json.dumps(provenance, sort_keys=True).encode()})
    with open(path, 'xb') as handle:
        pq.write_table(table, handle)


def _read_channels(path, data, bundle_sha):
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=COLUMNS)
    raw = (table.schema.metadata or {}).get(PROVENANCE_KEY)
    provenance = json.loads(raw) if raw else None
    if provenance is not None and provenance.get('bundle_sha256') != bundle_sha:
        raise ValueError(f'channels were scored from a different prepared bundle: {path}')
    rows = table.to_pylist()
    candidate_sets = {q: set(c) for q, c in data['cands'].items()}
    for row in rows:
        q, lid, score, seeds = (row[key] for key in COLUMNS)
        if q not in candidate_sets or lid not in candidate_sets[q]:
            raise ValueError(f'unknown read/candidate in channels: {(q, lid)}')
        if not isinstance(score, numbers.Real) or not math.isfinite(score):
            raise ValueError(f'non-finite/invalid channel score for {(q, lid)}')
        if isinstance(seeds, bool) or not isinstance(seeds, numbers.Integral) or seeds < 0:
            raise ValueError(f'invalid seed count for {(q, lid)}')
    return rows, provenance


def _check_coverage(data, by_read):
    for q in data['ids']:
        rows = by_read.get(q, [])
        if [row['locus_id'] for row in rows] != data['cands'][q]:
            raise ValueError(f'missing, duplicate or out-of-order candidate channels for {q}')


def cmd_prepare(a):
    import pysam
    from tracete.extract.annotations import RepeatIndex, iter_repeat_annotation
    from tracete_lm.d6_likelihood import reverse_complement
    if a.flank < 0:
        raise ValueError('flank must be non-negative')
    seqs = read_fasta(a.reads)
    idx = RepeatIndex(iter_repeat_annotation(a.rmsk))
    cands = {q: set() for q in seqs}
    with pysam.AlignmentFile(str(a.bam), 'rb') as handle:
        for rec in handle:
            q = rec.query_name
            if q not in cands or rec.is_unmapped or rec.reference_name is None:
                continue
            pos = rec.reference_start
            for op, length in rec.cigartuples or ():
                if op in (0, 7, 8):
                    cands[q].update(iv.locus_id for iv in idx.query(rec.reference_name, pos, pos + length)
                                    if iv.overlap_bp(pos, pos + length) > 0)
                    pos += length
                elif op in (2, 3):
                    pos += length
    cands = {q: sorted(value) for q, value in cands.items()}
    windows = {}
    with pysam.FastaFile(str(a.genome)) as fa:
        lengths = dict(zip(fa.references, fa.lengths))
        for lid in sorted({lid for value in cands.values() for lid in value}):
            iv = idx.get(lid)
            if iv.chrom not in lengths or iv.start < 0 or iv.end > lengths[iv.chrom]:
                raise ValueError(f'candidate outside the reference: {lid}')
            start = max(0, iv.start - a.flank)
            end = min(lengths[iv.chrom], iv.end + a.flank)
            window = fa.fetch(iv.chrom, start, end).upper()
            if not window or not set(window) <= set('ACGTN'):
                raise ValueError(f'candidate window must contain ACGTN bases: {lid}')
            windows[lid] = reverse_complement(window) if iv.strand == '-' else window
    data = {'ids': sorted(seqs), 'fit_ids': [q for q in sorted(seqs) if cands[q]],
            'seqs': seqs, 'cands': cands, 'windows': windows}
    with open(a.out, 'xb') as handle:
        pickle.dump(data, handle, protocol=4)
    print(json.dumps({'ids': len(seqs), 'fit_ids': len(data['fit_ids']), 'windows': len(windows)}))


def cmd_score(a):
    from tracete_lm import pipeline_core
    from tracete_lm.stream_scores import matrix2d_scores_streaming
    from tracete_lm.post_em_guard import guard_seed_counts
    full = load_bundle(a.d)
    ids = full['ids'] if a.ids is None else [q.strip() for q in Path(a.ids).read_text().splitlines() if q.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError('shard contains duplicate read IDs')
    unknown = set(ids) - set(full['ids'])
    if unknown:
        raise ValueError(f'shard contains {len(unknown)} unknown read IDs: {sorted(unknown)[:3]}')
    fit_ids = [q for q in ids if full['cands'][q]]
    lids = {lid for q in fit_ids for lid in full['cands'][q]}
    data = {'ids': ids, 'fit_ids': fit_ids,
            'seqs': {q: full['seqs'][q] for q in ids},
            'cands': {q: full['cands'][q] for q in ids},
            'windows': {lid: full['windows'][lid] for lid in lids}}
    started = time.monotonic()
    # Freeze the research streaming chunk size without mutating another module.
    def scorer(model, read, windows, **kwargs):
        kwargs['batch_size'] = 4
        return matrix2d_scores_streaming(model, read, windows, **kwargs)
    values = pipeline_core.score(RELEASE, data, ['joint_v1_pool_v2_em'], scorer=scorer,
                                 checkpoint=a.ckpt, pool_params=a.pool_params)
    guards = guard_seed_counts(data)
    rows = [{'read_id': q, 'locus_id': lid,
             'score_joint': float(values['joint'][q][j]),
             'seed_count_guard_4096cap': int(guards[q][j])}
            for q in fit_ids for j, lid in enumerate(data['cands'][q])]
    if any(not math.isfinite(row['score_joint']) for row in rows):
        raise ValueError('scorer produced non-finite emissions')
    provenance = {'bundle_sha256': pipeline_core.sha(a.d), 'checkpoint_sha256': pipeline_core.sha(a.ckpt),
                  'pool_sha256': pipeline_core.sha(a.pool_params), 'device': 'cpu', 'precision': 'float32',
                  'stream_batch_size': 4, 'configuration': 'joint_v1_pool_v2_em'}
    _write_channels(rows, a.out, provenance)
    print(json.dumps({'ids': len(ids), 'fit_ids': len(fit_ids), 'rows': len(rows),
                      'seconds': round(time.monotonic() - started, 1), **provenance}))


def cmd_merge(a):
    from tracete_lm.pipeline_core import sha
    data = load_bundle(a.d)
    bundle_sha = sha(a.d)
    paths = []
    for pattern in a.shards:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise ValueError(f'no shard files match {pattern}')
        paths.extend(matches)
    seen, provenance, initialized = {}, None, False
    for path in dict.fromkeys(paths):
        rows, source = _read_channels(path, data, bundle_sha)
        if initialized and source != provenance:
            raise ValueError('shards have different scoring provenance (model, pool or bundle)')
        provenance, initialized = source, True
        for row in rows:
            key = row['read_id'], row['locus_id']
            if key in seen:
                old = seen[key]
                if old[COLUMNS[3]] != row[COLUMNS[3]] or abs(old['score_joint'] - row['score_joint']) > 1e-6:
                    raise ValueError(f'redundant shards disagree on {key}')
            else:
                seen[key] = row
    by_read = defaultdict(list)
    for row in seen.values():
        by_read[row['read_id']].append(row)
    _check_coverage(data, by_read)
    rows = [row for q in data['ids'] for row in by_read[q]]
    if provenance is None:
        warnings.warn('legacy channels have no model/bundle provenance; only row integrity was checked')
    _write_channels(rows, a.out, provenance)
    print(json.dumps({'shard_files': len(set(paths)), 'rows': len(rows),
                      'reads_with_candidates': len(data['fit_ids'])}))


def cmd_fit(a):
    from tracete_lm import pipeline_core
    from tracete_lm.post_em_guard import apply_post_em_guard
    import pandas as pd
    full = load_bundle(a.d)
    rows, provenance = _read_channels(a.channels, full, pipeline_core.sha(a.d))
    by_read = defaultdict(list)
    for row in rows:
        by_read[row['read_id']].append(row)
    _check_coverage(full, by_read)
    if provenance is None:
        warnings.warn('legacy channels have no model/bundle provenance; only row integrity was checked')
    emissions = {q: [row['score_joint'] for row in by_read[q]] for q in full['fit_ids']}
    guards = {q: [row[COLUMNS[3]] for row in by_read[q]] for q in full['fit_ids']}
    winners, posterior, em, edges = pipeline_core.fit(full, emissions)
    decisions = apply_post_em_guard(full, guards, em.gamma, edges[2])
    converged = bool(em.converged) if full['fit_ids'] else None
    fit_status = ('CONVERGED' if em.converged else 'MAX_ITER_REACHED') if full['fit_ids'] else 'NO_CANDIDATE_READS'
    out = []
    for q in full['ids']:
        base = {'read_id': q, 'n_candidates': len(full['cands'][q]), 'method': 'CF_LM_v1_EM',
                'fit_converged': converged, 'fit_status': fit_status, 'winner': winners.get(q, ''),
                'posterior': posterior.get(q), 'called': q in winners,
                'report_reason': 'ASSIGN_ALL' if q in winners else 'NO_CANDIDATE'}
        out.append(base)
        decision = decisions[q]
        passed = decision['call'] == 'UNIQUE'
        out.append({**base, 'method': 'CF_LM_v1_EM_guard',
                    'winner': winners.get(q, '') if passed else '',
                    'called': passed, 'report_reason': decision['reason'] if full['cands'][q] else 'NO_CANDIDATE'})
    columns = ['read_id', 'n_candidates', 'method', 'fit_converged', 'fit_status', 'winner', 'posterior', 'called', 'report_reason']
    frame = pd.DataFrame(out, columns=columns)
    if Path(a.out).suffix.lower() == '.tsv':
        frame.to_csv(a.out, sep='\t', index=False, float_format='%.12g', mode='x')
    else:
        with open(a.out, 'xb') as handle:
            frame.to_parquet(handle, index=False)
    meta = {'n_ids': len(full['ids']), 'n_fit': len(full['fit_ids']), 'em_converged': converged,
            'fit_status': fit_status,
            'em_iterations': int(em.n_iter), 'n_loci': len(edges[3]), 'n_winner': len(winners),
            'n_unique_guard': sum(decision['call'] == 'UNIQUE' for decision in decisions.values()),
            'scoring_provenance': provenance, 'channels_sha256': pipeline_core.sha(a.channels)}
    with Path(a.receipt or str(a.out) + '.json').open('x') as handle:
        handle.write(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='stage', required=True)
    sp = sub.add_parser('prepare')
    sp.add_argument('--bam', required=True); sp.add_argument('--reads', required=True)
    sp.add_argument('--rmsk', required=True); sp.add_argument('--genome', required=True)
    sp.add_argument('--flank', type=int, default=500); sp.add_argument('--out', required=True)
    sp.set_defaults(fn=cmd_prepare)
    sp = sub.add_parser('score')
    sp.add_argument('--d', required=True)
    sp.add_argument('--ids', help='optional file of unique read IDs; omit to score the whole bundle')
    sp.add_argument('--out', required=True)
    sp.add_argument('--device', choices=['cpu'], default='cpu', help='frozen Joint CPU/FP32 protocol')
    sp.add_argument('--ckpt', type=Path, default=RELEASE / 'assets/ckpt_step30000.pt')
    sp.add_argument('--pool-params', type=Path, default=RELEASE / 'assets/pool_frozen_params_v2.json')
    sp.set_defaults(fn=cmd_score)
    sp = sub.add_parser('merge')
    sp.add_argument('--d', required=True); sp.add_argument('--shards', nargs='+', required=True)
    sp.add_argument('--out', required=True)
    sp.set_defaults(fn=cmd_merge)
    sp = sub.add_parser('fit')
    sp.add_argument('--d', required=True); sp.add_argument('--channels', required=True)
    sp.add_argument('--out', required=True)
    sp.add_argument('--receipt', help='optional path for EM diagnostics; defaults to <out>.json')
    sp.set_defaults(fn=cmd_fit)
    args = parser.parse_args(argv)
    outputs = [Path(args.out)]
    if args.stage == 'fit':
        outputs.append(Path(args.receipt or str(args.out) + '.json'))
    if len({path.resolve() for path in outputs}) != len(outputs):
        parser.error('output and receipt must be different files')
    for path in outputs:
        if path.exists():
            raise FileExistsError(f'output already exists: {path}')
    args.fn(args)


if __name__ == '__main__':
    main()
