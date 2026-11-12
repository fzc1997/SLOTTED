//! m2d_bands: bit-identical Rust port of the tracete_lm.m2d_matrix band
//! construction core (encode / kmer_codes / CodeIndex / match_pairs /
//! vote_diagonal / path_events / band_arrays / window_lut /
//! group_other_unions / build_pair_band / candidate_group_bands).
//!
//! Semantics contract (verified by tests here AND by the Python-side
//! bit-identity harness against tracete_lm.m2d_matrix):
//! * all index arithmetic in i64, identical tie-breaking (argmax first
//!   occurrence, stable sort, strict `>` for the reverse-strand vote),
//! * uint16 k-mer codes with the same modulo wrap for N-carrying windows,
//! * per-code candidate cap of 32, left-sided searchsorted,
//! * float64 position priors rounded to float32 on store, clip before cast,
//! * bool OR unions are exact; prefix/suffix ORs reproduce the per-candidate
//!   other_union lookups bit for bit.

use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray1, PyArray3, PyReadonlyArray1};
use pyo3::prelude::*;

const K_ANCHOR: usize = 8;
const N_CHANNELS: usize = 9;
const PER_CODE_CAP: i64 = 32;
const LUT_SIZE: usize = 1 << (2 * K_ANCHOR);

// channel ids (must match tracete_lm.m2d_matrix)
const CH_ANCHOR: usize = 0;
const CH_MATCH: usize = 1;
const CH_MISMATCH: usize = 2;
const CH_SUB: usize = 3;
const CH_INS: usize = 4;
const CH_DEL: usize = 5;
const CH_DISC: usize = 6;

fn ascii_code(b: u8) -> u8 {
    match b.to_ascii_uppercase() {
        b'A' => 0,
        b'C' => 1,
        b'G' => 2,
        b'T' => 3,
        _ => 4,
    }
}

fn encode_u8(seq: &[u8], cap: usize) -> Vec<u8> {
    seq.iter().take(cap).map(|&b| ascii_code(b)).collect()
}

fn reverse_complement_u8(seq: &[u8]) -> Vec<u8> {
    const LUT: [u8; 6] = [3, 2, 1, 0, 4, 255];
    seq.iter().rev().map(|&b| LUT[b as usize]).collect()
}

/// Rolling 2-bit codes of the k-mers (A0 C1 G2 T3); N-carrying windows get a
/// wrapped code plus has_n=true (the wrap mirrors numpy's uint32->uint16 cast).
fn kmer_codes(seq: &[u8], k: usize) -> (Vec<u16>, Vec<bool>) {
    let n = seq.len() as i64 - k as i64 + 1;
    if n <= 0 {
        return (Vec::new(), Vec::new());
    }
    let n = n as usize;
    let mut codes = Vec::with_capacity(n);
    let mut has_n = Vec::with_capacity(n);
    for s in 0..n {
        let win = &seq[s..s + k];
        let mut hn = false;
        let mut code: u32 = 0;
        for (i, &b) in win.iter().enumerate() {
            if b >= 4 {
                hn = true;
            }
            code += (b as u32) << (2 * i);
        }
        codes.push(code as u16);
        has_n.push(hn);
    }
    (codes, has_n)
}

/// Candidate-side inverse index: code -> sorted original positions.
struct CodeIndex {
    positions_sorted: Vec<i32>,
    uniq: Vec<u16>,
    start: Vec<i64>,
    counts: Vec<i64>,
}

impl CodeIndex {
    fn build(codes: &[u16], has_n: &[bool]) -> CodeIndex {
        let mut kept: Vec<(u16, i32)> = codes
            .iter()
            .enumerate()
            .filter(|&(p, _)| !has_n[p])
            .map(|(p, &c)| (c, p as i32))
            .collect();
        kept.sort_by_key(|&(c, _)| c); // stable, like np.argsort(kind="stable")
        let positions_sorted: Vec<i32> = kept.iter().map(|&(_, p)| p).collect();
        let mut uniq = Vec::new();
        let mut start = Vec::new();
        let mut counts = Vec::new();
        let mut i = 0usize;
        while i < kept.len() {
            let c = kept[i].0;
            let s = i as i64;
            let mut j = i;
            while j < kept.len() && kept[j].0 == c {
                j += 1;
            }
            uniq.push(c);
            start.push(s);
            counts.push((j - i) as i64);
            i = j;
        }
        CodeIndex { positions_sorted, uniq, start, counts }
    }
}

/// (i, j) arrays of exact anchor matches; per-code candidate positions capped.
fn match_pairs(read_codes: &[u16], read_has_n: &[bool], idx: &CodeIndex) -> (Vec<i64>, Vec<i64>) {
    let rpos: Vec<i64> = (0..read_codes.len())
        .filter(|&p| !read_has_n[p])
        .map(|p| p as i64)
        .collect();
    if idx.uniq.is_empty() || rpos.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut hits: Vec<(i64, usize, i64)> = Vec::new(); // (read pos, loc, m)
    for &r in &rpos {
        let c = read_codes[r as usize];
        let mut loc = idx.uniq.partition_point(|&x| x < c); // searchsorted left
        if loc > idx.uniq.len() - 1 {
            loc = idx.uniq.len() - 1;
        }
        if idx.uniq[loc] == c {
            let m = idx.counts[loc].min(PER_CODE_CAP);
            if m > 0 {
                hits.push((r, loc, m));
            }
        }
    }
    let total: usize = hits.iter().map(|&(_, _, m)| m as usize).sum();
    let mut i_rep = Vec::with_capacity(total);
    let mut j = Vec::with_capacity(total);
    for &(r, loc, m) in &hits {
        let s0 = idx.start[loc];
        for k in 0..m {
            i_rep.push(r);
            j.push(idx.positions_sorted[(s0 + k) as usize] as i64);
        }
    }
    (i_rep, j)
}

/// argmax diagonal of (1,2,1)-smoothed j - i anchor votes; ties -> smallest.
fn vote_diagonal(mi: &[i64], mj: &[i64], span: i64, h: i64) -> i64 {
    if mi.is_empty() {
        return 0;
    }
    let mut min_d = i64::MAX;
    let mut max_d = i64::MIN;
    for k in 0..mi.len() {
        let d = mj[k] - mi[k];
        min_d = min_d.min(d);
        max_d = max_d.max(d);
    }
    let width = span.max(h).max(-min_d).max(max_d) + 1;
    let lo = -width;
    let n_bins = (2 * width + 1) as usize;
    let mut votes = vec![0i64; n_bins];
    for k in 0..mi.len() {
        votes[(mj[k] - mi[k] - lo) as usize] += 1;
    }
    let mut best_val = -1i64;
    let mut best_bin = 0usize;
    for b in 0..n_bins {
        let mut s = votes[b] * 2;
        if b > 0 {
            s += votes[b - 1];
        }
        if b + 1 < n_bins {
            s += votes[b + 1];
        }
        if s > best_val {
            best_val = s;
            best_bin = b;
        }
    }
    best_bin as i64 + lo
}

/// Greedy local alignment along the anchored diagonal -> sparse error events.
fn path_events(r: &[u8], c: &[u8], diag: i64, band: i64) -> Vec<(usize, i64, i64)> {
    let h = r.len() as i64;
    let w = c.len() as i64;
    let mut events = Vec::new();
    let mut i: i64 = 0;
    let mut o: i64 = 0;
    while i < h {
        let j = i + diag + o;
        if 0 <= j && j < w && r[i as usize] == c[j as usize] {
            i += 1;
            continue;
        }
        if 0 <= j + 1
            && j + 1 < w
            && r[i as usize] == c[(j + 1) as usize]
            && (i + 1 >= h || j + 2 >= w || r[(i + 1) as usize] == c[(j + 2) as usize])
        {
            if o <= band && -o <= band {
                events.push((CH_DEL, i, o));
            }
            o += 1; // read-side deletion
            continue;
        }
        if i + 1 < h
            && 0 <= j
            && j < w
            && r[(i + 1) as usize] == c[j as usize]
            && (i + 2 >= h || j + 1 >= w || r[(i + 2) as usize] == c[(j + 1) as usize])
        {
            if o <= band && -o <= band {
                events.push((CH_INS, i, o));
            }
            i += 1;
            o -= 1; // read-side insertion
            continue;
        }
        if 0 <= j && j < w && o <= band && -o <= band {
            events.push((CH_SUB, i, o));
        }
        i += 1;
    }
    events
}

/// Per-k-mer-position flag: the 8-mer at that position is intact and absent
/// from the other candidates (disc_flags in tracete_lm.m2d_matrix).
fn disc_flags(cand_codes: &[u16], cand_has_n: &[bool], other_union: &[bool]) -> Vec<bool> {
    cand_codes
        .iter()
        .enumerate()
        .map(|(p, &c)| !cand_has_n[p] && !other_union[c as usize])
        .collect()
}

/// Dense (C, H, 2*band+1) float32 band image around `diag`.
fn band_arrays(
    read_u8: &[u8],
    cand_u8: &[u8],
    read_codes: &[u16],
    read_has_n: &[bool],
    cand_index: &CodeIndex,
    diag: i64,
    band: i64,
    disc_flag: Option<&[bool]>,
) -> Array3<f32> {
    let h = read_u8.len();
    let w = cand_u8.len();
    let width2 = (2 * band + 1) as usize;
    let mut out = vec![0f32; N_CHANNELS * h * width2];
    if h > 0 && w > 0 {
        let wl = w as i64;
        let at = |ch: usize, i: usize, oi: usize| (ch * h + i) * width2 + oi;
        for i in 0..h {
            for oi in 0..width2 {
                let j = i as i64 + diag + (oi as i64 - band);
                let valid = 0 <= j && j < wl;
                let jc = j.clamp(0, wl - 1) as usize;
                let eq = valid && read_u8[i] == cand_u8[jc];
                out[at(CH_MATCH, i, oi)] = eq as u8 as f32;
                out[at(CH_MISMATCH, i, oi)] = (valid && !eq) as u8 as f32;
            }
        }
        for (ch, i, o) in path_events(read_u8, cand_u8, diag, band) {
            out[at(ch, i as usize, (o + band) as usize)] = 1.0;
        }
        let (mi, mj) = match_pairs(read_codes, read_has_n, cand_index);
        for k in 0..mi.len() {
            let off = (mj[k] - mi[k]) - diag;
            if off <= band && -off <= band {
                out[at(CH_ANCHOR, mi[k] as usize, (off + band) as usize)] = 1.0;
            }
        }
        if let Some(flag) = disc_flag {
            if !flag.is_empty() {
                let mut fl = vec![0f32; w];
                for (p, &f) in flag.iter().enumerate() {
                    fl[p] = f as u8 as f32;
                }
                for i in 0..h {
                    for oi in 0..width2 {
                        let j = i as i64 + diag + (oi as i64 - band);
                        if 0 <= j && j < wl {
                            out[at(CH_DISC, i, oi)] = fl[j.clamp(0, wl - 1) as usize];
                        }
                    }
                }
            }
        }
        let hf = h as f64;
        let wf = w as f64;
        for i in 0..h {
            for oi in 0..width2 {
                let j = i as i64 + diag + (oi as i64 - band);
                let valid = 0 <= j && j < wl;
                out[at(7, i, oi)] = if valid { ((i as f64) / hf) as f32 } else { 0.0 };
                out[at(8, i, oi)] = if valid {
                    (((j as f64) / wf).clamp(0.0, 1.0)) as f32
                } else {
                    0.0
                };
            }
        }
    }
    Array3::from_shape_vec((N_CHANNELS, h, width2), out).expect("band buffer shape mismatch")
}

/// Two-strand 8-mer presence lookup for ONE candidate window (65536 bools).
fn window_lut_impl(seq: &[u8], cand_cap: usize) -> Vec<bool> {
    let mut lut = vec![false; LUT_SIZE];
    let fwd = encode_u8(seq, cand_cap);
    let rc = reverse_complement_u8(&fwd);
    for strand in [&fwd[..], &rc[..]] {
        let (codes, has_n) = kmer_codes(strand, K_ANCHOR);
        for (p, &c) in codes.iter().enumerate() {
            if !has_n[p] {
                lut[c as usize] = true;
            }
        }
    }
    lut
}

/// Per-member union of the OTHER members' k-mers (prefix/suffix OR; exact).
fn group_other_unions_impl(luts: &[Vec<bool>]) -> Vec<Option<Vec<bool>>> {
    let n = luts.len();
    if n <= 1 {
        return vec![None; n];
    }
    let or_with = |a: &[bool], b: &[bool]| a.iter().zip(b).map(|(x, y)| *x || *y).collect::<Vec<bool>>();
    let mut pre: Vec<Vec<bool>> = Vec::with_capacity(n);
    pre.push(luts[0].clone());
    for k in 1..n {
        pre.push(or_with(&pre[k - 1], &luts[k]));
    }
    let mut suf: Vec<Option<Vec<bool>>> = vec![None; n];
    suf[n - 1] = Some(luts[n - 1].clone());
    for k in (0..n - 1).rev() {
        suf[k] = Some(or_with(suf[k + 1].as_ref().unwrap(), &luts[k]));
    }
    let mut out: Vec<Option<Vec<bool>>> = Vec::with_capacity(n);
    for k in 0..n {
        if k == 0 {
            out.push(suf[1].clone());
        } else if k == n - 1 {
            out.push(Some(pre[n - 2].clone()));
        } else {
            out.push(Some(or_with(&pre[k - 1], suf[k + 1].as_ref().unwrap())));
        }
    }
    out
}

/// Same contract as tracete_lm.m2d_matrix.build_pair_band.
fn build_pair_band_core(
    read: &[u8],
    candidate: &[u8],
    other_union: Option<&[bool]>,
    band: i64,
    read_cap: usize,
    cand_cap: usize,
    orient: bool,
) -> Array3<f32> {
    let cand_u8 = encode_u8(candidate, cand_cap);
    let (cand_codes, cand_has_n) = kmer_codes(&cand_u8, K_ANCHOR);
    let cand_index = CodeIndex::build(&cand_codes, &cand_has_n);
    let read_fwd = encode_u8(read, read_cap);
    let read_rc = reverse_complement_u8(&read_fwd);

    let orient_vote = |seq: &[u8]| -> (Vec<u16>, Vec<bool>, i64, i64) {
        let (codes, has_n) = kmer_codes(seq, K_ANCHOR);
        let (mi, mj) = match_pairs(&codes, &has_n, &cand_index);
        let d = vote_diagonal(&mi, &mj, cand_u8.len() as i64, seq.len() as i64);
        let votes = mi
            .iter()
            .zip(mj.iter())
            .map(|(&a, &b)| (a, b))
            .filter(|&(a, b)| ((b - a) - d).abs() <= band)
            .count() as i64;
        (codes, has_n, d, votes)
    };

    let (f_codes, f_has_n, f_diag, f_votes) = orient_vote(&read_fwd);
    let (read_u8, read_codes, read_has_n, diag) = if orient {
        let (r_codes, r_has_n, r_diag, r_votes) = orient_vote(&read_rc);
        if r_votes > f_votes {
            (read_rc, r_codes, r_has_n, r_diag)
        } else {
            (read_fwd, f_codes, f_has_n, f_diag)
        }
    } else {
        (read_fwd, f_codes, f_has_n, f_diag)
    };

    let disc_flag = other_union.map(|u| disc_flags(&cand_codes, &cand_has_n, u));
    band_arrays(
        &read_u8, &cand_u8, &read_codes, &read_has_n, &cand_index, diag, band,
        disc_flag.as_deref(),
    )
}

// ---------------------------------------------------------------- Python API

#[pyfunction]
#[pyo3(signature = (read, candidate, other_union=None, *, band=32, read_cap=4096, cand_cap=4096, orient=true))]
fn build_pair_band<'py>(
    py: Python<'py>,
    read: &str,
    candidate: &str,
    other_union: Option<PyReadonlyArray1<'py, bool>>,
    band: i64,
    read_cap: usize,
    cand_cap: usize,
    orient: bool,
) -> PyResult<&'py PyArray3<f32>> {
    let union: Option<Vec<bool>> = other_union.map(|a| a.as_array().iter().copied().collect());
    let arr = build_pair_band_core(read.as_bytes(), candidate.as_bytes(), union.as_deref(), band, read_cap, cand_cap, orient);
    Ok(arr.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (read, windows, *, read_cap=4096, cand_cap=4096))]
fn candidate_group_bands<'py>(
    py: Python<'py>,
    read: &str,
    windows: Vec<String>,
    read_cap: usize,
    cand_cap: usize,
) -> PyResult<Vec<&'py PyArray3<f32>>> {
    // Inputs converted under the GIL; the pure-Rust compute releases it so a
    // Python-side producer thread can overlap band building with GPU feeding.
    let read_bytes = read.as_bytes().to_vec();
    let core: Vec<Array3<f32>> = {
        let luts: Vec<Vec<bool>> = windows.iter().map(|w| window_lut_impl(w.as_bytes(), cand_cap)).collect();
        let unions = group_other_unions_impl(&luts);
        py.allow_threads(|| {
            windows
                .iter()
                .zip(unions.iter())
                .map(|(w, u)| build_pair_band_core(&read_bytes, w.as_bytes(), u.as_deref(), 32, read_cap, cand_cap, true))
                .collect()
        })
    };
    Ok(core.into_iter().map(|a| a.into_pyarray(py)).collect())
}

#[pyfunction]
#[pyo3(signature = (sequence, *, cand_cap=4096))]
fn window_lut<'py>(py: Python<'py>, sequence: &str, cand_cap: usize) -> &'py PyArray1<bool> {
    window_lut_impl(sequence.as_bytes(), cand_cap).into_pyarray(py)
}

#[pyfunction]
fn group_other_unions<'py>(
    py: Python<'py>,
    luts: Vec<PyReadonlyArray1<'py, bool>>,
) -> Vec<Option<&'py PyArray1<bool>>> {
    let ls: Vec<Vec<bool>> = luts.iter().map(|a| a.as_array().iter().copied().collect()).collect();
    group_other_unions_impl(&ls)
        .into_iter()
        .map(|o| o.map(|v| v.into_pyarray(py)))
        .collect()
}

#[pymodule]
fn m2d_bands(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_pair_band, m)?)?;
    m.add_function(wrap_pyfunction!(candidate_group_bands, m)?)?;
    m.add_function(wrap_pyfunction!(window_lut, m)?)?;
    m.add_function(wrap_pyfunction!(group_other_unions, m)?)?;
    Ok(())
}

// -------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_and_rc() {
        assert_eq!(encode_u8(b"aCgTnX", 16), vec![0, 1, 2, 3, 4, 4]);
        assert_eq!(encode_u8(b"ACGTACGT", 4), vec![0, 1, 2, 3]);
        assert_eq!(reverse_complement_u8(&[0, 1, 2, 3]), vec![0, 1, 2, 3]);
        assert_eq!(reverse_complement_u8(&encode_u8(b"ACGTT", 8)), encode_u8(b"AACGT", 8));
    }

    #[test]
    fn kmer_codes_values() {
        let (codes, has_n) = kmer_codes(&encode_u8(b"ACGTACGT", 16), 8);
        assert_eq!(codes, vec![58596]);
        assert_eq!(has_n, vec![false]);
        let (codes, has_n) = kmer_codes(&encode_u8(b"ACGT", 16), 8);
        assert!(codes.is_empty() && has_n.is_empty());
        let (_, has_n) = kmer_codes(&encode_u8(b"ACGTNNNN", 16), 8);
        assert_eq!(has_n, vec![true]);
    }

    #[test]
    fn match_pairs_cap_and_lookup() {
        // candidate: 47 A's -> 40 windows of code 0
        let cand = vec![0u8; 47];
        let (ccodes, chas) = kmer_codes(&cand, 8);
        let idx = CodeIndex::build(&ccodes, &chas);
        let read = vec![0u8; 10];
        let (rcodes, rhas) = kmer_codes(&read, 8);
        let (mi, mj) = match_pairs(&rcodes, &rhas, &idx);
        assert_eq!(mi.len(), 3 * 32); // 3 read windows, each capped at 32 hits
        assert_eq!(mi[0], 0);
        assert_eq!(mj[0], 0);
        assert_eq!(mj[31], 31);
    }

    #[test]
    fn vote_and_events() {
        let mi: Vec<i64> = (0..3).map(|k| 10 * k).collect();
        let mj: Vec<i64> = (0..3).map(|k| 10 * k + 5).collect();
        assert_eq!(vote_diagonal(&mi, &mj, 100, 30), 5);
        assert_eq!(vote_diagonal(&[], &[], 100, 30), 0);

        let r = encode_u8(b"ACGT", 16);
        let c = encode_u8(b"ACGGT", 16);
        assert_eq!(path_events(&r, &c, 0, 32), vec![(CH_DEL, 3, 0)]);
        let r = encode_u8(b"AACGT", 16);
        let c = encode_u8(b"ACGT", 16);
        assert_eq!(path_events(&r, &c, 0, 32), vec![(CH_INS, 1, 0)]);
        let r = encode_u8(b"AAAA", 16);
        let c = encode_u8(b"AAAC", 16);
        assert_eq!(path_events(&r, &c, 0, 32), vec![(CH_SUB, 3, 0)]);
        let r = encode_u8(b"ACGTACGT", 16);
        assert!(path_events(&r, &r, 0, 32).is_empty());
    }

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 33
        }
    }

    #[test]
    fn group_unions_match_naive() {
        let mut rng = Lcg(42);
        for k in 1..=10usize {
            let luts: Vec<Vec<bool>> = (0..k)
                .map(|_| (0..LUT_SIZE).map(|_| rng.next() % 8 == 0).collect())
                .collect();
            let got = group_other_unions_impl(&luts);
            for j in 0..k {
                if k == 1 {
                    assert!(got[j].is_none());
                    continue;
                }
                let naive: Vec<bool> = (0..LUT_SIZE)
                    .map(|b| (0..k).any(|m| m != j && luts[m][b]))
                    .collect();
                assert_eq!(got[j].as_ref().unwrap(), &naive, "k={k} j={j}");
            }
        }
    }

    #[test]
    fn band_shape_and_finite() {
        let read = encode_u8(b"ACGT".repeat(200).as_slice(), 4096);
        let cand = encode_u8(b"ACGTN".repeat(100).as_slice(), 4096);
        let (ccodes, chas) = kmer_codes(&cand, 8);
        let cidx = CodeIndex::build(&ccodes, &chas);
        let (rcodes, rhas) = kmer_codes(&read, 8);
        let arr = band_arrays(&read, &cand, &rcodes, &rhas, &cidx, 7, 32, None);
        assert_eq!(arr.shape(), &[9, 800, 65]);
        assert!(arr.iter().all(|v| v.is_finite()));
        let empty = band_arrays(&[], &cand, &[], &[], &cidx, 0, 32, None);
        assert_eq!(empty.shape(), &[9, 0, 65]);
    }
}
