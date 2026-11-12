# SLOTTED

**Single-cell/spatial Long-read Origin Tracing for Transposable-Element Disambiguation**

SLOTTED assigns long RNA-sequencing reads to transposable-element (TE) loci.
It combines an error-aware sequence likelihood with a learned alignment-band
model and supports whole-population expectation–maximization (EM) assignment
with an optional selective reporting guard.

**Version:** 1.0.0

## 1. Installation

### 1.1 Clone the source repository

```bash
git clone https://github.com/fzc1997/SLOTTED.git
cd SLOTTED
```

The repository includes the model, calibration and chr21 example inputs.

### 1.2 Create a Python environment and install dependencies

The tested environment is **Linux x86-64 with Python 3.12**. A GPU is not needed
for the bundled demonstration. We recommend at least **4 GB of available RAM**.
The source checkout occupies approximately **150 MiB**,
excluding the Python environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-demo.txt
python -m pip install --no-deps .
python scripts/run_chr21_demo.py
```

Dependency versions are specified in `requirements-demo.txt`.

Optional acceleration (not needed to run SLOTTED):

```bash
python -m pip install '.[numba]'

# Optional Rust backend; requires a Rust toolchain
python -m pip install '.[rust]'
bash scripts/rebuild_rust_wheel.sh
```

## 2. Usage

### 2.1 SLOTTED-Joint

Provide the aligned reads, read sequences, repeat annotation and indexed
reference genome:

```bash
python scripts/slotted.py --bam alignments.bam --reads reads.fa --rmsk repeats.rmsk.txt --genome genome.fa
```

SLOTTED runs the complete workflow and creates a timestamped output directory
automatically. The final result is `SLOTTED_joint_predictions.tsv`; intermediate
files and logs are kept in `intermediates/`. The bundled model and calibration
are used by default. Joint scoring runs on CPU.

| Argument | Input |
|---|---|
| `--bam` | BAM alignments, including retained alternative mappings |
| `--reads` | Read FASTA, plain or gzip-compressed; unique IDs and A/C/G/T/N bases |
| `--rmsk` | UCSC RepeatMasker annotation or supported repeat GTF |
| `--genome` | Indexed reference FASTA matching the alignments and annotation |
| `--out` | Optional new output directory; existing directories are rejected |

Run `python scripts/slotted.py --help` for additional options.

### 2.2 SLOTTED-LM: learned-model assignment

The LM-only command supports CPU and CUDA:

```bash
python scripts/assign_reads_m2d.py --bam alignments.bam --reads reads.fa --rmsk repeats.rmsk.txt --genome genome.fa --ckpt assets/ckpt_step30000.pt --out SLOTTED_lm_assignments.tsv --device cpu --verify-exact --selective
```

`--verify-exact` verifies batched winners against the per-read reference
recipe and fails if a mismatch cannot be resolved. `--selective` adds
`UNIQUE` or `ABSTAIN` reporting using probability margin and exact-seed support.
Use `--device cuda` for GPU inference and `--band-backend python` to explicitly
use the Python band backend. Rust is optional.

For your own data, provide reads aligned against the same reference used for
the annotation. Alignment is an upstream step; SLOTTED does not run an aligner
or infer cell barcodes automatically. Preserve read-to-cell/UMI metadata for
downstream analysis; read assignments are not UMI-deduplicated counts.

## 3. Example

The supplied example contains the **full GRCh38 chr21 reference**
(46,709,983 bases), 64,812 RepeatMasker annotation records and **12 real ONT
reads** from public LongBench run [SRR30947478](https://www.ebi.ac.uk/ena/browser/view/SRR30947478).
Four reads have no TE candidate, four have one, and four have multiple candidates.
All required inputs and expected outputs are included.

```bash
python scripts/run_chr21_demo.py
```

The command runs LM assignment and the complete Joint workflow, checks input
hashes, and compares the outputs with the supplied expectations. Success is
reported as:

```text
EXPECTED_OUTPUTS_MATCH
```

The example takes approximately **5 minutes** on CPU and uses about
**1.5 GiB** of memory (one numerical thread, Intel Xeon Platinum 8358P).
Runtime varies with hardware.

Each run creates a directory named `chr21_YYYYMMDD_HHMMSS_<id>` under
`demo_output/`, where `<id>` is a six-character random suffix. Result filenames
stay fixed inside it. Use `--out PATH` to choose an exact directory; an existing
path is rejected. Assignment commands also reject existing output files. See [example provenance and inputs](examples/chr21/README.md)
and [expected output tables](examples/chr21/expected/).

The example demonstrates read assignment on a small selected population;
it should not be used to estimate accuracy or sample-wide expression.

## 4. Output

The real chr21 example produces:

```text
demo_output/chr21_YYYYMMDD_HHMMSS_<id>/
├── SLOTTED_joint_predictions.tsv   # Joint assignments and guarded calls
├── SLOTTED_lm_assignments.tsv      # LM-only assignments and guarded calls
└── intermediates/
    ├── input.pkl                  # Prepared input
    ├── scores.parquet             # Candidate scores
    ├── channels.parquet           # Merged scores
    ├── joint_fit.json             # EM convergence and provenance
    ├── lm_receipt.json            # LM run details
    ├── demo_receipt.json          # Demo verification and timing
    ├── README.md
    └── *.log                      # Processing logs
```

### 4.1 LM assignment table (`SLOTTED_lm_assignments.tsv`)

| Field | Description |
|---|---|
| `read_id` | Input read identifier |
| `n_candidates` | Number of candidate TE loci |
| `winner_locus` | Highest-scoring candidate locus |
| `winner_logit` | Learned-model logit for the winner |
| `margin` | Top-two logit difference; not a probability |
| `prob_margin` | Top-two softmax probability difference, with `--selective` |
| `winner_seeds`, `max_other_seeds` | Exact-seed support for the winner and strongest alternative |
| `call` | `UNIQUE` or `ABSTAIN`, with `--selective` |

Reads with no TE candidates are omitted from the LM table and counted in
its receipt. The chr21 example therefore produces **8 LM rows**.

### 4.2 Joint prediction table (`SLOTTED_joint_predictions.tsv`)

| Field | Description |
|---|---|
| `read_id`, `n_candidates` | Input read identifier and candidate count |
| `method` | `CF_LM_v1_EM` or `CF_LM_v1_EM_guard` |
| `winner` | Assigned locus; empty when no assignment is reported |
| `posterior` | EM posterior of the assign-all winner; not a calibrated correctness probability |
| `called` | Whether this endpoint reports a locus |
| `report_reason` | Assignment, abstention or `NO_CANDIDATE` reason |
| `fit_converged`, `fit_status` | Convergence status of the shared EM fit |

Joint retains every input read for both endpoints: **24 rows** in this example.
For the guarded endpoint, inspect `called` and `winner` before interpreting a
reported locus. No-candidate populations are explicitly marked without
performing an EM fit.

Assignment is limited to candidates present in the supplied alignments.
Selective calls require a probability margin of at least 0.10, at least five
exact k=31 seed hits, and stronger seed support than every alternative. A
single-candidate read requires the seed floor; exact probability ties abstain.

## 5. Support

Report bugs or usage questions through [GitHub Issues](https://github.com/fzc1997/SLOTTED/issues).
Include your command, software version, environment and relevant error message.
To run the tests, install `python -m pip install -e '.[dev]'` and run
`python -m pytest tests -q`.

## 6. License

Copyright © 2026 Zhican Fu and co-authors. All rights reserved.

Free for academic, educational and not-for-profit research use during peer
review. Other uses require permission. The final open-source license will
be specified upon publication. See [LICENSE](LICENSE).
