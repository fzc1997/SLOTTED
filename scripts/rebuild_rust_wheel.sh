#!/usr/bin/env bash
# Rebuild the m2d_bands Rust wheel (bit-identical band construction kernel)
# and install it into the active Python environment.
#
# Usage:  bash scripts/rebuild_rust_wheel.sh [python]
# Requires: a current rustc/cargo toolchain and maturin (pip install maturin).
set -euo pipefail
PY="${1:-python}"
PYTHON_EXE="$("$PY" -c 'import sys; print(sys.executable)')"
export PYO3_PYTHON="$PYTHON_EXE"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE/rust/m2d_bands"

# Keep Cargo's target-directory setting; isolate only this build's wheel output.
WHEEL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/slotted-m2d-wheel.XXXXXXXX")"
WHEEL_DIR="$(cd "$WHEEL_DIR" && pwd)"
trap 'rm -rf -- "$WHEEL_DIR"' EXIT

cargo test --release --locked
"$PYTHON_EXE" -m maturin build --release --locked --features extension-module \
  -i "$PYTHON_EXE" --out "$WHEEL_DIR"
shopt -s nullglob
WHEELS=("$WHEEL_DIR"/*.whl)
if (( ${#WHEELS[@]} != 1 )); then
  printf 'Expected one freshly built wheel in %s; found %s\n' "$WHEEL_DIR" "${#WHEELS[@]}" >&2
  exit 1
fi
if [[ "${WHEELS[0]##*/}" != m2d_bands-*.whl ]]; then
  printf 'Unexpected wheel filename: %s\n' "${WHEELS[0]}" >&2
  exit 1
fi
"$PYTHON_EXE" -m pip install --force-reinstall --no-deps "${WHEELS[0]}"
"$PYTHON_EXE" - <<'EOF'
import numpy as np, m2d_bands
g = m2d_bands.candidate_group_bands("ACGT" * 300, ["ACGT" * 200 + "TT", "ACGT" * 200 + "GG"])
print("m2d_bands OK:", len(g), g[0].shape, g[0].dtype)
EOF
