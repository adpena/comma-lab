#!/usr/bin/env bash
# release_tac_oss.sh — produce a sanitized export of `tac` for public OSS release.
#
# DO NOT execute this without first reviewing the export tree manually.
# DO NOT push to GitHub from this script. The user controls the public publish step.
#
# Outputs: ~/tac_oss_export/ — clean, sanitized, ready for `git init && git push`.
#
# Reference: experiments/results/oss_tac_release_plan_20260504_claude/release_plan.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EXPORT_DIR="${TAC_EXPORT_DIR:-$HOME/tac_oss_export}"

if [[ -e "$EXPORT_DIR" ]]; then
  echo "FATAL: $EXPORT_DIR already exists. Move or delete before re-running."
  exit 1
fi

echo "[1/8] Building export at $EXPORT_DIR"
mkdir -p "$EXPORT_DIR"

echo "[2/8] Copying whitelisted source tree (rsync with --include + --exclude)"
# We use rsync with explicit include patterns. Alternative: explicit cp -r per item.
rsync -av \
  --include="LICENSE" \
  --include="src/" \
  --include="src/tac/***" \
  --include="experiments/" \
  --include="experiments/pipeline.py" \
  --include="experiments/contest_auth_eval.py" \
  --include="experiments/canonical_local_auth_eval_smoke.py" \
  --include="experiments/modal_auth_eval.py" \
  --include="docs/" \
  --include="docs/architecture.md" \
  --include="docs/scoring_formula.md" \
  --include="docs/deterministic_archive_contract.md" \
  --include="docs/faq_runtime_closure.md" \
  --include="docs/paper/" \
  --include="docs/paper/EXTERNAL_SOURCE_ATTRIBUTION_C067.md" \
  --include="tools/" \
  --include="tools/lane_maturity.py" \
  --include="tools/review_tracker.py" \
  --exclude="**/__pycache__" \
  --exclude="**/.pytest_cache" \
  --exclude="**/.ruff_cache" \
  --exclude="**/.mypy_cache" \
  --exclude="**/*.pyc" \
  --exclude="**/.DS_Store" \
  --exclude="*" \
  "$REPO_ROOT/" "$EXPORT_DIR/"

echo "[3/8] Replacing pyproject.toml with sanitized version (no cloud extras)"
# We keep the dependency declaration but drop the [project.optional-dependencies.cloud]
# block (Modal/Lightning/Vast — operator-specific). We also keep cryptography for
# Lane C compliance gate.
.venv/bin/python "$REPO_ROOT/experiments/results/oss_tac_release_plan_20260504_claude/_sanitize_pyproject.py" \
  --src "$REPO_ROOT/pyproject.toml" \
  --dst "$EXPORT_DIR/pyproject.toml" || {
    # Fallback: just copy as-is and let user manually strip cloud extras.
    cp "$REPO_ROOT/pyproject.toml" "$EXPORT_DIR/pyproject.toml"
    echo "[WARNING] pyproject sanitizer not run — manual strip of [cloud] required"
  }

echo "[4/8] Replacing README.md with public-facing version"
cp "$REPO_ROOT/experiments/results/oss_tac_release_plan_20260504_claude/PUBLIC_README.md" "$EXPORT_DIR/README.md"

echo "[5/8] Stripping known-private references via sed"
# Tailscale IP scrubbing
declare -a TAILSCALE_IPS=(
  "100.81.85.28" "100.125.140.94" "100.120.99.124" "100.114.131.54" "100.65.24.39"
)
declare -a OPERATOR_PATHS=(
  "/Users/adpena/" "/Users/adpena/Projects/pact/" "/teamspace/"
)
declare -a OPERATOR_TOKENS=(
  "vastai_api_key" "modal_token" "lightning_studio_token"
)
for f in $(find "$EXPORT_DIR" -name "*.py" -o -name "*.md" -o -name "*.toml" -o -name "*.sh"); do
  # Tailscale IPs → REDACTED
  for ip in "${TAILSCALE_IPS[@]}"; do
    sed -i.bak "s|$ip|<TAILSCALE_REDACTED>|g" "$f" 2>/dev/null || true
  done
  # Operator absolute paths → ~ or generic placeholder
  sed -i.bak "s|/Users/adpena/Projects/pact|<REPO_ROOT>|g" "$f" 2>/dev/null || true
  sed -i.bak "s|/Users/adpena/|~/|g" "$f" 2>/dev/null || true
  sed -i.bak "s|/teamspace/[a-zA-Z0-9_/-]*|<LIGHTNING_STUDIO_REDACTED>|g" "$f" 2>/dev/null || true
  rm -f "$f.bak"
done

echo "[6/8] Adding .gitignore"
cat > "$EXPORT_DIR/.gitignore" <<'EOF'
__pycache__/
*.pyc
*.pyo
.venv/
venv/
.pytest_cache/
.ruff_cache/
.mypy_cache/
.DS_Store
*.egg-info/
build/
dist/
.coverage
.tox/
results/
EOF

echo "[7/8] Adding .github/workflows/ci.yml"
mkdir -p "$EXPORT_DIR/.github/workflows"
cat > "$EXPORT_DIR/.github/workflows/ci.yml" <<'EOF'
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ['3.11', '3.12']
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
        with:
          version: latest
      - name: Set up Python
        run: uv venv --python ${{ matrix.python-version }}
      - name: Install
        run: uv pip install -e ".[dev]"
      - name: Lint
        run: uv run ruff check .
      - name: Test
        run: uv run pytest src/tac/tests -m "not cuda and not slow" --timeout=60
EOF

echo "[8/8] Writing CHANGELOG.md from module → score-landmark map"
cat > "$EXPORT_DIR/CHANGELOG.md" <<'EOF'
# Changelog

## v1.0.5 — initial public release (2026-05-04)

The `tac` package as published in this release tracks contributions from
2026-04-09 onwards across the comma.ai video compression challenge. Score
landmarks per module:

- `qp1_pose_codec.py` — C-067 anchor 0.31561703 [contest-CUDA T4 A++]
- `pfp16_codec.py` — PFP16 A++ baseline 1.044 [contest-CUDA T4]
- `qzs3_renderer_codec.py` — Lane G v3 1.05 [contest-CUDA RTX 4090]
- `water_filling_codec_v2.py` — Lane Ω-W-V2 40.98% byte savings [empirical]
- `arithmetic_qint_codec.py` — PD-V2 0.9974 [contest-CUDA RTX 4090] (first sub-1.0)
- `submission_archive.py` — strict-scorer-rule packer (foundational)
- `eval_roundtrip_gate.py` — closes proxy-auth gap (foundational)
- `preflight.py` — 90+ STRICT preflight checks; permanent bug-class extinctions
- `sjkl_basis.py` — Score-Jacobian KL basis primitive (Wave-Ω paradigm)
- `sensitivity_map.py` — β-Fisher 1.016 LANDED [contest-CUDA T4]
- `lane_c_compliance.py` — Ed25519 cryptographic compliance gate
- `mask_grayscale_lut.py` — Selfcomp 0.36 paradigm (PR #56 reverse engineering)
- `henosis_pr82_transfer.py` — PR #82 Henosis 30-byte-header decoder
- `pr85_bundle.py` — PR #85 Otto adaptive masking
- `pr86_hpac_codec.py` — PR #86 HPAC hybrid coder

See `README.md` for project overview, methodology, and acknowledgments.
EOF

echo ""
echo "===================================================================="
echo "DONE. Export tree at: $EXPORT_DIR"
echo "===================================================================="
echo ""
echo "Recommended next steps for the operator:"
echo "  1. Manually review export tree for leaked private references:"
echo "     grep -r 'TAILSCALE_REDACTED\\|LIGHTNING_STUDIO_REDACTED' $EXPORT_DIR"
echo "  2. Run pytest in isolation:"
echo "     cd $EXPORT_DIR && uv venv && uv pip install -e '.[dev]' && uv run pytest src/tac/tests -m 'not cuda and not slow'"
echo "  3. Initialize git:"
echo "     cd $EXPORT_DIR && git init && git add . && git commit -m 'initial public release of tac (Task-Aware Codec)'"
echo "  4. Create the public repo on GitHub:"
echo "     gh repo create adpena/tac --public --description 'Task-Aware Codec for the comma.ai video compression challenge'"
echo "  5. Push:"
echo "     git remote add origin git@github.com:adpena/tac.git && git push -u origin main"
echo "  6. Tag a release:"
echo "     git tag v1.0.5 && git push origin v1.0.5"
echo "  7. Replace placeholder URLs in PR107 body (in commaai/comma_video_compression_challenge#107) with the actual public URL."
echo ""
