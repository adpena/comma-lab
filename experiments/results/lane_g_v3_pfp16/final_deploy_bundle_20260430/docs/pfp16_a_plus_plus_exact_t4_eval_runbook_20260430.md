# PFP16 A++ Exact T4 Auth Eval Runbook

Goal: evaluate the exact Lane G v3 + PFP16 archive
`0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`
with `experiments/contest_auth_eval.py` on a Tesla T4 CUDA/NVDEC host.

## Fastest Safe Path

Do not rebuild the lane. Upload the already-built archive and run
`contest_auth_eval.py` directly.

Local exact artifact:

```bash
ARCH=experiments/results/lane_g_v3_pfp16/archive_lane_g_v3_pfp16.zip
EXPECTED_SHA=0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f
EXPECTED_BYTES=686635

test "$(shasum -a 256 "$ARCH" | awk '{print $1}')" = "$EXPECTED_SHA"
test "$(stat -f%z "$ARCH" 2>/dev/null || stat -c%s "$ARCH")" = "$EXPECTED_BYTES"
```

Lightning T4 dispatch template:

```bash
export LIGHTNING_USER=s_...   # fill in the active Lightning SSH user
export REMOTE="${LIGHTNING_USER}@ssh.lightning.ai"
export SSH_KEY="${HOME}/.ssh/lightning_rsa"
export REMOTE_PACT=/home/zeus/content/pact
export REMOTE_UPSTREAM=/home/zeus/content/upstream

ARCH=experiments/results/lane_g_v3_pfp16/archive_lane_g_v3_pfp16.zip
EXPECTED_SHA=0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f
EXPECTED_BYTES=686635
RUN_ID=pfp16_a_plus_plus_t4_$(date -u +%Y%m%dT%H%M%SZ)

ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$REMOTE" \
  "mkdir -p '$REMOTE_PACT/auth_eval_input' '$REMOTE_PACT/experiments/results/lane_g_v3_pfp16/$RUN_ID'"

scp -i "$SSH_KEY" -o StrictHostKeyChecking=no \
  "$ARCH" "$REMOTE:$REMOTE_PACT/auth_eval_input/pfp16_exact_0af839_archive.zip"

ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$REMOTE" \
  "REMOTE_PACT='$REMOTE_PACT' REMOTE_UPSTREAM='$REMOTE_UPSTREAM' RUN_ID='$RUN_ID' bash -s" <<'REMOTE'
set -euo pipefail

EXPECTED_SHA=0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f
EXPECTED_BYTES=686635

cd "$REMOTE_PACT"
source /system/conda/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate cloudspace 2>/dev/null || true
source "$REMOTE_PACT/env.sh" 2>/dev/null || true

if [ -x /opt/conda/bin/python ]; then
  PYBIN=/opt/conda/bin/python
else
  PYBIN="${PYBIN:-$(command -v python)}"
fi
export PYBIN

export PYTHONPATH="$REMOTE_PACT/src:$REMOTE_UPSTREAM:$REMOTE_PACT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ARCHIVE="$REMOTE_PACT/auth_eval_input/pfp16_exact_0af839_archive.zip"
OUT="$REMOTE_PACT/experiments/results/lane_g_v3_pfp16/$RUN_ID"
mkdir -p "$OUT"

ACTUAL_SHA=$("$PYBIN" -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$ARCHIVE")
ACTUAL_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
echo "archive_sha256=$ACTUAL_SHA" | tee "$OUT/archive_sha256.txt"
echo "archive_bytes=$ACTUAL_BYTES" | tee -a "$OUT/archive_sha256.txt"
test "$ACTUAL_SHA" = "$EXPECTED_SHA"
test "$ACTUAL_BYTES" = "$EXPECTED_BYTES"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "gpu_name=$GPU_NAME" | tee "$OUT/gpu.txt"
case "$GPU_NAME" in
  *T4*) ;;
  *) echo "FATAL: A++ evidence requires Tesla T4 provenance; got '$GPU_NAME'." >&2; exit 6 ;;
esac

bash "$REMOTE_PACT/scripts/probe_nvdec.sh" --ensure-dali

rm -rf "$OUT/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
  --archive "$ARCHIVE" \
  --inflate-sh submissions/robust_current/inflate.sh \
  --upstream-dir "$REMOTE_UPSTREAM" \
  --device cuda \
  --keep-work-dir \
  --work-dir "$OUT/eval_work" 2>&1 | tee "$OUT/auth_eval.log"

cp "$OUT/eval_work/contest_auth_eval.json" "$OUT/contest_auth_eval.json"
cp "$OUT/eval_work/provenance.json" "$OUT/eval_provenance.json"
cp "$OUT/eval_work/report.txt" "$OUT/report.txt"

"$PYBIN" - "$OUT/contest_auth_eval.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
prov = p["provenance"]
assert p["archive_size_bytes"] == 686635, p["archive_size_bytes"]
assert prov["archive_sha256"] == "0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f"
assert prov["device"] == "cuda", prov["device"]
assert prov.get("gpu_t4_match") is True, prov.get("gpu_model")
assert p["n_samples"] == 600, p["n_samples"]
print("PFP16_A_PLUS_PLUS_READY")
print(json.dumps({
    "final_score_reported_rounded": p["final_score"],
    "score_recomputed_from_components": p["score_recomputed_from_components"],
    "avg_posenet_dist": p["avg_posenet_dist"],
    "avg_segnet_dist": p["avg_segnet_dist"],
    "rate_unscaled": p["rate_unscaled"],
    "archive_size_bytes": p["archive_size_bytes"],
    "archive_sha256": prov["archive_sha256"],
    "gpu_model": prov.get("gpu_model"),
    "gpu_t4_match": prov.get("gpu_t4_match"),
}, indent=2, sort_keys=True))
PY
REMOTE
```

Fetch evidence:

```bash
mkdir -p "experiments/results/lane_g_v3_pfp16/$RUN_ID"
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$REMOTE" \
  "cd '$REMOTE_PACT/experiments/results/lane_g_v3_pfp16/$RUN_ID' && \
   tar czf - auth_eval.log contest_auth_eval.json eval_provenance.json report.txt archive_sha256.txt gpu.txt" \
  | tar xzf - -C "experiments/results/lane_g_v3_pfp16/$RUN_ID"
```

## Expected Output

Successful run must contain:

- `archive_sha256=0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`
- `archive_bytes=686635`
- `gpu_name=Tesla T4` or equivalent `nvidia-smi` name containing `T4`
- `DaliVideoDataset on rank 0 with 1 files.`
- `RESULT_JSON:` from `experiments/contest_auth_eval.py`
- `PFP16_A_PLUS_PLUS_READY`
- `contest_auth_eval.json` with:
  - `archive_size_bytes: 686635`
  - `n_samples: 600`
  - `provenance.device: cuda`
  - `provenance.gpu_t4_match: true`
  - `provenance.archive_sha256: 0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`

Expected metrics should be close to the existing exact CUDA evidence:

- reported rounded final score: `1.04`
- recomputed score: about `1.0440481283330025`
- PoseNet distortion: about `0.00346020`
- SegNet distortion: about `0.00400830`
- rate: about `0.01828808`

## Existing Evidence And Blocker

Existing local evidence directory:
`experiments/results/lane_g_v3_pfp16/exact_cuda_20260430T1353Z/`

It evaluates the exact archive SHA with `contest_auth_eval.py --device cuda`
and recomputes score `1.0440481283330025`, but it is not A++ because its
provenance is RTX 4090:

- `gpu_model: NVIDIA GeForce RTX 4090`
- `gpu_t4_match: false`

Do not use these wrappers for A++ evidence:

- `experiments/modal_train_lane.py`: forces `AUTH_EVAL_DEVICE=cpu` in Modal.
- `experiments/modal_auth_eval.py`: does not run `contest_auth_eval.py` and
  evaluates with CPU/PyAV.
- `scripts/lightning_auth_eval.sh`: old postfilter-specific direct eval path,
  not the PFP16 exact archive through `contest_auth_eval.py`.

Operational blocker: an accessible T4 CUDA/NVDEC host with the synced repo,
upstream videos/models, and working `uv`/DALI. There is no core codec blocker.

## Successful Run - 2026-04-30T16:20Z

The blocker is cleared. Lightning AI produced exact A++ evidence.

Artifacts:

- `experiments/results/lane_g_v3_pfp16/pfp16_a_plus_plus_t4_20260430T1620Z_codex/contest_auth_eval.json`
- `experiments/results/lane_g_v3_pfp16/pfp16_a_plus_plus_t4_20260430T1620Z_codex/auth_eval.log`
- `experiments/results/lane_g_v3_pfp16/pfp16_a_plus_plus_t4_20260430T1620Z_codex/report.txt`
- `experiments/results/lane_g_v3_pfp16/pfp16_a_plus_plus_t4_20260430T1620Z_codex/eval_provenance.json`

Facts:

- Remote: `s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai`.
- Hardware: Tesla T4, driver `580.126.09`.
- Remote staged tree:
  `/home/zeus/content/pact_pfp16_exact_20260430T1625Z`.
- Upstream tree: `/home/zeus/content/upstream`, git `c5e1274`.
- Eval chain:
  `contest_auth_eval.py --device cuda -> inflate.sh -> upstream/evaluate.py`.
- `gpu_t4_match=true`, `n_samples=600`.
- Archive SHA:
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`.
- Archive bytes: `686635`.
- Rounded score: `1.04`.
- Recomputed score: `1.043987524793892`.
- PoseNet: `0.00346442`.
- SegNet: `0.00400656`.
- Rate: `0.01828808`.

Verdict: A++ evidence grade.
