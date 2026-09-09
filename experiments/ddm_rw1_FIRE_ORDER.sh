#!/usr/bin/env bash
# ddm_rw1 fire order — the whole chain, in the order it must run, with the gate stated.
#
# GATE (MAIN's sequencing rule, 2026-09-09): do NOT start step 3 until BOTH
#   (a) `tools/cell_admission.py cells` shows 0 Metal occupants, AND
#   (b) `.omx/tmp/codex_runs/ddm_sj1_pass4_r2.done.done` exists
# sj1's five CPU shards starve a Metal cell's host thread, and a render-admitted
# move must land before this arm's admission runs (this arm moves ALL 600 renders,
# so MAIN sequences it LAST).
#
# RE-BASE: steps 0-2 are cheap and MUST be re-run after any pointer move; the module
# refuses on its own if the live archive sha stops matching what it is pinned to.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO}/.venv/bin/python"
ARM="${REPO}/experiments/ddm_rw1_renderer_edge_foldback.py"
SHARDS_SH="${REPO}/experiments/ddm_rw1_admission_shards.sh"
WORK="/Volumes/VertigoDataTier/pact/ddm_rw1_renderer_edge_foldback"
LAUNCH="${PY} ${REPO}/tools/launch_detached_process.py"

echo "== 0. controls (seconds; re-run after ANY pointer move) =="
echo "${PY} ${ARM} control --threads 3"
echo "${PY} ${ARM} section --threads 2"
echo "${PY} ${ARM} export --threads 2 --out-dir ${WORK}/nullbuild --out ${WORK}/receipts/EXPORT_NULL.json   # MUST reproduce the live archive byte for byte"

echo
echo "== 1. pose geometry (~11 min, 1 proc; only if the carrier section changed) =="
echo "${LAUNCH} --output-dir ${WORK}/logs/prep --done-receipt rw1_prep.done --nice 10 --nice-best-effort -- ${PY} ${ARM} prep --batch 25 --threads 3"

echo
echo "== 2. rate law (~35 s; only if the semantic section changed) =="
echo "${LAUNCH} --output-dir ${WORK}/logs/ratelaw --done-receipt rw1_ratelaw.done --nice 10 --nice-best-effort -- ${PY} ${ARM} rate-law --threads 2"

echo
echo "== 3. TRAIN — GATED, one Metal occupant =="
cat <<'TRAINNOTE'
MEASURED cost (25-step MPS compatibility+timing smoke, batch 4, 2 threads, while the
machine was loaded): 2.17 s/step -> 3,000 steps ~= 1.81 h, plus ten n600 evaluations.

LR, RE-DERIVED against the measured rate law.  The first derivation targeted an O(1)-code
drift over the horizon (lr = 2/T = 6.7e-4) because the rate was believed expensive.  The
rate law says a FULL rewrite of all 12,672 codes costs 176.7 B = 139 repaired cells, so an
O(1) drift is now needlessly timid: at 6.7e-4 the measured drift after 25 steps is 0.0086
codes (predicted 0.0084, ratio 1.02) and NO code crosses a rounding boundary.  Target an
O(3)-code drift instead: lr = 2*3/3000 = 2e-3.  The n600 evaluation every 300 steps IS the
line search -- the best-by-instrument checkpoint is kept, so an ft1-style excursion is
caught at step 300 rather than at the end.
TRAINNOTE
echo "# check the gate first:"
echo "${PY} ${REPO}/tools/cell_admission.py cells | head -2"
echo "ls ${REPO}/.omx/tmp/codex_runs/ddm_sj1_pass4_r2.done.done"
echo "${LAUNCH} --output-dir ${WORK}/logs/train --done-receipt rw1_train.done --nice 10 --nice-best-effort -- ${PY} ${ARM} train --device mps --steps 3000 --batch 4 --lr 2e-3 --eval-every 300 --checkpoint-every 300 --run-dir ${WORK}/runs/foldback --out ${WORK}/receipts/TRAIN.json --threads 3"
echo "# resumable: add --resume-from ${WORK}/runs/foldback/ckpt.periodic.stepNNNNNN.pt"

echo
echo "== 4. admission (CKPT = the best-by-instrument checkpoint the trainer names) =="
echo "CKPT=\$(${PY} -c \"import json;print(json.load(open('${WORK}/receipts/TRAIN.json'))['best']['path'])\")"
echo "STAGE=render CKPT=\${CKPT} SHARDS=4 bash ${SHARDS_SH}    # ~10 min: seeds the 3.66 GB decode, then 4 strided shards"
echo "STAGE=seg    SHARDS=4 bash ${SHARDS_SH}                   # ~5 min: n600 realized flips, merged; REFUSES partial coverage"
echo "STAGE=pose   SHARDS=4 bash ${SHARDS_SH}                   # ~1.4 h: per-pair re-solve, resumable per row"
echo "${PY} ${ARM} admit --seg ${WORK}/admission/SEG.json --pose-rows ${WORK}/admission/pose_rows_*.jsonl --checkpoint \${CKPT} --threads 3"

echo
echo "== 5. seal (only if ADMISSION.json says admits=true) =="
cat <<'SEALBLOCK'
5a. stage the candidate archive into a copy of the LIVE tree and re-pin the receiver's
    own ARCHIVE_SHA256 / ARCHIVE_BYTES in its inflate.py.
5b. produce the public-entrypoint smoke PAIR with pc2's producer (both roles; the
    frontier role is the LIVE pointer tree, LATE-BOUND on purpose):
      .venv/bin/python experiments/ddm_pc2_carrier_kwidth_rankcut.py smoke \
          --out <smoke_dir> --candidate-runtime <staged_tree> \
          --frontier-runtime <live_pointer_tree>
5c. check it with the validator's OWN checker before firing:
      .venv/bin/python -c 'import json,sys;from tac.candidate_seal import _public_smoke_problems;print(_public_smoke_problems(json.load(open(sys.argv[1])),None,None))' \
          <smoke_dir>/public_entrypoint_smoke.json
5d. seal -- every flag below verified against tools/make_candidate_seal.py add_argument:
      .venv/bin/python tools/make_candidate_seal.py \
          --candidate-id ddm_rw1_renderer_edge_foldback \
          --runtime-dir <staged_tree> --axis contest_cuda --out <seal.json> \
          --public-entrypoint-smoke <smoke_dir>/public_entrypoint_smoke.json \
          --admit-bar-net-ds -2e-5 --verify-archive-sha <candidate_sha> \
          --bound-base-receipt <live_row_MODAL_REMOTE_RESULT.json> \
          --falsifier <...> --retained-path <...> --json
    There is deliberately NO flag that takes a hand-typed bound; --bound-base-receipt
    COMPUTES it. The candidate id carries no v<digit> token.
    MAIN fires. This arm never calls Modal.
SEALBLOCK
echo
echo "== 6. WHAT THE 2026-09-09 RUN LEFT FOR THE NEXT ARM =="
cat <<'NEXTNOTE'
The 3,000-step minibatch run FIRED falsifier (a): the residual ROSE to -11.37 % at 66
changed codes.  Two controls then made that number readable:

  perturb-control : a RANDOM code change of the same size costs 2.1-4.1x MORE, so the
                    surrogate IS steering -- it avoids 75.6 % of random collateral at
                    66 codes and needs 100 % to break even.
  full-field      : step 1 moves the latent by exactly lr and step 2 reports a
                    byte-identical loss, because the forward is piecewise constant in
                    the latent while AdamW normalises PER PARAMETER.  Every code marches
                    at the same rate, which destroys the sparsity the rate law and the
                    collateral both call for.

So the next probes, cheapest first:

  1) gradient-topk -- one exact n600 gradient, then realized flips for the k largest
     |dL/dcode| each moved ONE step down their own gradient.  No optimizer in the way.
        .venv/bin/python experiments/ddm_rw1_renderer_edge_foldback.py gradient-topk \
            --device mps --k 1,4,16,64,256 --batch 4
  2) if top-k turns positive at some k, sweep k and tau, then run the admission (4a-4c).
  3) only if BOTH stay negative is the actuator closed at formulation scope, and only
     then is the charter's widening (--widened, +10,080 codes in blocks.2) worth an hour.
NEXTNOTE
echo

echo "== FALSIFIERS (count them plainly, do not narrate around them) =="
echo "(a) instrument residual falls < 3 % after 3,000 steps -> widen ONCE to blocks.2 (--widened) at the same LR;"
echo "    still < 3 % -> the renderer-weight actuator is CLOSED on this object at formulation scope."
echo "(b) admitted total dS >= -2e-5 after re-solve + real encode -> no candidate; record the per-pair"
echo "    seg/pose/rate distribution and stop."
