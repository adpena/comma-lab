# ddm_mx1h — CPU-torch authority verdict mode + LIVE dress rehearsal (the unbuilt endpoint rung)

**Critical-path clause:** campaign #984 Row-1's next rung after the ARM-CAP burn is "CPU-torch n32
verdicts vs fp1 floor 0.008305" — and NO tool exists for it (trainer modes are
probe/torch-smoke/mlx-parity/mlx-train/mem-probe; none is an eval-only verdict on an MLX
checkpoint). Building + rehearsing it NOW, mid-burn, converts the endpoint from hope to a proven
chain and yields the first proxy↔authority gap reading. This arm directly shortens time-to-next-
exact-row: the ARM-CAP vs ARM-VEH selection (which gates the n120 dispatch) reads THESE verdicts.

**Recall-first (do not re-derive):**
- `experiments/ddm_mx1_pr130_semantic_renderer.py::run_torch_smoke` (line ~522) already loads the
  lifted `SemanticTokenRenderer`, upstream SegNet on CPU (`_load_upstream_segnet`), stratified
  pair selection, and the token caches. Reuse all of it; DROP the optimizer (verdict = eval-only).
- `--mode mlx-parity` already contains the MLX↔torch parameter mapping (npz param names → torch
  state_dict). Reuse that mapping code for npz loading; do NOT invent a new one.
- The MLX checkpoint format: `save_stage_checkpoint_npz` — params under their MLX names,
  `meta::history_json` (uint8 bytes → json), `meta::step`, `opt::*`, `extra` carries `pair_ids`,
  `score_claim: False`, `axis`. The training pair set is IN the checkpoint — the verdict MUST use
  the same `pair_ids` (read them from the npz, never re-derive via seed).
- Batch shape is part of the forward instrument (memory
  `batch_shape_is_part_of_the_forward_instrument_20260806`): pin and RECORD the SegNet batch size
  used; run the whole n32 in ONE fixed batching scheme and write it into the receipt.

## Deliverable 1 — `--mode torch-verdict`
Add to `experiments/ddm_mx1_pr130_semantic_renderer.py` (same file, same argparse; new choice
`torch-verdict`):
- Args: `--init <mlx npz checkpoint>` (accept npz; detect by suffix; keep torch .pt path working
  for parity), `--input-cache/--target-cache` as in torch-smoke, `--out <receipt json>`.
- Load npz → torch state_dict via the parity mapping; model.eval(); `torch.no_grad()`.
- Forward the checkpoint's OWN `pair_ids` through the model → contest-faithful roundtrip to
  (384,512) → upstream SegNet argmax → mismatch vs target argmax = d_seg per pair + aggregate.
- Receipt json: per-pair d_seg, aggregate d_seg, pair_ids, checkpoint path+sha256, step,
  segnet batch size, axis `"[macOS-CPU advisory torch upstream SegNet]"`, `score_claim: false`,
  and the comparison row: MLX in-training d_seg_batch at the same step (read from
  `meta::history_json`) so the proxy↔authority gap is IN the artifact.
- CPU-only, no MLX import required on this path, no Metal. Peak RAM must stay modest (~6-8GB
  transient); process the n32 pairs in chunks if needed (record chunk size — instrument pin).

## Deliverable 2 — LIVE dress rehearsal
Run it on the LIVE ARM-CAP checkpoint:
`.omx/research/ddm_mx1e_20260807/regen2/launch_arm_cap/n32_metal/mlx.latest.npz`
- COPY the npz to your receipts dir first (atomic-renamed by the trainer, but copy anyway; record
  the copied file's sha256 + the step you got).
- Report: authority d_seg vs the MLX proxy (~0.001038 at step 1000) vs the fp1 floor 0.008305.
- DO NOT touch the live run dir otherwise. DO NOT launch anything on Metal. The trainer
  (pid 77351) keeps running; you are read-only toward it.

## OPTIMAL FORM
- Family reference form: the REAL upstream SegNet + contest-faithful roundtrip at (384,512),
  ste_round semantics matching the trainer's own in-training verdict path. No proxy scorer.
- SCOPE reduction (legal): n32 pairs — this is the arm-selection instrument the campaign design
  specifies; it is NEVER a family/n600 verdict. Say so in the receipt (`verdict_scope: n32
  arm-selection instrument`).
- MECHANISM reductions: NONE permitted. If the parity mapping is incomplete for any tensor,
  FAIL CLOSED with the tensor names — do not skip tensors silently.

## Discipline
- Commit via `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256`
  per file; tags `[no-triality] [p0-ledger-ok]`; run
  `tools/review_tracker.py mark-file <f> --status reviewed` twice per .py; NO Claude/AI
  attribution or Co-Authored-By trailer of any kind — commits are the operator's alone.
- Tests: at least npz→torch mapping round-trip on a synthetic tiny checkpoint + a receipt-schema
  test. Findings doc: `.omx/research/ddm_mx1h_20260807/MX1H_FINDINGS.md`.
- Axis honesty everywhere: [macOS-CPU advisory]; score_claim=false; never "contest" language.
