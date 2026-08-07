# ddm_mx1t — checkpoint-series FACET analyzer + tail-average A/B (the iteration instrument)

**Critical-path clause:** ARM-CAP's telemetry shows loss −29% while d_seg_batch sits FLAT at
~0.00104 (steps 250→1000+). That divergence is THE iteration question: are the remaining wrong
pixels near-flip (margins shrinking → loss descent IS progress, keep training / more steps pays)
or far-margin stuck (capacity/objective wall → next config needs different mechanism)? NO current
instrument can answer it — the history carries one scalar, the mx1h receipt is aggregate-only.
Per the holistic-facet doctrine (per-class d_seg · per-pair · never a headline composite), the
arm-selection and the n120 config BOTH need this decomposition. Everything required is already
on disk: checkpoints every 250 steps (weights + pair_ids) + deterministic caches + frozen
scorers. This arm builds the analyzer that converts persisted signal into iteration telemetry.

**Recall-first (do not re-derive):**
- `--mode torch-verdict` (d7f557bb7c + rr14's tensor fail-closed tests) already loads MLX npz →
  torch and runs the upstream SegNet path. EXTEND it (or a sibling mode `torch-facets`) —
  do NOT build a new loader.
- Checkpoints: `.omx/research/ddm_mx1e_20260807/regen2/launch_arm_cap/n32_metal/
  mlx_stage_step{000250..}.npz` (every 250 steps; COPY before reading; record sha per ckpt).
- Class order is CANONICAL comma10k `0=Road 1=Lane 2=Undrivable 3=Movable 4=MyCar` — never
  luma-sort (CLAUDE.md SegNet section; the ×3 historical bug).
- Batch shape is part of the forward instrument: pin ONE batching scheme across all checkpoints
  and record it (memory `batch_shape_is_part_of_the_forward_instrument_20260806`).
- Tail-average tooling precedent: dy2 (#964) built a plateau-anchored tail-average EMA mode —
  check `git log --grep dy2` / its findings for reusable averaging code before writing new.

## Deliverable 1 — facet trajectories across the checkpoint SERIES
For EVERY saved checkpoint (250 → latest at run time), emit one receipt row with:
- aggregate d_seg (must reproduce mx1h's 0.0010689 at step 1500 — the cross-instrument anchor;
  if it does not match, STOP and diagnose before proceeding);
- PER-CLASS d_seg (mismatch rate per canonical class, both directions: GT-class c mispredicted,
  and predicted-c false positives);
- PER-PAIR d_seg (all 32 pairs, the full vector — not top-k);
- MARGIN HISTOGRAM at mismatched pixels: (top1 − top2) logit gap distribution (fixed bins,
  e.g. 0-0.05/0.05-0.1/0.1-0.25/0.25-0.5/>0.5) + the same at CORRECT boundary-band pixels for
  contrast. The TREND of this histogram across steps is the near-flip-vs-stuck discriminator.
- flip-set churn: |mismatch-set(step_k) Δ mismatch-set(step_{k-1})| / |mismatch-set| — is the
  model trading the SAME pixels back and forth (oscillation) or converging on a stable residual?

## Deliverable 2 — tail-average vs final A/B (the EMA-gap cure, $0)
The lifted trainer has NO EMA (verified: no shadow in the MLX loop — doctrine gap inherited
from the lift). Do NOT touch the live trainer (A/B symmetry with ARM-VEH must hold). Instead:
- average the LAST K saved checkpoints (K∈{2,4,8}; simple mean of params; reuse dy2 machinery
  if compatible) → run the SAME facet verdict on each average → table: final vs avg-K d_seg.
- If averaging wins, that is a FREE post-hoc quality lever for BOTH arms + n120, applied
  symmetrically at selection time; if it loses, the EMA gap is measured harmless HERE (scoped:
  this vehicle, this stage — not a general EMA verdict).

## Deliverable 3 — the iteration verdict (typed, scoped)
One findings table answering: (a) near-flip vs stuck (margin trend); (b) which classes/pairs
own the residual; (c) churn regime; (d) tail-average verdict; (e) the RECOMMENDED next-config
deltas for the n120 fire (steps/LR/etc.) each tagged DERIVED-from-which-measurement. Axis
labels everywhere: [macOS-CPU advisory torch upstream SegNet], score_claim=false, n32
arm-instrument scope — never a family/n600 claim.

## Constraints
CPU-only, NO Metal, NO scorer slot conflict (slot is free; this IS the scorer-slot occupant —
mark it), co-tenant RAM ≤ ~8GB transient (chunk pairs), read-only toward the live run dir
(copy checkpoints out). Wall-clock: full series × 32 pairs is fine serially; if slow, do
{250, 1000, 2000, latest} first and append the rest.

## Discipline
Serializer + POST-EDIT `--expected-content-sha256` per file; tags `[no-triality] [p0-ledger-ok]`;
review_tracker ×2 per .py; NO Claude/AI attribution or Co-Authored-By trailer — commits are the
operator's alone. Findings: `.omx/research/ddm_mx1t_20260807/MX1T_FINDINGS.md` + receipts JSONL.
