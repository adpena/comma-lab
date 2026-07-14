# Codex adversarial findings — Task #494 throughput authority ladder

**UTC:** 2026-07-14T03:10:02Z  
**Lane:** `throughput_authority_ladder`  
**Review:** round 1, independent re-derivation from code and receipts  
**Status:** `FIXED_N600_REDERIVED; DYNAMIC_N600_RUNNING; DEVICE_GATES_OWED_MAIN`  
**Authority:** `[macOS-CPU Torch one-thread advisory/QDQ feasibility]`  
**Flags:** `research_only=true` · `score_claim=false` · `pointer_moved=false`

## Pointer and verdict scope

The submittable pointer remains `0.19108282419209976 [contest-CPU]` (`ad02b012…`). The
exact borrowed-lineage defensive bank remains `0.1880443979880752 [contest-CPU]`
(`196acd18…`). This review changes neither. It reviews throughput MEANS only.

Negatives below are `INSTANCE` or `FORMULATION` scoped. None kills fixed-point arithmetic,
integer lowering, Metal, ANE, CUDA, or the witness-compiler paradigm.

## Findings

### F1 — fp32 control could falsely satisfy the fixed-point admission gate — FIXED

The initial n600 summary searched all arms for a minimum exact arm. `fp32_control` was exact by
construction, so the summary emitted `ARGMAX_FIXEDPOINT_FEASIBLE` even though every actual WnAn arm
failed. The finalizer now excludes all non-fixed-point controls from selection, requires an arm name
of the form `w<integer>a<integer>`, and emits `NO_ADMITTED_PRECISION_IN_LADDER` with both minimum
fields null. Regression coverage makes the control exclusion load-bearing.

**Verdict scope:** implementation bug in summary selection, not numerical-row invalidity. The
original row-producing source SHA and the later summary-finalizer SHA are recorded separately; no
numerical row was recomputed.

### F2 — an n600 label was possible without exact pair-index custody — FIXED

Row count alone could not prove pairs 0..599 were unique and complete. Summaries now record count,
unique count, expected and observed pair-index SHA-256, and require their equality before
`full_real_n600=true`. The fixed receipt now passes that exact custody predicate.

### F3 — legacy cache labels could overwrite the computed CPU authority in the Metal harness — FIXED

The first custom-Metal fidelity child recomputed one-thread CPU-Torch logits/argmax, then replaced
the reference labels with the legacy cache before counting flips. That silently demoted the intended
authority. The overwrite is removed. CPU-computed argmax and margins own fidelity; legacy cache
differences are audit-only telemetry. This matters because the cache differs by one pixel in one
pair under its historical thread geometry.

### F4 — QDQ feasibility was at risk of being presented as native integer speed — FIXED

The PyTorch probe performs weight/activation QDQ followed by fp32 convolution accumulation. It can
answer argmax/error feasibility, but cannot answer integer placement or latency. Receipts now state
`native_integer_speed_claim=false`; the typed policy refuses a receipt that does not explicitly make
that disclaimer. Only the custom-Metal host receipt can satisfy the speed gate.

### F5 — fixed calibration has a high-bit clipping plateau — MEASURED FORMULATION negative

On real pairs 0..599, W24A24 still flips 8,960 / 117,964,800 pixels, aggregate
`7.595486111111111e-05`, worst pair `9.358723958333334e-04`, with 133,890 interval-uncertified
pixels. W18 through W24 plateau near the same result while continuous Pose debt continues to shrink.
This implicates held-out activation-range clipping in the fixed-calibration formulation; it does not
show that more arithmetic precision is ineffective. The separate label-free dynamic max-absolute
arm removes that mechanism.

### F5b — dynamic scaling removes the plateau but is not exact through W24 — MEASURED INSTANCE

On exact pairs 0..599, dynamic W24A24 leaves 19 flips / 117,964,800 pixels, aggregate
`1.6106499565972223e-07`, worst pair `5.086263020833333e-06`, and 244 conservative
uncertified pixels. W20 is the first arm satisfying the registered aggregate and worst-pair
tolerance. Dynamic scaling reduces the fixed W24 flip mass about 471.6x, so it is a real lever, but
`minimum_argmax_exact_arm` remains null. The 19 flips occur on 19 distinct pairs.

The real SegNet maximum Conv2d fan-in is 4,248. Uniform W26A26 has bound
`4,782,822,519,189,016,728 < 2^63`; W27A27 exceeds int64. W25/W26 are being measured as a
finite single-int64 ceiling check. If neither is exact, the negative scope is the uniform,
per-operator-dynamic-scale, single-int64 formulation—not mixed precision, multi-limb accumulation,
sparse correction, or the fixed-point family.

### F6 — the full-R equation anchor originally targeted the wrong receipt schema — FIXED

The full-R probe emits `pythagorean_exact_arithmetic_full_r_n600.v2`. The anchor builder expected a
different schema string and would have rejected valid host evidence. Schema and tests now agree.

### F7 — integer R admission needed exact raw-state parity, not only a dequant bound — FIXED

A small fp32 dequant error does not prove the integer recurrence implemented the intended exact
state. The benchmark now compares every real frame's final raw int32 state hash against the NumPy
authority, requires cross-repeat identity, checks the static no-overflow bound, and only then applies
the dequant error and speed gates. It streams frames instead of retaining 1,200 full-frame views.

### F8 — direct-int64 Metal is a candidate kernel formulation, not authority by construction — GUARDED

The kernel replaces all 125 frozen-SegNet Conv2d instances, including grouped/depthwise and the
explicit head. The converter audit showed every construction path reaches one of the two patched
converter symbols, and a post-conversion set/unique-count predicate refuses incomplete coverage.
Exact integer accumulation removes reduction-order variation, but dynamic scaling, fp32
dequant/bias, device compiler support, end-to-end logits, cross-process digests, and latency still
require the M5-Max receipt. The worker environment has no evaluated Metal and does not manufacture
those measurements.

### F9 — strict interval certification is reported separately from empirical exact argmax — OPEN POLICY

The host harness records both full-corpus exact argmax and the conservative rule
`reference top1-top2 margin > 2*max-class absolute logit error`. The latter is sufficient but stronger
than direct exhaustive n600 argmax equality. The current policy intentionally requires both before a
default-off local candidate is exposed. If the device receipt is exact/reproducible/fast but has a
small conservative uncertified set, that is a policy decision to relax the certificate, not evidence
that the Metal formulation failed argmax preservation.

### F10 — ANE higher precision is blocked by the public formulation, not by ANE as a family — GUARDED

The settled calibrated CoreML W8A8 formulation flipped 45.836809% of held-out pixels and must not be
rerun as if new. Public CoreML activation quantization exposes an 8-bit compute formulation; it does
not provide the W16+ custom fixed-point surface selected by this ladder. The ticket compiler therefore
emits a typed `PUBLIC_ANE_PRECISION_UNREPRESENTABLE` blocker when appropriate, while retaining CoreML
fp32 as forward-only advisory evidence.

### F11 — the decisive wall is local forward verdict, not the fast differentiable teacher — CORRECTED

Measured n96 one-thread CPU-Torch local verdict time is 59.615 seconds, 0.621 seconds/pair, with
SegNet/PoseNet shares 0.774/0.226. The n600 372.6-second figure is a DERIVED linear projection. The
separate MLX single-call teacher backward/forward probe is not an epoch decomposition and must not
redirect Task #494 toward the wrong cost center. Integer R is retained for reproducibility; custom
fixed-point SegNet forward is the throughput P0.

## Review disposition

- Fixed-scale numerical receipt: **accepted as MEASURED FORMULATION negative** after independent
  summary re-derivation and custody repair.
- Dynamic-scale QDQ receipt: **pending full 0..599 completion**; no partial result will be promoted.
- Full-R, integer-R, custom-Metal, Pose dry-start: **built but OWED MAIN host measurements**.
- CoreML/ANE W8A8: **settled FORMULATION refusal**; no duplicate run.
- Contest CPU/CUDA authority: **unchanged and not inferred** from local substrate results.

## STORES CONSULTED

Full `CLAUDE.md`; full `AGENTS.md`; canonical frontier, lane, task, subagent, and equation stores;
the v7.5/v8 operating specs; relevant #348/#449/#456/#477/#478/#482/#490 research artifacts;
the exact Task #494 source files and receipts; both live inboxes through the timestamps recorded in
the session checkpoint. No paid provider, protected live run, run stop, or contest evaluator was
actuated.
