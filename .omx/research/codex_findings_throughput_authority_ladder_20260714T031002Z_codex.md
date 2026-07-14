# Codex adversarial findings — Task #494 throughput authority ladder

**UTC:** 2026-07-14T03:10:02Z  
**Lane:** `throughput_authority_ladder`  
**Review:** round 1, independent re-derivation from code and receipts  
**Status:** `QDQ_UNIFORM_AND_GEOMETRY_INT64_N600_REDERIVED; WEIGHT_L1_INT64_N600_RUNNING; DEVICE_GATES_OWED_MAIN`
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

### F5b — dynamic scaling removes the plateau but QDQ/fp32 is not exact through W26 — MEASURED FORMULATION

On exact pairs 0..599, dynamic W24A24 leaves 19 flips / 117,964,800 pixels, aggregate
`1.6106499565972223e-07`, worst pair `5.086263020833333e-06`, and 244 conservative
uncertified pixels. W20 is the first arm satisfying the registered aggregate and worst-pair
tolerance. Dynamic scaling reduces the fixed W24 flip mass about 471.6x, so it is a real lever, but
`minimum_argmax_exact_arm` remains null. The 19 flips occur on 19 distinct pairs.

The real SegNet maximum Conv2d fan-in is 4,248. Uniform W26A26 has bound
`4,782,822,519,189,016,728 < 2^63`; W27A27 exceeds int64. The corrected finite ceiling receipt is
complete: W25 has 13 flips and 139 uncertified pixels; W26 has 3 flips and 83 uncertified pixels.
The W26 flip pairs are 64, 371, and 587. Receipt SHA-256 is
`a04a8e2672981faeda9a2a1adb086c8e1a4c073c0e1319dcd78ee1536c594c91`.

The negative scope is the uniform, per-operator-dynamic-scale **QDQ with fp32 Conv accumulation**
formulation—not direct exact-int64 accumulation, mixed precision, multi-limb accumulation, sparse
correction, or the fixed-point family.

### F5c — W26 qmax was not representable in the original fp32 clamp — FIXED

Positive W26 `qmax=33,554,431` rounds to `33,554,432` in fp32. The original float-domain clamp could
therefore admit a code outside the signed W26 range. Torch, NumPy, and Metal paths now round and clamp
in exact integer space. Endpoint regressions are load-bearing. The predecessor diagnostic receipt
(SHA-256 `d6ccc273c0b2a9f1313588237eeb412773757c91a1e21e9b06c09dd9280a8a41`) is explicitly
non-authoritative even though its final flip totals happened to match the corrected receipt.

### F5d — QDQ/fp32 cannot stand in for the actual exact-int64 kernel — FIXED AND MEASURED

The QDQ candidate converts codes back to fp32 and calls fp32 Conv2d. Above the fp32 significand range,
it cannot retain every odd W26 code, and its reduction is not exact int64. Treating its three flips as
a direct-int64 negative would cross a formulation boundary. The CPU twin keeps signed codes through
all 125 Conv2d, proves every static accumulator bound, performs exact signed-int64 convolution, and
uses one fp32 finalization. Its exact real 0..599 receipt has 4 / 117,964,800 flips at pairs 64,
362, 371, and 507, aggregate `3.390842013888889e-08`, 77 conservative uncertified pixels, and
maximum absolute logit error `2.525597810745239e-04`. Receipt SHA-256 is
`b4bd48f580501926492d826a8a2504f5420fa266d6270f4aff915e7820f60af2`.

**Verdict scope:** `INSTANCE` negative for uniform W26 direct-int64 on this frozen SegNet and real
n600 corpus. It is not a negative for exact integer convolution, mixed precision, multi-limb
accumulation, Metal, or the fixed-point family.

### F5e — worst-layer uniform precision wastes the int64 budget — MEASURED INSTANCE NEGATIVE

Only five Conv2d layers require W26 under the static no-overflow contract. A label-free geometry rule
assigns each layer the largest precision in W26..W30 satisfying `fan_in*qmax^2 <= 2^63-1`, yielding
W26:5, W27:30, W28:22, W29:19, W30:49. The rule does not inspect labels, margins, or the four flip
locations. The full exact CPU twin closed real pairs 0..599 with 1 flip / 117,964,800 at pair 11,
aggregate `8.477105034722222e-09`, worst pair `5.086263020833333e-06`, 38 conservative
uncertified pixels, and maximum absolute logit error `7.62939453125e-05`. Training tolerance
passes, but exact authority does not. Receipt SHA-256 is
`129e9d39d09ff2e019cdab7ac04f699b64a846d319390d71d3bd12d9497959f5`.

**Verdict scope:** `INSTANCE` negative for the geometry-only W26..W30 allocation on this frozen
SegNet and real n600 corpus. The flipped reference pixel is an exact zero-margin tie; tighter static
bounds, multi-limb accumulation, sparse tie policy, Metal, and the fixed-point family remain open.

### F5f — frozen-weight L1 is the tighter static bound — DISTINCT SUCCESSOR RUNNING

The geometry bound still assumes every frozen weight code has magnitude `qmax`. For output channel
`oc`, exact integer convolution instead satisfies
`|acc_oc| <= activation_qmax * sum_i |weight_q[oc,i]|`. This uses only frozen model bytes and the
chosen precision; it is independent of frames, labels, logits, margins, and the observed flip set.
At the maximum safe signed-int32 code width W31, it yields W27:4, W28:28, W29:32, W30:41,
W31:20. The worst proven accumulator is `9,035,402,569,620,285,889`, leaving
`187,969,467,234,489,918` of signed-int64 headroom. The CPU twin, pair-atomic n600 runner, custom
Metal adapter, receipt-bound policy, and canonical anchor are built. The full geometry-only
predecessor closed negative, the fail-closed gate admitted this arm, and its full n600 run is now
resumable in progress; partial rows carry no verdict.

### F5g — the surviving error is currently tie semantics, not accumulator width — SUCCESSOR BUILT

The only partial W27..W31 flip observed so far is the same pair-11 pixel as the geometry arm. The
one-thread reference logits for classes 0 and 4 are exactly equal at `5.2026519775390625`, so
`torch.argmax` selects lower class 0. The W27..W31 candidate differs by only
`1.430511474609375e-06` and selects class 4. A focused diagnostic corrects that pixel first at
epsilon `2^-19` by choosing the lowest class within epsilon of the candidate maximum; epsilon
`2^-15` already changes an additional pixel, proving that an arbitrary wide tie band is unsafe.

The production probe therefore preregisters `0, 2^-24..2^-10`, selects the minimum
calibration-exact epsilon using pairs 0..119 only, and validates heldout pairs 120..599 without
reselection. It is pair-atomic/resumable, refuses an incomplete or already-exact W27..W31
predecessor, binds transitive corrected-QDQ custody, and has NumPy plus MLX decision heads. The
custom-Metal host includes the selected MLX head inside synchronized latency and refuses any tie
receipt lacking calibration, heldout, and full exactness.

**Verdict scope:** pair-11 behavior is `INSTANCE` diagnostic until the full split-honest receipt
closes. No tie-snap feasibility claim is made from this one pair.

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

### F9 — strict interval certification is reported separately from empirical exact argmax — FIXED POLICY

The host harness records both full-corpus exact argmax and the conservative rule
`reference top1-top2 margin > 2*max-class absolute logit error`. The latter is sufficient but stronger
than direct exhaustive n600 equality and can mark exact or tiny ties uncertified. Because the source
video is fixed and every real n600 output pixel is measured, local-candidate admission now requires
exhaustive exact argmax, one digest across 10 processes, and positive speed; the interval fraction is
reported independently and cannot overrule direct equality. Terminal score authority remains exact
contest CPU/CUDA on archive bytes.

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

### F12 — OSS offers mechanisms, not evaluator-equivalent authority — CONFIRMED

Primary documentation confirms that PyTorch does not promise CPU/GPU-identical results, MLX exposes
the required custom-Metal construction surface, Core ML public activation quantization is W8A8, and
TensorRT's explicit low-precision schemes do not promise CPU-identical scorer argmax. The Task #494
custom path is not duplication of an off-the-shelf authority backend. External results remain design
evidence only; the local exact n600/cross-process/latency receipts stay load-bearing.

## Review disposition

- Fixed-scale numerical receipt: **accepted as MEASURED FORMULATION negative** after independent
  summary re-derivation and custody repair.
- Dynamic-scale QDQ receipts: **accepted as MEASURED through the W26 single-int64 ceiling**; no exact
  QDQ/fp32 arm, with the negative scoped to that formulation.
- Uniform exact-int64 W26 CPU twin: **accepted as MEASURED INSTANCE negative** with 4 single-pixel
  flips; training tolerance passes, but authority does not.
- Geometry-safe mixed W26..W30 CPU twin: **accepted as MEASURED INSTANCE negative** with one
  exact-zero-margin flip at pair 11; training tolerance passes.
- Frozen-weight-L1-safe W27..W31 CPU twin: **full 0..599 run in progress**; no partial result
  promoted.
- Full-R, integer-R, custom-Metal, Pose dry-start: **built but OWED MAIN host measurements**.
- CoreML/ANE W8A8: **settled FORMULATION refusal**; no duplicate run.
- Contest CPU/CUDA authority: **unchanged and not inferred** from local substrate results.

## STORES CONSULTED

Full `CLAUDE.md`; full `AGENTS.md`; canonical frontier, lane, task, subagent, and equation stores;
the v7.5/v8 operating specs; relevant #348/#449/#456/#477/#478/#482/#490 research artifacts;
the exact Task #494 source files and receipts; both live inboxes through the timestamps recorded in
the session checkpoint. No paid provider, protected live run, run stop, or contest evaluator was
actuated.
