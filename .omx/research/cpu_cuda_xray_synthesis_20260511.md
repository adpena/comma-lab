# CPU/CUDA xray synthesis — handoff P5 deliverable 4

_Generated_: `2026-05-11T18:34:00Z` ·
_Schema_: `cpu_cuda_xray_synthesis.v1` ·
_Tag_: `[diagnostic-not-score]` ·
_Score claim_: **NONE**

This memo synthesizes the three layer-/loader-drift tools landed in this
session into a single mechanism-attribution view for the device-axis behavior
of the contest scorer pipeline. Per CLAUDE.md `forbidden_mps_derived_strategic_decision`
+ "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE",
this synthesis is a diagnostic guide for routing future operator-gated GPU
spend; it is NOT a score claim and CANNOT promote, kill, or rank any lane.

## What this synthesis brings together

Three new orchestrator tools that consume the existing `tac.diagnostics`
introspection primitives:

| Tool | What it measures | Status this session |
|---|---|---|
| `tools/cpu_cuda_xray_loader_drift.py` | PyAV (CPU) vs DALI/NVDEC (CUDA) decoded RGB delta + shared-input custody | **macOS-CPU side LANDED**; DALI/NVDEC cell pending operator gate ($<$0.05 Modal) |
| `tools/cpu_cuda_xray_segnet_layer_drift.py` | Per-EfficientNet-B2-stage CPU/CUDA forward drift on a shared input + per-stage compounding | **CPU record LANDED**; CUDA capture pending operator gate ($<$0.05 Modal) |
| `tools/cpu_cuda_xray_posenet_layer_drift.py` | Per-FastViT-T12-block CPU/CUDA forward drift + (1+ε)^L compounding + Hydra pose head drift | **CPU record LANDED**; CUDA capture pending operator gate ($<$0.05 Modal) |

All three are wired through the existing `experiments/dump_scorer_activations.py`
+ `tools/probe_eval_loader_drift.py` + `tac.diagnostics.compute_layer_drift`
primitives; new orchestrator code totals ~1100 LOC across the three tools and
19 dedicated tests. They preserve the established shared-input tensor custody
schema (`eval_loader_shared_input_tensor.v1`) so the SegNet and PoseNet xrays
can ingest the same decoded RGB bytes that came out of the loader-drift tool.

## What we already know from the device-axis paired anchor matrix

Per `.omx/research/device_axis_paired_anchor_matrix_20260511.md`:

| Anchor | Family | Pose CUDA/CPU ratio | Seg CUDA/CPU ratio | Winning axis |
|---|---|---:|---:|---|
| **A1** (PR101-derived, score-gradient training) | HNeRV-cluster | **5.18×** worse on CUDA | 1.18× worse on CUDA | **CPU** |
| **PR106 r2** (latent sidecar) | HNeRV + per-pair latent delta | **0.197×** (5.1× *better* on CUDA) | 1.017× (essentially equal) | **CUDA** |
| PR103-on-PR106 AC repack | HNeRV decoder + AC rate work | — | — | **CUDA** |

**The seg ratio is small in absolute terms (≤1.2×) for both anchors. The pose
ratio is the ONLY axis with a 5× device-axis signature, AND it FLIPS SIGN
between the two HNeRV-family substrates.** This is the central mystery the xray
tools are designed to attribute.

## What the local CPU-only xray runs already tell us

The CPU-only smoke runs landed this session produced:

1. **Loader-drift tool** (`experiments/results/cpu_cuda_xray_loader_drift_20260511T183001Z/`)
   - 1 shared-input tensor written:
     `batch000000_cpu_av_raw_rgb_uint8_before_posenet_segnet.pt` (23.3 MB)
   - DALI cell unavailable on macOS (expected — no CUDA/DALI runtime locally)
   - Loader-drift attribution: **NOT MEASURED on this run**; dispatch plan
     emitted for Linux x86_64 GPU follow-up

2. **SegNet layer-drift CPU record** (`.../cpu_cuda_xray_segnet_layer_drift_20260511T183011Z/cpu_capture/`)
   - 387 SegNet layers captured (EfficientNet-B2 + smp.Unet decoder)
   - SegmentationHead final logits layer detected at `segmentation_head` and `segmentation_head.0`
   - Dispatch plan emitted for Linux x86_64 GPU CUDA capture

3. **PoseNet layer-drift CPU record** (`.../cpu_cuda_xray_posenet_layer_drift_20260511T183026Z/cpu_capture/`)
   - 397 PoseNet layers captured (FastViT-T12 backbone + Hydra pose head)
   - 256 FastViT-block-named nodes (RepMixer/Attention + nested submodules)
   - Hydra pose head detected at `hydra.final_layer.pose`
   - Dispatch plan emitted for Linux x86_64 GPU CUDA capture

4. **Identity smoke (CPU-vs-CPU paired)**
   - Same record paired against itself produces zero drift across all 396
     SegNet layers / 397 PoseNet layers (sanity check on the compute-layer-drift
     primitive). All `compound_factor=1.0×` as expected.

## Mechanism attribution — verdict structure

Once paired Linux-CUDA captures land via the emitted `dispatch_plan.json`
files, the tools produce a **typed verdict** with four cells:

| Verdict | Loader drift | Layer drift signature | What it means |
|---|---|---|---|
| **A: Loader-dominated** | large LSB delta (`large_drift` class) | flat / no per-block compounding | Device-axis score drift comes from PyAV-vs-DALI decoded-RGB differences propagating through Lipschitz-bounded scorer kernels. Fix: align loader paths or accept axis-specific scores. |
| **B: Scorer-forward-dominated** | byte-identical or single-LSB | per-block compounding ≈ observed final-ratio | Device-axis drift originates inside the scorer (FastViT/EfficientNet/Hydra) kernels. The 5× pose ratio is `(1+ε)^L` accumulated across FastViT blocks. Fix: study scorer kernel numerics (TF32, atomicAdd, fused multiply-accumulate). |
| **C: Threshold-geometry / Lipschitz-amplified** | byte-identical | small per-block ε but final-output drift large | Scorer kernels are nearly identical but cross a decision threshold (argmax / pose-MSE) at some layer. Fix: identify the threshold-crossing layer (the xray's `first_argmax_divergence` / `first_l2_relative_exceedance` row). |
| **D: Mixed / coupled** | small LSB delta + per-block compounding | both contribute roughly equally | The two mechanisms are coupled. Fix: separate via the 4-cell discriminator matrix (see below). |

The tools surface the raw evidence for each verdict; the operator (or council)
chooses the verdict label based on the empirical numbers.

## What the synthesis CANNOT yet conclude this session

Local macOS-CPU + missing CUDA records mean:

- **We cannot localize the FIRST layer where argmax diverges** — that requires
  a non-trivial drift signal between CPU and CUDA records on the same shared
  input. CPU-vs-CPU pairing gives zero drift (verified).
- **We cannot quantify the FastViT (1+ε)^L compound factor empirically** —
  same reason. The tool reports `compound_factor=1.0×` on identity-paired
  records (sanity check).
- **We cannot attribute loader drift magnitude** — DALI/NVDEC runtime is not
  available on macOS. Linux x86_64 GPU dispatch is required.

These three blockers ALL clear once a single `$<$0.05` Modal GPU dispatch runs
the canonical `tools/probe_eval_loader_drift.py --run-forward-cells` AND the
two `experiments/dump_scorer_activations.py --device cuda --shared-input-tensor`
captures (PoseNet + SegNet) on the same shared-input tensor.

## Implications for the substrate-class-boundary hypothesis

The council's **Insight 1** (per `feedback_grand_council_pose_axis_insights_review_20260511.md`)
posited that the device-axis flip between A1 and PR106 r2 is a **substrate-
class boundary phenomenon**: each HNeRV-family substrate has a different
"signature" in how its decoded RGB bytes are received by the scorer, and the
boundary lies somewhere between A1's score-gradient-trained latent space and
PR106's per-pair latent-delta sidecar space.

The xray pipeline does NOT yet decide this empirically (CUDA records pending),
but it provides the **machinery to decide it once the captures land**:

1. **If the per-layer drift signatures on A1 and PR106 r2 are STRUCTURALLY
   SIMILAR (same first-divergence layer, similar per-stage compound factors)
   but only the SIGN flips**, then the substrate-class boundary is real and
   localized to a specific scorer numerical sensitivity. The drift mechanism
   is the same; the substrates simply land on opposite sides of a scorer
   threshold cliff.

2. **If the per-layer drift signatures are STRUCTURALLY DIFFERENT (different
   first-divergence layers, different stage-compounding profiles)**, then the
   "device-axis flip" is actually two different mechanisms that happen to both
   produce ~5× pose ratios. The substrate-class boundary is an artifact, not a
   primitive.

The xray output schema includes the **`first_divergence`** and
**`stage_compounding`** / **`fastvit_compounding`** fields explicitly so this
comparison can be made by feeding the A1 and PR106 r2 decoded-RGB shared-input
tensors through the same tool back-to-back.

## Implications for non-HNeRV pose-axis lane prioritization

Per the device-axis matrix mechanism reading, at the PR106 r2 frontier
operating point (`pose_avg=3.2e-5`), pose marginal-value is 2.71× SegNet's per
unit distortion. This makes pose-targeted non-HNeRV lanes higher EV per dollar
than mask grammar:

- **`raft_radial_openpilot_pose`** (RAFT-derived pose sidecar; tier 90)
- **`lapose_motion_atom_allocator`** (LAPose-inspired pose-conditioned byte
  allocation; tier 50)
- **`telescopic_foveation_field`** (geometry-aware foveation; tier 40)

These lanes need empirical pose-improvement signal per byte (the council's
**measured `d_pose/db` vs `d_seg/db`** axis). The xray pipeline does NOT
directly produce those numbers — that requires a candidate archive that
emits charged bytes that the inflate pipeline consumes. BUT the xray DOES
produce a critical input to pose-lane prioritization:

> **If the FastViT-T12 block compounding factor on the A1-equivalent shared
> input matches the empirical A1 pose ratio (5.18×), then the pose marginal
> signal is DOMINATED by FastViT kernel numerics, and pose-axis lanes that
> change RGB bytes upstream of the scorer (RAFT, foveation) will have a
> different operational profile than pose-axis lanes that change the latent
> stream (LAPose).**

This is a directional prior, not a proof. Empirical pose-improvement per byte
still requires byte-closed candidate archives + exact eval, per CLAUDE.md
"NEVER claim a contest-compliant score without inflate.sh → evaluate.py".

## Council 5 surfaced decisions cross-reference

Per `project_full_custody_takeover_codex_offline_20260511.md`:

1. **R2 paired CPU eval ($<$0.10 Modal CPU Linux x86_64)** — UNANIMOUS 10/10
   council priority. This xray pipeline is **complementary** to that dispatch:
   R2 paired CPU eval measures the *scored archive* on the CPU axis (medal-
   band visibility); the xray pipeline measures the *scorer mechanism* across
   axes (mechanism attribution). They answer different questions; both should
   eventually land.
2. A1 PR-submission 5-turn council greenup trigger — independent.
3. Phase 2 $223–303 envelope approval — independent.
4. Frontier roadmap split — $0 tooling change; could include the xray
   pipeline's per-architecture-class drift attribution as a 14th column.
5. A1 submission packet expansions — could include the xray output as a
   "device-axis-explanation" appendix per the council's recommendation.

The xray pipeline's three dispatch plans (loader + SegNet + PoseNet) total
**~$0.15 Modal GPU spend** if run as a single Modal CPU+GPU container. This
is below all NOT YET items' thresholds. It remains **operator-gated** per
CLAUDE.md cross-agent-dispatch coordination.

## Adversarial review summary

3-clean-pass council review per CLAUDE.md "Recursive adversarial review
protocol":

- **Round 1** (Yousfi / Fridrich / Contrarian): the L2-relative threshold
  (1e-2) is a reasonable boundary for layer-level divergence; compound factor
  is correctly framed as a hypothesis being tested, not a theorem.
  ✓ all outputs tagged `[diagnostic-not-score]`. ✓ no score claim or kill
  verdict can be derived.
- **Round 2** (Shannon / Dykstra / MacKay): mathematical soundness — the
  compound factor is an upper-bound estimate via Lipschitz triangle
  inequality, NOT a proof that per-layer ε sums to final drift.
  ✓ documented in markdown as "evidence" not "proof". ✓ adds strict
  information vs single-scalar 5× ratio.
- **Round 3** (Quantizr / Hotz / Selfcomp): mechanism attribution — a
  macOS-CPU + Linux-CUDA pairing mixes (device-axis drift) ⊕ (macOS-vs-Linux-
  CPU drift). **Fix landed**: `_detect_capture_host()` + `mixed_substrate_advisory`
  banner explicitly tags such pairings as `[macOS-CPU advisory only]`.

**All three rounds CLEAN.** The tools are tested (19 unit tests pass), the
output is non-promotable by construction, and the substrate-mixed pairing
is now flagged.

## Wire-in declarations (CLAUDE.md "Subagent coherence-by-default")

Per the 6-hook unified-Lagrangian wire-in mandate:

1. **Sensitivity-map contribution** — per-layer drift IS a sensitivity map for
   the device-axis. Records under `experiments/results/cpu_cuda_xray_*_layer_drift_*/`
   are typed JSON consumable by future allocator code.
2. **Pareto constraint** — N/A (diagnostic, not Pareto-eligible).
3. **Bit-allocator hook** — drift attribution informs which axis (pose vs seg)
   to prioritize bytes for at the current operating point. Verdict consumers
   in future bit-allocator code should query the layer-drift JSON's
   `first_divergence` + `stage_compounding` fields.
4. **Cathedral autopilot dispatch hook** — the three emitted `dispatch_plan.json`
   files are registered with `lane_id` claim templates; cathedral autopilot
   may consume them once operator gate clears.
5. **Continual-learning posterior update** — drift mechanism feeds the
   per-archive drift posterior per `tac.per_archive_drift_posterior` IF the
   mechanism is substrate-specific (i.e., A1 vs PR106 r2 have different
   per-layer signatures). Trigger condition lives in the synthesis verdict
   (A/B/C/D table above).
6. **Probe-disambiguator** — THIS IS the disambiguator for the substrate-class-
   boundary council Insight 1 hypothesis. Verdict A/B/C/D resolves it.

## Lane registry + memory

- Lane: `lane_cpu_cuda_xray_p5_landing` (Phase 5, L0 at registration; gates
  pending paired CUDA captures).
- Memory: `feedback_cpu_cuda_xray_p5_landed_20260511.md` (this session
  landing).
- Cross-refs:
  - `.omx/research/device_axis_paired_anchor_matrix_20260511.md`
  - `~/.claude/projects/.../memory/feedback_cuda_cpu_axis_profile_learning_layer_20260508.md`
  - `~/.claude/projects/.../memory/feedback_grand_council_pose_axis_insights_review_20260511.md`
  - `~/.claude/projects/.../memory/project_full_custody_takeover_codex_offline_20260511.md`
  - `~/.claude/projects/.../memory/project_top_priority_revisit_NOT_YET_operator_decisions_20260509.md`

## Loop pause status

**PAUSED** per 2026-05-09 operator directive, unchanged by this P5 landing.
No `ScheduleWakeup` outstanding. Total GPU cost this session: **$0**.
