# Stacking architecture: cross-lane composition + artifact reuse

**Status**: design contract, binding for all Cycle-1+ lanes
**Date**: 2026-04-28
**Owner**: skunkworks council (Yousfi + Fridrich + Hotz + Quantizr + Contrarian)

This document codifies how lanes compose at the artifact level, so that
deadline-pressure work can be stacked WITHOUT a refactor of every lane.
Every new lane MUST conform to the conventions in this doc on creation.

---

## 1. Anchor reuse pattern

The canonical anchor for the May 3 deadline is **Lane G v3**, located at:

```
experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip
```

Reproducible-from-saved-artifacts contest-CUDA score **1.05** (verified
2026-04-28). All Cycle-1+ lanes warm-start from this anchor unless they
explicitly replace the renderer (see section 3).

**Rule**: any lane that needs a "starting point" MUST consume Lane G v3
artifacts directly (renderer.bin / masks.mkv / poses.pt / metadata.json
extracted from `archive_lane_g_v3.zip`). Do NOT silently retrain a
near-equivalent baseline; that wastes GPU and forks the anchor lineage.

---

## 2. Artifact reuse table

For each lane, what artifacts it CONSUMES from prior lanes vs PRODUCES
for downstream stacking. Sidecars are byte-additive to the archive.

| Lane | Consumes from anchor | Produces (for downstream) | Slot |
|------|----------------------|--------------------------|------|
| Lane G v3 | (origin)              | renderer.bin + masks.mkv + poses.pt | renderer-base |
| Lane J-JBL | renderer + masks + poses | retrained renderer (Jaccard-Boundary loss) | renderer-replacement |
| Lane J-NWC | renderer | NWC-encoded renderer.bin (smaller) | renderer-encoder |
| Lane J-IMP | renderer | 89%-sparse renderer.bin | renderer-replacement |
| Lane J-NWCS (NEW) | renderer + Lane W hard-pair signal | sensitivity-aware NWC-encoded renderer.bin | renderer-encoder |
| Lane EC | any renderer + GT pairs | gradient_corrections.bin sidecar | sidecar-additive |
| Lane MOS | masks corpus | prior.npz sidecar | sidecar-additive |
| Lane HF | renderer + scorer | foveation_params.bin sidecar | sidecar-additive |
| Lane GP | poses.pt | pose_gp.bin (replaces poses.pt) | pose-replacer |
| Lane FL | RAFT flows | poses_raft.pt (replaces poses.pt) | pose-replacer |
| Lane Ω-V2 | renderer + Hessian | per-weight bit allocation table | renderer-encoder |
| Lane W | renderer + scorer | hard_pair_weights.npy (signal-only) | signal-sidecar |
| Lane F-V5 | renderer | FP8 renderer.bin (hardware-disclosed) | renderer-encoder |
| Lane J-NWCS-EC (NEW) | renderer + Lane W signal + GT pairs | composed archive (NWCS renderer + EC sidecar) | renderer-encoder + sidecar-additive |

---

## 3. Composition rules: stack vs exclusive

### 3a. Renderer-replacement (mutually exclusive — pick ONE)

These lanes each produce a complete `renderer.bin`. Stacking two of these
in the same archive is undefined behavior; the archive can only hold one
renderer.bin.

```
{Lane G v3, Lane J-JBL output, Lane J-IMP-90%, Lane I Cool-Chic, Lane V Quantizr, Lane F-V2}
```

### 3b. Renderer-encoder (apply on TOP of any renderer-replacement output)

These lanes RE-ENCODE an existing renderer.bin into a smaller form. They
compose with any renderer-replacement output.

```
{Lane J-NWC, Lane J-NWCS, Lane F-V5 FP8, Lane Ω-V2 per-weight bits}
```

**Compatibility within renderer-encoder**: J-NWC and J-NWCS are mutually
exclusive (J-NWCS supersedes J-NWC). J-NWCS and Ω-V2 can stack (Ω-V2
runs first to set per-weight bit budgets, J-NWCS quantizes within
that budget). F-V5 FP8 is exclusive with both NWC variants (different
weight representation).

### 3c. Sidecars (composable additively)

Each occupies a distinct archive slot and contributes byte-additively
to the rate term. They are independent.

```
{Lane EC corrections, Lane MOS prior, Lane HF foveation}
```

All three can be stacked in a single archive. Each carries its own
inflate-time decoder hook in `inflate.sh`.

**Renderer-encoder × sidecar-additive composition rule (explicit
example)**: a renderer-encoder lane (Lane J-NWC, J-NWCS, Ω-V2, F-V5)
produces a `renderer.bin` whose magic bytes are in the strict-scorer-rule
allowlist; a sidecar-additive lane (EC, MOS, HF) produces a single
load-bearing artifact discovered by name. These two slots NEVER
conflict at the inflate dispatcher (the renderer arm reads
`renderer.bin`, the sidecar arm reads a different file). Therefore any
renderer-encoder × any sidecar-additive composition is structurally
clean and can be assembled by `tac.stack_compositions`. The first
concrete instance is `compose_jnwcs_with_ec` (J-NWCS renderer-encoder ×
Lane EC sidecar) — see section 4 "J-NWCS + EC" stack candidate.

### 3d. Pose replacers (mutually exclusive — pick ONE)

```
{Lane GP poly-fit, Lane FL RAFT-derived, original poses.pt from anchor}
```

### 3e. Signal-only (no archive bytes; pure training-time signal)

```
{Lane W hard-pair weights}
```

These lanes do NOT add to the archive but supply a training signal
consumed by other lanes. Lane W's hard-pair weights are required input
for Lane J-NWCS.

---

## 4. Stack candidates (per Jack-from-skunkworks Cycle-1 plan)

### Conservative stack — predicted [0.85, 0.95]

```
Lane G v3 renderer
  → replaced with Lane J-JBL output (renderer-replacement)
  → re-encoded with Lane J-NWC (renderer-encoder)
  → + Lane MOS prior (sidecar-additive)
```

Estimated archive: ~520 KB (vs Lane G v3 678 KB). Risk: low — every
component has been validated standalone.

### Aggressive stack — predicted [0.65, 0.85]

```
Conservative stack
  + Lane EC gradient corrections (sidecar-additive)
  + Lane HF foveation (sidecar-additive)
```

Estimated archive: ~580 KB. Risk: medium — Lane EC and Lane HF interact
with mask quality; needs an integration auth-eval.

### J-NWCS + EC + Lane G v3 anchor — predicted [0.78, 0.92]

```
Lane G v3 renderer (renderer-base, anchor)
  → re-encoded with Lane J-NWCS (renderer-encoder)
  + Lane EC gradient corrections (sidecar-additive)
```

Estimated archive: ~480 KB at default split (4 bits/weight × 30KB EC).
Cost ~$9 / 14h on RTX 4090. Cycle position: #3 (after the conservative
and aggressive stacks have measured their integration baselines).
Composed via `tac.stack_compositions.compose_jnwcs_with_ec` and deployed
via `scripts/remote_lane_j_nwcs_ec_stack.sh`. The two artifacts attack
the rate wedge from STRUCTURALLY ORTHOGONAL layers — J-NWCS at the
weight-bit allocation layer, EC at the inflate-time pixel residual
layer — so the composition is COMPLEMENTARY, not redundant. The user's
"perhaps lane j-nwcs might be supplemented by engineered corrections /
or maybe that's accomplished by the hard-pair-aware codec retraining"
question resolves to: J-NWCS handles weight-bit allocation
(hard-pair-aware retraining of the WEIGHT codec); EC handles per-pixel
inflate-time residuals (a SCORER-derived signal that bypasses the
weight codec entirely). Both belong in the stack.

### Moonshot stack — predicted [0.40, 0.65]

```
Lane J-IMP-90% renderer (renderer-replacement, 89% sparse)
  → re-encoded with Lane J-NWCS (renderer-encoder, sensitivity-aware)
  → re-encoded with Lane Ω-V2 (per-weight bits)
  + Lane J-DCAE mask codec
  + Lane J-EFD distill applied at compress time
```

Estimated archive: ~280 KB (sub-Quantizr). Risk: high — three-encoder
stack on a sparse renderer is novel; auth-eval gap unknown.

---

## 5. Reuse conventions: lanes-in-development checklist

Every new lane MUST satisfy ALL of the following on creation (enforce
at PR-review time; preflight check in flight as `check_lane_documents_anchor_and_slot`):

1. **Document its anchor**: typically `experiments/results/lane_g_v3_landed/`.
   The lane script's header comment must name the anchor zip path.

2. **Specify its slot**: declare in the script header which slot it
   occupies, using one of: `renderer-base`, `renderer-replacement`,
   `renderer-encoder`, `sidecar-additive`, `pose-replacer`,
   `signal-sidecar`. This determines which other lanes it can stack with.

3. **Provide a `stack-with` matrix**: in the script header or in a
   companion `lane_<name>_stacking.md`, list the lane IDs this lane is
   compatible with vs exclusive against. The composition rules in
   section 3 govern; lane authors flag deviations explicitly.

4. **Do NOT silently re-train artifacts that already exist in the anchor**.
   If a lane needs `poses.pt`, it consumes the anchor's `poses.pt` (or
   the output of an upstream pose-replacer). It does NOT re-run pose
   TTO from scratch unless that is the lane's explicit purpose.

5. **Sidecar lanes MUST emit a single, clearly-named artifact** that
   `inflate.sh` can decode independently. No cross-sidecar coupling.

6. **Renderer-encoder lanes MUST be a pure function of an input
   renderer.bin** (plus optional signal sidecars). They MUST NOT depend
   on the source renderer's training distribution beyond what is
   captured in the renderer.bin itself.

7. **Every lane's auth-eval script reports** the slot it occupies and
   the anchor it consumed, so cross-lane stacking experiments can
   reconstruct provenance.

---

## 6. Cross-lane integration cycles

Per CLAUDE.md "no premature kills" rule: lane composition must be
empirically validated, not predicted. The integration cycle is:

1. Land each component lane standalone (auth-eval against Lane G v3
   1.05 baseline).
2. For each candidate stack, run a single integration auth-eval on
   the assembled archive.
3. Promote the stack only if its auth-eval beats Lane G v3 1.05 by a
   margin larger than the proxy-auth gap floor (~0.05).

Stacking a lane that regresses standalone may still help in a stack
(see `feedback_dont_abandon_high_score_lanes_for_stacking_20260428`).
Do not predict stack value from standalone scores.

---

## 7. Open questions

- Whether Lane Ω-V2 (per-weight bits) and J-NWCS (per-block codebook)
  can co-encode without bit-overlap; needs a joint dry-run.
- Whether Lane HF foveation interacts destructively with Lane J-IMP
  sparsity (foveation re-introduces dense regions that sparsity
  removed).
- Whether pose-replacers (GP, FL) preserve enough fidelity for the
  PoseNet scorer at 90%-sparse renderers.

These questions are deferred to integration cycles; design-doc revs
will follow each integration result.
