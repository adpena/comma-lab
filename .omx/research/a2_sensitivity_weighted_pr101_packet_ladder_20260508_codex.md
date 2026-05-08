# A2 Sensitivity-Weighted PR101 Packet Ladder

Date: 2026-05-08
Owner: Worker C / Codex
Scope: Track 1 Phase A2 sensitivity-weighted lossy coarsening, PR101 substrate.

## Change

Added a packet-ladder builder for A2 selected-K schedules:

- `tools/build_a2_sensitivity_weighted_pr101_packet.py`
- consumes `tools/sensitivity_weighted_lossy_coarsening.py` manifests;
- materializes each `weighted_k_allocations[].selected_Ks` row into a local
  runtime packet with `archive.zip`, archive SHA-256, member SHA-256, and
  runtime-tree custody;
- patches only the generated packet-local `src/codec.py` to support an A2
  length-prefixed decoder layout;
- leaves the source PR101 runtime untouched.

The A2 K-selection manifest now points at this builder through
`packet_ladder_builder` and
`implementation.byte_closed_packet_ladder_builder`.

## Wire Contract

The packet-local archive member layout is:

```text
A2K1 || decoder_len:u32le || PR101 decoder_blob || PR101 latent_blob || sidecar
```

The generated runtime keeps PR101's stock fixed-offset parser as a fallback and
adds an `A2K1` branch that consumes `decoder_len` before slicing the unchanged
latent and sidecar sections. The source runtime is not modified.

## Custody / Authority

Builder outputs are byte-closed local custody artifacts, not score artifacts.
Every top-level and per-variant manifest keeps:

- `score_claim: false`
- `promotion_eligible: false`
- `rank_or_kill_eligible: false`
- `ready_for_exact_eval_dispatch: false`
- `dispatch_attempted: false`
- `remote_gpu_run: false`

No exact CUDA, contest-CPU, lane claim, or score claim is produced here.

## No-Op Guard

Per-variant manifests include `noop_detection`. All-ones schedules,
source-matching decoder blobs, source-matching members, or source-matching
archives add blockers and keep the variant non-promotable. This prevents a
layout-only packet rewrite from being promoted as a semantic coarsening result.

## Adversarial Hardening

Codex review added three fail-closed guards before promotion:

- output directories must not overlap the source runtime tree or contain input
  files, so `--force` cannot delete custody inputs;
- source ZIP member names must be safe, non-hidden, relative names before the
  member name is reused in candidate packets;
- the supplied state dict must reproduce the source PR101 decoder blob with an
  all-ones K schedule before any lossy schedule is materialized.

These guards keep A2 from silently becoming a source/runtime mutation, unsafe
ZIP rewrite, or state/source mismatch experiment.

## Real Contract Smoke

One real PR101 A2 schedule was materialized locally with no dispatch and no
score claim:

- command: `tools/build_a2_sensitivity_weighted_pr101_packet.py --a2-manifest experiments/results/track1_phase_a2_sensitivity_quant_20260508T154125Z/A2_result.json --variant-limit 1`
- output manifest:
  `experiments/results/track1_phase_a2_packet_ladder_codex_20260508T160150Z/a2_packet_ladder_manifest.json`
- variant: `weighted_k_00_rms_0p0386`
- candidate archive bytes: `159491`
- candidate archive SHA-256:
  `bfb912ff7dbbd843b3bf6e5d12ff876eeab359e38113204d0ccae4277fd35d27`
- candidate member bytes: `159391`
- candidate member SHA-256:
  `3906da037c6e6604669a863cacd6be88efb3453ea1acbf1b885bf22bb5771a78`
- decoder bytes: `143389`
- decoder SHA-256:
  `ec94df1c1d9e3233ffa209414f471418630ab1e7ca185ee8df0a6212f7dff056`
- state/source all-ones closure: passed
- no-op detection: false
- remaining blockers: no Level-2 lane claim, no exact CUDA auth eval, no
  contest-CPU auth eval, no operator score-claim review, no packet-local
  inflate parity.

## Remaining Blockers

Before any A2 packet can be used as score evidence:

1. run local source-vs-candidate inflate parity on the selected packet;
2. run pre-submission compliance on the reviewed packet surface;
3. claim the Level-2 lane before any exact eval dispatch;
4. run exact CUDA and contest-CPU auth eval;
5. perform operator score-claim review.

Until those are complete, this lane remains an empirical byte-closure actuator
only.
