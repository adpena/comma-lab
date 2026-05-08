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

## Runtime Closure Probe

Added `tools/probe_a2_packet_runtime_closure.py` as a scorer-free closure
probe for A2 packets. It imports only the packet-local runtime, parses the
stock source archive and A2 candidate archive through that runtime, and checks
strict model state loading. It does not render raw frames, load PoseNet/SegNet,
dispatch remote work, or claim a score.

Real probe on the same `weighted_k_00_rms_0p0386` packet:

- output:
  `experiments/results/track1_phase_a2_packet_ladder_codex_20260508T160150Z/variants/weighted_k_00_rms_0p0386/a2_runtime_closure_probe.json`
- status: `runtime_closure_verified_no_score`
- source parser path: stock fallback, 28 decoder tensors, latents `[600, 28]`
- candidate parser path: `A2K1` length prefix, 28 decoder tensors, latents
  `[600, 28]`
- score-affecting payload changed: true
- manifest SHA-256 excluding self:
  `4d5376006cf473816976488e9ff5c845a1686bc228f8e3e5adcc316431067185`
- cleared blocker: `packet_local_inflate_parity_not_run`
- remaining blockers: no Level-2 lane claim, no exact CUDA auth eval, no
  contest-CPU auth eval, no operator score-claim review.

## Remaining Blockers

Before any A2 packet can be used as score evidence:

1. run pre-submission compliance on the reviewed packet surface;
2. claim the Level-2 lane before any exact eval dispatch;
3. run exact CUDA and contest-CPU auth eval;
4. perform operator score-claim review.

Until those are complete, this lane remains an empirical byte-closure actuator
only.

## Adversarial Review Closure

Follow-up review found five packet-ladder risks and the builder was hardened:

- upstream A2 authority is now validated (`schema`, source tool, false
  authority fields, and `dispatch_attempted=false`);
- upstream diagnostic/stub blockers are preserved into packet manifests, while
  blockers actually cleared by the packet build are listed under
  `packet_closure.cleared_blockers`;
- packet-local parser smoke now runs during build and controls
  `runtime_consumes_changed_archive_bytes`;
- `proxy_vs_materialized` reconciles analytical selector bytes against actual
  archive/member/decoder bytes, with `candidate_archive.bytes` as the
  authoritative byte field;
- ZIP custody now rejects unsafe names, backslashes, control characters,
  drive-letter names, and local/central header name mismatch, and records
  method/flag/header metadata.

Hardened real one-variant build:

- output manifest:
  `experiments/results/track1_phase_a2_packet_ladder_codex_hardened_20260508T161558Z/a2_packet_ladder_manifest.json`
- candidate archive bytes: `159491`
- candidate archive SHA-256:
  `bfb912ff7dbbd843b3bf6e5d12ff876eeab359e38113204d0ccae4277fd35d27`
- packet-local parser smoke: passed, `A2K1`, 28 decoder tensors, latents
  `[600, 28]`
- selector reported bytes: `159544`
- materialized authoritative bytes: `159491`
- proxy-vs-materialized delta: `+53` reported bytes over actual archive bytes
- cleared blockers: `no_byte_closed_runtime_packet_built`,
  `packet_local_inflate_parity_not_run`
- remaining blockers: `cpu_local_allocator_proxy_only`,
  `diagnostic_or_stub_sensitivity_map_not_score_authority`, `is_stub=true`,
  `tag contains 'stub'`, `score_sensitivity_artifact_must_be_certified_before_promotion`,
  no Level-2 lane claim, no exact CUDA auth eval, no contest-CPU auth eval,
  no operator score-claim review.

Non-final pre-submission compliance on the hardened packet passed all 21/21
checks:

- compliance JSON:
  `experiments/results/track1_phase_a2_packet_ladder_codex_hardened_20260508T161558Z/variants/weighted_k_00_rms_0p0386/pre_submission_compliance.nonfinal.json`
- standalone runtime closure probe:
  `experiments/results/track1_phase_a2_packet_ladder_codex_hardened_20260508T161558Z/variants/weighted_k_00_rms_0p0386/a2_runtime_closure_probe.json`
