# Codex findings — per-layer/per-channel genuine adaptive width

**UTC:** 2026-07-14T12:16:00Z  
**Lane:** `margin_adaptive_perlayer_followon`  
**Authority:** `[research-only throughput MEANS; local saved-artifact verification]`

## Pointer status

**UNMOVED.** Submittable `[contest-CPU Linux x86_64]` pointer:
`0.19108282419209976`. The `0.1880443979880752` defensive bank is borrowed
and non-submission. No archive, contest replay, or score claim is produced.

## Re-derived findings

1. **MEASURED predecessor boundary:** cap28/cap30/cap31 have zero observed
   n600 flips but physically use int32 storage; cap16 uses int16 and has 5,446
   flips. The earlier global-cap experiment is an `INSTANCE` result and cannot
   establish physical adaptive-width acceleration.
2. **DERIVED no-fake accounting:** the current typed operand kernels sign-extend
   into the same int64 MAC. Thus conventional MAC/FLOP reduction is exactly
   `0.0`; only physical operand traffic and measured same-map native-storage
   latency may be credited to adaptive width.
3. **DERIVED decisive A/B:** `U32_M` vs `A32_M` controls logical-map structure;
   `A32_M` vs `AN_M` holds all logical arithmetic fixed and isolates native
   physical widths; `U32_C` is a numerical placement control. CPU-to-Metal
   alone is never a width result. If CPU is a reference implementation its
   placement row is `CPU_REFERENCE_IMPLEMENTATION_BOUND`.
4. **DERIVED SE formulation:** 23 SE blocks / 46 SE conv paths are a tiny
   `0.01325748%` of Conv MACs. A MEASURED real-pair shape trace plus explicit
   operation convention yields `0.15992356%` for SE conv + GAP + gate and
   leaves `99.84007644%` in the bulk treatment. Full-fp SE is an efficient
   per-layer/channel control, not a solution to spatial global dependency.
5. **DERIVED certificate law:** the fixed-n600 primary certificate is exhaustive
   tie-aware winner identity plus ten-process digest. Interval and affine
   bounds are distinct, stronger robustness summaries; unsound endpoint
   linearization is not affine arithmetic.
6. **MEASURED custody trap:** cached `lstars` differs from live NumPy-fp32
   authority by one pixel at design pair 195. The executor therefore
   re-derives all 600 reference decisions, matches predecessor hashes, and
   stores only a verified compact correction manifest for resume.
7. **BUILT executor:** the MAIN path now owns real seed execution, nonmonotonic
   per-layer/per-channel search, frozen validation, same-map forced-int32/native
   timing, ten fresh-process digests, SSD checkpoint waterfall, and certified
   success cleanup. No Metal row is claimed locally.
8. **REVIEW-SEALED controls:** `U32_C` rounds into int64 and clamps to the exact
   integer qmax before int32 storage, including the fp32-inexact 27/28-bit
   limits. The typed DSL and plan parser expose only the built `full-fp` SE
   treatment; the unimplemented `adaptive-int` choice is rejected rather than
   silently ignored. Grouped-convolution high-bit parity is regression-tested.

## Ranked formulations

| Rank | Formulation | n600 argmax certificate | Genuine width / traffic | Latency authority |
|---:|---|---|---|---|
| 1 | Per-channel within layer, native buffers | Built, `OWED_MAIN` execution | Finest built map; exact bytes `OWED_MAIN_SEARCH` | `A32_M/AN_M`, `OWED_MAIN` |
| 2 | Per-layer native buffers | Built, `OWED_MAIN` execution | Coarser built map; exact bytes `OWED_MAIN_SEARCH` | `A32_M/AN_M`, `OWED_MAIN` |
| 3 | Full-fp SE plus adaptive-int bulk | Built structural control; map certificate `OWED_MAIN` | `99.84007644%` bulk remains eligible under the explicit operation convention | Same four-arm timing, `OWED_MAIN` |
| 4 | Sound affine tightening | Primitive built; `UNANCHORED_SCREEN_ONLY` until nonlinear remainder custody exists | Potentially admits lower widths; no credited cut | No timing admission |

The regenerated plan receipt is byte-identical at its experiment and research
mirrors, SHA-256
`f89f8fb0a5cf9924a6850f33da7cdaf9bf9886de8dbb31fa7eb6d41d37afa4e0`.

## Admission and remaining owed evidence

The selected map is admissible as a local native-width timing candidate only
after: frozen-design search on `0..263`; untouched validation `264..599`;
all-pixel exhaustive identity; deterministic exceptional witness; ten-process
digest; physical buffer proof; and positive measured `A32_M -> AN_M` timing.
Any absent row is `OWED_MAIN`, not inferred from bytes or logical bit counts.

`BEST_FOUND_CERTIFIED` is the only honest search label for the finite
best-improvement/per-channel-prefix procedure. Width feasibility is explicitly
nonmonotone: cap28 passes, cap29 fails, cap30 passes, so binary search and
monotone pruning are prohibited.

## Verdict scope and reformulation queue

There is no negative family verdict. A failed map or timing is scoped to that
finite searched `INSTANCE` / `FORMULATION`, then queues: richer per-channel
grouping/exchange; sound affine propagation through nonlinearities and SE;
exact global-summary closure for spatial allocation; and a packed native
narrow-MAC kernel. This incorporates the operator's 11:49 optimal-form rule
and the 11:57 correction: public/monolithic int16 is TESTED-BACKEND absence,
not a mixed-width family `NO_GO`.

## Stores consulted

Frozen build spec; predecessor exact-int implementation and n600 receipt;
canonical lane/subagent state; CLAUDE.md/AGENTS.md; latest intake findings; and
inboxes through `2026-07-14T13:10:48Z` (per-arm empty). No Metal, paid,
provider, evaluator, or live-run action occurred.

## Later binding correction — v9 integration is still owed

The fleet directive at `2026-07-14T13:22:43Z` makes v9·CGauge live wiring a
promotion prerequisite. These additive files are therefore
`BUILT_AND_REVIEWED_BUT_NOT_V9_LIVE_WIRED`, not a landed live-vehicle lever.
The current typed policy compiles the external measurement probe, but it is not
yet a canonical v9 `Lever` with a custodied `LawRef`, proven-base/trainer
consumer, and launch receipt bijection.

The explicitly assigned `provenance_canonicalize_fix_all_fakes` owner must wire
that edge; this lane will not edit its witness-autoconfig/config/bijection hot
surfaces. The correct consumer is the asynchronous/local SegNet verdict-forward
path, not the training-loss backward path, because this exact-integer kernel is
inference-only. Admission must consume
`margin_adaptive_perlayer_metal_n600.v1` only after full-n600 identity, ten
fresh processes, and positive same-map `A32_M/AN_M` timing. Until the owner
returns a compiled argv/consumer receipt, status is
`V9_INTEGRATION_BLOCKED_OWNER`; official evaluator score authority is unchanged.

## Serializer handoff — sandbox blocked

After three clean review passes, the canonical serializer was invoked with an
explicit 18-file allowlist, per-file expected SHA-256, `base=new`, and triality
`dag,dsl,equations`. Its first attempt correctly waited for an old worker
checkpoint; after that worker wrote a fresh complete/no-edits checkpoint, the
normal retry reached `git add` and failed rc128 because this sandbox cannot
create Git temporary/index database files (`Operation not permitted`). There
is no commit SHA and no override was used.

Privileged MAIN should serialize exactly the five lane artifacts, six
implementation/probe/host files, six focused tests, and the DSL/equation pair
enumerated by the final lane checkpoint and operator handoff. Do not absorb
the many unrelated sibling files in the shared dirty tree. Exact final hashes
are supplied in the parent handoff after this note's bytes are sealed.
