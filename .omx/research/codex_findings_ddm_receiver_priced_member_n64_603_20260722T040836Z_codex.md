---
title: Codex adversarial round-1 review of DDM receiver-priced member solve v2
date_utc: 2026-07-22T04:08:36Z
task: 603
lane_id: lane_ddm_mdl_member_solve_v2_priced_603_20260722
review_round: 1
review_disposition: ACCEPT_SCOPED_FORMULATION_WALL_WITH_CACHE_CROSSCHECK_CAVEAT
research_only: true
score_claim: false
main_landing_review_required: true
---

# Adversarial disposition

The measured negative is valid only for the current fixed-record Task #603 chart grammar at n64.
It does not close variable-length chart syntax, entropy-coded semantic payloads, curvelet/shearlet
residual carriers, pre-uint8 member optimization, n600, or the wider DDM/member family.

# Falsification attempts

1. **Was this Task #602's reference-to-source identity failure in new clothes? — FALSIFIED.** The
   selected output is a standalone six-member archive. All members, including Pose6, are parsed and
   consumed; source raw is not consulted during decode. Archive SHA is
   `f3f98457ff8495dfefbfad2fb04549c8936eea15a1087d12c852144b5be5ae35`.
2. **Were the proposals no-ops, making the zero-byte delta vacuous? — FALSIFIED.** Low/mid/high
   proposals changed 24,531 / 24,159 / 23,955 semantic scalars and produced three distinct proposal
   archive hashes. Each decoded successfully. Their final byte lengths nevertheless remained exactly
   274,664 because every residual record and ZIP member has fixed extent.
3. **Was diagnostic zlib or an estimated rate smuggled back into selection? — FALSIFIED.** Every
   proposal was encoded twice by `compile_chart_archive`; `len(final ZIP bytes)` was read directly.
   No zlib value, source-array byte count, or projected n600 estimate entered admission.
4. **Did tolerance happen after selection? — FALSIFIED.** Each rung bound allowed per-stratum escape
   inside the selector. Exact byte break-even rejected all proposals before distortion spend, and the
   selected artifact was independently remeasured at each rung by the established batch16 oracle.
5. **Did aggregate membership hide class collapse? — CONFIRMED CAVEAT.** Membership 0.493605613708
   is almost entirely Undrivable. Road, Lane, MyCar, and Movable are zero; boundary membership is
   0.118697769367. The aggregate is not an efficacy claim.
6. **Does the gt raster crosscheck establish exact target identity? — NO.** Same-batch target versus
   cached `gt_n600.lstars` agreement is 0.999873638153. The batch16 target-versus-described numerator
   remains internally paired and deterministic, but the cache mismatch must remain visible.
7. **Could Fisher/margin ranking have found a better proposal? — NOT REACHED.** The operator's
   reverse-waterfill directive stops at exact rate break-even. With `delta_bytes=0` for every proposed
   stratum, spending any membership or Pose debt is inadmissible; a Fisher-ranked successor only
   becomes meaningful after a receiver-proven variable-length syntax creates a positive byte saving.
8. **Did the run establish contest score or promotion evidence? — FALSIFIED.** The axis is local
   macOS-CPU frozen-SegNet advisory. No PoseNet score, contest evaluator, contest CPU/CUDA replay, or
   candidate archive exists.

# Round-1 required successor

Replace fixed residual record bodies with a deterministic variable-length semantic code whose exact
decoder, unique-home ledger, sampled no-op proof, and resume identity are proven before optimization.
Then rerun the same five-rung selector at n64 and n600, retain the pre-uint8 state, and spend bytes by
Fisher/margin EV only after a strict positive byte saving exists.

The review consumed `docs/operating_manual_craft_handoff.md`; MAIN must re-review the formulation
boundary, cached-raster caveat, and exact archive bytes at merge time.

0.1910828242 [contest-CPU] — unchanged.

