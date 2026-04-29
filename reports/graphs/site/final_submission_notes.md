# final submission notes

## submission posture (current — Era 2 / renderer)

- target submission: contest-CUDA Lane G v3 = `1.05` (or whatever Selfcomp-paradigm derivative lands [contest-CUDA] sub-1.05 before May 3)
- frame the historical Era 1 (`1.73` Track B post-filter) as the originating arc, not the submission candidate
- frame Track A as transparency-only, never as a submission candidate
- emphasize that every honest promotion is contest-CUDA verified on the EXACT submission archive bytes
- emphasize the engineering-rigor story (78 STRICT preflight checks, eval_roundtrip non-negotiable, MPS-vs-CUDA drift discovery) — that is differentiated and not competitively load-bearing

## leaderboard context (fetched 2026-04-29)

- Quantizr 0.33 (#1)
- Selfcomp 0.38 (#2)
- Mask2mask 0.60 (#3)
- our 1.05 would rank ~4th if shipped today; live work targets sub-0.30

## milestone sequence (Era 1 — codec + post-filter, historical)

- `4.06` - honest Track B baseline
- `3.25` - best x265 floor
- `2.20` - repaired AV1 path became competitive again
- `2.12` - colorspace hardening reduced evaluator mismatch
- `2.08` - encoder-side `sharpness=1` became the clean AV1 floor
- `2.05` - tiny learned int8 post-filter became the new honest floor
- `2.01` - saliency-weighted learned int8 post-filter
- `1.99` - long-500 QAT+EMA learned int8 post-filter
- `1.95` - long-500 QAT+EMA h32 learned int8 post-filter
- `1.92` - long1000 QAT+EMA h16 learned int8 post-filter
- `1.85` - long1000 QAT+EMA h32 learned int8 post-filter
- `1.84` - weighted ensemble of `1.85` and best Monte Carlo refinement
- `1.73` - long1000 QAT+EMA h64 learned int8 post-filter (Era 1 final floor)

## milestone sequence (Era 2 — neural renderer, current)

- `0.90` - first reproducible-from-saved-artifacts contest-CUDA baseline (dilated-h64 + CRF=50 + matched poses, 2026-04-25)
- `1.15` - Lane A pose TTO from baseline poses (2026-04-27)
- `1.05` - Lane G v3 KL distill weight=0.002 + pose TTO retry (2026-04-28, Modal repro 1.04 on 2026-04-29)

The 0.90 → 1.05 ordering looks inverted but reflects archive-content tradeoffs: Lane A and Lane G v3 carry an additional 401KB pose tensor that the 0.90 baseline did not. Within the 694KB archive size, Lane G v3 is the lowest contest-CUDA score we have measured.

## milestone sequence (Era 3 — Selfcomp paradigm, live)

No [contest-CUDA] landings yet. Eight Modal lanes in flight:
- MM (grayscale-LUT mask), SA (94K SegMap clone), SC++ (SA + KL distill T=2.0), SO (SC++ + Hessian block-FP)
- in parallel: q_faithful_v3, sz_phase2_v2, mae_v_v2, lane_w_v2

## concise submission framing

This submission is the result of a measured progression across two paradigms.

Phase 1 (Era 1) was a tiny CNN post-filter trained against frozen scorer gradients on top of an SVT-AV1 encoded archive. It established the lab's measurement discipline (smoke + scorer + review-gate), produced a `1.73` floor, and a publishable mathematical investigation showing why CNN strategies are mathematically necessary for the rank-1 PoseNet Jacobian.

Phase 2 (Era 2) abandoned the codec entirely. A small neural renderer (dilated-h64) is trained directly against the scorer gradients and bypasses pixel-level codec decisions. Pose TTO at compress time and KL distillation on the SegNet logits (T=2.0, weight=0.002) closed the proxy-auth gap and produced the current `1.05` [contest-CUDA] floor.

Phase 3 (Era 3 — live work) attacks the remaining gap to the leaderboard via the Selfcomp paradigm: grayscale-LUT mask encoding, single-mask + 6-DOF affine duality, analytical pose, block-FP weight self-compression, and a smaller 94K-param SegMap renderer. Multiple lanes are in flight on Modal.

## current non-promoted research follow-ons

- Lane M-V2 (radial-zoom rank-1 hypothesis): `1.84` [contest-CUDA], regression. Dead unless someone wants to revisit pose-pad asymmetry handling.
- Lane GP v3 (Gaussian-process pose fit): `89.67` [Modal-T4-CPU]. Runge phenomenon at degree-10 polynomial. Dead at this basis; could be revived with DCT or B-spline.
- Lane UNIWARD v8: `1.14`. Council 5/5 KILLED standalone. Encoder pipeline is no-op on the bitstream.
- The strongest Era 1 non-promoted family is still bounded Monte Carlo / layer-scale search; preserved for the writeup arc.
- The SegNet-native branch transferred honestly but weaker, at `1.90`.

## strategic-secrecy notes (non-negotiable per CLAUDE.md)

- Do NOT publicize the Cloudflare site URL until the human explicitly says it is time.
- Do NOT expose Lane W / Lane Ω / Lane DARTS-S internals on public surfaces.
- The Selfcomp-paradigm sequencing is internal-only until a [contest-CUDA] landing makes it part of the submitted archive.
