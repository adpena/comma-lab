# #307 Contour-string flip coding — n600 MEASURED row (FEED-07b row 6)

**Status: COMPLETE (2026-07-07). VERDICT: NO-GO** vs the pre-registered `b < 0.65 B/flip` bar —
and the coder DID improve on the #280 floor (0.876 → **0.820 B/flip**), so the NO-GO now carries
the measured reason: the flip residual is **fragmented confetti, not codeable strings**.

Run: `tools/measure_contour_string_flip_coding.py` on the mod32cap live-run snapshot
(`snapshot_ema_BEST.npz`, ep425 best d_seg 0.0036364, sha256
`9f123bac950af1ec8eecc938c042f6d0e0662d0a427a16d5ba147e3f1fc62d93`; run
`levelset_n600_witness_mod32cap_20260706T115554Z`), **all 600 pairs** (the n600 evidence bar),
BOTH surfaces (witness-alone + analytic-lane-band composed), reusing the #280 leverd_stage0 data
path verbatim (#202 byte-close render authority: int8-dequant weights + self-orient fixed point +
contest R + frozen CPU-torch SegNet). Coder: connected-component chain-code DFS walk (absolute
8-dir symbols, digital-straightness contexts) + anchors/counts/classes streams over the in-tree
`RangeEncoder`; **every frame decode-verified bit-exact** before any byte was reported. Authority:
**[macOS-CPU advisory] NON-PROMOTABLE** — a coder-rate measurement, never a score. Pointer 0.19110
UNMOVED (MEANS). Artifacts: `experiments/results/contour_flip_coding_n600_20260707/`.

## Measured rows (n600)

| surface | d_seg | n_flips | n_components | **b_contour (B/flip)** | bz2 full-grid baseline | rate_S if stored |
|---|---|---|---|---|---|---|
| witness-alone | 0.003741 | 441,329 | 142,270 | **0.8201** | 0.8797 | 0.24101 |
| + lane-band composed | 0.003805 | 448,914 | 145,347 | **0.8183** | 0.8770 | 0.24459 |

Stream decomposition (alone): anchors **162,934 B (45%)** + chain 148,275 B + classes 50,036 B +
counts 708 B = 361,953 B. Chain: 740,388 symbols (299,059 backtracks) ≈ **1.60 bits/symbol** — the
digital-straightness contexts DO work (8-dir alphabet coded at 1.6 b); the cost center is the
**component count**: 142,270 anchors ≈ 1.15 B each.

## Coherence decomposition (the NO-GO branch deliverable: the incoherent fraction, quantified)

| component size | components | flips carried | share of flips |
|---|---|---|---|
| 1 px (singletons) | 63,458 (44.6% of comps) | 63,458 | **14.4%** |
| 2–3 px | 45,658 | 106,424 | 24.1% |
| 4–7 px | 21,120 | 106,174 | 24.1% |
| 8–15 px | 8,977 | 94,173 | 21.3% |
| 16+ px | 3,057 | 71,100 | 16.1% |

Mean component size **3.1 px**; 38.5% of flips live in components ≤3 px. The published
1–1.5 bits/contour-px regime assumes LONG coherent strings; this residual's per-component anchor +
walk overhead over 3-px islands is why b floors at ~0.82. Even with a FREE chain+class stream, the
anchors alone cost 0.37 B/flip — reaching 0.65 needs mean component size ≈ 3× larger, i.e. the
residual itself must become more coherent (a TRAINING outcome, not a coder trick).

## Verdict + consequences

- **NO-GO** (0.8201 ≥ 0.65 on both surfaces): Lever-D flicker-residual coding stays CLOSED
  (re-confirms the #280 NO-GO at the ep425 checkpoint, now at full n600 and with the stronger
  coder). d_seg stays IN-TRAINING — consistent with the islands/#301/#274 arms.
- The contour coder itself is the new measured floor (0.820 < 0.876 bz2): registered as a MEASURED
  anchor on `leverd_flicker_residual_reactivation_economics_v1`.
- Band-composed ≈ witness-alone (surfaces within 0.2% in b; composed d_seg 0.003805 slightly WORSE
  than alone 0.003741 at ep425 — the band's net value at this checkpoint is ≈0/negative on the
  flip count; consistent with the mod32cap baseline design where the band was not trained-with).
- Also measured en route: witness-alone n600 d_seg through byte-close render = 0.003741 vs the
  live trainer verdict 0.003636 (faithful render-authority sanity, Δ ≈ 2.9%).

## Process note

Extraction ran resumable per-pair (600 cached flip npz) driven in bounded foreground chunks after
the detached-daemon kills (see the #336 memo's process note; instrumented RSS flat 2.4–2.7 GiB
across the whole extraction — no tool memory spike; kills were environmental).

## Checkpoint custody addendum (2026-07-07, clean-pass-#2 F1 remediation — APPEND-ONLY)

The evidence JSON's `ckpt.dir` cites a session-scratchpad path (transient). The exact ep425
EMA-BEST snapshot the measurement ran against (sha256
`9f123bac950af1ec8eecc938c042f6d0e0662d0a427a16d5ba147e3f1fc62d93`) has been copied into this
memo's durable results directory as `snapshot_ema_BEST.npz` alongside a `CHECKPOINT_CUSTODY.md`
note (sha verified equal post-copy). The sha-pinned evidence JSON is untouched; bit-exact
re-runnability is restored independent of scratchpad GC. Source run:
levelset_n600_witness_mod32cap_20260706T115554Z (snapshot 2026-07-07T02:37Z).
