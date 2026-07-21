# Codex findings — predictor R3 causal/surgical pass

`Task #578 round 3` · `lane_id=predictor_r3_causal` · `research_only=true` · `[macOS-CPU advisory]` · `MAIN_REVIEW_REQUIRED`

## Verdict

`R3_MEASURED_DESCRIPTION_CURVE_ONLY`. The real-n600 description curve reaches a 216,207-byte knee with description `d_seg=0.016095318264431422`, but this is not a receiver-closed archive and not a score. The contest-CPU pointer remains exactly `0.1910828242`.

Verdict scope is narrow: the exact round-2 predictor masks, the R3 mod-8 phase/curvature/log-xi majority response, exact coherent-component packets, and reversible BASE coding. It does not kill causal phase response, curve carriers, or event alphabets as families.

## D1 — causal boundary response (`MEASURED`)

The input inventory reproduces 1,772,327 boundary misses. The zero-table online model—fit strictly from decoded frames earlier than the frame being predicted—hits 174,526 seed-free sites (9.8472798756%) and introduces 45,430 wrong sites, for net 129,096 fewer misses before exception packets. Its exact surviving-exception packets total 1,087,552 bytes, so no boundary-exception row survives the global box after higher-EV component packets enter.

The fixed n64 model costs 478 bytes, hits 150,848 sites, introduces 41,886, and nets 108,962. The online form is better on this formulation and needs no video-derived rule table. It executes #424 `cross_scored_frame_xi_interp`; it consumes the #425 measured jitter prior/Huffman lengths; and it records LawRef `partition_temporal_transport_amortization_jitter_bound_v1`.

## D2 — surgical component partition (`MEASURED`)

There are 10,919 coherent components and 1,339,907 pixels. Every component receives its own exact zlib-9, self-delimiting packet and independent λ* decision. The all-component independent stream is 490,133 bytes.

On the same masks, the exact #234-style adjacent-frame Hungarian birth/match/death grammar with XOR residuals is 181,904 bytes versus 274,844 bytes for PBS1. The event stream has 3,525 births, 7,394 matches, 3,513 implicit deaths, and 648,189 XOR sites; its decoder replays every component exactly. Thus event coding wins this aggregate same-mask A/B by 92,940 bytes (`MEASURED`, formulation scope only). The composed curve still uses individual packets because D2's delegated rule requires per-component admission rather than an indivisible aggregate bundle.

## D3 — BASE accounting conflict (`MEASURED` + `DERIVED`)

The apparent conflict was a layer mismatch:

- `169,855` was never a measured base. It was the round-2 composed knee: a `DERIVED` algebraic implied base of 73,777 plus 96,078 measured variable bytes.
- `262,498` was a mixed-layer declared base: 221,195 raw PXCH bytes plus 41,303 Brotli-q11 LBND2 bytes.
- R3 applies reversible coding to both sections. #553-style derived ternary ties reduce PXCH to a two-bitplane quotient, then the #557 race selects per-section Brotli-q11 + raw LZMA1. The exact self-contained BASE materialization is 36,011 bytes; the grouped control is 37,080 bytes.

JRD #453 is `NULL_SAFE_NOT_APPLICABLE` because these sections have no ordered lossy coefficient prefix. L20–L32 signed-int8 byte maps are also `NULL_SAFE_NOT_APPLICABLE` to semantic label/container bytes. Neither inapplicable transform was faked.

## D4 — composed curve v3 (`MEASURED` bytes, `DERIVED` description fraction)

The decomposed knee is exact:

- BASE entropy: 36,011 bytes.
- Adaptive predictor parameters: 0 bytes.
- Admitted boundary exceptions: 0 bytes.
- 1,802 admitted self-delimiting component packets: 180,196 bytes.
- Total: 216,207 bytes, 15 bytes below the delegated box.
- Corrected original misses: 1,268,835; causal introduced misses still eaten: 45,430.
- Remaining description misses: 1,898,681 / 117,964,800 = `0.016095318264431422` (`DERIVED` from exact masks).

The packet bundle is materialized at `/Volumes/VertigoDataTier/pact/evidence/predictor_r3_20260721/canonical_r3_20260721/components/admitted_knee_components.pcomp3`, 180,196 bytes, SHA-256 `32b41a7db1c6927e6812a49bdea568cd00fd2c19ca9798e38666385e8ed2e68d`. All 9,129 rejected candidates remain in the receipt's eaten ledger, including their λ and box gates. Per-class/per-stratum admitted/eaten totals are also in the receipt; Road cell-interior dominates admission (130,736 bytes / 868,864 corrections).

## STORES CONSULTED

- `reports/latest.md` and `.omx/state/lane_registry.json` for pointer/lane custody.
- `.omx/state/subagent_progress.jsonl` and the live Codex inboxes for ownership/directives.
- `.omx/research/predictor_r2_missdelta_measurements_20260721.json` and its SSD n600 chunks.
- `.omx/research/predictor_upgrade_xi_chart_measurements_20260721.json` lineage, #424/#425 code, #234 tracker, #453 JRD, #553 pattern, #557 coder primitives, and the canonical jitter/breakeven laws.
- Frozen n600 cache `/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz` under its recorded SHA-256 custody.

## Promotion boundary

No archive, inflate receiver, uint8/R replay, Pose term, contest-CPU replay, or contest-CUDA replay exists here. Therefore no score or promotion claim is authorized. MAIN must independently review the commit before any landing or downstream use.

