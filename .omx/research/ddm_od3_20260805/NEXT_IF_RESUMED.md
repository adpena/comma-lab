# OD3 Next If Resumed - 2026-08-05

OD3 terminality is measured for the OD2 n32 pair set. Resume from the artifacts below; do not rerun OD3 unless a receipt hash check fails.

## Measured State

- Raw terminal run: `/Volumes/VertigoDataTier/pact/ddm_od3_20260805/od3_js1_n32_terminal_seg100_cprime_k4.json` SHA `5f7f934e6bafa440572577509e0e733ab3c5e80940d8cf11178e91ecb93bd4ca`.
- Aggregate: `.omx/research/ddm_od3_20260805/OD3_AGGREGATE.json`.
- Stage-1 cap ambiguity is closed on this n32 advisory subset: 32/32 semantic stops, 0/32 `safety_bound_REPORTED`, derived ceiling 100 steps, max actual step 75.
- Stage-1 pooled eta is `0.604882865092900`; delta S_seg subset is `-0.107113520304362`.
- Terminal k=4 carriage preserved seg 32/32 and produced mean d_pose `0.000791809037082`, below same-row baseline `0.000801428562340` but above OD2 registered target `0.0007588698333620414`.
- k=8/k=12 worst-four fallback panel exists at `/Volumes/VertigoDataTier/pact/ddm_od3_20260805/od3_capacity_sweep_k8_worst4.json` and `/Volumes/VertigoDataTier/pact/ddm_od3_20260805/od3_capacity_sweep_k12_worst4.json`. k=8 was best on that panel; k=12 was worse than k=8.

## OD4 Fire Order

1. FIRED-BY-OD3: close Stage-1 terminality and terminal k=4 carriage proof. Status: done in `OD3_TERMINALITY_RECEIPT.md`.
2. QUEUED-WITH-FIRE-ORDER: OD4 builds the receiver-closed archive path for the staged composition. Use OD3's measured terminal coefficients and account for:
   - k=4 frame_0 DCT coefficients as counted payload: 96 B/pair, projected 57,600 B if applied to all 600 pairs.
   - DCT basis as deterministic rule-118 free code.
   - Stage-1 representation bytes as an open pricing question; do not assume zero and do not hide video-derived payload in code.
3. QUEUED-WITH-FIRE-ORDER: OD4 decides selective higher-k carriage only after archive/rate accounting. The instance-scoped k=8 panel recovered `0.000271546421573` mean d_pose on four worst rows at +288 B/pair; k=12 was not monotone and should not be defaulted.
4. QUEUED-WITH-FIRE-ORDER: after byte-closed archive construction, run n>=32 scorer survival first, then n600 only under the common scorer-slot policy. Recompute S from components; do not use evaluate.py's rounded final score.
5. FOLDED: ST2 pair ranking for OD3 exact rows. It had zero overlap with the OD2 n32 pair ids and did not legally reorder this run. A future targeter can re-enter only with a receiver-consumed table keyed to the target pair ids/cells.

## Boundaries To Preserve

- OD3 is `[macOS-CPU frozen-scorer advisory]`, not contest authority.
- No receiver-closed archive was built by OD3.
- No full n600 scorer job was run by OD3.
- The pose leg is still m96 pose-easy (`0.42628664334579025x` population), while the seg leg is matched (`1.0099888594483923x`).
- Protected files remain off-limits.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
