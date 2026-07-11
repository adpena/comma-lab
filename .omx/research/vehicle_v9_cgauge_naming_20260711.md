# Vehicle designation: **V9 · CGauge** (the Covariant-Gauge witness) — canonical naming record (2026-07-11)

**Operator decision (2026-07-11):** the synthesis vehicle is named **V9 · CGauge**
("V9?" → yes; "use gauge or a combo like CGauge" → CGauge = **C**ovariant + **Gauge**).
Pointer 0.19108282 [contest-CPU] **UNMOVED** — this is a naming/architecture-generation
record, it moves no score.

## Why V9 (a synthesis, not an increment)
The v7.5.x line asked "which trunk"; v8 asked "which per-class carrier." The
covariance-totality audit (`covariance_totality_texture_trunk_verdict_20260710.md` +
`witness_general_covariance_totality_v1`) **collapsed both into one answer**: a SINGLE
covariant trunk (the v7.5.2 lineage) WITH the v8 carriers reinterpreted as **gauge-zero-mode
stores** (per-boundary integration constants, not per-pixel). V9 therefore *supersedes and
unifies* the v7.5.x single-trunk line AND the v8 carrier line — leapfrogging to 9 is the honest
marker that it eats both. It is NOT a v7.5.4 increment and NOT a v8.1 increment.

## Why "CGauge" (the name carries the physics)
- **C = Covariant** (the FOUNDATION): general covariance of the witness — all pair-dependence
  factors through **(ξ, R)** (the ego-screw connection + the measurement operator). No absolute
  per-pixel content; the pixel lattice is the aether. The Einstein half.
- **Gauge** (the MECHANISM): `d_seg = d_cov + d_gauge`; at convergence d_cov≈0 so descent IS
  gauge-fitting; the carriers are literally **gauge-zero-mode** stores. The build/store half.

The bare number would inherit the muddled 7.5.2 / 7.5.3 / 8 ambiguity; the name does not.

## What V9 · CGauge IS (the architecture generation)
- SINGLE covariant trunk (v7.5.2 lineage) — texture trunk **DROPPED** (2 kills: #417 build-INERT
  + covariance residual is smooth events/gauge, not hf-texture).
- **Appearance-phase endgame** machinery: T1 phase-advection (#424) + phase-carrier (#425) +
  Law-5 floor→phase-tail curriculum + `label_floor_detector` costate.
- **Anisotropic rank-8 parametrization**: mod-dim ~17 (Whitney 2·8+1, rank-8 doubly-measured),
  hidden-dim sized to rank-8 + gauge margin (NOT scaled hoping for d_seg — capacity is WASTED
  when d_cov≈0). Bandwidth concentrated along the ξ-covariant + along-tangent boundary
  directions, not isotropic.
- v8 carriers as gauge-zero-mode stores: Movable reaction-event sidecar (85%) + Lane/hood
  gauge-phase zero-mode.

## Sub-clause — CLOSED (d_pose mirror audit landed 2026-07-11, `dpose_covariance_mirror_audit_20260711.md`, commit 40cf964f6)
**V9 · CGauge = DEFINITIVELY a single covariant trunk on BOTH axes (v7.5.2).** The texture trunk
is **dead-for-pose by DOMINANCE** (asymmetric mirror: d_seg killed it by *absence* — residual has
zero texture, hf/lf 2e-4; d_pose kills it by *dominance* — the HF band IS pose-legible, but 63–76%
is a constant ~24-byte-correctable bias, and the pair-specific ceiling after linear-ξ calibration
is ~1.3e-3 ≈ R1's already-banked 0.001610 at 7.2KB, 10–40× above the S2 target at ≥100–1000× the
bytes). **#395 fully dispositioned: DROP** (reactivation only if the S1/S2 output-space solve rungs
fail ≤0.0011 at n600 byte-close — and even then this trunk's ceiling barely matches R1, so a *new*
mechanism, not this one, would be required).

**Sector count = the trunk carries ONE covariant sector; pose rides a SEPARATE dxi channel.** The
mirror measured pose adds **+2.1–2.4 local DOF** beyond the rank-8 partition code (TwoNN, both
codebook widths) → joint *global* dim ≈ 10 — BUT those +2 DOF are **partition-INVISIBLE** (pose
dims 3/4 kNN R² 0.03/0.02; pose-residual↔luma-HF Spearman −0.089 ≈ 0). So they do **not** widen the
trunk's mod vector; they route through the dedicated **dxi/steering channel (6+k)** — exactly the
S2 output shape and the already-banked R1 dxi archive shape.

### The V9 · CGauge parametrization spec (MEASURED — the operator's mod/hidden-dim answer)
- **mod-dim: 17–19, UNCHANGED** (Whitney 2·8+1 on the doubly-measured rank-8 d_seg manifold; #223/#299
  target UNCHANGED by the pose leg — pose adds no trunk-mod DOF). Bandwidth anisotropic along the
  ξ-covariant + along-tangent boundary directions, not isotropic mod-N.
- **hidden-dim: sized to rank-8 + gauge margin** — NO texture-trunk hidden budget on EITHER axis
  (capacity is wasted where d_cov≈0).
- **pose: +2 DOF through the dxi/steering channel (6+k)**, NOT the trunk — confirms the v7.5.2
  conditioning-gate + banked-dxi design.

All advisory `[macOS-CPU / CPU-torch research-signal]`, NON-PROMOTABLE; owed-through-byte-close =
the S1/S2 output-space solve n600 rows (the sole reactivation gate). Pointer 0.19108282 UNMOVED.

## Propagation status (this pass, on pose-audit landing 2026-07-11)
- [DONE] Sub-clause CLOSED (this memo, above) — single covariant trunk BOTH axes + dxi channel.
- [DONE] DAG **FEED-v9-cgauge** naming block + the mod/hidden-dim corollary (references the agent's
  FEED-dpose-mirror + the registered `dpose_photometric_band_mirror_audit_20260711` anchor).
- [DONE — measurement leg] `canonical_equations`: the +2-DOF / partition-invisible / route-via-dxi
  corollary is registered as the mirror anchor on `posenet_luma_chroma_sensitivity_asymmetry_v1`
  (agent, commit 40cf964f6). The SIZING conclusion (mod-17-19 unchanged + hidden rank-8+gauge) is
  routed to **#223** (its in-progress Whitney/Nyquist parametrization derivation is the proper
  home — a separate partial equation here would fragment #223) + **#299** (mod-dim/boundary-capacity
  A/B: the pose leg leaves the mod target UNCHANGED).
- [DONE] **#395** disposition finalized: **DROP** on both axes (reactivation = S1/S2 output-space
  solve fails ≤0.0011 at n600 byte-close).
- [OWED — next launch config] `SPEC_v9_cgauge` (author from SPEC_v75 §8 contract + the covariance
  verdict + this parametrization spec) + the DSL designation label — deferred to the actual
  V9·CGauge config build for the next witness launch (no lever change; CGauge is the identity, the
  levers are the v7.5.2 + phase-advect + carrier set).

## Triality legs (this record)
- **equations** = N/A here (naming; the covariance law is already `witness_general_covariance_totality_v1`).
- **DAG** = FEED block deferred to the pose-audit pass (see propagation plan).
- **DSL** = N/A (designation, not a lever).
`[no-triality]` — pure naming/apparatus record.
