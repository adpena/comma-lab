# ddm_aa1 Findings - Asymmetric-Advantage Composition

Date: 2026-08-07  
Axis: `[macOS-CPU advisory, byte-only archive-grammar audit]` unless a row says external.  
`score_claim=false`; `promotion_eligible=false`; `pointer_moved=false`.

## Verdict First

The first AA1 measured row is FIRED: PR130/CPR1 has **0 counted camera-resolution payload bytes**, so the
#401 blind-coordinate / resize-nullity direct byte reclaim is **0 B** on this archive shape.

Measured audit receipt: `.omx/research/ddm_aa1_20260807/aa1_pr130_payload_blind_audit.json`.

| quantity | measured |
|---|---:|
| PR130 CPR1 archive bytes | 191,052 |
| member `p` bytes | 190,952 |
| compressed model bundle bytes | 73,968 |
| semantic token stream bytes | 116,980 |
| raw semantic renderer bytes | 40,252 |
| raw CPR1 pose carrier bytes | 23,054 |
| raw HPAC model bytes | 20,179 |
| counted camera-resolution payload bytes | **0** |
| direct blind-coordinate reclaim | **0 B** |

Scope: FORMULATION direct audit. CPR1 stores learned renderer/model bytes, a 24x32 pose carrier, and
range-coded 600x384x512 semantic labels on the scorer grid. It does **not** store 874x1164 camera pixels.
The 22.6969% blind-camera-coordinate proof remains valid, but it has no direct CPR1 payload section to
delete. A naive `0.226969 * 116,980 = 26,551 B` token saving is invalid because those are scorer-grid
semantic tokens feeding a learned renderer, not camera-grid pixel values.

## Attack Sheet

Machine-readable rows are in `.omx/research/ddm_aa1_20260807/AA1_ATTACK_SHEET.jsonl` (13 JSONL rows,
one per term x lever).

High-signal rows:

| term | lever | status | isolated delta if it transferred | boundary |
|---|---|---:|---:|---|
| rate | direct #401 blind-coordinate on CPR1 | MEASURED_ZERO_DIRECT | 0 B / 0 S | folded for this archive shape |
| rate | #869 adaptive per-cell L | PROJECTED from our IX2 stream | -113,555 B / -0.075612 S | unmeasured on PR130 semantic tokens; scorer leg queued |
| rate | PR130-style HPAC on our labels | UNMEASURED on our payload | -9,857 B / -0.006563 S vs tq1c KT anchor | live hb1 owns this; no duplicate |
| seg | tq1c snap moves | MEASURED on own base | -0.000322 S if same dseg transfer | unmeasured on PR130 base |
| seg | sq2 solve line | MEASURED negative composition | no adopted delta | pose erosion dominated current formulation |
| pose | PR130 neutral-gray carrier | EXTERNAL measured cure candidate | -0.069379 S vs our current pose term | must fit conditioned semantic base |
| composition | #827 seg+rate bank | DERIVED/MEASURED but blocked | -0.086798 S seg+rate | measured pose wall dominates unless pose-carrying base cures it |

Do not sum the projected deltas. #869 changes the token distribution, HPAC trains on that distribution,
and #827 depends on a pose-legible base. The dependency order is the actual result:

1. Choose or create a pose-carrying semantic base: ARM-CAP endpoint -> ARM-VEH -> n120 selection.
2. Train/exact-decode HPAC on that base's semantic labels; consume hb1 instead of duplicating it.
3. Apply terminal snap/seg finishers to the selected base, measuring realized dseg/dpose.
4. Apply CPR1/model self-compression only to owned carrier/model sections that prove exact round-trip.
5. Exact-eval only after the composed archive is byte-closed and the scorer slot is assigned.

## Blockers

The pose-carrying-base question does **not** automatically unblock #827/#934. CR1 measured the burn
seg+rate prize, but also measured a photometric pose wall for the ep854 burn base with the warp-base
pose carrier. PR130 proves a semantic-pose vehicle can carry low pose, but compatibility with our
banked seg/rate surfaces is still a live measurement, not a theorem.

The full n600 scorer slot is not owned by AA1. `main_hot_state` has ARM-CAP/ARM-VEH and live process
coordination; AA1 therefore did byte-only/scorer-free work and queued scorer-dependent follow-ons.

## Recall Evidence

Sources searched and read:

- Governing context: `.omx/tmp/codex_runs/_common_contract.md`, `PROGRAM.md`, `CLAUDE.md`/`AGENTS.md`,
  `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`, `HANDOFF.md`,
  `SYSTEM_MAP.md`.
- Queries included: `PR130|semantic-pose|HPAC`, `#869|ddm_tw1|adaptive waterfill|token waterfill`,
  `blind_coordinate|CERTIFIED_ZERO_WEIGHT_BLIND_MASK|nullity`, `tq1|snap`, `#827|#934|pose-carrying`,
  `sq2|eg1|fd2`, across `.omx/research`, `docs`, `reports`, `.omx/state`, and `src/tac/canonical_equations`.
- Beyond the charter seeds, the search changed the plan by finding:
  - `ddm_hb1` already owns the HPAC-on-our-labels byte race and reports HPAC not yet measured locally;
  - task ledger row `tz1_adaptive_percell_869_joint_remeasure_20260804` already queues the #869 scorer leg;
  - `blind_coordinate_rate_lever_v1` scopes #401 direct savings to camera-resolution-storing payloads;
  - `ddm_cr1` measures the #827 seg+rate prize and the pose wall, so pose-base compatibility is the named blocker;
  - `main_hot_state` routes ARM-CAP -> ARM-VEH -> n120 selection before composition #984.

## Follow-on Dispositions

- FIRED: CPR1 blind-coordinate direct payload audit.
- FOLDED: direct #401 blind-coordinate/nullity reclaim on CPR1 as shipped.
- QUEUED-WITH-FIRE-ORDER: #869 scorer leg after scorer slot frees; do not claim transfer to PR130 token stream.
- QUEUED-WITH-FIRE-ORDER: hb1 HPAC row; consume its exact-decode packed result when available.
- QUEUED-WITH-FIRE-ORDER: tq1 snap pass on the selected pose-carrying semantic base after ARM-CAP/ARM-VEH.
- QUEUED-WITH-FIRE-ORDER: #827/#934 only through jd5/ARM-VEH pose base; do not spend n600 on ep854+warp.

Own-vehicle frontier remains **S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]**.
Contest pointer remains borrowed/unmoved.
