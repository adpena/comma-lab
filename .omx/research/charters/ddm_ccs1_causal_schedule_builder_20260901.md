# ddm_ccs1_causal_schedule_builder — BUILD dcc1's rank-1 successor: the versioned receiver-causal G/M schedule on the UNCHANGED AFR1 field, fit + real-coder encode + exact parse-back, decided SCORER-FREE at the 137,986 B gate (tasks #1374 / #1182; fire order in ddm_dcc1_decoder_causal_conditioning_verdict_20260901.md §First-measurement + ddm_gmf1_fitted_crossgroup_gm_verdict_20260901.md §NEXT_IF_RESUMED)

## MANDATE

dcc1 derived the conditioning-transport law (`H(E_i(C_i) | D_<i, p_i) = 0` for free
conditioning), closed the census 8-satisfy/3-violate with zero falsifiers, and ranked ONE
successor first: a **causal G/M schedule on the unchanged AFR1 field** whose every context is
decoder-available at consumption time — previous decoded classes, a boundary state updated
from the decoded prefix, and deterministic position cells. This replaces SFP1's dead
encoder-side four-label schedule (gmf1 RECALL-CLOSED) with the lawful formulation. The rung
is DECISIVE WITHOUT A SCORER: the field is byte-identical to AFR1, so distortion is FIXED —
if the complete archive exceeds 137,986 B the schedule instance is CLOSED by arithmetic; if
it fits, MAIN owns an authority replay on the exact bytes. This arm builds the schema, fits
the model, encodes with real coders, proves parse-back, and lands the typed verdict.

## SCOPE

0. **$0 FIRST — register the law.** dcc1's Catalog #344 waiver deferred registration only
   because its source anchors were unlanded; ALL are now committed on main (qx3/qx4/gmf1
   memos + the dcc1 verdict). Register `decoder_causal_condition_transport_v1` in
   `tac.canonical_equations` as an operational-domain extension of
   `wyner_ziv_decoder_side_information_conditional_entropy_savings_v1` (per the post-#400
   Catalog #299 consolidation rule — an extension, not a new gate): the exact-causal-time
   equivalence-class obligation + the `B_T >= ceil(H(E(C)|D,p)/8)` transport floor, anchors =
   the qx3 510,404 B bridge + gmf1's 3/3 source closure, producers/consumers named.
1. **Author the versioned receiver schema v1** (the gmf1 repair contract): for every coded
   group, schedule/context = {previous decoded replacement classes · boundary state updated
   from the decoded prefix · deterministic position cells}. Define integer CDF selection,
   parser ordering, reset state, and exact parse-back. Serialize and COUNT every video-derived
   schedule/model byte. The schema must pass dcc1's law at every coding step by construction.
2. **Fit ONE seeded held-out nonlinear model** on the unchanged AFR1 field X (117,964,800 B,
   SHA `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`), lm1's train/test
   protocol (the pk3/pk4 lesson: 23/23 in-sample = 0/23 held-out — no in-sample-only claims).
   Seed recorded; deterministic; resumable-from-disk checkpoints (P0).
3. **Encode with real coders + prove identity.** Count model + schedule + stream + headers +
   complete archive. Decode TWICE: the AFR1 field AND the final rendered output must be
   byte-identical both times. Retain everything (model weights, coded streams, receipts,
   deterministic repeat).
4. **The gate (scorer-free decisive):** complete archive > 137,986 B ⇒ this fixed-distortion
   schedule instance is CLOSED — write the measured pool plainly (vs shipped 126,926 B and
   allowance 87,403.86 B). Complete archive ≤ 137,986 B ⇒ emit a typed fire order handing
   MAIN the exact archive for authority replay (`upstream/evaluate.py` on the exact bytes;
   same-distortion score arithmetic stays a PROJECTION until then). Either way: preserve the
   three pinned SFP1 field hashes untouched (comparability anchors).

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight).
- The local SCORER LANE belongs to MAIN, always. Do NOT write who currently holds it into a
  charter (the #1210 stale-precondition genus — MEASURED 2026-08-29 when
  `ddm_bz2_bornsmall_capacity_ceiling` refused on a stale occupancy claim). This rung needs
  NO scorer by design; if any follow-on does, emit a typed fire order and let MAIN fire it.
- Serializer commits w/ post-edit `--expected-content-sha256`; bundle-fallback on
  .git/objects denial (#1293). `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD (P0 DEF CON 1000): model weights + schedule + raw and coded streams
  + complete archive + repeat + checkpoints + command/config + hashes, all to
  `/Volumes/APDataStore/pact/ddm_gmf1_fitted_crossgroup_gm/` (the named consumer store from
  the gmf1/dcc1 fire orders; verify free bytes AT START, preserve the 1 GiB AP reserve).
- DETACHED >30-MIN COMPUTE: the fit launches outside the arm session via the canonical
  detached launcher (script paths avoid claude/codex tokens — the fleet-reaper argv
  predicate), pidfile + crash-resumable checkpoints + durable done-receipt; the arm MONITORS.
  HPAC-family training is proven LOCAL (hb1/rx2 lineage) — no GPU dispatch.
- CLOSED-FORM-FIRST: the fitted coding model is learned BY CONSTRUCTION (the model IS the
  object under test — gmf1's recorded one-line reason); ALL byte arithmetic stays exact; no
  entropy estimates cited as bytes (SCREEN labels only, refusal-ceiling use).
- RATE ROWS ONLY — the field is byte-identical to AFR1 so NO distortion claim exists to make;
  any claim beyond rate is fabrication.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- gmf1 RECALL-CLOSED (`ddm_gmf1_fitted_crossgroup_gm_verdict_20260901.md`): SFP1's
  source/target/boundary labels have no decoder semantics — this schema replaces them; do NOT
  smuggle any encoder-only label back in (that re-opens the closed formulation, not this one).
- jf1 same-schedule refit (found by gmf1's recall): 13,463 B model + 114,143 B stream =
  127,606 B ≈ shipped 126,926 B — model refits AT THE SHIPPED SCHEDULE recover ≈ the shipped
  pool. This arm changes the SCHEDULE BASIN; jf1 is the measured prior, not this object.
- Sharp-optimum law (#1214, five arms): same-basin perturbations are closed — the entire
  premise here is a basin change; the memo must confront the law, not dodge it.
- rr9 (#1244): reorder-only closure, explicitly leaves re-architected schedules open · mi1
  (#1266): 47.4× is ADD-ON economics, inapplicable to replacement · lm1 (#1285):
  discrete/linear replacement closed, nonlinear open — the gmf1 stage-0 race table binds.
- dcc1 DEAD-ENDS bind verbatim: no fixed-G/M stand-ins · no position-only fit · B1/B2/B3
  stay folded (sg2b measured their decoded-field edits Seg/Pose-negative; a lossless refit
  cannot repair rendered output) · QX4's six forms are not this arm's business.

## OPTIMAL FORM

- Family exemplar: the SHIPPED HPAC ITSELF — the census's positive control, whose
  decoded-neighbor contexts satisfy the law by construction at 126,926 B; source receipts
  `hpac_integer.py` SHA `6e6b4f4d0b293fb60cc1b751958756a4cd6c2ce7bcff68c6f03e20277856803f` +
  `inflate.py` SHA `e01325d65c42223d5e1ca8169f2bef0f62ae59bdcfeabf321e681fa2cd07d4e2`
  (`src/tac/pr130_runtime/dv1_cpu_runtime/`). Evaluation reference = lm1's train/test
  falsifier protocol. Provenance pins: dcc1 verdict memo (this fire order's authority) ·
  gmf1 RECALL_CLOSURE.json SHA
  `95f90363ea4d58b52bc00cd5370a7996dc3502b971f203d3aded6a6e71b17598` · base X SHA
  `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` · shipped HPAC section
  SHA `602115b323b0e403d08287af9b273a2d4fb23e026d83c1f6e4609ed77ef98f98` (13,515 B) + RC64
  113,411 B · afr1 archive SHA
  `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` (180,002 B).
- SCOPE reductions legal: fit on a seeded RANDOM pair sample (n≥120, seed recorded, never a
  prefix — m88/bp2) before the full-field confirmation encode; boundary-state richness may
  start minimal (decoded-prefix class transitions) before a loop-until-dry feature sweep.
  MECHANISM reductions FORBIDDEN: real coders · held-out coding · counted model bytes ·
  full-field final encode · exact double-decode identity · the three declared context
  families (dropping one is a different, weaker object — name it if forced).
- **PRIOR-LAW PREDICTION (falsifiable):** sharp-optimum + jf1 predict the fitted causal pool
  lands within ±10% of the shipped 126,926 B — the shipped decoded-neighbor schedule already
  captures most causal structure, so the likely outcome is CLOSED at the archive gate
  (~180 KB complete). FALSIFIER: complete archive ≤ 137,986 B — the SCMDL diagonal opens for
  real and MAIN buys the authority row; count it plainly either way.

## DELIVERABLE

`.omx/research/ddm_ccs1_causal_schedule_builder_verdict_20260901.md` — typed rows: law
registration receipt · schema v1 spec + serialized bytes · fit receipts (seed, split,
held-out coding) · the counted pool table {model B, schedule B, stream B, headers, complete
archive B} vs {126,926 / 87,403.86 / 137,986} · double-decode identity proof · verdict
{CLOSED-AT-GATE w/ measured pool / FIRE-ORDER-TO-MAIN w/ exact archive sha} · DEAD-ENDS +
denominator. Commit via the serializer. End with the own-vehicle frontier line
(S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600], afr1 sha cbb8d928…d405bf25 —
UNMOVED unless the fire order lands).
