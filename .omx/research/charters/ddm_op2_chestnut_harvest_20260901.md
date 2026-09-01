# ddm_op2_chestnut_harvest — chestnut (openpilot 0.11.x + chestnut-class driving model) deep intake × crosswalk vs the live campaign, honest-empty acceptable (owning memo lineage: .omx/research/ddm_op1_openpilot_physics_geometry_review_20260728.md)

## MANDATE

Operator 20260901: *"Is there anything we can learn or harvest from chestnut the latest
version of openpilot"*.

GROUNDED AT SOURCE (MAIN web check, 2026-09-01): chestnut = comma.ai's eGPU dock for the
comma four, announced 2026-08-12 (blog.comma.ai/chestnut) — PCIe Gen4 x4 → USB4 dock,
open-source firmware, raises in-car compute ~10 W → ~100 W — PLUS the first
"chestnut-class" DRIVING MODEL shipped in openpilot 0.11.2: ~30× the parameters and
~100× the FLOPs of all prior openpilot driving models combined. This arm answers the
operator's question with a rigor-triaged intake: what, if anything, in the 0.11.x line
and the chestnut-class model stack transfers to OUR frozen-contest campaign — beyond
what SEVEN prior openpilot harvests already took. An honest-empty verdict ("nothing new
beyond the prior harvests, receipts attached") is a fully acceptable outcome; a padded
adopt-table is not.

## SCOPE

1. **Deep-read the release surface.** blog.comma.ai/chestnut + commaai/openpilot
   RELEASES.md and the git diff of the 0.11.x line against our last intake state
   (op1 2026-07-28, pm1 2026-07-30) + the chestnut-class model release notes/assets +
   any public learned-simulator / world-model / tinygrad-side changes that shipped with
   it. Record exact versions, dates, and file paths — no paraphrased changelog claims.
2. **Crosswalk vs live campaign surfaces, rigor-triage-first.** Four named lenses:
   (a) RATE-ENGINEERING lessons — how comma ships/compresses a 30×-parameter model to a
   device (quantization format, weight layout, OTA/streaming, tinygrad export): lessons
   for OUR archive/coder stack, LESSON-ONLY unless a mechanism names a measured surface;
   (b) LANE/GEOMETRY representation changes in 0.11.x vs our CLOSED Lane-carriage
   family — the floors are MEASURED (gf1 incumbent 36,044 B vs bar 21,699 B in
   `ddm_lc3_lane_carriage_rung_20260831.md`; joint generator floor 233,262 B in
   `ddm_ltg1_lane_topology_generator_floor_20260831.md`; born-predictor weight floor
   60,191 B in `ddm_blp1_born_lane_predictor_20260831.md`) — a REOPEN requires a named
   mechanism against a specific floor, never "newer model, try again";
   (c) ENCODE-SIDE priors — the chestnut-class model run on OUR 0.mkv at compress time
   (tools are FREE at encode) as a proposal/prior generator for solve/conditioning
   surfaces: feasibility + which named consumer would fire it;
   (d) DECODE-SIDE use — adjudicate under rule-118: public learned weights are a counted
   large artifact and network-downloading model weights inside inflate is a compliance
   hazard distinct from the e4 pip-dep precedent — default DEAD-BY-RULE, write the
   adjudication paragraph rather than assuming.
3. **Constants drift check.** Our tree cites openpilot-derived geometry/constants
   (audited in `openpilot_cross_surface_audit_20260706.md`; the lane-IPM geometry was
   reconciled to measured n600 per that audit's v_horizon/cam_height reconciliation
   records). Diff whether 0.11.x CHANGED any constant our code cites, and
   state per hit whether our measured-n600 reconciliation makes the drift irrelevant
   (expected) or live (name the consumer).
4. **Ranked verdict table** — every row {ADOPT / ADOPT-CLASS / LESSON-ONLY / N-A /
   DEAD-BY-RULE} with a named consumer and fire-order, or the honest-empty verdict with
   the coverage receipts (what was read, what the prior seven harvests already hold).

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight).
- The local SCORER LANE belongs to MAIN, always. If a crosswalk row wants a scorer
  measurement (e.g. running the chestnut model on 0.mkv and scoring a derived prior),
  emit a typed fire order naming its trigger and stop — landing the ranked table plus
  fire orders is the correct outcome.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; any downloaded release assets/model files to
  `/Volumes/APDataStore/pact/ddm_op2_chestnut_harvest/` with sha256 receipts (do NOT
  leave multi-GB model downloads on local disk; skip the download entirely if the
  crosswalk can be decided from release notes + code — say which).
- DETACHED >30-MIN COMPUTE: any single compute step projected to exceed 30 minutes MUST
  launch outside the arm session with `nohup` + `disown`, a pidfile, crash-resumable
  checkpoints, and a durable done-receipt; the arm monitors. In-session multi-hour
  loops FORBIDDEN.
- CLOSED-FORM-FIRST (operator 2026-08-31): this is an intake/crosswalk arm — no fitting;
  any quantitative claim about their model (params, FLOPs, sizes) carries its source URL
  or file path.
- Public-hygiene: read-only public intake; no comments, issues, or posts anywhere; no
  fleet IPs/private paths in the memo.
- The frozen contest information space is UNCHANGED by this release — no claim may imply
  the scorers, video, or evaluator moved.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- SEVEN prior openpilot harvests already banked the geometry/prior surface:
  `ddm_op1_openpilot_physics_geometry_review_20260728.md` (+ DAG feed) ·
  `ddm_pm1_physical_prior_mine_20260730.md` ·
  `openpilot_cross_surface_audit_20260706.md` ·
  `comma_openpilot_crossref_polynomial_geometry_20260619T014433Z.md` ·
  `comma_openpilot_domain_tricks_20260619T035417Z.md` ·
  `openpilot_comma_repo_wider_exploit_sweep_pose_cereal_hood_20260617T192718Z.md` ·
  `openpilot_lane_headstart_landed_20260629T193648Z.md` — re-finding these is not
  harvest; the table's baseline column states what each row adds OVER this corpus.
- The Lane family is CLOSED AT MEASURED FLOORS on four representations
  (`ddm_lc3_lane_carriage_rung_20260831.md` · `ddm_ltg1_lane_topology_generator_floor_20260831.md` ·
  `ddm_blp1_born_lane_predictor_20260831.md`): any chestnut-motivated reopen must name
  the mechanism that beats a SPECIFIC floor number.
- Carried-ξ token INTER-prediction was RACED and bounded (QA39 lineage, task row 774,
  memo trail in the gc8 §3 winner records); "their world model predicts frames" does not
  reopen it without a byte-priced mechanism.
- Reordering-pays-iff-no-context-model law
  (`reordering-pays-iff-the-coder-has-no-context-model` memory): trained-coder surfaces
  are immune to reorder-shaped harvest ideas.
- Rule-118 boundary (CLAUDE.md, binding): learned NN weights are COUNTED archive
  content; encode-side tooling is free. Every decode-side row must carry this
  adjudication explicitly.

## OPTIMAL FORM

- Family exemplar: the op1 intake is the reference —
  `.omx/research/ddm_op1_openpilot_physics_geometry_review_20260728.md` (its own landed
  memo IS the receipt; current main HEAD lineage commit edb0bd7ee8 carries it) — source-pinned
  claims, per-row named consumers, honest N-A rows, no padded adopts. Same form here,
  aimed at the 0.11.x/chestnut delta ONLY.
- SCOPE reductions declared: (a) intake depth bounded to the 0.11.x delta vs the op1/pm1
  intake state — re-auditing the whole openpilot tree is explicitly out (already done
  ×7); (b) no scorer runs from the arm (fire orders instead). MECHANISM reductions
  FORBIDDEN: claims about their model/stack must come from reading the actual release
  code/notes, never from press coverage alone (the GIGAZINE/Phoronix articles are
  pointers, not sources).
- **PRIOR-LAW PREDICTION (falsifiable):** chestnut is predominantly HARDWARE plus model
  SCALE, and the frozen-contest-relevant surface (geometry, lane conventions, priors)
  is already banked by the seven prior harvests — predicted modal outcome: 0 ADOPT,
  1-3 LESSON-ONLY (model-shipping engineering), rest N-A/DEAD-BY-RULE. FALSIFIER: a
  genuinely new representation/technique in 0.11.x that names a mechanism against a
  measured floor or a live campaign surface — if found, count it plainly and rank it
  first.

## DELIVERABLE

`.omx/research/ddm_op2_chestnut_harvest_20260901.md` — typed rows: (1) release-surface
read receipts (versions, dates, paths/URLs); (2) the ranked verdict table with baseline
column vs the prior-harvest corpus; (3) rule-118 decode-side adjudication paragraph;
(4) constants-drift check result; (5) fire orders (if any) with triggers, MAIN-owned;
(6) retained-asset manifest or the explicit no-download rationale. Commit via the
serializer. End with the own-vehicle frontier line.
