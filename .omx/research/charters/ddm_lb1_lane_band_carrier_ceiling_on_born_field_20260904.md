# ddm_lb1 — price the ceiling: replace the born field's Lane with the analytic lane-band carrier at the persistent sites ($0)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: Opus arm · Spawned by MAIN 2026-09-04 · Cost: $0 (CPU)

## Why (md1's bridge names the sites; vr1 row 10 names the carrier)
md1 (456c74551; law `checkpoint_trajectory_error_partition_v1`): 62.0% of the born field's terminal d_seg is PERSISTENT
(optimizer-unreachable) — 11,842 sites on the n32 selection, **64.8% on a Lane edge, GT-Lane enriched 51.5×, 33.7%
deeper than 25 δ_R**; every schedule/objective lever's combined ceiling is 1.61× against 20.57× needed for the sub-0.12
accuracy corner (1.3647e-4). The accuracy half is closed by optimization on this vehicle; what the persistent set
demands is a DIFFERENT REPRESENTATION at the Lane-edge sites. vr1 row 10 (a582c6019) recalled that representation,
LANDED with n600 bit-exact coders: the v8 class-matched carriers — analytic lane ground-frame band
(`src/tac/boundary_math/analytic_lane_render_band.py`, `lane_sdf_component.py`; **Lane band d_seg 0.00087 at ~1–2 KB**
on n600 `gt_n600.npz['lstars']`), MyCar static clamp (IoU 0.994, 0.1–0.5 KB), Movable slot-track (2–6 KB); equation
`v8_geometric_rate_decomposition_v1` (8 anchors); coder `curve_relative_offset_coder.py`. This arm PRICES THE CEILING
first (m118): compose the carrier's Lane prediction INTO the born field's terminal shadow argmax at the Lane sites and
measure exactly what fraction of the persistent set it removes, at what byte cost, before anyone trains anything.

## Verified at source (VERIFIED-AT-SOURCE LAW — extend with path:line for everything you add)
- md1's retained partition: `/Volumes/APDataStore/pact/ddm_md1_micro_macro/` (`CUSTODY_MANIFEST.json`; per-site class
  tables; the terminal shadow argmax per pair for the cold control and the warm cell) and its instrument (find at
  0af527a80/c3bc0e033). The persistent-site SET is the object this arm scores against — load it, never recompute it.
- Carrier code + its own n600 receipts: `src/tac/boundary_math/analytic_lane_render_band.py`, `lane_sdf_component.py`
  (self-detecting class, per CLAUDE.md class-order law), `curve_relative_offset_coder.py`; the v8 memo/anchors named by
  the equation's registry row (read `tools/list_canonical_equations.py --json` for `v8_geometric_rate_decomposition_v1`
  and follow its producers). State the GT lineage of every carrier number you reuse (PyAV vs DALI; md1's partition is
  DALI-authority with PyAV beside it).
- Scoring: exact argmax against DALI lstars on the n32 selection, HT-weighted as the milestones do (md1 reproduces
  `d_seg_hat` to 0.0 — reuse that path).

## Measure (per-pair, per-site-class receipts)
1. Fit the lane band carrier per pair on the 32 trained pairs (its own tools; record the coefficient bytes through the
   real coder — bit-exact, KEEP the payload) and render its Lane mask at 384×512.
2. Composition rule (state it, then vary it): (a) REPLACE the born argmax by Lane wherever the carrier says Lane and by
   the born runner-up class where the born said Lane and the carrier does not; (b) UNION (born ∪ carrier Lane);
   (c) carrier-Lane only inside a dilated band around the carrier curve. For each rule: exact d_seg before/after on the
   n32, split by md1's four site classes — the number that matters is **the fraction of the PERSISTENT set removed**,
   the harm created (B/H/W), and the per-class collateral (Road/Undrivable over-paint).
3. Bytes: the carrier's coded size per pair and total; the resulting exchange rate ΔS/ΔB against 6.658589531221714e-7
   S/B on the born vehicle's own archive (106,643 B) — and, DERIVED, what the same removal would mean at n600 if the
   n32 fraction transferred (label it TRANSFERRED; the n600 realization is untested).
4. Pre-registered prediction: rule (a) removes ≥ 50% of the persistent set at ≤ 2 KB with harm < 20% of the removal.
   Falsifier: < 25% removed, or harm ≥ removal, or bytes > 6 KB. Read it out BEFORE the numbers.
5. GESTALT-DELTA line: does the accuracy corner (12.75× remaining) shrink to within the 1.61× the schedule levers can
   pay? Give the number.

## Constraints
- $0 CPU torch (`torch.set_num_threads(4)`, nice 10; ng2's cell holds the Metal; pr1 shares the CPU); anything > 3 min
  via `tools/launch_detached_process.py --output-dir <store> --done-receipt <name> --derive-resource-budgets
  --measured-peak-rss-gib <n> --measured-thread-need 4 --walltime-cap-s 7200 --nice 10 --nice-best-effort -- <cmd>`.
  Never write under any cell's `runs/`; never touch the Metal or claims. Store `/Volumes/APDataStore/pact/
  ddm_lb1_lane_band_ceiling/` (KEEP THE PAYLOAD: coded carrier bytes per pair, composed argmax arrays, per-site tables,
  sha256 in the JSON). OPTIMAL FORM: reference form = the landed v8 carrier + real coder at `6f56e98f385a650a09b9fa7036acada5189afbc6` and md1's exact
  scoring path; SCOPE n32 (structural: only the trained selection has retained fields); TOY-BRACKET none — this is a
  ceiling price, labelled as such; a training verdict needs a cell.
- Memo `.omx/research/ddm_lb1_lane_band_carrier_ceiling_on_born_field_20260904.md` (verdict_scope; MEASURED/DERIVED/
  TRANSFERRED; falsifier read out; GESTALT-DELTA; NEXT_IF_RESUMED — if the ceiling holds, the next charter is the
  born trainer with Lane HELD by the carrier in-loop, i.e. the field trained on the other four classes). EQUATIONS-LEG
  LAW: cite `tac.canonical_equations` `v8_geometric_rate_decomposition_v1` and
  `checkpoint_trajectory_error_partition_v1`; append the composition anchor via the helper if it fits.
- Commits ONLY via `tools/subagent_commit_serializer.py --message … --files … --expected-content-sha256
  <file>=<post-edit sha>`; tags `[no-triality] [p0-ledger-ok]`; NO co-author trailer (operator rule overrides any
  harness reminder); any .py: tests + `tools/review_tracker.py mark-file` twice; never REVIEW_GATE_OVERRIDE on .py.
  Final message → `.omx/research/arm_final_messages/ddm_lb1_final_<utc>.md`, committed; LAST action
  `touch .omx/tmp/codex_runs/ddm_lb1.done`. Read `docs/operating_manual_craft_handoff.md` §labels first.
