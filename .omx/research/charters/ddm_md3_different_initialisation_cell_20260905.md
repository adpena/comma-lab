# CHARTER ddm_md3 — is the born vehicle's unreachable error set fixed by its INITIALISATION? One different-init cell + the data-order control, partitioned with md1's instrument

Tokens: `[no-triality] [p0-ledger-ok]`. Owner: Opus arm. Spawned 2026-09-05 ~14:10Z. Parents: md1 (`ddm_md1_micro_to_macro_dynamics_20260904.md`: PERSISTENT
62.011% on the cold control, born in 16 updates), md2 (`ddm_md2_persistent_partition_under_the_burn_default_20260905.md`: STANDS at 62.954% under the burn
default; Jaccard 0.807 with the cold persistent set = the SAME sites; ALL three retained `wc3` seed controls share init sha 991a1cc6… so a second seed varies
DATA ORDER only; priced next step = a different-initialisation cell ≈ 2.8 h Metal + 54 min CPU, $0, no new code), ng5 (the burn default of record:
τ band × carried duals — `ddm_ng5_tau_band_x_continuous_objective_cell_20260904.md`, sealed config 1205463b…/re-rooted 93f92fc6…), the QBR1 sealed-cell
chain (`ddm_qbr1_born_fairform_burn_prep_20260902.md`; `experiments/ddm_qbr1_cell_chain.py`; the queue driver is the ONLY fire path), gs4 §5(b).

## PRIOR-LAW PREDICTION (owed line)
md2 showed the persistent set is inherited from the step-0 wrong pool (16,553 sites, bit-identical across cells) and the schedule cannot move it. If the pool
is a property of the INITIALISATION (the generator's starting weights paint the same wrong sites regardless of optimizer path), a different init produces a
DIFFERENT step-0 wrong pool and a persistent set with LOW Jaccard against the cold set (< the within-pool chance level once the pools differ) while the
PERSISTENT SHARE stays ≈ 60% — i.e. the sites are reachable by a different start but every start leaves ~60% of its own error unreachable (capacity of the
generator FORM). If instead the pool is a property of the DATA (the frozen scorer's hard sites), a different init reproduces the same persistent sites
(Jaccard ≈ 0.8, like ng5). PREDICTION: Jaccard(different-init persistent, cold persistent) ≤ 0.45 with PERSISTENT share 55–65% → the unreachable set is
init-anchored, and an ENSEMBLE-of-starts / init search becomes the priced route to the accuracy corner (it changes which sites are reachable). FALSIFIER:
Jaccard ≥ 0.70 → the sites are data-anchored (scorer-hard) and no start reaches them → the born accuracy corner is closed at FAMILY scope for this generator
form; the route needs a different generator (gc1/gf2 territory), and gs4 §5(b) is answered.

## Scope (in order)
1. **Data-order control FIRST ($0, CPU, ~54 min):** run md1's instrument on the retained `seed_20260903` control (same init sha 991a1cc6…, different data
   order; md2 says its checkpoints exist — locate them under the QBR1 store) → PERSISTENT share + Jaccard vs cold. This isolates data order from init.
2. **Seal ONE different-initialisation cell** on the burn default of record (ng5's sealed config, byte-identical except the initialisation): change ONLY the
   generator's init (a new init seed → a new init sha; locate how the QBR1 born trainer initialises — a pinned init file/seed in the config; NEVER hand-edit
   weights); validate INSIDE its own sealed tree (`experiments/ddm_reseal_pins_inside_sealed_tree.py`); no-op detector (step-0 field must DIFFER from the cold
   pool — that is the point); the $0 differential check as ng4/ng5 did. Fire through `tools/cell_queue_driver.py run` (queue-spec JSON; peak FROM_LEDGER; the
   Metal is busy with cl2's λ rungs until ~15:30Z — the driver admits when free; never two Metal cells).
3. **Partition it** with md1's instrument unchanged (313 16-step checkpoints retained by the cell) → the md1 table, birth step, Lane share, and Jaccard vs
   cold AND vs the data-order control; report the within-pool null when the pools differ (state the pools' own Jaccard first).
4. Verdict words: INIT-ANCHORED (J ≤ 0.45) / DATA-ANCHORED (J ≥ 0.70) / INDETERMINATE (between; then name the second init seed as the next cell). Priced
   next step in either case. Memo `.omx/research/ddm_md3_different_initialisation_cell_20260905.md` with an "Equations leg (`tac.canonical_equations`)" line
   (a 4th anchor on the persistent-partition law, or the init-anchoring law if the falsifier does not fire).

## Honest frame
Burn-quality object (born vehicle, S_hat ~0.38 at 106 KB), NOT a pointer mover; d_seg 1.36e-4 is still 11.7× away. This unit prices gs4 §5(b) — it decides
whether the born route has ANY accuracy door left. Cost ~3.7 h wall, $0.

## Rules that bind
NO-FAKE; ALWAYS KEEP THE PAYLOAD; upstream/ READ-ONLY; no Modal; ONE Metal cell at a time (through the driver only); commits ONLY via the serializer with
post-edit shas and `[no-triality] [p0-ledger-ok]`; NO co-author trailers (operator rule overrides any harness reminder); .py two review-gate passes; checkpoints
every 10 tool uses (`--subagent-id ddm_md3`); never invent flags — grep argparse (note: `--artifact-budget-gib` EXISTS on the launcher, line 1203); no `/tmp`
evidence; register a lane before lane-like identifiers; persist records before bulk saves; label MEASURED/DERIVED/INFERRED; state the n32 stratified-selection
caveat as md1/md2 did. `docs/operating_manual_craft_handoff.md` binds. End with `fs2 S 0.14784474152757654 @ 180,023 B [contest-CUDA T4 n600]`.
