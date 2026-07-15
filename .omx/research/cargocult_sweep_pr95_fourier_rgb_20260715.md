# Three-family cargo-cult sweep — PR95 · Fourier · RGB — LIVE stack — 2026-07-15

Task: #508 (operator 2026-07-15 verbatim: *"Also need to sweep for more pr95 and Fourier and RGB
cargo cult, RGB only has place as a finisher at the edges and annulus and longest end of long tail
or other optimal formulation and application."*)
Doctrine: `rgb_only_finisher_edges_annulus_longtail_cargocult_sweep_20260715.md`.
Scope: the LIVE stack only — `spec_v9_cgauge.py` (v9_cgauge_ideal_mod19 chain) +
`spec_c1_optimal_form_20260715.py` + `curriculum_dsl` levers + the levelset trainer
(`experiments/train_levelset_witness_realized_through_R_mlx.py`) — via the COMPILED c1 argv
(the actual launch surface), not the historical archive.
Authority: `$0` source/compile inspection; research-only; no score claim; pointer
0.19108 [contest-CPU] / 0.18804 [borrowed bank, non-submission] UNMOVED.

## Stores consulted (recall-first)

- `tools/graph_memory_recall.py "PR95 curriculum cargo cult blinded structural derivation l7 defect hosc beta"`
  → elementwise-audits-launder-structural-cargocult (#302 derivation link), curriculum-still-PR95,
  ancestor-vehicle, don't-rerun-PR95-reskinned.
- Memories: `fake_curvelet_basis_is_isotropic_fourier_in_v9_cgauge_20260714`,
  `no_fourier_basis_DAG_FEED_20260715` (via MEMORY.md hook), `constants_are_poison_…_20260715`,
  `negcure_join_second_exemplars_msal_sR_and_hosc_betaanneal_20260710`.
- Prior sweep (do-not-re-derive): `.omx/research/codex_findings_rgb_cargocult_scrutiny_optimal_replacement_20260714_codex.md`
  (16-row formulation classification) + its DAG FEED (`FEED-500-rgb-cargocult-taskspace-replacements`).
  THIS sweep's delta over 07-14: which of those formulations are **ACTIVE in the compiled launch
  argv**, plus the PR95/Fourier families the 07-14 arm did not own.
- Live argv obtained by compiling `compile_c1_optimal_form_launch_config()` ($0, pure).

## Verdict scope

Every negative below is scoped on the ladder (instance < formulation < family < paradigm). No family
closure and no exact-eval verdict is made here. "ACTIVE" = present in the compiled c1 argv or an
argparse default the compiled argv silently inherits.

## Findings table

| # | Family | Surface (file:line) | Finding | Severity | Re-scoped form | Route | verdict_scope |
|---|---|---|---|---|---|---|---|
| P1 | PR95 | trainer:2409 + `--l7-start-epoch` default 800 (trainer:13599), `--seg-loss` default `ce` | The PR95 chain `ce → tau_softplus → l7_softplus` is still the trainer-DEFAULT curriculum; l7 is the CLAUDE.md-documented measured DEFECT ("L∞ sharpening inside a viscosity flow"). The live config avoids it ONLY via explicit `--seg-form-unify-tau` (constant continuous-τ form; trainer:2617). One dropped flag re-activates the defect stage silently at ep800. | hygiene (score-risk on config drift) | default `l7_start_epoch` → disabled sentinel (≥epochs) requiring explicit opt-in, or a fail-loud banner when the default chain schedules l7 without an explicit `--l7-start-epoch` | reformulation-queue (trainer-default change = behavior-affecting for other configs; not a same-turn patch) | formulation (l7-as-default; the l7-defect verdict itself is MEASURED) |
| P2 | PR95 | levelset trainer:13690 (`--l7-threshold` default 1.0) vs base trainer:3153 (default 0.42) | Twin-parser inherited-constant divergence on an l7 constant; inert in the live config (unify_tau). | hygiene | one owner (DSL LawRef) if l7 is ever re-opened; else delete with the l7 demotion | queue (ride P1) | instance |
| P3 | PR95 | live argv `--tau-softplus-tau 0.3` + trainer:5506/6259 | Under `--seg-form-unify-tau` the live τ is COUPLED to the render softmax-temp; the emitted 0.3 is a fallback that the unify path supersedes — an emitted-but-superseded PR95-named constant (#417-adjacent custody smell, not an inert-lever fake). | hygiene | constants-manifest note `superseded_under_unify_tau` (constants-are-poison audit row) | queue → the Fable v9-signal constants audit (2026-07-15 arm) | instance |
| P4 | PR95 | live argv (verified-clean) | hosc β custody honest: `--hosc-beta 1.0 --hosc-beta-end 3.177 --hosc-beta-anneal linear` (annealed, NOT fixed β=4, NOT the inherited 10.0; spec_v9_cgauge:796 documents the V7 manifest-drift catch). Curriculum is the witness-native EVENT-GATED schedule (lane_nucleus / annulus_plateau / powerlaw_meat / birth-completion / sigma_min_plateau) per the #302 derivation — NOT the PR95 8-stage skeleton. Muon retained per L78 with warm-start momentum + lr-final 0.1. | — (clean) | — | — | — |
| F1 | Fourier | spec_v9_cgauge `_V9_CGAUGE_DELTA` (was: absent) → trainer:13436/13454 defaults | The basis family — the #1 measured lever family — rode trainer ARGPARSE DEFAULTS: the compiled argv carried neither `--basis` nor `--self-orient`, violating the spec's own principle ("trainer-side argparse defaults are not a scientific configuration owner") and the "off is a tracked queue" law. The legacy-Fourier default is DELIBERATE (no-Fourier doctrine: A/B control, never a silent curvelet flip) but was an UNTRACKED silent state. | score-relevant custody (behavior-identical values) | **FIXED this sweep**: explicit `--basis legacy_fourier_ab_control` + `--no-self-orient` in `_V9_CGAUGE_DELTA` with V9_CGAUGE_PROVENANCE rows citing the A/B-control doctrine + the owed n600 through-R curvelet A/B | immediate fix (landed; argv value == prior default ⇒ training behavior identical) | — |
| F2 | Fourier | `witness_autoconfig.py:403` (FIXED) · trainer:12824 parser description "softmax-of-SDF + curvelet" · `curvelet_feats`/`curvelet_directional_B` names · law id `cgauge_curvelet_parabolic_bank_v1` · manifest key `curvelet_cols` | NO-FAKE #7 naming residue: "curvelet" labels on the legacy directional-FOURIER bank (per `fake_curvelet_basis_…_20260714` the remediation was ordered 07-14 but the autoconfig label survived). | hygiene (NO-FAKE label honesty) | honest mechanism names; law-id rename is APPEND-ONLY custody (supersession row, not mutation) | autoconfig label **FIXED**; trainer-description + function/law renames QUEUED (trainer edit deferred: a dry-start run is live and source-hash provenance binds the trainer file) | — |
| F3 | Fourier | live argv `--dseg-aware-taper` (strength 1.0, scale auto) | The #121 Fourier-column amplitude taper is ACTIVE at launch while its own ledger row is RE-VALIDATE-AT-CONVERGENCE ("+18% NO-GO RETRACTED (under-converged); converged anchors flip sign to −8%"; module docstring says ASSUMED_AWAITING_VERIFICATION). Annulus-DIRECTED (margin-saliency reweight → scope-compliant), byte-neutral, rule-118 free — but the ON-state rests on an owed converged A/B. | watch (owed measurement, not a fake) | the owed converged byte-close n600 A/B row | queue (A/B instrument already named in the lever ledger) | instance |
| F4 | Fourier | `spec_c1_optimal_form_20260715.py` + `basis_control.py` + trainer:4074-4159 (verified-clean) | c1's curvelet handling is doctrine-compliant: typed SLOT, receipt-gated fold (`curvelet_optimal_form_receipt`, fail-closed on a missing path), never a silent default flip; genuine `windowed_curvelet`/`compact_shearlet` exist opt-in with numpy-authority MLX parity gates; the honest name `legacy_fourier_ab_control` is the runtime id. No Fourier re-introduction found in the new #507 module. | — (clean) | — | — | — |
| R1 | RGB | trainer:13426 (`--palette-anchor` default=True) + live argv emits it | Mean-GT-RGB per-class palette INIT is live-active. 07-14 rank-3 classification: CARGO-CULTED — INITIALIZER OPTIMALITY (the empirical base is mean-RGB > one weak luma ramp, not task-optimality). Palette shape (K,3) + final RGB render are hard constraints; the CENTROID objective is not. | score-relevant (H3) | decision-optimal 15-D palette solve in through-R winner/rival geometry with Pose-6 trust (07-14 program 2); pre-registered $0 measurement: finite-difference all 15 palette coords through exact R + frozen scorers vs the mean-RGB init | reformulation-queue (07-14 H3 row re-affirmed as LIVE-ACTIVE) | formulation |
| R2 | RGB | live argv `--seg-chroma-boundary-weight 0.1 --seg-chroma-boundary-margin-band 1.0 --seg-chroma-boundary-start-event annulus_plateau` | GT-chroma match is annulus-SCOPED (margin band) + event-GATED (fires at annulus_plateau) — an existing, correctly-scoped RGB-as-finisher-at-the-annulus instance under the new doctrine. The Euclidean GT-chroma METRIC remains the 07-14 rank-4 queued reformulation (luma-null chroma direction projected through the decision Jacobian). | scope COMPLIANT; metric queued (H4) | keep scope; replace metric with the decision-projected chroma direction once the sibling metric receipt lands | queue (07-14 H4, unchanged) | formulation (metric only) |
| R3 | RGB | live argv (verified-clean) | NO full-frame RGB reconstruction/supervision anywhere in the compiled launch surface: no pixel-MSE tether flag emitted or default-on; texture trunk absent (L87 honored, `--texture-trunk` store_true default False); UNIWARD RGB-texture prior REPLACED by the S_R through-R reachability treatment (`--margin-saliency-reachability`, leg-A — the L76 cure); the appearance-phase advection term is the LONG-TAIL finisher instance (ep726 terminal band, gt_advected ref); `--structured-init` consumes cached L* argmax (task-space, not RGB); pose rides the dxi carrier (banked R1 shape), not an RGB/luma INR. The live vehicle IS the non-RGB task-space witness with RGB only at the render boundary + the annulus/long-tail finishers. | — (clean; the doctrine's positive checks all pass) | — | — | — |
| R4 | RGB | `witness_autoconfig.py` `lever_priors.deferred_levers["uniward"]` | Stale advisory note still advertises the UNIWARD texture-cost warm-start although msal_uni was MEASURED INERT (at chance) vs through-R S_R (L76). Notes-only; the lever is deferred, not active. | hygiene | annotate the deferred row with the L76 inert verdict + point at S_R as the replacement | queue (notes fix, ride the next autoconfig touch) | instance |
| A1 | apparatus (pre-existing, discovered; SISTER FIX IN-FLIGHT) | `src/tac/tests/test_dsl_compile_hash_enforcement.py` (1 failed + 7 errors at clean HEAD) via `v9_provenance_gates.build_dsl_compile_provenance_document` | The #332 DSL-compile self-recompile gate REFUSES the live `v9_cgauge_ideal` config at HEAD ("reconstructed #332 flag-to-Lever bijection manifest differs"). Root cause MEASURED independently here: lever-carried `lawref_equation_ids` (the 4 DsegAwareTaper flags + `--seg-margin-satisfice-msafe`) do not survive the `model_dump(json)` → `model_validate` round-trip — reconstructed Levers lose their `constant_refs` LawRef custody. CONVERGENCE: the blocked sibling arm `p0_merge_to_main_20260715` (its session had no writable `.git`) left the FIX uncommitted in the shared working tree while this sweep ran — a symmetric `lawref_to_declaration`/`lawref_from_declaration` codec in `lawref.py`, `TypedLever.to_dsl()` rehydration in `typed_config.py`, one-authority record canonicalization in `v9_provenance_gates.py`, and the `_typed_ideal_lever` refactor in `spec_v9_cgauge.py`. Verified on the composed tree: `test_dsl_compile_hash_enforcement` 32 passed. | P0-apparatus | (sister fix matches the measured root cause exactly) | LANDED mid-sweep by the merge-reconciliation arm (commit `fef6d3cc2b`, "dsl_compile_hash round-trip CLOSED": lawref codec + typed_config rehydration + one-authority canonicalization); its companion `_typed_ideal_lever` codec-consumer hunk in `spec_v9_cgauge.py` + the msafe golden hunks in `test_v9_provenance_gates.py` were left in-tree and CO-LAND in this sweep's commit (shared files, both changesets attributed in the commit body; reviewed here as the second pair of eyes) | n/a |
| A2 | apparatus (pre-existing) | `src/tac/tests/test_typed_launcher_dsl_composition.py` — 4 failures at clean HEAD AND on the composed tree (identical set) | epochs=5 smoke compositions trip the 07-13 curriculum epoch-budget feasibility guard (start-epochs 300-800 > epochs=5). Corroborates P1: the violation message itself lists the argparse-default `--l7-start-epoch=800` leaking into feasibility accounting. | hygiene (test-fixture reconciliation owed to the epoch-budget-guard owner) | fixture epochs or guard exemption for boot-runnability smoke compiles | queue (not this sweep's surface) | n/a |

## Fixes landed this sweep (small + safe, behavior-identical)

1. `src/tac/witness_dsl/spec_v9_cgauge.py` — `_V9_CGAUGE_DELTA` + `V9_CGAUGE_PROVENANCE`: explicit
   `--basis legacy_fourier_ab_control` + `--self-orient False` (compiles to `--no-self-orient`).
   Values equal the prior argparse defaults ⇒ the trainer behavior is IDENTICAL; the change is
   custody-only (the basis family becomes a typed, reasoned, A/B-control-labelled state).
2. `src/tac/witness_autoconfig.py:403` — the NO-FAKE label residue: "self-orient curvelet basis" →
   "self-orient directional-FOURIER feats … NOT a curvelet frame" (notes-only).
3. `src/tac/tests/test_v9_provenance_gates.py` — bijection-collector golden counts
   `[199,224,224,224] → [201,226,226,226]` with an amendment comment (the documented amendment
   pattern already used for the C1a +5).

Verification: v9 + c1 compile clean with the new custody flags; targeted suites
`test_spec_v9_cgauge` + `test_spec_c1_optimal_form_20260715` + `test_v9_provenance_gates` +
`test_witness_cloud_launcher` + `test_witness_autoconfig` = 273 passed; ruff `--select F` clean on
all touched files (the 25 style findings on those files pre-exist unchanged);
`test_dsl_compile_hash_enforcement` failures confirmed PRE-EXISTING via stash/pop on clean HEAD (A1).

## Six-hook wire-in

1. Sensitivity map: no change (custody-only edits).
2. Pareto: no change; exact (d_seg, d_pose, bytes) remains terminal.
3. Bit allocator: no change.
4. Cathedral autopilot: findings routed via this memo + the DAG FEED; no new consumer.
5. Continual learning: this memo + the standing-sweep doctrine memory are the anchors; A1 root cause
   recorded for the apparatus owner.
6. Probe disambiguator: R1 (palette) and F3 (taper) each carry a pre-registered $0/owed measurement;
   no new tool needed (07-14 programs + the lever ledger's own A/B instrument).

## Pointer-delta honesty

No candidate archive, no exact evaluation, no pointer movement. Submittable pointer 0.19108282
[contest-CPU Linux x86_64] UNMOVED; 0.18804 remains the borrowed non-submission bank row. Everything
above is custody/hygiene/routing — MEANS.
