# Pointer status — UNCHANGED

`[contest-CPU Linux x86_64]` remains **0.1910828242** (`reports/latest.md:57-62`, archive
prefix `ad02b0124cbb`). The locally banked `0.1880443979880752` row is explicitly the
`lane_pr128_click_import_NONSUBMISSION_defensive_bank_20260712` architecture and has no
submitted PR; it is not promoted here. This audit launched no run, changed no score artifact,
and makes no pointer claim.

# Hostile NO-FAKE audit — live V9·CGauge stack

- **Lane:** `lane_v9_cgauge_fake_implementation_audit_20260714`
- **Mode:** `research_only=true`; read-only code/config/receipt re-derivation
- **Verdict time:** 2026-07-14 UTC
- **Surfaces actually traced:** the compiled `v9_cgauge_432` and `v9_cgauge_ideal` argv,
  `experiments/train_levelset_witness_realized_through_R_mlx.py`, the base RGB trainer,
  the level-set receiver twins, `src/tac/witness_dsl/`, `witness_autoconfig`, canonical
  equations, the R1 byte-close receipt, and the current pointer surfaces.
- **Method:** names and memos were treated as allegations. A claim passed only when the
  compiled argv, trainer producer, serialized payload, receiver consumer, and authority
  receipt formed the required path. Tests that only validate metadata shape were not accepted
  as behavioral proof.
- **Verdict-scope:** all basis performance negatives are **FORMULATION**-scoped to the
  bounded owed16 warm-start arms. True localized curvelet/shearlet families and fresh-start
  training remain open. No family is killed by this audit.

## Executive verdict

The live V9·CGauge stack contains **five confirmed claim/mechanism gaps**, of which three are
score-critical:

1. The `banked R1` degeneracy fallback selects and prints a marker but never imports, switches,
   or serializes the banked R1 twist. This is the canonical forbidden-class **#1** failure.
2. The live object named a generic `curvelet/shearlet` basis is a global polar bank of Fourier
   plane waves. It has orientation coverage, but no translated spatial windows, localized
   envelope, or actual curvelet/shearlet frame. This is forbidden-class **#7**.
3. `witness_autoconfig` still globally declares `self-orient curvelet basis (-48% exponent); ON`
   even though current v7.5.2/V9 compilation explicitly sets self-orient **OFF**, and `-48%` is
   only an n96 circular synthetic direct-partition result. This is **#7 + #8**.

Two additional confirmed gaps are high-value reproducibility/semantic debt: 43 of 44 active
V9 provenance-declared flags have no matching constants-manifest entry, and the one matched
432 flag (`hosc_beta_end`) records `10.0` while the live argv emits `3.177`; and the compiled
`--curriculum-event-triggered` flag is deliberately inert under unified tau. The latter is
honestly surfaced by the trainer, so it ranks below the hidden score-path gaps.

## Ranked confirmed fake ledger

Rank is `score relevance × severity`, not code aesthetics.

| Rank | Claim | What the code actually does | Class | Severity / score relevance | Two-landing remediation |
|---:|---|---|---|---|---|
| 1 | Degenerate or never-fired pose gate “ships/selects banked R1” | The current carrier is built from current calibration and its trainable table starts at zero (`warp_real_luma_frame0.py:535-548,646-710`). On degeneracy, `resolve_pose_finish_engage` returns `_banked_sel`, but the consumer only prints `backstop_banked_r1_row`; it does not load an R1 checkpoint, replace `xi_eff`, change a model parameter, or set an export selector (`train_levelset...py:11448-11492`). Current V9 argv has `--pose-carrier --pose-carrier-source generated --pose-carrier-residual-mode table`, but no byte-close `--pose-carrier-xi-from-ckpt`. The real R1 connector exists only in the separate byte-close CLI (`tools/levelset_byte_close_and_eval.py:3300-3313,3466-3497`). | **#1** canonical markers without work; **#7** name/mechanism | **CRITICAL.** The vehicle assumes the measured pose contribution `0.1269`; a non-firing gate can instead preserve current untrained/zero residual bytes while claiming the banked floor. | **L1 mechanism:** either rename to `pose_finish_disengaged_no_banked_payload` and fail closed, or add a receiver-closed selector that loads one compatible, complete R1 checkpoint/archive and records source hash/frame geometry. **L2 guard:** inject a distinct known R1 twist and assert model/export/archive SHA and decoded frame0 bytes change; refuse “banked selected” unless the exact selected payload survives parse-back and full-n600 authority is explicitly labeled. |
| 2 | `generic curvelet/shearlet bank`, `curvelet_directional_B`, “actual parametric polar grid” | `curvelet_directional_B` constructs only frequency columns `(f cos θ, f sin θ)` and `curvelet_feats` returns global `sin(2πX@B), cos(2πX@B)` (`lever_b_levelset_generator.py:136-184`). There is no translated location index, spatial window, scale-localized envelope, wedge window, or compact/decaying atom. Trainer and both receiver twins reproduce those same global plane waves (`train_levelset...py:4000-4008,4100-4111`; `archive_grammar.py:686-712`; `torch_levelset_inflate.py:102-127`). The bank is deterministic, GT-free, receiver-closed, multiscale and direction-covered—real properties—but it is not a genuine curvelet/shearlet frame. | **#7** borrowed/name mechanism not honored | **CRITICAL.** This sits directly on the open d_seg representation axis. Orientation coverage exists, but the advertised localized parabolic approximation benefit is not in the live forward. | **L1 mechanism:** rename current family to `polar_directional_fourier_plane_wave`, or wire the non-colliding Task #497 basis-family compiler to a genuinely localized equal-budget frame. **L2 guard:** a family may claim `curvelet` only if a structural test proves translated localization/windows and parabolic scale/orientation support; reject implementations reducible to global `X@B` sin/cos. Preserve train/inflate parity. |
| 3 | Global proven-base claim: `self-orient curvelet basis (-48% exponent); ON` | The stale claim is literal at `witness_autoconfig.py:399-407`. Current V9 calls `derive_crucible_v752_config(..., self_orient=False)` (`spec_v9_cgauge.py:349-366`), and autoconfig drops all self-orient flags and the directional lever (`witness_autoconfig.py:2925-2951,3030-3045`). Executable compile confirms `--self-orient`, `--n-dir-freqs`, `--freq-across`, and `--freq-along` are absent. The `-48%` source is n96 circular synthetic direct partition, explicitly not realized n600 (`witness_measured_findings_20260701.py:303-327`). Owed16 measured OFF `0.004244` vs along8 ON `0.004259` at ep675; along26 was `0.004286`, all warm-start/advisory (`owed16_bounded_ab_and_drystart_20260710.md:180-216,225-259`). | **#7** stale name/active-state claim; **#8** proxy/ancestor number presented on a live stack | **CRITICAL.** It falsely says the best-known d_seg basis lever is live and carries a 48% exponent, obscuring that the actual vehicle lacks that mechanism. | **L1 mechanism:** supersede the global proven-base row with vehicle-scoped state: `polar directional Fourier bank ON; self-orient OFF; -48% n96 direct-partition prior only`. Real localized basis work routes to Task #497. **L2 guard:** compile each named vehicle and require every `ON` claim to map to an emitted flag plus trainer-consumed feature-width delta; any `%` claim must cite axis, n, receipt, and domain of validity. |
| 4 | V9 “EVERY constant placed on the value-provenance ladder” and 432 “never changes LawRef constants” | `V9_CGAUGE_PROVENANCE` is a plain metadata dictionary. Executable re-count finds 45 rows, 44 active in each live argv, but only one active flag has a corresponding constants-manifest key: **43/44 lack executable manifest custody**. The 432 compiler blindly inherits the v7 manifest (`spec_v9_cgauge.py:675-712`): it emits `--hosc-beta-end 3.177` but records `hosc_beta_end.value=10.0`. The ideal compiler's `_derive_manifest_from_emitted_argv` repairs that one matched value only and explicitly leaves historical 432 untouched (`spec_v9_cgauge.py:758-796`). The provenance test checks only rung/form/law string shape (`test_spec_v9_cgauge.py:113-122`); it never resolves the laws or compares them to emitted argv. | **#2** constants/tests instead of behavior; **#1** canonical metadata without executable work | **HIGH.** A launch can carry materially false reproducibility custody. It may not directly lower S, but it invalidates causal attribution and resumption/replay claims. | **L1 mechanism:** make every active scientific flag compile from a `LawRef`/resolved constant into the same argv owner, or explicitly label non-law metadata; backfill 432 parity without silently rewriting historical receipts. **L2 guard:** strict coverage gate requires active provenance flags = emitted scientific flags with matching manifest value/type/equation; fail on 43 uncovered flags or `10.0 != 3.177`. Tests must mutate a law input and prove emitted behavior changes. |
| 5 | An active compiled `--curriculum-event-triggered` surface implies the curriculum controller fires | Both 432 and ideal argv co-emit `--curriculum-event-triggered` and `--seg-form-unify-tau`. Unified tau bypasses the discrete CE→tau→l7 dispatch (`train_levelset...py:2498-2523`), and the trainer explicitly logs `event_curriculum_inert_under_unify` (`train_levelset...py:15322-15350`). The independent `--tau-advance-mode event` controller is real (`train_levelset...py:3370-3421,11784-11846`) and must not be conflated with the inert discrete curriculum flag. | **#7** active-name/mechanism gap, narrowly scoped | **MEDIUM.** A stale active flag can misroute analysis, but the trainer is honest and loud and the live tau controller still works. | **L1 mechanism:** on new lineages remove the inert discrete flag, or rename/persist it as a legacy resume-compatibility marker outside the active-lever surface. **L2 guard:** compile-time receiver-consumption bijection refuses any active flag the selected loss family classifies inert; legacy resumes may carry it only with an explicit `inert_compat` status. |
| 6 | “n600 authority” / “authority-scale” for the R1 pose receipt can be read as exact contest authority | The R1 mechanism itself is real: the receipt measured all 600 decoded pairs, `d_pose=0.001609547`, `d_seg=0.004549120`, archive `89,772 B`, and exact frame0 carrier parity (`reports/r1_dxi_238/n600_shipdxi.json:150-186`). But that same receipt says `[macOS-CPU advisory] NON-PROMOTABLE`, `promotion_claim=false`, and `exact_eval_upstream_evaluate.ran=false` (`:1-5,162-164,189-197`). The memo says `n600 authority`/`authority-scale` while also qualifying the axis (`r1_dxi_shippability_byteclose_20260708.md:10-13,93-114`). | **#8** exact-vs-surrogate/axis overstatement | **MEDIUM.** The bytes and advisory score are hard-earned; only the authority word is wrong. More importantly, that real R1 artifact is not imported by current V9 (rank 1). | **L1 mechanism:** rename to `full-n600 byte-closed macOS-CPU advisory`; reserve `authority` for exact `upstream/evaluate.py` on a declared contest axis. **L2 guard:** schema rejects `authority=true` or authority wording when `exact_eval_upstream_evaluate.ran != true` or hardware is advisory; transfer claims must also name the consuming config/archive SHA. |

## Suspected / one check still owed

These are not promoted to confirmed fakes.

| Suspected claim | What is known | Exact missing check / verdict scope |
|---|---|---|
| A genuine localized basis will improve V9 d_seg | Task #497 has now landed a family catalog/compiler, but no genuine different-frame equal-budget real-n600 row exists. | Fresh-start, equal-budget, receiver-closed n600 A/B through R using `metric_id=argmax_native_vjp_fidelity_v1`; full-n600 metric selection currently remains `NO-VERDICT_DATA_CUSTODY`. Family remains open. |
| Banked R1 pose transfers cleanly into a current V9 trunk | R1's own checkpoint/archive path is real on the advisory axis. Current V9 differs in trunk/basis/mod-dim and does not import it. | Build a compatibility-checked whole-payload selector and exact parse-back A/B. Do not infer transfer from identical field names or output shape. |
| V9's 43 uncovered provenance rows all need LawRefs | They are active declared constants, but some may be honest configuration metadata rather than scientific constants. | Classify each row as executable law-bound constant or non-law metadata. The current blanket “EVERY constant” claim fails either way; do not manufacture LawRefs for bookkeeping values. |

## Active-lever receiver-consumption audit

The exhaustive sweep did **not** find a pile of inert named levers. The following mechanisms are
implemented and consumed. These are clears, not findings:

| V9 active lever | Actual trainer consumer | Audit verdict |
|---|---|---|
| `seg_form_unify_tau` | unified loss/validation at `train_levelset...py:2498-2523` and live loss path | **REAL**; only the inherited discrete curriculum flag is inert |
| `tail_k_warm_restart` | tail restart resolution at `:10820-10865` | **REAL** |
| `n323_ladder_island_homotopy` | mask preparation at `:5817-5917`, loss at `:6445-6466` | **REAL** |
| `R7_polyak_finisher` | live averaging/candidate logic at `:7946-7966` | **REAL** |
| `v75_area_constraint_birth` | setup `:5926-5953`, loss `:6467-6487` | **REAL** |
| `v75_birth_completion_event` | controller/setup `:5967-6016`, observation `:8813-8818`, applied multipliers `:11684-11712`, resume sidecar `:10661-10668` | **REAL**; an older observability comment is superseded by the actual ramp consumer |
| `n287_dash_comb` | composed lane render at `:5079-5095` | **REAL** |
| `temporal_screw_consistency` | params/loss/gate at `:5460-5476,6371-6414,11524+` | **REAL** |
| `pose_finish_conditioning_gate` | detector and engagement logic at `:8407-8469,11448-11523` | **REAL gate; FAKE fallback payload selection** |
| `phase_advection_consistency` | params/loss/engage at `:5480-5535,6415-6433,11433-11447` | **REAL** |
| ideal `unified_tau_eikonal_hold` | live tau-rung coupling at `:11784-11846` | **REAL** |
| ideal `n292_closed_loop_eikonal_control` | closed-loop controller path beginning `:8773` | **REAL** |
| ideal `R7_beta2_window_rewarmup` | optimizer reset/LR boundary at `:11808-11835` | **REAL** |
| ideal `FEED_08a_length_sigma` | configured at `:5615-5617`, consumed in loss | **REAL** |
| ideal `tie_locus_displacement` | params/loss at `:5411-5458,6295-6308` | **REAL** |
| ideal `margin_band_satisficing` | params/loss at `:5537-5552,6334-6351` | **REAL** |

## Measured-number custody table

| Number/claim | Actual grade | Audit disposition |
|---|---|---|
| `-48%` directional basis | **MEASURED** n96 circular synthetic direct-partition; not real-n600 realized V9 (`witness_measured_findings_20260701.py:303-327`) | **FAKE only when presented as the live ON effect.** Canonical equation itself is now honestly scoped. |
| owed16 OFF `0.004244`, along8 `0.004259`, along26 `0.004286` | **MEASURED** through-R n600 `[macOS-CPU advisory]`, bounded warm-start, single seed (`owed16...md:180-216,225-259`) | Valid formulation evidence; fresh-start/localized family remains open. |
| R1 `d_pose=0.001609547`, pose term `0.126868` | **MEASURED** on all 600 inflated pairs, byte-closed `[macOS-CPU advisory]`; no exact upstream eval (`n600_shipdxi.json:150-197`) | Mechanism real on its own R1 artifact; **not** automatically current V9's shipped floor. |
| R1 pose section `7,195 B`; archive `89,772 B` | **MEASURED** exact bytes for the R1 artifact (`r1_dxi...md:69-79`) | Real. “about 7.2 KB” is honest for that R1 section only. |
| chroma `7.54%`, `4.38%`, `93.4%` | **MEASURED removal/WORTH diagnostic**, not add-back gain (`chroma_rung_design_20260710.md:35-48,197`) | Honest where currently used; no V9 treatment-effect claim admitted. |
| `12.5–51×` family | Search resolves these to older v8/nscs06 sample-count or n8→n600 diagnostic ratios, not a current V9 exact score claim | Not consumed as V9 authority; **CLEARED** from this ledger. |
| `0.1910828242` | Exact registered `[contest-CPU Linux x86_64]` current submittable pointer | Authority pointer; unchanged. |
| `0.1880443979880752` | Exact local defensive-bank record imported from open PR128; no submitted PR for the current frontier pointer | Real local custody, but **NON-SUBMISSION** and not promoted. |

## Honest mechanisms explicitly cleared

- The base RGB trainer really uses a fixed random isotropic Fourier matrix
  (`train_witness_realized_through_R_mlx.py:313-318,455-503`). Current level-set V9 does **not**:
  it uses the deterministic polar directional plane-wave bank. The audit corrects the trigger's
  over-broad wording instead of repeating it.
- The level-set trunk is a real task-space/SDF latent that intentionally renders RGB through
  palette/texture. Producing RGB for the frozen scorer does not make the task-space latent fake.
- The pose carrier, its joint gradient, current-checkpoint byte-close connector, and receiver path
  are real (`train_levelset...py:4858-4933,5238-5288,7325-7373`; byte-close CLI
  `:3300-3313,3466-3497`). Only the alleged **banked-R1 fallback transfer** is fake.
- Chroma boundary loss is consumed on the shared through-R render. Its mechanism is real; its V9
  add-back score gain remains unmeasured.
- The typed witness DSL compiles actual trainer argv and parses on the real parser. It is a real
  compiler, not forbidden-class #6 search masquerading as a solver.
- V8 per-class carriers are a gated child design, not a claimed current V9 mechanism. No current-V9
  fake is manufactured from a future design.

## Triality and routing

- **DSL leg:** supersede stale active-state/basis/provenance claims; bind scientific values to
  executable `LawRef`s; make receiver consumption a compile-time bijection.
- **DAG leg:** standalone feed is
  `.omx/research/v9_cgauge_fake_implementation_audit_DAG_FEED_20260714.md`.
- **Equation leg:** retain `curvelet_directional_basis_dseg_reduction_v1` only with its current
  n96/warm-start domains. Equal-budget selection routes through canonical
  `metric_id=argmax_native_vjp_fidelity_v1`; receipt schema
  `reachable_decision_geometry_fidelity.v1`; selector schema
  `reachable_decision_preconditioner_selection.v1`; candidate
  `winner_rival_margin_fisher_natural`.
- **Ownership:** Task #497 owns the real-basis catalog/compiler/probe. The #500 metric arms own the
  metric implementation and curriculum surface. This audit owns the round-1 hostile review and
  does not duplicate sibling code.

## Stores consulted

`CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`; `docs/operating_manual_craft_handoff.md`;
`docs/vehicle_operating_system.md`; v7.5/v8 canonical SPECs; `reports/latest.md`;
`.omx/state/canonical_frontier_pointer.json`; lane registry; subagent progress; canonical equations;
V9 configs/memos; current trainers and receivers; R1 JSON/memo; latest sister findings/design/council
memos; last-24h directives; both live inboxes through 2026-07-14T13:08:14Z.

## Round-1 self-review

- Re-counted executable provenance after the initial hypothesis: exact result is **43 of 44 active
  provenance flags uncovered**, not 44 of 44. The receipt preserves the corrected number.
- Separated the real current polar directional bank from the base trainer's random isotropic bank.
- Separated real pose carrier/byte-close machinery from the telemetry-only fallback selector.
- Separated the inert discrete curriculum flag from the live event tau controller.
- Scoped every basis negative to the measured formulation and kept the optimal localized/fresh-start
  reformulation queue open.
- No code was changed, no heavy job was launched, and no advisory number was promoted.

