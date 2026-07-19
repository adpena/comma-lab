# Duty-Queue Fire Tickets — Sealed Blocked Handoff

Date: 2026-07-19  
Lane: `duty_queue_fire_tickets_20260719`  
Authority SHA-256: `c8718f7d30b38d639d635b53578b4271a37986205ac2da0c1d7bc7fbd3cc6aa0`

## Verdict first

All four requested tickets were materialized in the required order, but all four are
`BLOCKED / NOT FIRE-READY`. The ticket `launch.sh` files are deliberate refusal wrappers;
they contain no trainer command and return rc=6. No trainer, governed dry-run, GPU, paid
dispatch, evaluator, or score path ran.

The exact pointer remains **`0.1910828242 [contest-CPU Linux x86_64]` UNMOVED**. This landing
is MEANS. The isolated-branch commit requires independent MAIN landing review and does not
authorize a launch.

| order | ticket | duty prior | sealed artifact | verdict | decisive blocker |
|---:|---|---:|---|---|---|
| 1 | `DsegAwareTaper` | 78.9% | `01_dseg_aware_taper` | `BLOCKED` | The canonical treatment removes the structural epoch-0 Lever from the ON control; removing it only after an ep725 warm start is not the same contrast. |
| 2 | `HorizonWeightedMargin` | 47.3% | `02_horizon_weighted_margin` | `BLOCKED` | The 0.15 loss-share declaration is DERIVED-LIVE but no ep725 boundary receipt resolves its nonzero weight or the required measured `MarginStepCap`. |
| 3 | `StepNativeActivation` | 34.2% | `03_step_native_activation` | `BLOCKED` | The treatment changes HOSC endpoint/schedule at ep725 without a matching #518 reanchor, v0-position, or response-window receipt. |
| 4 | #497 curvelet matched bytes | n/a | `04_curvelet_matched_bytes_p0_497` | `BLOCKED` | Current pure compiles are clean, but the existing wrapper lacks current paired custody, resume/#518 proof, one-factor isolation, powered verdict rules, and enforced equal-byte completion. |

## Custody and current-source truth

MEASURED from the defensive bank, read-only:

- BEST EMA: 460,448 bytes, SHA-256
  `b0a431e9259cd3c54ae53b677076823f36e096b27eb0d9ba74ed7c54c9113cef`.
- `levelset_best.json`: epoch 725, `d_seg=0.003457972208658854`, SHA-256
  `66950d15bf8575eb4bae28c39c0276b0960f8e14364ed45da24b30b960f27d9e`.
- `PROVENANCE.md`: SHA-256
  `80422b7f88fd31abbe9362e95d9e0b95ac6136084c254739c1f2f19b33df9d42`.
- The bytes and sidecar identity match the requested checkpoint. They do not prove optimizer/RNG
  state, BEST-boundary persistence, or arm comparability. The fork must therefore be
  weights-only with a fresh optimizer, but no full fork configuration is authorized here.

The worktree source base is `920fda3e5a847cd800b0a8aed19eb12356275f3e`. At the artifact's
`2026-07-19T15:01:05Z` snapshot label, MAIN resolved to
`e7b35814a8eea576fdc45119773ec8d9e3f89bac`; MAIN was a descendant and all 15 enumerated
authority inputs matched four ways: source checkout, source-HEAD blob, MAIN checkout, and
MAIN-HEAD blob. The machine-readable receipt is
`duty_queue_fire_tickets_20260719/main_source_custody.json`, SHA-256
`b5003f59201638b8e0cb1e1b7eb24900a184e48235e389aadea51986919c5c40`. It covers the
factories, trainer, #497, launcher/governor/safety/readiness surfaces, NCDE source, both beta2
laws, #518 memo, and frontier pointer. This is a sealed snapshot, not a claim that MAIN cannot
change afterward; MAIN landing review must revalidate drift.

## Tickets 1–3: typed treatment declarations, not fake full compiles

The three factories and trainer consumers are present. Every emitted flag is accepted by the
real trainer parser and has a source consumer hit. Each ticket records exactly one signed
control-to-treatment contrast anchored by a scientific `Lever` declaration, structured
LawRefs/constant manifests, and a canonical treatment-delta hash. For taper, the factory
declaration belongs to the ON control and the treatment removes the complete Lever:

| factory source | typed signed-contrast SHA-256 | existing V9 isolation declaration |
|---|---|---|
| `curriculum_dsl.py:2783` `DsegAwareTaper` | `13df7b9396dba63df0699183fd6c000f30989a14b75f5108d13579be40ec8f43` | ON control has strength 1.0, AUTO scale 0.0, floor 0.05; treatment removes all four taper flags/LawRefs |
| `curriculum_dsl.py:5026` `HorizonWeightedMargin` | `eb51f3e9747f25f43fe4e5fa918ba4b17e5cd0155da3bd10bc15717ed279c197` | treatment adds 0.15 derived-live share request, target 0.5, GT-margin [0.3,0.5], rows [96,288), boundary 726 |
| `curriculum_dsl.py:5423` `StepNativeActivation` | `c26f9e49bc23224a01a7b03f0fe6c17a89847acd43e1acaa719151b16c4e1748` | treatment changes only canonical signed argv delta HOSC beta-end 3.177→8.0 |

These are the already-declared V9 ISO parameters from
`compile_v9_cgauge_432_iso_launch_config`, not new constants. The existing ISO programs are
fresh mod19/3000-epoch programs, not ep725 c2/#518 forks. Therefore the ticket
`full_dsl_compile_hash` and argv are deliberately null: emitting a launch hash for an
uncomposed ep725 fork would grant false authority.

## #518 and control-arm comparability audit

The bank predates #518. It cannot be the control for a treatment fork that engages the #518
recipe. All three tickets select a **fresh paired OFF twin** from the same checkpoint, with the
same #518 recipe, seed, data order, observer cadence, checkpoint cadence, and treatment lever
OFF. No banked-control exception is claimed.

Current geometry state:

- DERIVED-PROVISIONAL beta2 window:
  `ceil(2 / ((1 - 0.999) * 75)) = 27 epochs = 2,025 optimizer steps`; the 8-vs-27 efficacy
  A/B remains owed.
- `ForkHeadSolve`: wired and default-off; no arm receipt proves engagement.
- `MarginStepCap`: wired and default-off; no measured/provenance-backed cap exists.
- v0 schedule positioning and resume-event reanchor: executable surfaces exist, but boundary
  state persistence/historical-event filtering and a matching ep725 receipt remain open.
- Base n600 memory projection: 71.54 GiB against an 89.6 GiB safe ceiling. This is not a
  per-arm certification because the final arms do not compile.
- `safe_run` is present. Per-arm memory, config-freshness, resumability, preserved stage
  checkpoints, and full dry-start receipts are absent and remain rc6-blocking prerequisites.

Catalog #506 apparatus hashes are preserved in every compiled-config audit. Key values are:

- launcher `05fddae253afac7ef6242f26d1e82219ea93b363d7391ad7b17b59c32e65202d`
- governor `26bcad63211c6a7c872c42ca0877617d995e707a7419738b5f97135bb28b528e`
- safe-run `a951ab7ec3baef5137f18472698a58b9f9864902f63f33bd7bbfa89294e3766b`
- readiness gate `30cb6152833d979b3cf58029ff3dc18d407d7078b16080a3195fd54029e25a0f`

## Window and verdict criteria: intentionally unsealed

MEASURED non-authorizing trajectory context:

- costate slope: `-0.0001280414014 S/epoch`;
- 95%-style half-width: `2.5222969e-5 S/epoch`;
- 25-epoch projected half-width: `0.0006305742 S`;
- only five points over epochs 975–1075;
- NCDE verdict-fit `R²=0.06020`, below the required 0.5.

This is single-trajectory error, not paired ON−OFF joint noise. The apparent NCDE response
time is unstable and is on `log_total`, not authoritative `d_seg`. Cadence yields only
observation floors of 127 epochs (`27+100`) or 302 epochs (`27+275`), not powered treatment
windows. Because lever response times, predicted term shares, and paired noise are not jointly
custodied, `FIRED-PAYS`, `FIRED-NEUTRAL`, and `FIRED-HURTS` remain `UNSEALED/BLOCKED` for all
three treatments. No threshold was invented from the duty percentages.

## Per-ticket confound contract

Each package carries the operator-amended confound audit. Before any future fire, both arms
must prove:

- identical checkpoint/observer cadence, avoiding the measured ~27-min/epoch checkpoint
  contamination class;
- `ep_loss > 0`, armed spike guard, and a deterministic positive-control sentinel;
- `lever_engage` plus treatment term-share binding, not flag presence alone;
- explicit live-vs-EMA verdict source and EMA-lag handling;
- resume events reanchored to resume epoch and geometry;
- n600 only, identical seed/RNG/data order, and a powered window against paired noise;
- resumable-from-disk state and distinct preserved end-of-stage checkpoints.

Every item is either evidenced or explicitly OPEN in each ticket. No OPEN item authorizes a
launch.

## Ticket 4: current #497 audit

MEASURED from pure production-equivalent compilation only; the fire script and launcher
`main()` were not invoked:

| arm | DSL compile hash | typed-config hash | canonical resolved-argv hash | schedule gate |
|---|---|---|---|---|
| control | `be96e7498b2f63d208187231d1f36c9b31a96dad1fd009b48dde9f147e35826a` | `36ee86bb385cffbdbd34d763676227d93c566ebc876e2017f2c1d9c63e630e6a` | `421a2855f20de332a184b85ac844124613743dae1d218852579da5fa8ea055d5` | rc0, 11 verdicts, 0 violations |
| treatment | `7ed4982087f723495ac8f8f2e41f6ac655655dab87749dd18bd294a24ce709a2` | `3de19c82df8ceb9b052b17d9ce063696d2fcc9d57114930c82d8c5d4ac978b26` | `4697f58c55a51ace52473ed02ac2d9230212ef5b814ae0cf27d41e6393d6e450` | rc0, 11 verdicts, 0 violations |

The audited script is 6,530 bytes, SHA-256
`cdfb05f4dbd0fc825cd6729f8bd2f71a47e1780af5164702213b00e575311bac`.
Positive static facts: both arms resolve at n600/3000 epochs, imports are present in the project
environment, operator-GO and governed dry-run routing exist, and the current schedule gate is
clean.

Fatal gaps: no warm-start/resume input; no #518 binding; basis plus AA/native-render changes make
the treatment more than one causal variable; only `front_end` telemetry is emitted; arm order,
mutual exclusion, completion, byte-close, equal-byte match, and finalize are documented but not
enforced by this wrapper; the c2 gate treats a missing run dir as quiescent and exposes
`--skip-c2-gate`; project-venv liveness inspection raises `PermissionError` in this sandbox while
the script catches only `ImportError`; current governed freshness/dry receipts and current arm
checkpoint products are absent. The historical arm directories contain manifests, not completed
training/byte-close/score products.

## Durable artifacts and verification

Root summary SHA-256: `e5573f9936e6f171367191f584796f028ea690f268eaf43755c0548d3a73241f`  
Root manifest SHA-256: `ec3c06a86f9b75bf12f2e1950997ab0a3efd779d50f2c0e7b8c7264b6b22c2e9`

Each ticket contains `launch.sh`, blocked compiled/audited config, provenance, a real refusal
receipt, verdict card, JSON/Markdown confound audit, and a byte/SHA artifact manifest. Tickets
1–3 additionally contain the typed treatment delta and explicitly unmaterialized
`witness_program.json`.

`GREEN_STATIC_REFUSAL` means only that the generated wrapper contains no trainer/subprocess
route and returned the expected rc=6. It never means trainer dry-start GREEN or launch-ready.

Verification completed before review:

- focused pytest: 4 passed;
- relevant DSL/lever/#518/hash suite: 113 passed;
- Python compile: passed;
- shell syntax and all four refusal wrappers: passed, rc=6 each;
- artifact-order, hash-manifest, checkpoint-custody, one-delta, paired-control, #497 hash, and
  no-launch assertions: passed;
- `git diff --check`: passed.

Independent fresh-eyes seal evidence is recorded separately in
`.omx/research/duty_queue_fire_tickets_20260719_review_receipt.json`. Absence of three
consecutive clean passes is itself a landing blocker.

## Operating-manual disposition and triality

The literal request for launch-ready tickets conflicts with the evidence required by the same
request. Following `docs/operating_manual_craft_handoff.md` §§1 and 7, the composed outcome is
four honest blocked tickets, not permissive scripts. Section 2 is implemented by separate
ticket packages plus independent review; §4 by re-deriving from factories, parser consumers,
actual checkpoint bytes, and current pure compiles; §5 by explicit MEASURED/DERIVED/OPEN labels;
§6 by resetting the clean-pass counter after every fix; and §8 by reporting MEANS and pointer
delta first.

Triality:

- DSL: three typed signed treatment contrasts, parser-consumer audits, and canonical delta
  hashes; taper direction is ON-control→whole-Lever removal; full fork compile hashes remain
  intentionally absent while blocked.
- DAG/apparatus: canonical task/lane status plus fail-closed ticket order and refusal edges.
- Equations: beta2 warmup law and noise arithmetic preserved; missing response/noise joins are
  blockers, never hardcoded constants.

## STORES CONSULTED

Delegated authority prompt; `CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`;
`docs/operating_manual_craft_handoff.md`; v7.5 SPEC §8; current lane/task registries; current
factory, trainer, #497, launcher/governor/readiness sources; defensive-bank checkpoint and
sidecars; #518 build memo; costate/NCDE artifacts; existing curvelet takeover/fire memos; MAIN
inbox amendment dated 2026-07-19T14:19:57Z.

## MAIN landing review required

MAIN must independently review the branch diff and specifically verify: (1) no launch authority
is hidden in any wrapper; (2) null full-compile hashes cannot pass #506; (3) checkpoint custody
still matches and the sealed MAIN-source snapshot has not drifted; (4) all OPEN confounds remain
fail-closed; and (5) no score/pointer claim is inferred from these artifacts.
