# FRESH SEEDED RUN — pre-launch SEAL round: recursive adversarial review of the config + control system

**2026-07-04. Operator: "a recursive round of adversarial review and double check the deep math and
engineering of the config for optimality based on everything we have learned." Role: ADVERSARY (refute,
not bless). $0; no launch; #205 (pid 29129) untouched/read-only. Pointer contest-CPU 0.19110 UNMOVED —
everything here is MEANS.**

Config under review: #205 `launch.sh` base + the 13 ledger changes (paint / seed-islands / eikonal
0.05→0.10 step / geometric+constant τ=1.0 / mod-19 / bank-6 / film-stiefel / muon warm-start+final-frac
0.1 / band 350 / rewarmup 20-cosine / event-triggered curriculum / closed-loop control). Trainer at HEAD
`3e3b9c697` (includes control-system builds 2a125ab62 / 2bf4ac94f / 9da07aa34 + the eikonal-tracks-fired-tau
SEAL fix). Review artifact argv: scratchpad `fresh_seeded_launch.sh` (session-local; not committed).

## FINAL VERDICT: **REVISE** (5 CRITICAL, 5 MED, 4 LOW; core seed/geometry levers verified sound)

The seed-fix core (paint + seed-islands + dilate-1 + eikonal step + constant τ + mod-19 + stiefel + muon
tuning + band-350 + rewarmup-20-cosine + closed-loop) survives adversarial attack. The TWO control-system
curriculum additions and the bank-6 memory footprint do NOT survive it — with receipts below, every claim
executed or code-cited.

## CRITICAL findings

| # | Finding | Evidence (executed / cited) | Fix |
|---|---|---|---|
| C1 | **`--curriculum-event-triggered` at defaults fires CE→tau ~ep150 while d_seg is still descending — MEASURED against #205's own CE log.** The trigger is \|rel slope\| ≤ `--curriculum-plateau-rel-eps` **1e-3** over 4 ep_loss points past min-stage 150. #205 CE-phase measured rel slopes (25-ep averaged, run.log): ep100→125 −1.14e-3, **ep125→150 −8.22e-4 (< 1e-3 ⟹ FIRES at ~ep151)**, while d_seg was 0.005473@150 still descending to 0.004752@300 (**15% CE-floor loss**) and `--persistence-warmup-epochs 300` protection would be at **50%** at MCF onset. The docstring's "CE terminal convergence ≈ −4.4e-7/ep ⟹ plateau" is true at ep275+ but the eps=1e-3 test CANNOT distinguish ep150 from ep275 (measured −9.38e-5 only at ep250→275). | `run.log` verdict rows parsed this session; `_stage_converged` trainer:999–1041 | Either **drop `--curriculum-event-triggered` from the primary run** (recommended; see C2/M1) or recalibrate: `--curriculum-plateau-rel-eps 1e-4` + `--curriculum-plateau-windows 25` + `--curriculum-min-stage-epochs 250` (then fire ≈ ep275–300 ≈ cap; event lever buys ~0–25 ep). |
| C2 | **Event-triggering can fire l7 — the MEASURED DEFECT stage the config intends to NEVER run — mid-run.** `_evt_resolve_seg_form` (trainer:1091–1095) fires the l7 boundary on tau-stage plateau with NO guard for the `--l7-start-epoch 1000 == epochs` "never" intent (`_fire` fires on `converged OR hit_cap`; a huge cap does not disable converge-fire). #205 tau-phase measured rel slope at ep450→475 = **−4.7e-4 < 1e-3** ⟹ at defaults l7 WOULD fire ~150 ep after the tau fire and run the L∞-defect loss for hundreds of epochs. `witness_autoconfig.py:1062` explicitly relies on "l7 collapses to ≤1 trailing epoch" — event mode silently breaks that guarantee. | trainer:1077–1095; run.log tau rows; autoconfig:1062 | Trainer guard (1 line, operator applies): in `_evt_resolve_seg_form`, treat the l7 boundary as DISABLED (return tau_softplus, no `_fire`) when `int(args.l7_start_epoch) >= int(args.epochs)`. Or drop event-triggering (C1). |
| C3 | **`--bank-n-scales 6` is memory-UNSAFE at n600 + self-orient.** MEASURED this session: bank-6 under `--max-bank-freq 64` → cols 84 → **in_feat 176** (vs 88 at bank-4). Preflight projection at the TRUE in_feat: cf_mx_cache **86.4 GiB**, projected peak **110.81 GiB > 89.6 GiB ceiling (0.70×128) ⟹ REFUSE**. bank-5 also unsafe (91.17). Only bank-4 (67.61 GiB) is SAFE under the current fp32 resident per-pair cache design. | `curvelet_directional_B` run this session; `wmp.project_peak_rss_gib(in_feat=176)` executed | **Primary run: `--bank-n-scales 4`.** Re-open bank-6 only after an fp16 / on-the-fly per-pair feats path lands (halving/eliminating the resident cache) — it was slope-gated anyway (ledger §4 exponent bet). |
| C4 | **`tools/witness_memory_preflight.py --launch-sh` is in_feat-BLIND → hands a FALSE SAFE for exactly this config.** Executed: `--launch-sh <fresh argv> --strict` → "SAFE 67.6 GiB" and the `--in-feat 176` override is silently ignored on the launch-sh path. `parse_launch_flags` (tool:129–148) parses neither `--bank-n-scales` nor `--max-bank-freq` nor `--n-dir-freqs`; `project_from_launch_sh` never passes `in_feat`. `launch_witness_run.py:329` step (b1) calls this exact path → the governed launcher would green-light the OOM config. This is the same surrogate-green class the tool was built to extinct (AXIS 9). | executed both invocations; tool:129–160; launcher:327–329 | Parse the bank/dir flags in `parse_launch_flags` and derive in_feat (2·cols(bank,max_freq)+4·n_dir_freqs), or at minimum honor `--in-feat` alongside `--launch-sh` and have the launcher pass the trainer-derived value. |
| C5 | **The governed launcher CANNOT express this config.** `tools/launch_witness_run.py` argv surface = `--config {proven_base, all_levers, sealed_205, store_nothing_205}` + ops flags only — no per-flag override, no passthrough; `witness_autoconfig` has no seeded/fresh derivation. None of the 4 configs carries `--seed-islands`/`paint`/`--eikonal-weight-end`/`--film-stiefel`/`--closed-loop-control`/etc. Raw heavy python launch is FORBIDDEN (P0 governor) ⟹ as things stand the run cannot legally launch. | launcher:206–274 read; autoconfig config surface grepped | Minimal fix: add `--extra-trainer-flags` (or `--launch-argv-file`) to the launcher, **validated against the already-existing `_trainer_flag_set()` (launcher:72–73)** — never-invent-flags enforced structurally; or add `derive_fresh_seeded_config()` + a `fresh_seeded` choice in autoconfig (keeps the config triality-canonical; preferred). |

## MED findings

| # | Finding | Fix |
|---|---|---|
| M1 | **The 3e3b9c697 SEAL fix covered ONE of FOUR epoch-anchored stage-relative levers.** Under event-triggering, eikonal now tracks the fired tau (verified trainer:3856–3864) but `--persistence-warmup-epochs 300` (linear ep/300, `persistence_topology_loss.py:245`), `--lane-band-start-epoch 350`, and the hosc-β anneal (over `--epochs`, β(ep150)=1.45 vs β(ep300)=1.90 at the boundary) all stay wall-clock-anchored — every one was calibrated against tau@300 and de-synchronizes when tau fires early. Class-level gap, not an instance. | Subsumed by dropping event-triggering (C1); else re-anchor all three to the fired boundary before any event-triggered run. |
| M2 | **`--closed-loop-control` + `--async-verdict` ≈ sync verdicts: ~2× run wall.** The decision point joins the *just-scheduled* verdict every eval epoch (trainer:4315→4356). #205 measured verdict wall 2062–2439 s ≈ the 25-epoch train wall (19/19 async verdicts completed, none skip-throttled) ⟹ GPU idles the full verdict at every eval point: ~1000-epoch run goes ~22 h → **~44 h**. Deterministic-by-design (FEED-04l), but the cost was never stated numerically. | Accept (recommended for run 1 — the closed loop IS the seed-survival safety net), or trainer option: decide on the PREVIOUS eval's verdict (1-eval lag = 25 ep ≪ the ~100-ep erosion timescale) which removes the stall entirely while staying deterministic. |
| M3 | **Eikonal + length regularize the FRAME-0 field only**: `phi0 = model.sdf(cf, c0)` (trainer:2493) — the MCF-erosion/survival story is about the SegNet-scored FRAME-1 partition; the 0.05→0.10 step protects it only via shared trunk+FiLM coupling. Pre-existing (#205 identical), but the σ-survival grounding for the knee implicitly assumed the scored field. | $0 probe: measure frame-1 |∇φ| drift through the tau stage on #205's ckpts; if decoupled, apply eik/length to `phi(cf, c1)` (or both) in a follow-up — not a launch blocker. |
| M4 | **hosc β never reaches 4.0**: the muon finisher freezes β at its ep726 value ≈ **3.18** (trainer:4098–4102 `_anneal_ep`). Pre-existing in #205, but the config DROPS the render-τ anneal (constant 1.0), leaving β as the ONLY interface-sharpening schedule — its terminal value now carries more weight. | Optional: `--hosc-beta-end 5.134` puts β(726)=4.00 exactly (β(ep)=1+(β_end−1)(ep−1)/999). Do NOT touch `--anneal-epochs` for this (it also drives the LR cosine). |
| M5 | `--curriculum-min-stage-epochs 150` floor is half the measured CE convergence (~ep300) — permissive only because the plateau test is supposed to hold the line, and C1 shows it does not at eps 1e-3. | Covered by C1's values (250/1e-4/25) or by dropping the flag. |

## LOW findings

- **L1** "l7 1000 = never" is off-by-one even in OFF mode: `ep < l7_start` at ep 1000 → the FINAL epoch runs `l7_softplus` (also true in #205; autoconfig acknowledges "≤1 trailing epoch"). Cosmetic hardening: `--l7-start-epoch 1001`.
- **L2** Constant τ=1.0 verified inert-exact: geometric = `start·(end/start)**prog` = 1.0 exactly ∀prog (no log/0-div path; guard at trainer:5555–5559 requires start>0 ∧ end>0 — passes); no code assumes end<start (cosine/geometric/hold all handle equality; muon temp-freeze inert at a constant).
- **L3** `--lane-band-start-epoch 350` under an early-fired tau widens the intended 50-ep deconflict gap to up to ~200 ep (benign; moot when event-triggering is dropped).
- **L4** Shape audit CLEAN for mod-19 + bank-6 + stiefel: `in_feat` derived from the actual post-cap bank (trainer:1491–1507, no hardcoded 32); `film = Linear(mod_dim, 2·hidden·n_hidden)` → weight (768,19) tall ⟹ Stiefel column-orthonormalization well-posed (freeze-decoder conflict correctly refused at trainer:1820–1828, N/A here); pose-carrier `table` mode is mod_dim-independent (`_pc_code_dim` only for `film` mode, trainer:1906); `--freq-across 32` is a frequency count, not mod-dim.

## AXIS 9 — measured-runnability (EXECUTED)

- **Memory preflight NUMBER:** tool output at this argv = "SAFE, projected peak **67.6 GiB** ≤ 89.6" — **FALSE for this config** (C4: in_feat-blind). True projection at the measured in_feat 176: **110.81 GiB ⟹ REFUSE**; at bank-4 (in_feat 88): **67.61 GiB ⟹ SAFE**. The launch-shaping conclusion: the config is runnable **only at bank-4**.
- **Bogus-flag probe:** full fresh argv + `--bogus-sentinel-flag` → argparse error `unrecognized arguments: --bogus-sentinel-flag` ONLY ⟹ all 13 change-flags (and every KEEP flag) parse against the real trainer argparse. (Note: the closed-loop knobs are `--closed-loop-stop-after-windows` / `--closed-loop-min-sustained-windows`; defaults 3/3 match the intended values, no flags needed.)
- **Control-system suites at HEAD:** `test_scheduled_eikonal_weight.py` + `test_event_triggered_curriculum.py` + `test_closed_loop_control.py` → **36/36 passed**.

## Cross-lever hunt — verified-clean items (same class as the eikonal/tau fix)

- Muon is **not** event-fireable: `_evt_resolve_seg_form` resolves tau + l7 boundaries only; `--muon-start-epoch 726` is a fixed optimizer boundary (trainer:3924) — warm-start-momentum + lr-final-frac anchor on `muon_start_epoch` (not the fire epoch) ⟹ resume-deterministic. Correct as-built.
- Closed-loop bump composes bounded-above on the schedule (`_cl_effective_eikonal` floors the cap at the scheduled value; ≤0 bump returns the schedule EXACTLY) — byte-identity contract verified.
- LR rewarmup + moment-reset boundaries include event-fired curriculum changes (via `seg_form != prev_seg_form`) AND band/lane/msal engagements (trainer:3879–3900) — consistent under both modes.
- paint + seed-islands + amplify: amplify weights and the seed residual/masks are built from **GT lstars** (trainer:2398–2412, 2446–2475), independent of the paint target ⟹ populated at ep0 regardless; paint shapes the structured-init φ-target (trainer:1744–1766). The ledger's measured ep0 acceptance gate (`part_frac[lane] > 0`, ≈0.0064) remains the binding check.
- Event controller determinism: within-stage `ep_loss` appended at epoch END (trainer:4474) ⟹ resolution reads only past epochs; resume persists `__evt_*` and `__cl_*` sidecar keys ON-only (byte-identical OFF).

## AXIS 8 — assumption-challenge

Two shared assumptions were at risk. One I **measured false** this round: *"ep_loss-plateau ⟹ stage done"* (C1 — the plateau test at eps 1e-3 cannot separate mid-descent from convergence on #205's own CE trace). The one the config still stands on: **"the σ-smoothing proxy (93% lane survival @ σ0.8) predicts the real tau/MCF flow's action on the painted seed."** The ledger already flags it directional-only (§5 NO-FAKE caveat). If wrong, the tau stage erodes the lane despite paint+eikonal — the #205 signature recurs. **In-run guard exists and is the right one:** closed-loop sustained-DIVERGING_ERASING → bounded eikonal bumps (cap 0.20) → clean early-stop with the best EMA shadow already preserved (`_maybe_preserve_best`), plus the external `witness_control_monitor` and per-stage CE checkpoint. This is why `--closed-loop-control` should STAY in the primary run even as event-triggering is dropped.

## The revised launch shape (recommendation to the operator)

1. **Drop `--curriculum-event-triggered`** for the primary run (C1+C2+M1): with honest thresholds it buys ~0–25 epochs and de-synchronizes 3 calibrated epoch-anchored levers + exposes the l7 defect path. Graduate it later as an isolated A/B after the l7 guard + boundary re-anchoring land.
2. **`--bank-n-scales 4`** (C3) — bank-6 re-opens behind the fp16/on-the-fly feats build + the ledger's slope gate.
3. **Keep `--closed-loop-control`** (the seed-survival safety net); accept the measured ~2× wall (M2) or land decide-on-previous-verdict first.
4. **Fix the preflight tool's in_feat blindness (C4) and add the launcher config/passthrough (C5) BEFORE launch** — the governed path must both express AND correctly gate this argv.
5. Everything else in the 13 SEALs as-is: paint, seed-islands, dilate 1, eikonal 0.05→0.10 (tracks tau@300 hardcoded), constant τ=1.0 geometric, mod-19, film-stiefel, muon warm-start + final-frac 0.1, band 350, rewarmup 20-cosine. Optional polish: `--hosc-beta-end 5.134` (M4), `--l7-start-epoch 1001` (L1).

**HARD GATE unchanged:** pointer 0.19110 UNMOVED; the run's first milestone after convergence is a
byte-closed `upstream/evaluate.py` n600 exact row, not another advisory verdict.
