# ddm_tp1 (#804) — v9-line telemetry PORT to the TR1 trainer (burn-4 §3.1 prereq 1)

**Pointer honesty FIRST: 0.1910828242 [contest-CPU] UNMOVED.** This is APPARATUS (score-neutral
read-only telemetry). No score claim; nothing here moves the pointer. Landing sha **15aad5a28b**.

**The owed port (vh1 row 7 / burn-4 charter §3 prereq 1):** the TR1 trainer had only 4 hits of the
v9 confound-cure vocabulary (`confound_alarm` frozen/gnorm/nonfinite + `weights_stepped`); burn-4's
F1–F4 halt rules need the MISSING signals to EXIST (the guards are only as good as the telemetry that
trips them). This port adds them, DEFAULT-OFF, byte-identical when off.

## STORES-CONSULTED (recall-first, path + where pinned)
- `.omx/research/ddm_vh1_v8v9v10_harvest_20260730.md` (2249f7955b) — **row 7** (the named owed port:
  loss_terms/term_domination/liveness/confound_alarm gap; #304/#321/#404) + row 14 (birth_completion
  key rides this item but is P2-owned).
- `.omx/research/ddm_burn4_charter_skeleton_20260731.md` (34d2354ac4) — §3 prereq 1 (the exact 4
  named signals: per-term loss rows #304 · term-domination + term_inert #321 · liveness accepted-frac/
  weights_stepped · positive-control #404) + the `⟪UNKNOWN: port landing sha⟫` slot filled here.
- `.omx/research/da_db_producers_20260713.md` — the #408 **Q1–Q7 closure table** (Q1 grad_clip · Q2
  term_domination+term_inert · Q3 verdict-live-gap · Q4 tail_cycle · Q5 would-fire · Q6 ladder_birth ·
  Q7 lever_engage) — the emission taxonomy this port maps TR1-applicability against.
- `experiments/train_levelset_witness_realized_through_R_mlx.py` (v9 trainer) — `_loss_terms_row`
  (5298), `_running_accepted_frac` (5411), inline `term_domination` streak (16892–16911), the
  `_dseg_canary_suite` positive control (11027–11038), the imported producers (332–345).
- `src/tac/witness_control/telemetry_producers.py` — the REUSABLE Q-producers this port imports rather
  than reimplements: `term_inert_rows`, `lever_engage_row`, `deterministic_strata`, `ProducerResumeState`.
- `src/tac/witness_control/verdict_trend_alarm.py` — `canary_suite()` (#404 positive-control, reused).
- `src/tac/confound_gates.py` — #402 `check_telemetry_verdict_rows_carry_liveness` (the STRICT gate my
  new `loss_terms` emitter must satisfy: it does — 0 tr1 violations).
- `experiments/train_tr1_partition_renderer_mlx.py` (the port target) + its existing telemetry
  (`topology_per_class`/betti/erasure @853, `realized_gate`/a1_gate, `confound_alarm` @1556/1581/1588,
  event_* rows) — so this EXTENDS, never duplicates.
- `src/tac/witness_dsl/spec_tr1_renderer_20260728.py` — the TR1 DSL SoT where the flag lands as a
  `Lever` factory (never-invent-flags AST validation).

## §1 INVENTORY DIFF (v9 telemetry surface × TR1 before → after)
| row / signal | v9 producer (path:line) | TR1 BEFORE | TR1 AFTER (this port) |
|---|---|---|---|
| **loss_terms (#304)** per-term shares | `_loss_terms_row` (v9 5298) | ABSENT (aggregate `ep_loss` only) | **PORTED** `tr1_loss_terms_row` @ gate cadence; keys {seg,rate,delta_sparsity} = the exact `batch_loss` addends; `sum_terms`/`sum_minus_total` self-check |
| **term_domination (#321)** | inline streak (v9 16892–16911) | ABSENT | **PORTED** `tr1_term_domination_alarms` (edge-triggered, 40% / 3 sustained rows) → `confound_alarm` |
| **term_inert (#321)** | `term_inert_rows` (producers 610) | ABSENT | **PORTED** — reused the producer over `{seg,rate,delta_sparsity}` engaged-map |
| **liveness accepted_frac (#402/C6)** | `_running_accepted_frac` (v9 5411) | `weights_stepped` only | **PORTED** accepted_frac + weights_stepped on every new loss_terms row |
| **positive-control (#404)** | `canary_suite` (verdict_trend_alarm 516) | ABSENT | **PORTED** — reused `canary_suite()` → `positive_control` row (once, first gate) |
| **lever_engage (Q7)** | `lever_engage_row` (producers 716) | ad-hoc `event_*` fields | **PORTED** canonical `lever_engage_row` companions (reads the existing `row`) |
| grad_clip_activation (Q1) | `ClipActivationAggregator` (producers 574) | `gnorm_last_batch` already emitted | **N/A** — TR1 does NO gradient clipping; gnorm observability already present |
| verdict_live_gap (Q3) | `verdict_live_gap_due`/`live_gap_fields` | A1 gate already rebases live↔shadow | **NOT PORTED (recorded)** — would add a 2nd scorer pass; DSL-default-OFF even in v9; outside the charter's named-4. Future opt-in if a live-gap decision is needed. |
| tail_cycle_endpoint (Q4) | `tail_cycle_endpoint_row` (producers 650) | — | **N/A** — TR1 seg burn has no tail-cycle chain |
| would_fire (Q5) | `would_fire_row` (producers 675) | — | **N/A** — different lever set (no powerlaw/annulus levers on TR1) |
| ladder_birth_complete (Q6) | `ladder_birth_complete_row` (producers 703) | `topology_per_class`/betti/erasure present | **P2-OWNED** — the `birth_completion` gate-key producer is charter P2 (vh1 row 14), rides this telemetry but not this task |
| component_wallclock (D-A) | `ComponentWallclock` (producers 90) | — | **out of scope** (timing observer, not a confound-cure signal) |

**DIFF = the port scope = the 6 CORE rows** (loss_terms · term_domination · term_inert · accepted_frac
liveness · positive_control · lever_engage). All 6 landed. Q1/Q3/Q4/Q5/Q6/D-A dispositioned above
(N/A · recorded-not-ported · P2-owned · out-of-scope) — not enum-padded with no-op rows (NO-FAKE #5).

## §2 THE PORT (files + flag)
- **Flag:** `--telemetry-v9-port {off,on}` (DEFAULT `off`) — trainer argparse.
- **Trainer** `experiments/train_tr1_partition_renderer_mlx.py`: pure builders `tr1_loss_terms_row`
  (#304) + `tr1_term_domination_alarms` (#321) + `TR1_LOSS_TERM_KEYS`/`TR1_TERMDOM_FRAC`(0.40)/
  `TR1_TERMDOM_MIN_ROWS`(3); loop wiring — a gated setup block (imports the reusable producers ONLY
  when on), a gated `lever_engage` companion block (reads the existing epoch `row`), and a gated
  gate-cadence emission block (loss_terms + domination + term_inert + positive_control on a fixed
  `deterministic_strata` subset, LIVE params).
- **DSL** `src/tac/witness_dsl/spec_tr1_renderer_20260728.py`: `lever_telemetry_v9_port(state)` Lever
  factory (`overrides={"--telemetry-v9-port": state}`) — never-invent-flags validated.
- **Tests** `src/tac/tests/test_ddm_tp1_v9_telemetry_port.py` — 15 tests (schema · off-identity ·
  alarm-fire · DSL validate); all pass. ruff F clean.

## §3 BYTE-IDENTITY RECEIPT (the sealed-lineage guarantee)
**Method.** Ran the NEW trainer on the n96 GT cache, tiny config (`--variant plain --num-pairs 4
--grid-downsample 16 --code-width 2 --renderer-width 8 --seed 3 --epochs 2 --gate-every 1
--batch-pairs 2 --ema-decay 0.95`), and deep-diffed every checkpoint `.npz` (all `param::`/`ema::`/
`opt::` arrays via `np.array_equal` + the `meta::json` blob). Controls for MLX float non-determinism.

**Result.**
- **CPU OFF vs ON (deterministic authority):** 112 `param/ema/opt` arrays across 4 checkpoints
  **ALL BIT-IDENTICAL**. The ONLY `meta::json` difference is `gate_wall_seconds` (a wall-clock float
  `time.monotonic()-t0` in the PRE-EXISTING `realized_gate` row — not my code).
- **CPU OFF vs OFF control (flag held off both):** IDENTICAL difference profile — trained bytes
  bit-identical, only `gate_wall_seconds` differs. ⇒ the flag introduces **ZERO** difference beyond
  the flag-INDEPENDENT wall-clock non-determinism.
- **GPU OFF vs OFF control:** `param/ema` differ by max|Δ|≈7e-6 (MLX GPU float non-determinism,
  flag-independent) — explains the cosmetic diff seen on the first GPU off-vs-on run.
- **ON emits, OFF does not:** ON telemetry.jsonl has loss_terms×2 + positive_control×1 +
  lever_engage×1 + telemetry_v9_port×1; OFF has 0 of each. Positive-control cleared
  (`verdict_clearance=True`).

**Verdict.** Trained/checkpoint WEIGHT bytes are **byte-identical with the flag off vs on**; the flag
is fully inert to training. Structural guarantees behind it: (1) the flag is threaded via `args` ONLY,
never `TR1Config` ⇒ `config_hash`/`canonical_json`/checkpoint `meta::cfg` are flag-invariant (unit-
tested); (2) every new emission is behind `if _tel_v9:` ⇒ OFF executes zero new code (⇒ off ≡
pre-change); (3) new rows go to `tlog`/JSONL ONLY, never `telemetry_tail` (the checkpoint-baked tail);
(4) the recompute uses the same deterministic forwards the loss uses (fixed dither bank + isolated
`order_rng`) ⇒ no RNG advance, no model/opt mutation.

## §4 FILLED P5 SLOT TEXT (for the burn-4 charter §3 prereq 1)
> **1. v9 telemetry port to TR1 — DONE (ddm_tp1 #804, sha 15aad5a28b).** `--telemetry-v9-port`
> (DEFAULT off) emits, when on: per-term `loss_terms` (#304, keys {seg,rate,delta_sparsity}),
> `term_domination` + `term_inert` alarms (#321), accepted-frac + weights_stepped liveness (#402), and
> a #404 positive-control sentinel (+ Q7 `lever_engage` companions). Byte-identical when off (CPU
> off-vs-on: 112 param/ema arrays bit-identical; only the pre-existing wall-clock `gate_wall_seconds`
> differs, matched by the off-vs-off control). F1–F4 halt rules now have their SIGNALS. Burn-4 turns
> it on via the `lever_telemetry_v9_port("on")` DSL Lever in the sealed ticket.

## §5 verdict_scope + NOTHING-MORE
- verdict_scope: apparatus (observability); no score, no family verdict. Score-neutral by construction.
- Q3 verdict-live-gap: recorded NOT-ported (2nd-scorer-pass cost; DSL-default-off in v9; outside the
  named-4). Reactivation: if burn-4 needs a live↔EMA gate decision, add it as a second opt-in flag
  reusing `verdict_live_gap_due`/`live_gap_fields` (both already imported-capable).
- birth_completion (Q6/vh1 row 14): P2-owned; this telemetry surface is ready for it to ride.
- Pre-existing UNRELATED test skew: `test_ddm_tb1_tr1_renderer.py::test_counted_ledger_keys_*` fails on
  HEAD too (a sister `byte_ledger_coder`/`rowband` landing added ledger keys the test wasn't updated
  for) — proven not mine (my diff is +429 insertions, none touching `counted_bytes_ledger`).

**Pointer 0.1910828242 [contest-CPU] UNMOVED.** [no-triality] [p0-ledger-ok]
