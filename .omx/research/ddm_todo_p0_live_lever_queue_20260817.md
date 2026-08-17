# The TODO queue is READABLE — it was just pointed at a retired vehicle, and the live vehicle keeps its own

**Status:** MEASURED (2026-08-17, MAIN, $0 local). Directive: operator, *"Pursue todo as p0."*
**Axis:** apparatus. No score claim. Frontier untouched.
**verdict_scope: INSTANCE** — the live trainer `tac.pr130_lift.train_semantic_quantized_resumable`
at HEAD, against 618 launch manifests on both SSD tiers.

## ANSWER FIRST

1. **Literal `TODO` on the live vehicle: ZERO.** `src/tac/pr130_lift` 0 · `src/tac/boundary_math` 0 ·
   `src/tac/mlx_score_aware` 0. (`tools/` carries 74, none on the frontier path.) The literal class
   is empty where it would matter, so "pursue TODO" has to mean the SEMANTIC class: never-fired levers.
2. **The lever registry describes a RETIRED vehicle.** `lever_registry.completeness()` returns
   `describes_live_vehicle: **False**`, `trainer_path: experiments/train_levelset_witness_realized_through_R_mlx.py`
   — and its own `LIVE_TRAINER_BASENAME` says `train_tr1_partition_renderer_mlx.py`, which is *also*
   not what we run. **The instrument is two vehicle-generations stale.**
3. **Its 199-row never-fired queue reaches the live trainer 0 times.** Name-mapping
   (`LeverName` → `--lever-name`) resolves **34 of 199** against the retired trainer's 443 flags and
   **0 of 199** against the live trainer's 38. The mapping works — it resolves where the registry
   points — so the zero is a real disjointness, not a naming artifact.
4. **The live trainer emits its OWN activation ledger, every run, and nothing ingests it.**
   `train_semantic_quantized_resumable.py:1061` prints `[b2e] editability levers ACTIVE:` with
   per-lever `active` + **`reason_if_off`** — the exact duty-to-measure record. It goes to `run.log`.
   `.omx/state/lever_activation_ledger.jsonl` has not been written since **2026-07-27** (21 days).
5. **The live queue, now MEASURED BY THE INSTRUMENT: 12 never-fired of 18 derived levers.**

⚠ **Corrected by building the cure.** This memo first said "11 never-fired, 3 fired" from a
hand-count over the table below. The landed ingester
(`tac.pr130_lift.live_lever_activation`), run over all 417 run logs and 618 manifests, returns
**18 derived levers · 6 fired · 12 never-fired**. My hand-count under-counted the never-fired by
one and classified `lr` / `ce_fraction` / `softplus_fraction` as "swept" rather than as *fired*,
which they are. The instrument wins; the table below is kept as the hand-audit it was.

⚠ **MAIN correction, same turn.** Earlier this session I wrote that the activation ledger's 251 rows
"carry no state, so 0 never-fired rows are extractable." **That was a wrong-key guess against an
event log**, not a defect: the ledger's schema is
`{agent, event, lever, reason, run_ref, ts, verdict_ref}` and state is DERIVED from events by
`activation_ledger.never_fired()` / `.duty_to_measure()`, which return **199** and **200** rows
respectively. The API works. I asserted absence from a failed query instead of reading the
producer — the same genus as the negative-existence-claim law
([[the-instruments-own-units-level-and-aggregation-are-part-of-the-claim-20260816]]).

## The live TODO queue (measured, not estimated)

Live trainer flag surface: **38** (19 of them plumbing — paths, caches, device, cadence, seeds).
Score-affecting levers and their observed values across all 12 live-trainer launch manifests:

| lever | default | observed | state |
|---|---|---|---|
| `--band-objective-weight` | 0.0 | `0.0, 1.0` | FIRED (rg1 probe) |
| `--float-warmup-steps` | 0 | `0, 100` | FIRED |
| `--weight-qat-q3q4` | off | `SET` | FIRED (mz2 q3/q4) |
| `--carrier-rank-penalty` | 0.0 | never passed | **NEVER FIRED** — ra2 carrier-rank is the named live head (#1079) |
| `--carrier-tensors` | `[]` | never passed | **NEVER FIRED** — same family, no tensors ever named |
| `--distill-weight` | 0.0 | never passed | **NEVER FIRED** — wd3 built scorer-aware distillation (#1069); this trainer-side path never ran |
| `--distill-max-seg` | 4e-4 | never passed | **NEVER FIRED** — paired with the above |
| `--film-row-dropout` | 0.0 | never passed | **NEVER FIRED** — mz2 measured −130..−2,051 B for FiLM-row sparsity, retained UNSCORED |
| `--film-row-dropout-protect-top` | 0 | never passed | **NEVER FIRED** |
| `--weight-perturb-robustness` | 0.0 | never passed | **NEVER FIRED** — F1, `reason_if_off: "sigma == 0 (default)"` |
| `--weight-perturb-shape` | `quantization` | never passed | **NEVER FIRED** |
| `--fixed-zero-mask` | off | never passed | **NEVER FIRED** |
| `--bits` | 4 | `4` only | never off default |
| `--ema-target-seed-fraction` | const | never passed | never off default |
| `--film-critical-multiplier` | const | never passed | never off default |

**Scope caveat, stated plainly:** 12 manifests is the population of runs that WROTE a manifest.
Absence there is a lower bound on firing, not proof of never-fired across all history. What is
*not* caveated is the trainer's own telemetry, which independently names four of these OFF with
their reasons on the currently-running job.

## Why this is the P0 the directive asked for

The duty-to-measure queue is the apparatus's answer to "what have we built and never tried."
It has been answering for a vehicle we retired. Three of its named-OFF levers have **measured
evidence already sitting in memos** — mz2's FiLM-row sparsity (−130..−2,051 B, retained but
unscored), ra2's carrier-rank (the live head), wd3's distillation (built, reactivated by operator
directive 08-15) — and none of them has ever been passed to the trainer that produces the frontier.
That is precisely the *built-elsewhere-unwired* grade ([[#864]], VALID_BUILD_GRADES[5]) at the
one place where it costs score.

**The cure is one consumer, not a new registry.** The trainer already emits the record with
`reason_if_off`. `.omx/state/lever_activation_ledger.jsonl` already has the schema and the derived
queries. Nothing needs designing — the two halves need connecting, and the registry needs to be
told which vehicle is live.

## THE CURE — LANDED, and it corrected me twice

`src/tac/pr130_lift/live_lever_activation.py` + 30 tests. Two ingest paths, because the
trainer's own emission turned out to be **partial**:

- **`ingest_run_log`** consumes the `[b2e]` emission — authoritative for the 5-lever editability
  family, carrying `reason_if_off` per lever. Records `fired` for ACTIVE levers ONLY; an off lever
  is deliberately never written, because `never_fired()` derives the queue from ABSENCE.
- **`ingest_launch_manifest`** consumes the launch argv — the *only* record for the other 14 levers,
  which have no per-run telemetry at all. A flag passed AT its default is not a firing, so the
  default is compared numerically (`2e-5` vs the source's `2e-05` is the same number).

**Anti-staleness:** the live lever set is **derived by AST** from the trainer's own
`add_argument` calls at call time. The registry went stale because its vehicle was a hardcoded
pointer; this one cannot describe a vehicle the code does not have. The single hand-declared thing
is `PLUMBING_FLAGS` (paths/caches/device/cadence/seeds), listed explicitly so every exclusion is
auditable rather than pattern-guessed.

**Building it corrected me twice, both caught by running it:**
1. The `[b2e]` emission alone left `lr`/`ce_fraction`/`softplus_fraction`/`band_objective_weight`/
   `float_warmup_steps` in the queue — levers I had *measured* as fired. The emission covers 4 of 18.
   That gap is why the manifest path exists.
2. The final count is 12/18, not the 11 I hand-counted.

The emission also carries `gate_aware_conditioning` — a lever with **no flag at all**, state
`DECLARED_UNBUILT_FOLLOW_ON`. That is the designed-stub grade surfacing honestly in telemetry;
a hardcoded registry would have dropped it.

## NEXT

1. **Re-point or scope the registry.** `describes_live_vehicle: False` should be repaired or made
   loud — a queue that silently describes a retired vehicle is the #936 adoption-decay genus with a
   vehicle label on it. The live queue now has its own reader; the retired one still answers by default.
2. **Drain the 12.** Ranked by existing evidence: `--film-row-dropout` (mz2 measured −130..−2,051 B,
   retained UNSCORED — and training FOR row sparsity is the #1074 P1 train-for-editability pattern)
   → `--carrier-rank-penalty` + `--carrier-tensors` (ra2 head, #1079) → `--distill-weight` +
   `--distill-max-seg` (wd3 built, #1069). Each is a single-variable window on a trainer whose
   windows cost ~6.8 min at n=600 by the corrected cadence law.
3. **`--fixed-zero-mask` and `--weight-perturb-robustness` have no evidence either way** — genuinely
   unexplored, which is what a TODO queue is *for*.
4. **Wire the ingest into the launcher** so the queue stays current without a manual sweep. Until
   then it is one command, and every new run leaves the record it needs.
