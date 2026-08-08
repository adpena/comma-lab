VERDICT: FINDINGS_RESET_COUNTER

# M1R5A mechanics review — config, guard, resume, memory, and seal gates

Tags: `[no-triality] [p0-ledger-ok]`

Axis: `[source/receipt inspection; scorer-free review]`.  
`score_claim=false`, `promotion_eligible=false`, `scorer_forwards_run_by_this_review=0`,
`metal_runs_by_this_review=0`, `launch_mutation=false`, `ticket_mutation=false`.

## Answer first

The M1 n120 fire is **not sealed**. This pass found five launch-blocking mechanics defects, one medium
inert-lever defect, and one low provenance defect in the pinned post-M1C1 artifact. The counter remains
`0/3`; every concurrent pass over this artifact is void under the charter's concurrency rule.

The highest-risk break is that the nominal real-config mem-probe cannot exercise the configuration it
clears: `mem-probe` mode explicitly disables the M1 controller policy, so it omits the full EMA shadow,
controller journal, controller-triggered checkpoints, and K8 materialization. The guard then accepts that
receipt without binding the local trainer/ticket source or the outer safe-run projection to the measured
peak. Separately, the seal orchestrator reports the child safe-run argv as `FIRE`, bypassing the controller
that alone survives a wall timeout; the checkpoint pair is not transactionally published; interpreter
options can pass invisibly through the flag classifier; and schema-less seal receipts can read as passes.

No burn, Metal job, scorer job, archive build, or score measurement was run.

## Frozen-subject re-verification

Re-hashed before source inspection:

| reviewed file | required prefix | re-derived SHA-256 | result |
|---|---:|---|---|
| `.omx/research/ddm_m1_20260808/launch_ticket_v5_event_driven.json` | `90cf28d390999ef9` | `90cf28d390999ef9cda47340d9ec01bc65a15fb9ab3f88c60625abc29b414ec9` | MATCH |
| `tools/mx1_fire_guard.py` | `60fc0501a65d8d09` | `60fc0501a65d8d09b9bacd57cafd414544eac340e4107fa52f0beccfa60bbee6` | MATCH |
| `experiments/ddm_mx1_pr130_semantic_renderer.py` | `8bad6a6b8be1b201` | `8bad6a6b8be1b20189791283f64638d01a1b406482a1a725bab712c77f27894c` | MATCH |
| `tools/ddm_seal_orchestrator.py` | `11c4368f009afc31` | `11c4368f009afc31aca67da37d7c9b2a9b109f3a00d20ca903800c26f3897a79` | MATCH |

The binding content-hash stop did not fire. Live HEAD was not used as a freeze key.

## Findings, ranked

### M1R5A-F1 — LAUNCH-BLOCKING — the measured memory path omits the fire-only controller/EMA footprint

**Verdict scope:** INSTANCE plus APPARATUS CLASS — this M1 probe/guard contract and any future receipt
whose source/config identity is not bound to the memory-bearing program.

**Mechanism.** `_mem_probe_args` forces `mode="mem-probe"`, and `run_mlx_train` defines
`mem_probe_mode` from that mode and sets `controller_policy=None` whenever it is true
(`ddm_mx1_pr130_semantic_renderer.py:2909-2929`, `3658-3667`). That removes exactly the post-M1C1
mechanics the fire adds: ticket-derived schedule/EMA policy (`:2929-2952`), the full parameter-shaped EMA
shadow (`:3044-3063`), initial controller evaluation and checkpoint (`:3310-3322`), per-step EMA updates
(`:3453-3457`), durable journal/controller decisions (`:3475-3555`), and K8 tail materialization after
enough stage files exist (`:3231-3241`, `3296-3306`). The K8 writer opens all member NPZs and stacks eight
float64 parameter copies per key (`:269-305`), a real later-phase CPU-RSS surface a three-step non-controller
probe never reaches.

The receipt does contain real fp16/per-step-hygiene/chunk-cache n120 telemetry and measured peaks
(`mem_probe_receipt.json:2-23`, `227-231`), but its `source_repo_head` identifies the borrowed PR130 source
repo, not the local trainer implementation (`mem_probe_receipt.json:1847-1848`). The ticket nevertheless
calls the resulting 10.515302 GiB total the real saturated config and hard-codes the derived 16.0 GiB
safe-run projection (`launch_ticket_v5_event_driven.json:780-790`). The guard validates receipt schema,
status, samples, memory-limit state, and a selected inner-argv tuple, but does not read the measured peak,
validate a local trainer/ticket hash, or compare the outer `--projected-gib`/`--rss-mb` values
(`mx1_fire_guard.py:417-540`). `_unwrap_safe_run` removes those outer safety flags before classification
(`:67-70`).

Several exclusion reasons are therefore false on this implementation. `steps`, `eval_every`, and
`checkpoint_every` control accumulated journal/history size and whether the eight-checkpoint K8 path is
reached; `resume_from` loads prior history/optimizer state and a separate EMA file; `run_dir` determines
the checkpoint members found by `glob`; and `launch_ticket_path`/`fire_argv_key` activate the controller
and EMA rather than serving as mere plumbing (`mx1_fire_guard.py:341-369`; trainer `:192-251`,
`3019-3063`, `3137-3161`, `3231-3306`, `3481-3518`, `3556-3558`).

**Exact failure scenario.** A fresh three-step probe passes at the recorded 10.52 GiB peak. The guard
passes the n120 child. The fire then allocates a full EMA shadow and later enters controller/K8/checkpoint
phases never measured by the probe. Even without an OOM, the ticket's 16.0 GiB system reservation can be
an under-stated admission claim. A source edit can also keep using a still-fresh receipt because no local
trainer SHA is compared.

**Smallest correct cure.** Make the probe execute the same controller-enabled memory phases as each fire
route, including fresh and resume load, full EMA residency/update, a representative controller eval,
live+EMA checkpoint, and K8 materialization. Record the local trainer SHA, ticket SHA, effective parsed
namespace, run-state geometry, and measured peak in the receipt; validate all of them in the guard. Derive
and compare the outer safe-run projection/RSS cap from that measured peak in the guard rather than trusting
ticket literals. Re-run both probes only after the cured source is frozen.

### M1R5A-F2 — LAUNCH-BLOCKING — the seal orchestrator's FIRE route bypasses `controlled-train`

**Verdict scope:** INSTANCE — the current M1 ticket/orchestrator join.

**Mechanism.** The ticket defines fresh and resume controller argvs at
`launch_ticket_v5_event_driven.json:198-220`; its doctrine says `controlled-train` is what survives a
safe-run timeout and queues the exact resume (`:859-890`). But the ticket has no top-level
`fire_argv_key`. The orchestrator therefore defaults to `argv_m1_n120_cap_saturated` and makes that child
the reported `FIRE` argv (`ddm_seal_orchestrator.py:237-259`). A source-derived `build_gates` control
returned:

```text
ticket_top_level_fire_argv_key=None
orchestrator_FIRE_argv_key=argv_m1_n120_cap_saturated
controller_routes=[argv_m1_controller_fresh, argv_m1_controller_resume]
```

Only `run_m1_controlled_train` waits for the safe-run child, consumes its durable status, survives rc=124,
evaluates the last journal state, writes the wall-cap receipt, and queues the resume argv
(`ddm_mx1_pr130_semantic_renderer.py:4320-4401`). The child argv alone has no process left to perform that
work after safe-run kills it.

**Exact failure scenario.** MAIN follows the orchestrator's `FIRE` line. The child reaches the 28,800 s
wall cap between in-process terminal events. `safe_run` kills it and returns 124. Because the controller
was bypassed, no wall-cap terminal receipt or controller resume disposition is produced; the ticket's
claimed wall-cap durability is inert on the advertised fire route.

**Smallest correct cure.** Give the ticket explicit, distinct `fire_execution_argv_key` and
`fire_guard_argv_key` semantics (fresh controller vs its child). The orchestrator must guard the routed
child but report the controller as `FIRE`; refuse a controller route that cannot be joined one-to-one to
its child, status receipt, terminal receipt, and resume controller. Add a live-ticket control asserting
the reported fire key is `argv_m1_controller_fresh`.

### M1R5A-F3 — LAUNCH-BLOCKING — atomic files do not make the live/EMA checkpoint bundle atomic

**Verdict scope:** FORMULATION — the current paired-file checkpoint publication order.

**Mechanism.** Each NPZ uses temp+rename (`mlx_semantic_renderer.py:358-400`), and the stage names are
distinct. The bundle is not transactionally published. `save_checkpoint_bundle` writes the stage live
file, then overwrites `mlx.latest.npz` with metadata already naming the not-yet-written stage EMA file,
and only afterward writes that EMA file (`ddm_mx1_pr130_semantic_renderer.py:3231-3280`). Resume first
loads `mlx.latest.npz`, extracts the referenced `ema_checkpoint`, and refuses/opens it
(`:3019-3023`, `3044-3063`).

**Exact failure scenario.** The process crashes after `mlx.latest.npz` is renamed at `:3253-3261` but
before `mlx_ema_stepNNNNNN.npz` is renamed at `:3266-3279`. The exact ticketed resume path points at a
live checkpoint whose required EMA file does not exist. A previous complete stage pair remains on disk,
but the canonical latest pointer no longer selects it, so crash-resume is broken.

**Smallest correct cure.** Materialize and load-verify both stage-specific files first, fsync them, then
atomically publish one bundle manifest/latest pointer as the final operation. On startup, select only the
newest manifest whose live and EMA paths and hashes both verify; otherwise fall back to the prior complete
manifest. Add an interruption control at every write boundary.

### M1R5A-F4 — LAUNCH-BLOCKING — executable-prefix/runtime options are invisible to classification

**Verdict scope:** APPARATUS CLASS — `_flag_value_map` classifies only GNU-style long options, not the
executable grammar that can change the measured process.

**Mechanism.** `_flag_value_map` records only tokens beginning with `--`; positional tokens and
single-dash interpreter options are ignored (`mx1_fire_guard.py:73-91`). `_validate_flag_classification`
then reasons only over that reduced map (`:372-388`). A transient read-only control inserted the valid
Python runtime option `-X tracemalloc` between the inner interpreter and trainer script, updated only the
temporary ticket identity, and called the real `evaluate_guard`. Result:

```text
status=passed  reason_code=fire_guard_passed
flag_classification=flag_classification_ok  unclassified=[]
runtime_option_visible=false
```

Python consumes that option and runs the trainer with allocation tracing enabled, so it is not a rejected
trainer positional. The same genus admits executable wrappers such as `env` and their memory-affecting
environment assignments.

**Exact failure scenario.** A future ticket adds `-X tracemalloc` (or a memory-affecting interpreter/
environment wrapper) for diagnostics. The probe receipt remains from the unwrapped interpreter. The
classifier sees the same long flags and the full guard passes, while the fire's peak-memory behavior has
changed.

**Smallest correct cure.** Validate the complete command grammar: exact safe-run executable/options,
delimiter count, exact inner interpreter and script, no unknown positional prefix or single-dash runtime
options, and no wrapper/environment prefix unless explicitly compared. Set the trainer parser to
`allow_abbrev=False`; canonicalize and compare its real parsed namespace. Add positive controls for
`--flag=value` and repeated-store last-value semantics, plus negative controls for `-X`, `env`, extra
`--`, short options, and abbreviated flags.

### M1R5A-F5 — LAUNCH-BLOCKING — schema-less and empty seal receipts can satisfy gates

**Verdict scope:** APPARATUS CLASS — seal-orchestrator receipt validation.

**Mechanism.** Missing files are correctly `PENDING`, but semantic malformation can pass.
`_eval_receipt_status` accepts any JSON object whose `status` is `passed`, without checking schema,
ticket/argv identity, reason code, source, or required fields (`ddm_seal_orchestrator.py:346-359`).
`_eval_mem_probe` likewise checks status and age but not the mem-probe schema, Metal clearance, sample
telemetry, or config (`:317-343`). `_eval_harvest` defaults missing `falsifiers` to `{}`; both its `fired`
and `unresolved` lists are then empty, so an empty JSON object becomes `SATISFIED`
(`:422-455`).

Transient scorer-free controls over those exact functions produced:

```text
missing guard receipt       -> pending
{"status":"passed"} guard  -> satisfied
{} harvest receipt          -> satisfied, "sigma measured, no falsifier fired"
```

**Exact failure scenario.** A truncated/manual/forged status-only guard receipt satisfies its gate, or an
empty harvest receipt vacuously clears all sigma falsifiers. After three review rows, the orchestrator can
report `FIRE READY` without the receipt evidence its own contract claims to require. The trainer's
in-process guard may catch a malformed burn-guard file, but it does not reconstruct the skipped sigma
harvest; the seal claim itself is already false.

**Smallest correct cure.** Validate an exact schema and complete required-field set per gate kind,
including ticket/argv/source identities and the exact registered falsifier-key set. Empty or unknown sets
must block. Add missing, invalid-JSON, wrong-top-level-type, wrong-schema, status-only, empty-falsifier,
missing-falsifier, fired-falsifier, and complete-good controls. The good control must still open.

### M1R5A-F6 — MEDIUM — `--verdict-batch-size` is inert on the MLX fire path

**Verdict scope:** INSTANCE — the `mlx-train` interpretation of this flag.

**Mechanism.** The fire argv carries `--verdict-batch-size 32`
(`launch_ticket_v5_event_driven.json:105-110`), the parser accepts it
(`ddm_mx1_pr130_semantic_renderer.py:4563`), and the guard calls it a peak-memory size term
(`mx1_fire_guard.py:322`, `358-360`). Source census found operational reads only in `torch-facets` and
`torch-verdict` (`ddm_mx1_pr130_semantic_renderer.py:1861`, `2104`); the other reads merely serialize it
into probe/ticket metadata (`:3740`, `4242`) or validate positivity (`:4586-4587`). `run_mlx_train` chunks
its in-loop d_seg evaluation with `microbatch_pairs`, not `verdict_batch_size` (`:3163-3203`).

**Exact failure scenario.** An operator changes the advertised verdict batch size expecting a different
MLX evaluation footprint. The guard may force a re-probe, yet both the probe and fire still evaluate at
the microbatch size. The lever name/config claim has no operational effect on the named path.

**Smallest correct cure.** Either wire `verdict_batch_size` into a distinct MLX evaluation chunker and
measure it, or remove it from `mlx-train` fire/probe argvs and from the equivalence tuple. Add a behavioral
control that changes the flag and observes the actual evaluation chunk size.

### M1R5A-F7 — LOW — the charter carries contradictory subject-tree commits

**Verdict scope:** INSTANCE — review provenance text, not the four content pins.

The main immutable-subject clause says the subject tree is `393d67d016`
(`CHARTER_r4a_mechanics.md:40-45`), while OPTIMAL FORM still says it is `1381ac84cb` (`:128`). Re-derived
Git-object hashes show the required four content hashes are exactly those at `393d67d016`; at
`1381ac84cb`, the ticket, trainer, and orchestrator have the older hashes. I followed the charter's
binding four-content-hash stop and the specific `393d67d016` clause. The smallest cure is to replace the
stale OPTIMAL FORM pin with `393d67d016` so transitive-source provenance is unambiguous.

## A1 — the fire argv actually runs what the ticket claims

**Conclusion: FINDING** (F2 and F6).

Every outer safe-run flag parses and has a real consumer: `--rss-mb`, `--timeout`, `--projected-gib`,
`--label`, `--status-receipt`, and `--child-pidfile` are declared at `safe_run.py:95-131`; they drive the
RSS cap, timeout, admission projection, durable status, and pid custody at `:380-431`, `439-480`, and
`509-547`. The `--` delimiter is split at `:66-70`.

Every controller flag is also real: `mode`, `launch_ticket_path`, `fire_argv_key`, and `out` are declared
at trainer `:4521-4581`; `controlled-train` dispatches at `:4595-4600`; its ticket/key route selects and
runs the exact child at `:4323-4334`; and `out` is written at `:4596-4598`.

For the inner fresh/resume child, this is the complete flag-to-consumer trace:

| flags | `add_argument` | operational consumer |
|---|---|---|
| `mode`, `device` | trainer `:4522-4537`, `4550` | mode dispatch/guard `:4602-4605`, MLX selection and memory limits `:2977-3002`, `4622-4650` |
| `pairs`, `seed` | `:4542`, `4545` | deterministic sample selection `:2923-2925`; controller N check `:2932-2933` |
| `steps`, `lr` | `:4543-4544` | schedule horizon/config/loop `:2927`, `2961-2966`, `3013`, `3323-3324` |
| `ce_fraction`, `softplus_fraction`, `bits` | `:4546-4551` | config and actual loss/quantization `:2956-2966`, `3336-3367` |
| `microbatch_pairs`, `microbatch_policy` | `:4555-4556` | plan derivation `:2767-2816`, executed chunks `:3065-3135`, `3390-3447` |
| `microbatch_hygiene`, `microbatch_chunk_cache` | `:4561-4562` | cache creation/cleanup `:3126-3135`, `3440-3461` |
| `train_compute_dtype` | `:4558` | dtype resolution/cast `:2979-2983`, `3348-3353` |
| `checkpoint_every`, `eval_every` | `:4553-4554` | ticket cadence checks and loop gates `:2934-2937`, `3475-3558` |
| `input_cache`, `target_cache`, `init` | `:4538-4540` | checkpoint/cache loads `:2953-2975` |
| `resume_from` | `:4570` | selection gate, live/optimizer/history load, paired EMA load `:2940-2952`, `3019-3063` |
| `run_dir` | `:4541` | sacred-run checks and checkpoint paths `:3137-3150`, `3231-3308` |
| `out` | `:4580` | final result write `:4661` |
| `fire_guard_verdict`, `launch_ticket_path`, `fire_argv_key` | `:4567-4569` | fresh in-process guard and identity checks `:4443-4517`; controller policy `:192-231` |
| `verdict_batch_size` | `:4563` | **no MLX-train operational consumer**; F6 |

Thus the inner training levers other than `verdict_batch_size` are wired. The overall fire claim still
fails because the orchestrator advertises the child rather than the controller that implements the
ticket's wall-cap semantics (F2).

## A2 — probe/fire peak-memory equivalence after the cure

**Conclusion: FINDING** (F1 and F6).

The exact/float tuple is centralized and the same constants feed comparison and classification
(`mx1_fire_guard.py:310-340`, `372-404`). That closes the old hand-list drift, but the semantic partition
does not match the current trainer:

| key | does equality bear on the current peak? | derivation |
|---|---|---|
| `mode`, `device`, `pairs` | YES | select execution path, allocator/device, and tensor population |
| `bits` | conservative | changes quantizer constants, not tensor shapes; no false clearance from comparing it |
| `microbatch_pairs` | YES | determines chunk tensor/gradient footprint |
| `microbatch_policy` | NO at this ticket | explicit `microbatch_pairs=4` wins; policy is metadata (`trainer:2767-2782`) |
| `cache_residency`, `microbatch_hygiene`, `microbatch_chunk_cache` | YES | change retained caches and cleanup |
| `verdict_batch_size` | NO | inert on `mlx-train` (F6) |
| `float_warmup_steps`, `train_compute_dtype`, `compile_train_loss` | YES | change quantization branch, dtype, and compiled graph |
| `perf_thread_pin` | conservative | changes CPU execution/thread state; comparing it cannot false-clear |
| `allow_soft_mem_limit` | safety-relevant | changes enforcement semantics even if not the workload graph |
| `input_cache`, `target_cache`, `init` | YES | determine loaded arrays/model configuration |
| `lr` | conservative | numeric optimizer input; topology is unchanged |
| `ce_fraction`, `softplus_fraction` | YES | can select different loss graphs |
| `mem_budget_gb` | YES | controls allocator/software-cap configuration |

Exclusion audit:

| excluded key | reason true? | conclusion |
|---|---|---|
| `out` | YES | final evidence path only |
| `mem_probe_steps` | conditionally | probe-only, but the chosen horizon must still reach every peak-bearing phase; this one does not |
| `seed` | YES for allocation shape | fixed `pairs` and fixed-shape caches keep allocation geometry |
| `fire_guard_verdict` | YES | guard-output identity path |
| `steps` | **NO** | changes accumulated history/journal and whether K8 materialization is reached |
| `run_dir` | **NO** | determines checkpoint member population and sacred-run state |
| `resume_from` | **NO** | loads prior history/optimizer and paired EMA state |
| `eval_every` | **NO** | changes journal/controller row population and decision/checkpoint cadence |
| `checkpoint_every` | **NO** | changes member count and K8 activation |
| `launch_ticket_path`, `fire_argv_key` | **NO** | activate/select the controller and EMA policy on the fire but are disabled by probe mode |

The outer `rss_mb`, `timeout`, and `projected_gib` safety values are in neither set because the wrapper is
discarded before classification. F1 is therefore a real false-clear path, not merely over-strict equality.

## A3 — classification-gate coverage

**Conclusion: FINDING** (F4).

`--flag=value` is parsed, hyphens normalize to underscores, and repeated long store-options follow the
same last-value behavior as argparse (`mx1_fire_guard.py:73-91`). Abbreviated long flags remain shorter
unclassified names and are refused; an extra inner `--` produces an unclassified empty key; and ordinary
extra trainer positional arguments are later rejected by argparse. Those shapes are not the bypass.

The bypass is the command prefix: valid Python single-dash runtime options and executable wrappers are
ignored by the map but consumed before trainer argparse. The real-guard `-X tracemalloc` control passed as
shown in F4. The classification gate therefore sees fewer memory-affecting options than the process
actually receives.

## A4 — resume and per-stage checkpoint durability

**Conclusion: FINDING** (F3).

The implementation does provide distinct step-encoded live, EMA, and K8 files (`trainer:3231-3241`),
periodic and controller-halt saves (`:3556-3560`), real EMA updates (`:3453-3457`), optimizer/history in
the live NPZ (`mlx_semantic_renderer.py:358-399`), load support (`:403-445`), and temp+rename atomicity per
individual file. It is not loop-end-only.

But the required resumable unit is the live+EMA pair, and its latest pointer is published between those
two writes. F3's crash window makes the exact ticketed resume path unloadable. Per-file atomicity is not
bundle durability.

## A5 — measured memory preflight

**Conclusion: FINDING** (F1).

The recorded receipt is genuinely n120, GPU-requested, fp16, microbatch 4, per-step hygiene, and chunk
cache, with 124 samples and measured MLX/RSS peaks (`mem_probe_receipt.json:2-23`, `227-231`,
`1851-1955`). It is not a formula-only B=8 estimate.

It is nevertheless not the fire's own saturated program: probe mode structurally disables the
controller/EMA/K8 path. The guard checks the receipt's measured-telemetry existence but never consumes its
peak to validate the ticket's 16.0 GiB safe-run reservation, and it does not bind the local source version.
The current receipt therefore cannot clear this post-cure fire.

## A6 — seal-orchestrator fail-closed behavior

**Conclusion: FINDING** (F2 and F5).

Missing receipt files are `PENDING`, wrong explicit statuses block or remain pending, stale mem-probes and
harvests are re-requested, and unmet dependencies cannot report satisfied
(`ddm_seal_orchestrator.py:267-314`, `317-359`, `422-455`). Those are clean.

Schema-malformed but superficially favorable JSON is not fail-closed: status-only guard/mem-probe receipts
and an empty falsifier map can satisfy gates (F5). The orchestrator also joins the final gate to the wrong
execution argv (F2). Consequently its own `READY` claim is not trustworthy yet.

## RECALL EVIDENCE

| scope | queries / sources | found beyond charter seeds | plan impact |
|---|---|---|---|
| Full mechanics corpus | Content searches for `mx1_fire_guard`, `launch_ticket_v5_event_driven`, `mem_probe_steps`, `one_sample_flip_S`, `M1R2-F2`, and `ddm_m1r4` over `.omx/research/`. | Found the earlier provenance-stop receipt, all round-4 reviews, and the later M1C1 cure receipt. The pinned hashes are the M1C1 post-cure artifact, not the pre-cure artifact described by older round prose. | Reviewed the live controller/EMA/checkpoint additions from source instead of re-reporting the earlier inert-stop finding. |
| Prior review genus | Opened `ddm_m1r2_20260808/M1R2_REVIEW.md`, `ddm_m1r4_20260808/M1R4A_REVIEW.md`, and `ddm_m1c1_20260808/M1C1_CURES.md`; searched RR10/RR11/RR13-RR15 surfaces for guard/receipt/resume antecedents. | Prior work focused on long-option tuple completeness, stale verdicts, and executable stopping. It did not close full command grammar, controller-enabled memory parity, bundle publication, or strict receipt schemas. | Turned those seams into the adversarial controls in F1-F5 rather than treating prior tests as authority. |
| Canonical equations | Ran `.venv/bin/python tools/list_canonical_equations.py --json` and filtered for `memory`, `resume`, `checkpoint`, `ema`, `probe`, `peak`, `fire`, `event`, `stopping`, and `trajectory`. | Found the run-geometry EMA law used by M1C1 and the standing complete-checkpoint doctrine; no equation authorized a non-controller probe as fire-memory authority. | Kept the EMA mechanism but audited its residency and transactional custody as part of the actual peak/resume program. |
| Research index/DAG/ledgers | Searched `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, task/landing/P0 ledgers for `M1`, `MX1`, `semantic_renderer`, `mem-probe`, `resume_from`, and `stage checkpoint`. | Found older M1 fire-chain and EMA/checkpoint precedents, but did not find a later artifact in that bounded scope that supersedes the pinned M1C1 files. | Used the pinned content as authority; no older fire receipt was promoted into a current clearance. |
| Live board | Read `.omx/state/main_hot_state.md`. | The retained board still described M1C1 as live while the pinned hashes and Git object `393d67d016` already contain the cure. | Treated the board as recall, not source authority; re-derived current state from content hashes and source. |
| Tests as attack surface | Searched existing guard/orchestrator/M1C1 tests for classification, schemas, controller routes, and checkpoint atomicity. | Existing orchestrator fixtures deliberately call schema-less `{"status":"passed"}` objects passing receipts; no test covered empty harvest falsifiers or command-prefix options. | Added transient both-direction controls without modifying the reviewed files or test suite. |

## Controls and self-attack

- Four SHA-256 pins matched before review.
- Transient real-guard control: injected `-X tracemalloc`; guard returned
  `passed/fire_guard_passed`, `flag_classification_ok`, `unclassified=[]`.
- Transient seal controls: missing guard receipt -> `pending`; status-only guard -> `satisfied`; empty
  harvest -> `satisfied`.
- Source-derived gate control: absent top-level fire key selected child
  `argv_m1_n120_cap_saturated`, not controller `argv_m1_controller_fresh`.
- Attacked the findings for false positives: extra exact-key equality is conservative and was not called a
  false-clear; missing receipts themselves do not pass; ordinary trainer positionals are rejected; each
  finding above has a path that survives those boundaries.
- All transient files were created and removed under a temporary directory. No persisted evidence cites a
  temporary path.

## Lens boundary

This pass covered mechanics only: ticket/config routing, safe-run and trainer argv consumers, guard
equivalence/classification, controller routing, checkpoint/resume custody, measured-memory preflight, and
seal receipt gates. It did **not** adjudicate the stopping rule's scientific validity, learning-rate or
EMA arithmetic, statistical bars, Seg/Pose authority, score arithmetic, archive bytes, receiver/public-wire
behavior, or contest CPU/CUDA promotion. No finding here kills the vehicle family; all verdict scopes are
instance/formulation/apparatus-class as stated.

Follow-ons are **QUEUED-WITH-A-FIRE-ORDER**: MAIN cures F1-F7, freezes new content hashes, re-runs a fresh
controller-complete and resume-complete measured memory preflight, then starts a new three-pass sequence
at `0/3`. No M1 fire is admissible before all of that is clean.

The exact contest pointer did not move. Nothing byte-closed or score-bearing was measured in this review.

Own-vehicle frontier unchanged: **S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]**.
