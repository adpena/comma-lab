# ddm_rg2 — RED developer-gate triage (task #860, blocks #905)

**Date:** 2026-08-16 · **Arm:** `ddm_rg2_red_gate_triage` · **Axis:** apparatus. No score claim; the
exact pointer is untouched by this unit and this work is a MEANS, not goal progress.

---

## Headline

**The filed numbers do not reproduce. MEASURED today: 8 of 25 dev gates RED, 231 violations — not
6 RED / ~316.** After landing the hygiene I own: **7 RED, 210 violations, 22.7 s wall clock.**

**Can the commit hook be flipped on? No — and timing is not what is stopping it.** The dev scope
already fits the budget (22.7 s vs 30 s). What stops it is that **the remaining RED gates cannot be
cured by fixing repo code**, because they fire on things a commit does not control: files outside
the repository, an untracked file, a doc the arm may not edit, a helper rename the gate was never
taught, and a sister arm's file that is still inside its own review cycle. Two are genuine open
design questions. Details in §5–§6.

Live confirmation of #905's premise, captured from the commit hook during this unit:
`[preflight-hook] preflight examined 0 gates this commit (fast --no-codebase mode). This hook is
NOT gate coverage.` The hook says so itself.

---

## 1. Method (canonical invocation, re-derived not assumed)

The prompt's `PREFLIGHT_FULL=0 ... --scope dev` was not verified against the code. Read
`src/tac/preflight.py:80353-80465`: `--scope` defaults to `dev` already, and no `PREFLIGHT_FULL`
env var exists on this path. The canonical invocation is:

```bash
.venv/bin/python -m tac.preflight --scope dev
```

Per-gate violation lists were collected by calling each check directly with `strict=False`
(they return `list[str]`), so counts are the gates' own, not parsed from a truncated
"first 3" error message.

Aggregation matters here: `_raise_aggregated_dev_gate_failures` (landed for task #852) is why a
full RED set is visible at all. Before it, `--scope dev` raised the FIRST red gate and abandoned
the rest — which is how a plan came to be written against one gate's violation count.

## 2. The reproduced RED set (MEASURED 2026-08-16)

Pre-fix run: **8 RED / 25 declared**, 24.3 s. Post-fix run: **7 RED / 25 declared**, 22.7 s.

| # | Gate | Violations | Class | State |
|---|------|-----------:|-------|-------|
| 1 | `check_state_writers_strict_load_for_mutating_path` | 1 | HYGIENE, **not mine to land** | RED (§5.7) |
| 8 | `check_substrate_trainer_pose_defaults_match_contest_formula` | 21 | HYGIENE | **FIXED** |
| 2 | `check_authoritative_tag_requires_custody_metadata` | 1 | **POLICY** | RED |
| 3 | `check_codebase_drift` | 25 (21 distinct files) | **STALE-GATE** | RED |
| 4 | `check_dispatch_claim_helper_present` | 1 | HYGIENE, **not mine to land** | RED |
| 5 | `check_subagent_landing_has_solver_wire_in` | 124 | **POLICY (out-of-repo scope)** | RED |
| 6 | `check_lane_pre_registered_before_work_starts` | 2 (1 root fact, untracked file) | **STALE-GATE** | RED |
| 7 | `check_substrate_score_aware_losses_use_canonical_scorer_contract` | 56 (52 = 1 root fact) | **STALE-GATE** + 1 POLICY | RED |

**Total 231 pre-fix → 210 post-fix.**

### 2a. Why the filed count was wrong, and the honest read of that

The dev-list wire-in comment records the 2026-08-01 state as *"19 green / 6 red / 2
not-independently-invocable of 27 declared."* That reconciles exactly with today's 25 (= 27 − the
2 that are not separate `dev_checks` entries), so the gate LIST has not changed. What changed is
the red set:

- `check_state_writers_strict_load_for_mutating_path` regressed **today**, in commit
  `17eebd418b` (a sister arm's deferral-expiry work added a new `_save_*` writer). MEASURED —
  `git log -S` on the function name.
- The 21 pose-default violations are **old** (files first added 2026-05-20; the gate has been in
  place since `c75d2e0c85`, 2026-05-12). So that gate was either already red on 2026-08-01 and the
  earlier enumeration missed it, or the enumeration was partial. I cannot distinguish these from
  the artifacts available. **Labelled INFERRED.**

The `~316` violation figure did not reproduce at all (231 measured). **Operating conclusion: do
not plan against the filed figures. Re-measure. The RED set moves on a timescale of one day** —
one of the eight arrived within hours of this triage.

### 2b. Counting discipline — one fact fanning out

The prompt's warning was correct and it bites twice:

- **Gate 3** reports 25 rows over **21 distinct files**; four files
  (`launch_tt1_detached.sh`, `launch_v4c_detached.sh`, `launch_v4d_detached.sh`,
  `run_capstone_capacity_ablation_2x2.sh`) are counted **twice**, once by the forbidden-pattern
  rule and once by the bash-allowlist rule.
- **Gate 7** reports 56 rows, of which **52 are one fact** (a helper was renamed and the gate was
  never told). See §5.3.
- **Gate 6** reports 2 rows for **one** token in one file.

Naive violation-summing overstates the real debt by roughly 3× on this population.

## 3. What I fixed (HYGIENE only)

**(a) `check_substrate_trainer_pose_defaults_match_contest_formula` — 21 → 0.**
21 substrate trainers declared `--gamma-pose` with argparse `default=1.0`, silently overriding
the correct dataclass default (`CONTEST_POSE_SQRT_WEIGHT = math.sqrt(10.0)`,
`score_aware_common.py:43`). The contest score is `100·d_seg + sqrt(10·d_pose) + rate`, so in the
score-domain Lagrangian `γ·sqrt(d_pose)` the DERIVED value is `γ = sqrt(10) ≈ 3.1623`. A default
of 1.0 trains against a pose weight 3.16× too small. This is exactly the "default override
antipattern" CLAUDE.md names.

The fix is not a taste choice — the value is derived from the contest formula, and green sibling
trainers (`block_nerv`, `siren`, `hi_nerv`, `ds_nerv`, `balle_renderer`, …) already carry
`default=math.sqrt(10.0)`. I copied their exact form rather than inventing one.

Default-override safety check, as CLAUDE.md requires: I grepped every `--gamma-pose` caller in the
repo. **Exactly one** passes it explicitly
(`scripts/remote_lane_substrate_time_traveler_l5_z7_lstm_predictive_coding.sh:419`) and that
trainer is **not** among the 21. So **zero of the 21 changed files have an explicit caller** —
MEASURED, no silent behavior change to any wired invocation.

Note for the gate's own record: `_check_174_default_is_sqrt10` accepts a float constant or a
`sqrt(10.0)` call, but **not** a `Name` reference. So the provenance-ladder-preferred form
(`default=CONTEST_POSE_SQRT_WEIGHT`) would keep the gate RED. Minor, worth knowing before someone
"improves" these to the named constant and is refused.

That is the only thing I landed. The state-writer gate's cure is written and verified but is
**not mine to commit** — see §5.7.

Ruff `--select F` clean on the 21 trainers; all 21 parse.

## 4. What I did NOT do, on purpose

**I landed zero waivers.** I did not mass-waiver, and I did not widen any gate's exemptions. I did
not touch gates 2/3/5/7 — their cures are decisions, and a gate quietly widened to go green is the
failure this apparatus exists to prevent. I wrote two waivers during the unit and reverted both
once measurement showed one was inert and the other was not mine to land.

I also **reverted** two waiver comments I had written into
`src/tac/witness_dsl/ep725_levelset_predictor_adapter.py`. Re-deriving the gate's logic showed the
per-line waiver is honored **only** on test paths (`if is_test and _line_or_window_has_fake_waiver(...)`,
`preflight.py:42121`). My waivers were **inert**. Leaving an inert waiver is worse than leaving the
gate red — it reads as protection that is not in force. That file is also **untracked** and sits in
a directory with ~40 other untracked modules, so it may belong to an in-flight sister arm; it is
now byte-restored and I did not otherwise edit it.

## 5. POLICY / STALE-GATE decisions owed to MAIN

### 5.1 Gate 2 — `check_authoritative_tag_requires_custody_metadata` (1) — **POLICY**

The gate demands every authoritative-tag site route through
`ContestResult.validate_custody` / `posterior_update`. The single hit is
`src/tac/submission_chain.py:230`, inside `axis_and_authority(device)` — a pure classifier that
maps a device to `("[contest-CUDA]", "authority")` / `("[macOS-CPU advisory]", "advisory")`, raises
on MPS, and refuses to infer one axis from the other. It persists nothing and emits no score. But
`grep validate_custody src/tac/submission_chain.py` returns **nothing across all 1,078 lines**: the
canonical submission surface decides "authority vs advisory" with its own logic and never consults
the custody validator. **Option A:** that is the real defect — wire `submission_chain` through
`validate_custody` (real integration work on the file MEMORY calls canonical, and the honest read
of the gate's intent). **Option B:** the axis classifier is a legitimately distinct primitive that
*enforces* the axis discipline rather than bypassing it, and earns a
`# CUSTODY_VALIDATOR_OK:<reason>` waiver. I recommend A and note it is not cheap; B is defensible
only if someone confirms no caller of `axis_and_authority` promotes on its verdict.

### 5.2 Gate 3 — `check_codebase_drift` (25 rows / 21 files) — **STALE-GATE**

`FORBIDDEN_FILE_PATTERNS` (`preflight.py:8403`) bans `experiments/launch_*.{sh,py}`,
`experiments/run_*.sh` and any bash in `experiments/` outside a five-entry submissions allowlist,
with the message *"Use scripts/deploy_vastai.py + pipeline.py instead."* That encodes the
2026-04-era Vast.ai deployment convention. The repo has since moved to the governed launcher +
local Metal, and the compute-split law now says Modal only when local is impossible. **The gate is
now refusing files that CLAUDE.md itself names as canonical**: the "MPS is a VALID TRAINING-GRADIENT
device" non-negotiable cites `experiments/launch_split_by_head_basin.py` as *"the canonical
reference any local substrate mirrors."* A STRICT gate and a non-negotiable cannot both be right.
The rest of the population is the same shape (`launch_mlx_witness_fleet.py`,
`experiments/manim_levelset/render.sh` — a viz renderer, `experiments/stage_*.sh` — staging
scripts). **Proposed narrowing:** match on **deploy semantics** (invokes a provider CLI —
vastai/modal/lightning, ssh-to-remote, GPU-instance create) rather than on filename glob +
directory. That is the gate's own stated intent: its inline comment says the rule exists to prevent
*"ad-hoc deploy scripts creeping in alongside the canonical pipeline."* **This narrowing does not
weaken the gate's live half:** `_scan_bash_text_for_forbidden` (the text-pattern rule that catches
inline `curl | sh` bootstraps) produced **0 of the 25** violations and is untouched by the proposal.
I did not land this; the scope call is MAIN's.

### 5.3 Gate 7 — scorer contract (56, of which 52 are one fact) — **STALE-GATE + 1 POLICY**

The gate requires an AST call named exactly `score_pair_components`
(`preflight.py:14635,14639`). **52 of the 56 flagged files DO route through the canonical contract**
— they call `score_pair_components_dispatch(...)`, documented at
`score_aware_common.py:234` as *"Canonical dispatch entry point for substrate score-aware losses
(F3)"*, which routes to `score_pair_components_with_cache` or `score_pair_components` and is
documented as *"mathematically identical."* The dispatcher landed 2026-05-14 (the GT-cache wire-in);
the gate was never taught its name. **Proposed fix: add `score_pair_components_dispatch` to the
accepted names.** This is intent-*restoring*, not widening — the gate asks "does this route through
the canonical scorer contract?", and the dispatcher **is** that contract now. I did not land it
because flipping 52 violations green in a two-line gate edit is a blast radius MAIN should sign,
and because a reasonable alternative exists (accept only the dispatcher, and treat direct
`score_pair_components` calls as the stale form).

The residue is genuinely 4, not 56:
- `atw_codec_v2`, `time_traveler_l5_autonomy` — route one level of **indirection** deeper, through
  `tac.codec.cooperative_receiver.atick_redlich`; `atw_codec_v2:42` already carries a
  `# SCORER_PREPROCESS_HANDLED_OK:` comment saying exactly this, which this gate does not read.
  Same root class: the gate follows zero levels of indirection.
- `tishby_ib_pure` — imports no scorer at all. Needs a look: either a pure-IB loss that legitimately
  never touches the scorer (then the gate's file-name-based scope is the issue), or a real gap.
- `d1_segnet_margin_polytope:158` — **the one genuine POLICY item.** It calls
  `self.seg_scorer(seg_input)` directly, *after* the canonical
  `self.seg_scorer.preprocess_input(...)`, to extract a top1-minus-top2 **logit margin** — a
  quantity `score_pair_components` does not return. Under CLAUDE.md's UNIQUE-AND-COMPLETE-PER-METHOD
  operating mode this is a textbook principled FORK (the canonical helper cannot express the
  substrate's distinguishing primitive). **Option A:** waive it as a documented fork. **Option B:**
  extend the canonical helper to expose margins so the fork is unnecessary. A is right if d1 stays
  dormant; B is right if margin-based losses generalize.

There is a broader question under all of this that MAIN should answer once rather than per-gate: a
STRICT gate that refuses every substrate not routing through a canonical helper is, structurally,
the canonicalization-by-default reflex that CLAUDE.md's UNIQUE-AND-COMPLETE-PER-METHOD section
identifies as the cause of the 0.196–0.199 plateau. The gate and that non-negotiable pull in
opposite directions. Worth a ruling.

### 5.4 Gate 5 — `check_subagent_landing_has_solver_wire_in` (124) — **POLICY, and the hardest one**

This gate scans `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_*_landed_*.md`
(resolved at `preflight.py:41449-41452`). MEASURED: that directory is
**outside the repository** (`inside repo: False`), holds **800** landed memos, and is not under git
at all. 124 of the 800 are missing at least one of the six wire-in declarations.

Three properties make this unfixable as hygiene, and each one independently blocks #905:
1. **Not commit-scoped.** A commit hook wired to this gate would refuse a commit based on files
   the commit does not contain and git does not track.
2. **Not host-portable.** The same repo commit gets a different verdict on a machine with a
   different memory directory. That breaks the deterministic-reproducibility spine.
3. **Monotonically growing and APPEND-ONLY.** Those memos are HISTORICAL_PROVENANCE under Catalog
   #110/#113 — retro-editing 124 of them to add declarations is forbidden, and even if it were
   done, the count returns with the next landing memo that omits a hook.

**Option A:** scope the gate to memos created in the current commit / current session (commit-scoped
enforcement, historical corpus exempt) — this preserves the gate's real purpose, which is to force
the declaration *at landing time*, not to audit history forever. **Option B:** remove it from the
dev/commit scope and keep it in `--scope all` release/custody sweeps only. **A is strictly better**:
B loses the at-landing-time enforcement that is the whole point. Either way, **this gate must move
before the hook can be flipped on.** It is not negotiable-around.

### 5.5 Gate 6 — `check_lane_pre_registered_before_work_starts` (2 rows, 1 fact) — **STALE-GATE**

`lane_render_band` is **not a lane**. It is one of four LVLS1 archive-manifest **block keys** —
`("lane_render_band", "pose_carrier", "chart_payload", "palette_residual")` — naming the render band
for SegNet **class 1 (lane markings)**, passed straight into `_take_block(..., label=key)`. Its three
siblings are obviously archive sections. Confirmed against the registry: **2,275 lanes are
registered and none contains "render" or "band."** The gate matched the `lane_` prefix
syntactically. This is the sister of the precedent named in my charter.

There is **no cure available inside the repo.** (i) Registering `lane_render_band` would manufacture
a fiction in a 2,275-row registry that other gates read — a NO-FAKE violation to satisfy a grep.
(ii) The per-line `# FAKE_LANE_OK:` waiver is honored **only on test paths** (verified above), so it
is inert here — I tried it and reverted it. (iii) The designed home is
`_LANE_ID_REFERENCE_BLOCKLIST` (`preflight.py:41621`), but that list is curated *generic identifiers*
(`lane_id`, `lane_name`, `lane_dir`), and adding a domain token to it is a repo-wide permanent
exemption — a scope decision. **Recommended:** allow the per-line `FAKE_LANE_OK` waiver in
production source (it already requires a written rationale, which is the actual safeguard), OR add
the token to the blocklist. **Compounding fact:** the offending file is **untracked**, so a commit
hook would today block commits over a file that is not being committed.

### 5.6 Gate 4 — `check_dispatch_claim_helper_present` (1) — HYGIENE, **routed, not landed**

The gate requires the literal string `"newer terminal row as closing"` in `AGENTS.md`
(`preflight.py:29359`). It is absent. Cause, MEASURED: `AGENTS.md` is now byte-identical to
`CLAUDE.md` (md5 `2b619524379eeb2a0f1a8cd72e04ea22`) after commit `48238b1213`
*"AGENTS.md: sync to exact copy of CLAUDE.md (operator directive 2026-07-25)"*. The phrase lived in
the older standalone `AGENTS.md`; `CLAUDE.md` never had it, so the operator-directed consolidation
dropped it. **The semantic is real and live**, not a magic token to paste:
`tools/claim_lane_dispatch.py:367` documents *"a newer terminal row closes an older nonterminal
row"*, implemented via `TERMINAL_PREFIXES` / `_is_terminal`. CLAUDE.md's dispatch section states the
obligation ("append a terminal row… do not leave completed jobs as phantom active claims") without
naming the closure rule the reader implements.

So the cure is one honest clause added to the CROSS-AGENT DISPATCH COORDINATION section, e.g.:

> The claim reader treats a **newer terminal row as closing** an older non-terminal row for the same
> `lane_id` / `instance-job_id`; that is why a terminal row is the closure and not a separate delete.

**I did not land it: that is an edit to CLAUDE.md/AGENTS.md, which no agent instruction may
authorize me to make.** Operator or MAIN owns this one. It is a one-line paste and it clears a RED
gate for the right reason.

### 5.7 Gate 1 — `check_state_writers_strict_load_for_mutating_path` (1) — HYGIENE, **routed**

`src/tac/probe_outcomes_ledger.py:1223`, `_save_appended_events_preserving_existing_text`. This is a
false positive **in the safe direction**: the function's "load" is a raw `read_text()` whose bytes
are re-emitted **verbatim** ahead of the new rows and are never parsed, so it structurally cannot
reset a corrupt ledger to `[]` — the exact harm the gate exists to prevent. A strict *parsing* load
here would **add** the drop-on-corrupt risk. The cure is the same-line waiver the gate's own
docstring provides for genuinely-additive writers; verified to take the gate 1 → 0:

```python
def _save_appended_events_preserving_existing_text(  # STATE_WRITER_STRICT_LOAD_OK:pure-append; the read is raw read_text() bytes re-emitted VERBATIM ahead of the new rows and never parsed, so a corrupt ledger cannot be reset to [] by this writer. A strict parsing load here would ADD the drop-on-corrupt risk the gate exists to prevent.
```

**I reverted it rather than commit it.** The file landed hours ago (`17eebd418b`) and is still inside
its own review cycle — the review gate reports `query_expired_deferrals: needs 1 more clean pass
(1/2)`. Committing my one-line waiver would have required me to certify a clean pass on a sister
arm's function, and on reading it I have a finding, so a clean pass would have been dishonest. The
file is byte-restored; the waiver above is ready to paste by its owner.

**The finding, handed back (MEASURED, not styled):** `query_expired_deferrals` guards its
`days_expired` computation with `except ValueError`, but the actual failure mode for a
parseable-yet-**naive** `expires_at_utc` is **`TypeError: can't subtract offset-naive and
offset-aware datetimes`**, which is not caught. One naive timestamp anywhere in the ledger crashes
the entire nag query — in a function whose whole purpose is resilience over 728 heterogeneous
historical rows. Reproduced:
`datetime.fromisoformat("2026-01-01T00:00:00".replace("Z","+00:00"))` → naive → `now - parsed`
raises TypeError. Suggested: `except (ValueError, TypeError)`, or normalize to aware before
subtracting. Secondary, lower severity: the expiry test `expires_at > now_iso` is a **string**
comparison, so an `+00:00`-form timestamp (rather than `Z`) sorts below `now_iso` and would be
reported as expired unconditionally. Safe for rows this module writes; latent for any written
elsewhere.

## 6. Can the commit hook (#905) be flipped on?

**Not today.** Timing is genuinely no longer the blocker — MEASURED 22.7 s against a 30 s budget,
consistent with MAIN's warm 19.87 s. The blockers are these, in the order they must fall:

0. **Gate 1 needs one paste by the owner of `probe_outcomes_ledger.py`** (§5.7) — cure written and
   verified, blocked only on that file's own open review cycle. Cheapest item, plus a real bug
   handed back.
1. **Gate 5 must be re-scoped** (§5.4). It gates commits on 800 un-versioned files outside the repo,
   on a per-host directory, with an APPEND-ONLY corpus that cannot be driven to zero. This one is
   structural: no amount of repo hygiene reaches it.
2. **Gate 6's untracked-file exposure** (§5.5). A hook that refuses a commit because of an untracked
   file's contents will fire constantly — `src/tac/witness_dsl/` alone currently holds ~40 untracked
   modules plus ~30 untracked tests.
3. **Gate 7 needs the two-line dispatcher-name decision** (§5.3), or 52 correct files stay red.
4. **Gate 3 needs the deploy-semantics narrowing** (§5.2), or the hook refuses commits touching a
   launcher CLAUDE.md itself calls canonical.
5. **Gate 4 needs one clause in CLAUDE.md** (§5.6) — cheapest item on the list, wrong owner for me.
6. **Gate 2 needs a real decision** (§5.1) — the only one of the six that is plausibly a true defect
   in live code.

A defensible intermediate that unblocks #905 without weakening anything: **flip the hook on with a
named subset** — the 18 currently-green gates plus the one I just cleared — and hold the seven under
adjudication out of the *commit* scope while leaving them in `--scope all`. That gives commits real
coverage now instead of the current effectively-zero, and it does not launder a single unresolved
decision into a pass. It also makes the remaining six visible as a short, owned list rather than an
undifferentiated "316 violations" that nobody can act on.

One standing risk, MEASURED rather than argued: **the RED set moved twice during the window this
task was open.** One gate (§3b) regressed within hours of this triage, from a sister arm's landing.
Whatever subset is chosen, the hook needs the aggregated reporter (already landed) plus a
regression alarm, or a green subset silently becomes a red one on the next commit.

## 7. Honest state at close

- **Fixed and landed:** 1 gate, 21 violations (the substrate `--gamma-pose` defaults).
- **Still RED:** 7 gates, 210 violations — of which the true independent-decision count is **7**
  (one per gate), not 210. The population is dominated by three single facts fanning out ×52, ×124,
  and ×25-with-4-double-counted.
- **Reverted:** 3 waiver lines — 2 that measurement showed were inert (test-path-only waiver
  mechanism), 1 that was correct but sat in a sister arm's file mid-review.
- **Handed back:** one real uncaught-`TypeError` defect in `query_expired_deferrals` (§5.7).
- **Exact pointer:** unmoved. This unit is apparatus. It does not lower the score and does not claim
  to.
- **Re-measure command:** `.venv/bin/python -m tac.preflight --scope dev` → at close
  **RC=1, 7 of 25 RED, 22.7 s.**

---

### Minor observation, not a decision

CLAUDE.md's "Mutation frontier" list does not include `src/tac/**`, yet several non-negotiables
(the STRICT-gate two-landing pattern) direct arms to edit `src/tac/preflight.py`, and current
practice does so constantly. The frontier list appears stale rather than binding-as-written. It did
not bind this unit in the end — everything I landed is under `experiments/**` and `.omx/**` — but
the inconsistency is worth resolving before the next arm reasons from it.
