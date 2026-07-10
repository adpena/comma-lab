# Hardcode + duplication hygiene audit — witness/level-set stack (2026-07-10)

Operator directive (verbatim): *"We are duplicating and hardcoding way too much. Terrible
hygiene. Bad code smell."* Scope: the live witness/level-set stack — the two trainers
(`experiments/train_levelset_witness_realized_through_R_mlx.py`,
`experiments/train_witness_realized_through_R_mlx.py`), the DSL/autoconfig/launcher/byte-close
tool ring around them, and the `witness_control`/`witness_curriculum`/`boundary_math`/
`local_acceleration` support modules. **READ-ONLY audit — no code was edited.**

**Already canonicalized, not re-flagged**: `src/tac/witness_run_artifacts.py` (run-artifact
filenames — being migrated by another agent right now), `src/tac/clip_profile.py` (per-clip
measured geometry/class-order), `tac.witness_dsl` (levers/flags as the DSL's job),
`tac.canonical_equations` (measured laws), `.omx/state/canonical_frontier_pointer.json`
(scores). **Already triaged by a prior sweep, not re-litigated in depth**: the `--tau/--l7`
dashboard-consumer hardcode class and the `874/1164`-in-display-tools class, both covered by
`.omx/research/hardcoded_run_constants_sweep_20260707.md` (task #340, `run_constant_gates.py`
WARN-gate). This memo covers **different** classes that sweep did not touch.

## Executive summary — top 5 by EV (blast-radius × drift-proneness)

1. **`witness_autoconfig._proven_base()` is a hand-maintained shadow copy of
   `curriculum_dsl.BASELINE`** — the two dicts encode ~overlapping-but-not-identical sets of
   "the proven sealed config" with **zero cross-check test**, and they have already started to
   diverge (`--w-pose`/`--verdict-pairs` present in one, absent in the other). This is the
   `config-must-be-DSL-defined` non-negotiable being violated **inside the DSL's own actuator
   module**. Highest EV: it is the literal single source of truth for what gets launched.
2. **The trainer's relative path string is re-declared as an independent module constant in 8
   non-test files** (`curriculum_dsl.TRAINER_REL`, `gauge.LEVELSET_TRAINER_REL`,
   `v2_compose/launch_command.DEFAULT_TRAINER`, `confound_gates` bare string,
   `launch_witness_run._TRAINER`, `tools/witness_autoconfig._TRAINER`,
   `mlx_gpu_determinism_probe._TRAINER`, `levelset_heldout_codefit_gate` bare string) plus ~40
   test files. A rename of the trainer file requires hand-editing 8+ production call sites.
3. **The trainer-argparse-flag-scanning regex is copy-pasted verbatim in 3 places**
   (`curriculum_dsl.py`, `tools/launch_witness_run.py`, `tools/witness_autoconfig.py`) — the
   exact same `re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', ...)` — instead of one shared
   `real_trainer_flags()` helper that already exists in `curriculum_dsl.py`.
4. **The fcntl-locked append-one-JSONL-row helper is reimplemented per-ledger** across
   `witness_dsl`/`witness_control` (byte-identical function name + docstring in
   `activation_ledger.py` and `curriculum_candidate_pool.py`; inline near-duplicates in
   `costate_posterior.py`, `shadow_controller.py`, `campaign_repl.py`, `decode_cache.py`) —
   this is a witness-stack-local instance of a much larger repo-wide pattern (~85-95 files use
   `fcntl.LOCK_EX` directly; out of narrow scope but worth naming).
5. **Canonical JSONL store-path constants exist but consumers re-spell the literal path
   string instead of importing them** — `CANONICAL_EQUATIONS_REGISTRY_PATH` (owned by
   `canonical_equations/registry.py`) is re-spelled in `dashboard_server.py` and
   `preflight.py`'s Catalog #359 check; `DEFAULT_LEDGER_PATH` (owned by
   `harness_failure_ledger.py`) is re-spelled in `tools/convene.py` and
   `witness_control/producer_bridge.py`.

Everything below is the full ranked table; items 6-8 are real but lower blast-radius, and the
"checked and NOT flagged" section documents hypotheses this audit falsified (so the next agent
doesn't re-check them).

## Ranked inventory

| # | literal/pattern | files + counts | existing canonical home | drift risk | cheapest fix | load-bearing |
|---|---|---|---|---|---|---|
| 1 | Full "proven base" launch-config dict (`w_seg=100, hosc_beta=4.0, freq_across=32, freq_along=4, eikonal_weight=0.01, length_weight=0.001, render_h=384, ema_decay=0.997, ...` — 28 keys) | `src/tac/witness_autoconfig.py:1006 _proven_base()` (28 literal keys, comment "recalled verbatim from the 0.003698 run_muon.log") vs `src/tac/witness_dsl/curriculum_dsl.py:1614 BASELINE.base` (a DIFFERENT, ~24-key dict with different key names, e.g. `--w-pose: 1.0` and `--verdict-pairs: 96` absent from `_proven_base()`). **No test cross-checks the two.** (Verified: `grep -rln _proven_base src/tac/tests` finds only tests that call `_proven_base()` in isolation, none that diff it against `BASELINE`.) | none — should be `curriculum_dsl.BASELINE` | **HIGH.** Whoever tunes `BASELINE` (the DSL, the thing CLAUDE.md says config must flow through) does not touch `_proven_base()`; the autoconfig actuator can silently emit a config that no longer matches "the proven sealed config" it claims to reproduce. Already observably diverged on `w_pose`/`verdict_pairs`. | Needs design, not pure mechanical: `_proven_base()` should derive its dict from `BASELINE.base` (translating `"--eikonal-weight"` key style to `"eikonal_weight"` key style) rather than re-typing values; requires understanding why the two key-naming conventions differ and whether `_proven_base()`'s consumer (`WitnessConfig.to_trainer_flags`) can consume `BASELINE.base` directly. | **YES — this IS the launch config.** |
| 2 | `"experiments/train_levelset_witness_realized_through_R_mlx.py"` re-declared as an independent constant | Non-test: `curriculum_dsl.py:38 TRAINER_REL`/`TRAINER_PATH`, `gauge.py:847 LEVELSET_TRAINER_REL`, `v2_compose/launch_command.py:45 DEFAULT_TRAINER`, `confound_gates.py:56` (bare string in a tuple), `tools/launch_witness_run.py:59 _TRAINER`, `tools/witness_autoconfig.py:34 _TRAINER`, `tools/mlx_gpu_determinism_probe.py:280 _TRAINER`, `tools/levelset_heldout_codefit_gate.py:73` (bare string). Test files: ~40 (each test independently re-derives `_REPO / "experiments/..."`, e.g. `_TRAINER_PATH`, `_MODPATH`, `_LEVELSET`, `_WITNESS` — 8 different local names for the identical string across `src/tac/tests/*.py`). Docstring-only mentions (not flagged, harmless): ~15 more files. | `curriculum_dsl.TRAINER_REL`/`TRAINER_PATH` (already exists, already the natural owner since the DSL is what compiles argv FOR this trainer) | **MEDIUM-HIGH.** The path is stable today, but per CLAUDE.md's own §CURRENT-STATE memory (`L1` "run-dir naming drifted... witness_checkin's fallback glob went blind") this EXACT class of "N independent copies of a name that should be ONE constant" already bit the run-artifact-filename surface once this week. Same mechanism, different name. | **Mechanical for production files** (8 files): `from tac.witness_dsl.curriculum_dsl import TRAINER_REL, TRAINER_PATH` and delete the local re-declaration. Tests are lower priority (each test is self-contained by convention) but could import the same constant to shrink the surface. | Indirect — feeds path resolution for launch/flag-validation, not the score itself. |
| 3 | `re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', <trainer source>)` — the "never invent a flag" argparse scanner | `src/tac/witness_dsl/curriculum_dsl.py:50` (inside `real_trainer_flags`, the presumed canonical owner), `tools/launch_witness_run.py:173`, `tools/witness_autoconfig.py:39`. Byte-identical regex in all 3. | `curriculum_dsl.real_trainer_flags` (already exists, already exported) | **MEDIUM.** If the trainer's argparse style ever changes (e.g. `add_argument('--foo'` single-quote, or a flag defined via a loop/helper instead of a literal `add_argument(` call), the regex needs updating in 3 places or the 3 copies silently diverge in what they consider "real." | **Mechanical**: replace both `tools/` copies with `from tac.witness_dsl.curriculum_dsl import real_trainer_flags`. | Safety-mechanism for the NO-FAKE "never invent a CLI flag" guard — protects against inventing flags, not a score input itself. |
| 4 | fcntl-locked "append one JSON row" helper | `src/tac/witness_dsl/activation_ledger.py::_append_locked_jsonl` and `src/tac/witness_dsl/curriculum_candidate_pool.py::_append_locked_jsonl` — **identical function name AND identical docstring** ("fcntl-locked APPEND of ONE json row (canonical .omx/state pattern; best-effort off-POSIX)."). Same pattern inlined (no factored helper) in `src/tac/witness_control/costate_posterior.py`, `shadow_controller.py`, `campaign_repl.py`, `decode_cache.py`. Repo-wide sister count (out of narrow scope): ~85-95 files touch `fcntl.LOCK_EX` directly. | none — no shared `tac.jsonl_ledger` module exists | **MEDIUM.** Each copy is small (~15 lines) and stable POSIX boilerplate, so it rarely breaks by itself, but a future correctness fix (e.g. handling `BlockingIOError` retry, or Windows/non-POSIX fallback improvements) has to be hand-propagated to N places; two are already verified byte-identical, meaning someone already copy-pasted rather than importing. | **Mechanical**: extract one `tac.jsonl_ledger.append_locked_jsonl(path, row)` (or similarly named) helper; the two byte-identical copies are a pure find-replace; the four inline variants need a quick read-through to confirm they don't have a ledger-specific twist before switching to the shared call. | Not score-input; affects observability-ledger durability under concurrent writers. |
| 5 | Canonical JSONL store paths re-spelled as raw strings instead of importing the owning module's constant | `CANONICAL_EQUATIONS_REGISTRY_PATH` (owned by `canonical_equations/registry.py:67`) re-spelled literally in `tools/dashboard_server.py:987` (`path = ".omx/state/canonical_equations_registry.jsonl"`) and `src/tac/preflight.py:81864` (`_CHECK_359_REGISTRY_PATH = "..."`). `DEFAULT_LEDGER_PATH` (owned by `harness_failure_ledger.py:63`) re-spelled in `tools/convene.py:224` and `src/tac/witness_control/producer_bridge.py:165` (inside a provenance dict, so lower severity — display-only). | `canonical_equations.registry.CANONICAL_EQUATIONS_REGISTRY_PATH`, `harness_failure_ledger.DEFAULT_LEDGER_PATH` (both already exist) | **LOW-MEDIUM.** If either store is ever relocated (per CLAUDE.md's "State JSONL archival policy" — files >10MB get archived/rotated), the constant-owning modules would update in one place but these 4 re-spelled consumers would keep reading/writing the OLD path silently. | **Mechanical**: import the constant instead of the literal string in the 4 call sites. `preflight.py`'s copy is the trickiest (it's inside a STRICT-gate scanner file with its own naming convention `_CHECK_359_REGISTRY_PATH`), everything else is a one-line import swap. | Not score-input; affects dashboard/preflight/convene reading the right file after a future rotation. |
| 6 | `CAMERA_H, CAMERA_W = 874, 1164` re-declared as a fresh module tuple instead of importing `tac.camera.CAMERA_H/CAMERA_W` | `tools/levelset_byte_close_and_eval.py:146`, `tools/witness_byte_close_and_eval.py:72` — both in the **live, exact-eval-adjacent byte-close path**. (Repo-wide sister count is large — dozens of old/dormant substrate files also re-declare this tuple, e.g. `optimize_grayscale_canvas.py`, `segmap_renderer.py`, `anr_token_renderer.py` — those are historical PR95-family substrates, out of this audit's scope, and the 07-07 sweep already explicitly classified "874/1164 in build/byte-close/bench tools (~100 files)" as a **deliberate, don't-retro-edit** class because those tools must reproduce exact measured bytes.) | `tac.camera.CAMERA_H`/`CAMERA_W` (already exists; `clip_profile.py` already imports from it correctly) | **LOW in practice** (0.mkv's resolution is fixed and unlikely to change), but the duplication IS real — two identical literal tuples, not one shared import. | **Mechanical AND safe** despite the 07-07 memo's general caution: `tac.camera.CAMERA_H == 874` and `CAMERA_W == 1164` today, so `from tac.camera import CAMERA_H, CAMERA_W` produces the byte-identical runtime value — this is importing-the-same-number, not changing-the-number, so the byte-identity risk the 07-07 memo was guarding against does not apply here. Worth doing specifically for these 2 files since they are live (unlike the ~100 historical files). | Score-adjacent (byte-close/exact-eval path) but the VALUE doesn't change, only its provenance. |
| 7 | Sealed-constant literals duplicated **within `curriculum_dsl.py` itself** across 2-4 internal program-builder call sites | `"--mod-dim": 32` at lines 1635 and 2935; `Regularizer("--eikonal-weight", 0.01)` at lines 1626 and 2854; `Regularizer("--length-weight", 0.001)` at lines 1627 and 2855. All same-file. | none — candidate for module-level `DEFAULT_MOD_DIM = 32`, `DEFAULT_EIKONAL_WEIGHT = 0.01`, `DEFAULT_LENGTH_WEIGHT = 0.001` | **LOW.** Same file, so a search-replace catches every copy; genuinely low drift risk versus the cross-file classes above — flagged mainly because these ARE the CLAUDE.md-cited "sealed constants" (mod32cap, per memory L2) and the value-provenance-ladder rule explicitly calls out bare literals as a bug class. | **Mechanical, tiny**: 3 named module constants, 4 call-site substitutions, same file. | These particular values ARE load-bearing (mod32cap is a council-designed sealed baseline per memory), but the duplication itself is low-risk. |
| 8 | `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` (and sibling `gt_n96.npz`/`gt_n6.npz`) path spelled out as a literal default across ~11 files | `curriculum_dsl.py:4161`, `launch_witness_run.py` (docstring + default), `witness_autoconfig.py` (tools, docstring), `dashboard_server.py` (3 dataclass field defaults + 3 env-var fallback defaults) | none, but **already triaged** by the 07-07 sweep as a "watch item... candidate for a `tac.gt_cache.default_path(n)` helper **later**" — i.e. explicitly deferred, not urgent | **LOW** per the prior triage's own judgment — every consumer exposes an overridable `--gt-cache`/env-var, so a wrong default is correctable at invocation time, not silently baked into a score. | Not recommended as a mechanical fix right now — re-flagging only to prevent re-discovery; the 07-07 memo already has the disposition. | No — advisory default only, always overridable. |

### Checked and NOT flagged (hypotheses this audit falsified — don't re-check these)

- **Render dims 384/512/874/1164 inside the two trainer files** — sampled every non-docstring
  occurrence in `train_levelset_witness_realized_through_R_mlx.py`; every hit is either (a) the
  single canonical `argparse.add_argument("--render-h", default=384)` declaration (correct,
  single-owner) or (b) a `# (384,512)` numpy-shape comment (non-load-bearing). **Not a
  duplication class** — my initial hypothesis (that render dims were scattered as bare literals
  across the trainer) was wrong.
- **`self_orient` `freq_across=32.0` / `freq_along=4.0`** — each appears exactly once, as the
  trainer's own argparse default. Not duplicated.
- **EMA decay `0.997`** — the CLAUDE.md-canonical Quantizr constant. Already explicitly
  "known-accepted, not the bug class" per the 07-07 sweep for tool-level provenance pins. The
  NEW thing this audit adds is that it's also one of the 28 values silently duplicated inside
  `_proven_base()` vs `BASELINE` (see #1) — that's a finding about the SHADOW-CONFIG pattern,
  not about `0.997` as a literal per se.
- **`witness_launch_readiness_gate.py` vs `witness_memory_preflight.py` vs
  `system_memory_governor.py`** — read all three; they compute genuinely different things
  (config-freshness-vs-launch.sh, config-based peak-RSS projection, live-process peak-RSS
  resolution from recorded/current values). Coincidentally-similar domain, independent by
  design. **Not duplication.**
- **`DEFAULT_SAFE_FRAC = 0.70` (memory_preflight) vs `DEFAULT_BAND_ENVELOPE_FRAC = 0.85`
  (system_memory_governor)** — two distinct, intentionally-different sealed ratios for two
  distinct gates (pre-launch single-config refusal vs live-envelope band), matching memory L51
  "safe-frac 0.85 (0.70 concurrent)". **Not duplication**, coincidental similarity of shape only.
- **sha256-file-hashing helpers (`_sha256_file`/`_sha256_bytes`/`_sha256`/...)** — real,
  systemic, ~20+ independent reimplementations, but essentially all of them live in
  OLD/dormant substrate or archive-tooling files (`qh0_renderer_codec.py`,
  `codec_op_admm_adapter.py`, `pr101_archive_state_loader.py`, etc.), **not** in the live
  witness/level-set stack (within scope, only `clip_profile._sha256_and_size` exists, used
  once). Noting as an adjacent repo-wide sister class for a future separate sweep, not ranking
  it here.
- **`src/tac/profiles.py::PROVEN_BASELINE`** — this is the OLD `experiments/pipeline.py`-era
  training-profile system (CLAUDE.md "Canonical pipeline standard" section), a completely
  different, non-witness pipeline. Its name superficially resembles `_proven_base()` but they
  are unrelated systems; not a duplication with the witness stack.

## Canonicalize-next queue

Ordered by EV; each tagged mechanical (safe to hand to a Sonnet agent, small tested diff) vs
needs-design (touches live launch config, needs care + a cross-check test before landing).

1. **[needs-design] #1 `_proven_base()` vs `BASELINE`.** Before touching this, a design pass
   must decide: should `_proven_base()` derive from `BASELINE.base` (translating key-naming
   conventions), or should `BASELINE.base` be regenerated FROM `_proven_base()`'s already-tested
   values, or should one be deleted entirely once the other is proven a strict superset? Any of
   the three needs a new test asserting the two configs agree wherever they overlap (or a single
   accepted mapping function is tested). This is the highest-EV item AND the riskiest — it
   changes what actually launches. Do not let a mechanical-only agent touch it.
2. **[mechanical] #2 trainer-path constant consolidation (production files only).** Import
   `TRAINER_REL`/`TRAINER_PATH` from `curriculum_dsl` in the 8 non-test files listed; delete the
   local re-declarations. Small, safe, verifiable by `ruff` + existing tests (each file's tests
   already assert the resolved path resolves correctly; only the *source* of the constant
   changes, not its value).
3. **[mechanical] #3 flag-scanner regex consolidation.** Import `real_trainer_flags` from
   `curriculum_dsl` in `tools/launch_witness_run.py` and `tools/witness_autoconfig.py`; delete
   the local regex + function. Verify via the existing tests that assert both tools refuse
   invented flags.
4. **[mechanical, but read each site first] #4 fcntl-jsonl-append helper extraction.** Create
   one canonical helper; the two byte-identical `_append_locked_jsonl` copies are a pure
   find-replace; the four inline variants (`costate_posterior.py`, `shadow_controller.py`,
   `campaign_repl.py`, `decode_cache.py`) need a 5-minute read each to confirm no ledger-specific
   deviation before switching. Out-of-scope repo-wide ~85-file sister pattern is a SEPARATE,
   larger future sweep — do not scope-creep this into it.
5. **[mechanical] #5 store-path constant imports.** Swap the 4 re-spelled literal paths for
   imports of `CANONICAL_EQUATIONS_REGISTRY_PATH` / `DEFAULT_LEDGER_PATH`.
6. **[mechanical, low priority] #6 camera-tuple import in the 2 byte-close tools.** Safe because
   the value doesn't change, only its source; do this alongside #5 in the same small PR since
   both are "import an existing constant instead of re-declaring it."
7. **[mechanical, tiny, low priority] #7 intra-DSL sealed-constant naming.** Add
   `DEFAULT_MOD_DIM`/`DEFAULT_EIKONAL_WEIGHT`/`DEFAULT_LENGTH_WEIGHT` module constants inside
   `curriculum_dsl.py` and use them at all 4 internal call sites. Purely cosmetic/hygiene; do
   last or skip if time-constrained.
8. **[no action]** #8 gt-cache path — already triaged 07-07, leave as-is.

## Pointer

Exact frontier pointer **0.19108282 UNMOVED** — this is a read-only hygiene audit (apparatus
means), not a score row. Feeds the next hygiene-canonicalization wave; does not itself move
`.omx/state/canonical_frontier_pointer.json`.
