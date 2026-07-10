# Canonicalization wiring sweep (#389) — U1↔U2↔U3 producer/consumer map + wire-in

**Date:** 2026-07-09 · **Operator GO:** #389 ("comprehensive sweep to identify upstream and
downstream producers and consumers and ensure all wired and integrated and working as expected").
$0, no GPU, run dirs read-only. **Pointer contest-CPU 0.19110 UNMOVED** — this is MEANS (measurement
+ coordination apparatus), not a lever, not an exact-eval row. `[macOS-CPU advisory · NON-PROMOTABLE]`.

The 3 canonical units landed with `# TODO(#389)` fences; this sweep resolves the fences, wires the
cross-unit consumption, migrates the one genuine duplicated live callsite, and runs round-1 review.

---

## PHASE 1 — producer / consumer map (executed, not asserted)

Greps run over `src/tac` (see the sweep for the exact queries): the discriminating signal for the
through-R d_seg pattern is the `d_seg_reference` + `per_class_flip_stats` compare-consumers, NOT the
too-broad `preprocess_input` (every renderer defines that method).

| Canonical unit | Upstream producers (feed it) | Downstream consumers (read it) | #389 action |
|---|---|---|---|
| **U1 `tac.through_r`** (`measure_through_r`, `compare`, `scaffold_assembler`) | `resolution_chain` (pinned R) · `boundary_math.seg_core.load_real_segnet` · `witness_control.perclass_verdict` · gt_n600 cache | `inc1a_harness.mask_dseg_meter` (**migrated**) · `inc1a_harness.composite_assembler` (already re-exports scaffold) · `experiments/probe_*` (read-only evidence, not migrated) · **U2** (rows) | emit `MeasurementRow`s (fence resolved) + share compare back-half |
| **U2 `tac.verdicts`** (`MeasurementRow`, `emit_verdict`) | **U1** `ThroughRResult.to_measurement_rows` (new) · any caller building a verdict | `session_bus` (verdict fan-in) · `triality_drift_detector` (serializer `--triality-legs`) · dashboards/costate (duty-to-measure) | posts `verdict_landed` bulletin (fence resolved) |
| **U3 `tac.session_bus`** (`bulletin`, `recovery_manifest`) | `review_counter.record_round` (WIRED pre-#389) · **U2** `emit_verdict` (new) · **U3** `recovery_manifest.register_inflight/complete` (new) | seal-round `staleness_check` · `#247` costate SENSE `events_since` · `session_recover.py report` | add 2 fail-open producers |

**Honest Phase-1 findings (verified by reading the code, not the prompt's suggestion — operating
manual §4):**
- **`movable_deshare.py` is NOT a through-R consumer.** It is a *rate-axis* byte dedup audit (numpy
  geometry on the frozen argmax cache → `encode_absolute_2d`); it never renders through R, never runs
  SegNet, never calls `d_seg_reference`. Migrating it to `measure_through_r` would be a FAKE migration.
  Its `measure_deshare_magnitude` / `pairwise_dedup_audit` byte/S outputs are candidate **U2**
  `MeasurementRow` (rate-axis) emitters — noted as a future wire, not forced in this sweep.
- **`segmap_renderer.py`** `argmax(dim=1)` is a renderer-internal training compare, not the
  gt_n600→R→SegNet verdict pattern. Out of scope.
- **`scaffold_assembler` / `laguerre_logit_offset`** use `d_seg_reference` / `per_class_flip_stats`
  for their OWN single-frame reconciliation / task-space lever — a *different* computation than the
  agg+per-class stack-compare. Not the duplicated fact. Out of scope.
- **The one genuine duplication (P1):** the "compare a realized/partition INT label stack vs `lstars`
  → aggregate d_seg + per-class rate + flip-share" **back half** was computed twice, inline, by
  `measure_through_r` and `measure_mask_dseg`. That is the fact now stored in ONE place.

---

## PHASE 2 — wirings landed

1. **U1→U2 (fence resolved).** `ThroughRResult.to_measurement_rows(*, git_sha, review_status, …)`
   builds canonical `MeasurementRow`s (1 aggregate `d_seg` + 5 per-class), `axis_tag=[through-R]`
   (non-authority by construction), `n_samples` = frame count (subset carries its reason),
   `provenance.inputs_sha256` = a new deterministic `sha256(lstars ‖ realized)` key computed in
   `measure_through_r`. OPT-IN (a method, not the hot loop) + LAZY import ⇒ `measure_through_r` stays
   **byte-identical by default** and the harness stays leaf-clean at import time. `review_status` is a
   REQUIRED arg (the harness cannot know if a fresh reviewer saw the number — operating manual §5).
   *Deviation from the fence's "per pair" wording:* aggregate+per-class is the verdict-useful
   granularity; per-pair scalars remain in `per_pair_dseg` for a caller who wants 600 rows.
2. **U2→U3 (fence resolved).** `emit_verdict` posts a fail-open `verdict_landed` bulletin AFTER the
   atomic JSON write (`subject = scope.scoped_to`), mirroring `review_counter`. Lazy import ⇒
   import-time deps stay stdlib+`tac.verdicts`; the JSON write is unchanged/byte-identical.
3. **Migration (the one genuine callsite).** New canonical `tac.through_r.compare.
   compare_label_stack_to_lstars` holds the shared back half; **both** `measure_through_r` and
   `measure_mask_dseg` delegate to it (each keeps its own front-half validation + error type +
   `extra` key name). `mask_dseg_meter` "delegates into `through_r`, keeps API" — the honest reading
   of the prompt (mask-level ≠ through-R front half; only the back half is shared).
4. **U3 producers.** `recovery_manifest.register_inflight → agent_spawned`, `complete →
   agent_completed` (fail-open, AFTER the durable checkpoint write). The heartbeat path is
   deliberately NOT wired (would spam `agent_spawned` each tick). **memo-landing helper:** N/A — no
   `tac` function lands a memo (memos land via git); the low-level `subagent_checkpoint.append_checkpoint`
   is deliberately not wired (heartbeat spam) — the semantic register/complete layer is
   `recovery_manifest`.
5. All commits via the serializer with `--triality-legs` (dogfooding U2's flag).

### No-regression proofs (numeric identity — re-derived, not recognised)
- **`compare` == the old inline formula, EXACTLY** (`test_compare_helper_matches_inline_formula_exactly`):
  independent inline recompute of agg/per-class/share/flips/pixels/std `==` the helper, on a synthetic
  5-class stack with genuine flips.
- **`mask_dseg_meter` migration is numeric-identity** (`test_mask_meter_migration_is_numeric_identity`):
  same inputs → identical `agg_dseg` / `per_class_dseg` / `flip_share` / `total_flips` / `total_pixels`
  / `extra["per_frame_std"]` before/after (the `extra` key preserved = API unchanged).
- **U1 identity canary still GREEN post-wiring** (`test_end_to_end_gt_frame_reproduces_lstars_dseg_zero`
  RAN, not skipped — real gt cache + real SegNet): GT frame → `measure_through_r` (through the new
  delegated back half) → `agg d_seg == 0.0`, `total_flips == 0`. The strongest possible identity proof.

---

## PHASE 3 — round-1 adversarial review (I own round 1 over U1+U2+U3+the wirings)

Attacks and dispositions:
- **(a) serializer `--triality-legs` under real use** → dogfooded on this sweep's commits; 58
  serializer + 21 triality-legs tests green; malformed-flag refuses before any git action (existing).
- **(b) numeric identity of migrated callsites** → proven exact (synthetic ==) AND through the real
  SegNet (canary d_seg==0). HELD.
- **(c) fail-open handlers swallowing real errors** → each new fail-open post is AFTER the durable
  write on a score-neutral notification path (operating manual §8.9), documented at the callsite;
  tested that a raising `post_event` never breaks the verdict/checkpoint write. ACCEPTED risk (same
  precedent as `review_counter`).
- **(d) U1 identity canary green post-wiring** → confirmed RAN + passed. HELD.
- **(e) P1 "any fact now in TWO places?"** → the compare back half is now ONE place (`compare.py`); no
  NEW duplication introduced (`inputs_sha256` only in harness; each fail-open helper in one module).

**FINDING (found + fixed — round 1 is therefore NOT_CLEAN, counter stays 0):** wiring the fail-open
bulletin producers into `emit_verdict` and `recovery_manifest.register/complete` made the EXISTING
unit tests write real events into the live LIVE_STATE store `.omx/state/session_events.jsonl` (a
concurrent seal agent's `staleness_check` reads it). Measured leak: `test_verdict_emit` +7,
`test_session_bus` +4, `test_review_counter` +30. **Class-fix:** bulletin-store isolation added to
every emitting test surface — autouse fixture in `test_verdict_emit.py` + `test_review_counter.py`,
and `tool_tmp` + the CLI test extended in `test_session_bus.py`. Post-fix in-process leak = **0**.
**Residual (documented, out of this sweep's scope):** `test_review_counter`'s multiprocessing
concurrency test still leaks **12** events from subprocess children (spawned children don't inherit
`monkeypatch`; U3's `record_round` has no `bulletin_path` param to thread) — a deeper U3 hardening
item (add a bulletin-path param / env override), NOT a new defect in the #389 wiring. Flagged for
rounds 2+.

Round-1 verdict: **NO-GO-to-SEAL** (found + fixed 1 class) → recorded `NOT_CLEAN`, findings=1. Main
dispatches rounds 2+ fresh; SEAL needs 3 consecutive CLEAN.

---

## Tests / hygiene
- Full affected suite **264 passed** (through_r 32 · inc1a 27 · verdict_measurement_row 24 ·
  verdict_emit 17 · session_bus 34 · review_counter 14 · subagent_checkpoint 54 · serializer 58 ·
  serializer_triality 21 + new: compare_helper 7 · canon389_wiring 6). New: **13** tests.
- `ruff check` clean on every touched file (`flip_inverse.py`'s pre-existing debt is an untracked
  sibling file, not touched here).
- review-tracker mark-file per `.py`; serializer commit with POST-EDIT shas; `--triality-legs` used.

## Triality legs
- **DAG:** `FEED-389` in `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- **DSL:** N/A-with-reason — apparatus wiring (measurement/verdict/coordination), no witness lever /
  trainer flag / curriculum surface for `witness_dsl` to hold.
- **equations:** N/A-with-reason — reuses the already-registered `d_seg` authority functional + score
  law; the producer/consumer wiring is the invariant, no NEW measured physics relation.

STORES CONSULTED: the 3 canon landing memos (`canonical_{through_r_harness,verdict_emission,
session_bus}_landed_20260709.md`), `docs/operating_manual_craft_handoff.md`, the DAG (FEED-canon-u1/u2/u3),
`tac.review_counter` (fail-open producer template), `tac.through_r.*`, `tac.verdicts.*`,
`tac.session_bus.*`, `tac.inc1a_harness.mask_dseg_meter`, `tac.boundary_math.movable_deshare`.

---

## R8 — the SEALING round did NOT seal (fresh-eyes at HEAD `acef0300c`, counter was 2/3)

**Verdict: NOT_CLEAN — 1 finding (found + fixed). Counter 2/3 → 0. No SEAL.**

Round 8 was the intended 3rd consecutive CLEAN → SEAL. The highest-bar adversarial-operator
read (angle: *"any doc claim not backed by a test? any comment that misstates the code?"*)
surfaced a genuine finding, so the seal is correctly REFUSED. This is the sealing round doing
its job — a comment that stated the literal opposite of runtime behavior, on a recovery-
surfacing path, unbacked by any test, is exactly what round 8 exists to catch before it is
certified.

**FINDING (comment-vs-code mismatch on a fail-toward-surfacing path — recovery_manifest.py):**
`recover_report`'s sort carried the inline comment *"None age (unparseable) sorts to the top."*
The sort key was `(e.age_seconds is not None, e.age_seconds or 0.0)` under `reverse=True`, which
ranks parseable ages (`is not None` == True) ABOVE None-age (False) — so an unparseable timestamp
sorted to the **bottom**, the literal opposite of the comment AND of the module's stated
*"fail toward surfacing, not hiding"* contract. Empirically confirmed:
`recover_report([PARSE_OLD age=9000, BADTS ts='not-a-timestamp'])` → `['PARSE_OLD','BADTS']`
(the corrupt-timestamp agent — the most suspicious crash, a likely torn write — buried last).
No test covered the None-age ordering (`test_recover_report_sort_most_stale_first` used two
parseable ages only), so the false comment shipped unchallenged through R4–R7.

**Class-fix (code aligned to the documented + intended behavior, not the reverse):** sort key →
`(e.age_seconds is None, e.age_seconds or 0.0)` so None-age (unparseable = most suspicious) sorts
to the TOP; parseable ages still order largest-first among themselves. Inline comment rewritten to
state the mechanism precisely. **+1 regression test** `test_recover_report_unparseable_timestamp_
sorts_to_top` asserts `['BADTS','VERY_OLD']` and `entries[0].age_seconds is None` — the doc claim
is now backed by a test. Existing `test_recover_report_sort_most_stale_first` (two parseable ages)
still passes (largest-first unchanged).

**Standing re-checks (all GREEN, re-derived):** 184 tests single-invocation (was 183; +1 new);
GT identity canary `test_end_to_end_gt_frame_reproduces_lstars_dseg_zero` **RAN** (real gt cache +
real SegNet, d_seg == 0, not skipped); serializer `--triality-legs` absent → `(None, None)`;
`emit_verdict` / bulletin write paths byte-identical (untouched); `ruff check` clean on both
touched files. review_counter round 8 recorded NOT_CLEAN via `record_round`.

**Rounds 9+ restart fresh; SEAL still needs 3 consecutive CLEAN from round 9.** The fix is
unreviewed new code (per the protocol, a fix resets the counter); a fresh reviewer must re-derive
the recovery-sort behavior at the next HEAD.

### Seal-boundary honesty — what 8 rounds did NOT cover (carry into rounds 9+)
Even had R8 sealed, the seal would bind ONLY the canon set at HEAD. Explicitly OUT of the seal:
- **`through_r/palette_realization.py` (+ its test) is UNCOMMITTED (untracked at HEAD)** — never
  entered the review window; any review of it is future work on rounds 9+.
- **The U3 bulletin PRODUCER set is still minimal** — only `verdict_landed` (via `emit_verdict` +
  `review_counter`) and `agent_spawned`/`agent_completed` (via `recovery_manifest`) are wired.
  `gate_ruled` / `spec_edited` / `memo_landed` / `agent_died` are declared kinds with NO canonical
  producer — a consumer that expects them will see none.
- **The `recovery_manifest.heartbeat` path is deliberately NOT wired to bulletin** (would spam
  `agent_spawned` per tick) — liveness is visible only via the checkpoint store, not the bus.
- **`MeasurementRow` has no parser-SIDE (read-back) surface** — `to_json_dict` is write-only; no
  canonical `from_json_dict`/loader validates emitted verdict JSON on read.
- **The review_counter multiprocessing residual (12 leaked child events)** flagged in R1 remains a
  deeper U3 hardening item (`record_round` still has no `bulletin_path`/env thread to children) —
  mitigated by `TAC_SESSION_BULLETIN_DISABLE`/`_PATH` env inheritance but not closed at the
  `record_round` API.
- **Anti-gaming boundary unchanged:** the ledger records CLAIMS a review happened; no gate proves
  a real review occurred (module docstring boundary) — the seal trusts the reviewed_commits trail.

pointer contest-CPU **0.19110 UNMOVED** — this whole wave is MEANS (measurement + coordination
apparatus), never a lever, never an exact-eval row. `[macOS-CPU advisory · NON-PROMOTABLE]`.

---

## R15 — seal-boundary REFRESH (fresh-eyes at HEAD `549eeb447`, counter was 1/3)

**Verdict: NOT_CLEAN — 1 finding (found + fixed; APPEND-ONLY, the R8 section above is preserved
verbatim). Counter 1/3 → 0. No SEAL.**

Round 15's fresh angle was a re-verification of the R8 "Seal-boundary honesty — what 8 rounds did
NOT cover (carry into rounds 9+)" carry-forward list against current HEAD truth (task-3: *stale
boundary claims = findings*). The list is a LIVE instrument — its whole purpose is "carry into
rounds 9+", and rounds 9–15 are exactly its readers. One item is now materially stale.

**FINDING (stale forward-claim on a live carry-forward coordination artifact — the same
doc-misstates-reality class R8 itself caught):** the R8 list's FIRST bullet asserts
`through_r/palette_realization.py` (+ its test) is *"UNCOMMITTED (untracked at HEAD) … any review of
it is future work on rounds 9+."* At HEAD that FORWARD claim is FALSE: the file and its test were
**COMMITTED at R11 (`7310b5506`)** and **REVIEWED across rounds 11–14** (they are in the r11/r12/r13/
r14 review scope). A rounds-9+ reviewer relying on the list would wrongly exclude a committed,
reviewed file from the seal — exactly the misdirection the sealing round exists to catch. Verified:
`git ls-files --error-unmatch src/tac/through_r/palette_realization.py` (tracked);
`git log -1 -- …palette_realization.py` → `7310b5506` (R11).

**Class-fix (append-only per Catalog #110/#113 — R8 preserved; this section supersedes its boundary
list as the CURRENT truth):** the boundary list refreshed to HEAD:

- **The currently-untracked scope-adjacent file is now `through_r/stem_perception.py`** (`git status`
  `?? …stem_perception.py`; `git ls-files --error-unmatch` errors = untracked). It is explicitly OUT
  of scope for this review wave by directive. `palette_realization.py` is no longer the untracked one.
- **`palette_realization.py` (+ test): COMMITTED (`7310b5506`, R11) + REVIEWED rounds 11–14** — IN the
  review window, NOT future work. (Supersedes the stale R8 bullet.)
- **U3 bulletin PRODUCER set: STILL minimal — UNCHANGED since R8.** Producers wired: `verdict_landed`
  (via `emit_verdict` + `review_counter._post_bulletin_verdict_landed`) and `agent_spawned` /
  `agent_completed` (via `recovery_manifest`). `gate_ruled` / `spec_edited` / `memo_landed` /
  `agent_died` are declared kinds with **NO canonical producer** (verified: the only non-test
  `post_event` callsites are review_counter, emit, recovery_manifest). A consumer expecting them sees
  none.
- **`MeasurementRow` parser-SIDE (read-back): STILL unbuilt — UNCHANGED since R8.** `to_json_dict` is
  write-only; there is no `from_json_dict` / loader in `tac.verdicts` (verified: grep finds none).
- **review_counter multiprocessing residual: CLOSED-via-ENV since R8 (PARTIALLY resolved).** The
  round-2 fix set `TAC_SESSION_BULLETIN_*` (env-inherited by spawned children) via an autouse fixture,
  and `test_subprocess_children_honor_bulletin_env_no_live_leak` proves **zero** child leak into the
  live store; R13 further made `read_events` env-symmetric. The `record_round` API still has no
  `bulletin_path` param — closure is via env inheritance, not an API param (the deeper hardening item
  remains open, but the leak itself is closed + tested).
- **Anti-gaming boundary unchanged:** the ledger records CLAIMS a review happened; no gate proves a
  real review occurred (module docstring boundary) — the seal trusts the `reviewed_commits` trail.

**Standing re-checks (all GREEN, re-derived — not recognised):** 178 affected tests single-invocation
(through_r 32/…, session_bus 34, verdict_emit 17, verdict_measurement_row 24, review_counter 14, …);
GT identity canary `test_end_to_end_gt_frame_reproduces_lstars_dseg_zero` **RAN** (real gt cache +
real SegNet, d_seg == 0, 4.0s — not skipped); serializer `--triality-legs` absent → `(None, None)`
(byte-identical no-op); torn-tail append tests green (bulletin + review_counter). **Findings-history
meta-audit — all 7 prior fixes intact at HEAD, each re-run + code-site read:** r4 `SEG_WEIGHT`
consumed from `tac.contest_score` (flip_inverse:68/537/602/624); r5 `_needs_newline_separator`
(bulletin:169 + review_counter:195); r8 recovery sort key `(age_seconds is None, …)` reverse=True
(recovery_manifest:338) + `test_recover_report_unparseable_timestamp_sorts_to_top`; r10 `top_k`
clamp `max(0, min(top_k, n_flips))` (flip_inverse:655) + empty-verify early-return before `np.stack`
(flip_inverse:703); r11 zero-pixel BOTH-denominator guard (palette_realization:483–486); r13
`read_events` env-symmetry (bulletin:258). **Constants sampled (provenance ladder):** `SEG_WEIGHT=100`
= MEASURED-ANCHOR (upstream/evaluate.py:92) consumed not re-hardcoded; `N600_EXPECTED=600` = anchored
to the n600 authority scale; `step_lsb=12.0` = honest actuator DEFAULT (effect MEASURED
prediction-vs-realized through real SegNet, not asserted-as-derived) — all CLEAN, no bare-literal
physics constant masquerading as derived.

**Rounds 16+ restart fresh; SEAL needs 3 consecutive CLEAN from round 16.** The fix is a
documentation edit; a fresh reviewer must re-verify the refreshed boundary list against the next HEAD.

pointer contest-CPU **0.19110 UNMOVED** — the boundary refresh is MEANS (coordination hygiene), never
a lever, never an exact-eval row. `[macOS-CPU advisory · NON-PROMOTABLE]`.

---

## SEALED (2026-07-10, rounds 1–18, 3 consecutive CLEAN r16–r18)

**#389 canon wave SEALED.** `review_counter.current_state('canon_wave')` → `consecutive_clean=3/3,
sealed=True` after R18 recorded CLEAN (findings=0). The seal binds the canon set at HEAD
`b15f3b46d`: `tac.through_r` (`harness`/`compare`/`flip_inverse`/`palette_realization`/
`scaffold_assembler`/`resolution_chain`), `tac.verdicts` (`measurement_row`/`emit`), `tac.session_bus`
(`bulletin`/`recovery_manifest`), the serializer `--triality-legs` flag, and the detector softening.

### The 8 substantive findings fixed across rounds 1–15 (each a real defect, each now test-guarded)
1. **R1** — fail-open bulletin producers leaked real events into the LIVE_STATE store during tests;
   class-fix = bulletin-store isolation on every emitting test surface (autouse fixtures + `tool_tmp`).
2. **R4** — `SEG_WEIGHT` re-hardcoded; consumed from `tac.contest_score` (flip_inverse:68/537/602/624).
3. **R5** — missing newline separator on torn-tail append (`_needs_newline_separator`, bulletin:169 +
   review_counter:195).
4. **R8** — `recover_report` sort comment stated the OPPOSITE of behavior; unparseable (most-suspicious)
   age sorted to the BOTTOM. Fix: key `(age_seconds is None, …)` reverse=True → None-age sorts TOP;
   `test_recover_report_unparseable_timestamp_sorts_to_top`.
5. **R10** — `top_k` unclamped + empty-verify `np.stack` on empty; `max(0, min(top_k, n_flips))` +
   early-return (flip_inverse:655/703).
6. **R11** — zero-pixel BOTH-denominator guard (palette_realization:483–486).
7. **R13** — `read_events` env-symmetry so spawned children honor `TAC_SESSION_BULLETIN_*` (bulletin:258)
   → zero child leak into the live store.
8. **R15** — stale forward-claim on the live carry-forward boundary list (`palette_realization.py`
   asserted UNCOMMITTED after it was committed at R11); boundary list refreshed to HEAD truth.

### The standing-check suite that now guards them (re-derived GREEN at every fresh HEAD)
- Full affected suite single-invocation (220 tests @HEAD b15f3b46d).
- **GT identity canary** `test_end_to_end_gt_frame_reproduces_lstars_dseg_zero` **RUNS** (real gt cache +
  real CPU-torch SegNet → agg d_seg == 0, total_flips == 0) — the strongest through-R identity proof.
- `ruff check --select F` clean on all 7 canon files.
- serializer `--triality-legs` absent → `(None, None)` (byte-identical no-op); malformed refuses before
  any git action.
- **live-store byte-identity**: emitting tests do not append to `.omx/state/session_events.jsonl`
  (before == after bytes) — the R1 leak class stays closed.
- torn-tail append + `TAC_SESSION_BULLETIN_DISABLE`/`_PATH` env-symmetry green.
- emit_verdict REFUSAL contract (missing scope/rows/composition-P12/constraint-carved-P10/
  negative-reformulation-queue) fully enforced + 10 refusal tests.

### R16–R18 fresh angles (the 3 sealing rounds)
- **R16** — HISTORICAL-vs-LIVE distinction on the boundary list; CLEAN.
- **R17** — CLEAN.
- **R18 (this seal)** — **production-consumer contract sweep**: every NON-TEST consumer of the 7 canon
  modules honors its producer's contract (delegation shim `composite_assembler`; `mask_dseg_meter →
  compare`; lazy fail-open `review_counter`/`recovery_manifest → post_event`; `harness → verdicts`
  opt-in). Docstring-vs-behavior on the 3 load-bearing publics (`emit_verdict`, `post_event`,
  `measure_through_r`) HELD + test-backed.

### Explicitly OUT of the seal (carry-forward — the seal does NOT certify these)
- **HANDOFF (cross-agent, not a canon defect):** tracked equation
  `canonical_equations/textured_power_diagram_20260710.py:119` (Fable/#394) sets
  `python_callable_module_path="tac.through_r.palette_realization:realize_partition_through_r"` — a name
  that does NOT exist (`palette_realization` exports `run_arm` / `realize_argmax_no_R`). It is dormant
  metadata (regex-only field validation, no test resolves the attribute; module-part-only import in the
  callable-resolution test), and `palette_realization` is correct/complete — so this is a **wrong name
  in the consumer's file**, not a producer defect. Fix belongs in the sibling equation file (out of this
  wave's mutation scope). Flagged for #394/Fable.
- **U3 bulletin PRODUCER set still minimal** — only `verdict_landed` + `agent_spawned`/`agent_completed`
  wired; `gate_ruled` / `spec_edited` / `memo_landed` / `agent_died` are declared kinds with no producer.
- **`MeasurementRow` parser-SIDE (read-back) still unbuilt** — `to_json_dict` write-only; no loader.
- **`recovery_manifest.heartbeat` deliberately NOT wired to the bus** (tick-spam avoidance).
- **Anti-gaming boundary unchanged** — the ledger records CLAIMS a review happened; the seal trusts the
  `reviewed_commits` trail (module docstring boundary).
- The untracked scope-adjacent `through_r/stem_perception.py` (R15 note) was COMMITTED by a sibling
  (DAG FEED stem-perception) between R15 and R18 — it entered the review window via that sibling, not
  this wave; its own review is the sibling's.

pointer contest-CPU **0.19110 UNMOVED** — the entire #389 wave is MEANS (measurement + coordination
apparatus), never a lever, never an exact-eval row. `[macOS-CPU advisory · NON-PROMOTABLE]`.
