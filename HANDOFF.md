# HANDOFF.md — Reading order for the next operator

**Audience:** A capable engineer who is not `adpena` and needs to make
material progress on this project within 30 days. This file is the
30-minute entry point. After reading this, read `SYSTEM_MAP.md` then
`CLAUDE.md` (in that order).

**Last refreshed:** 2026-05-16 (premortem consolidation wave). Re-run
`tools/regenerate_handoff_brief.py` if more than 30 days stale.

---

## 1. Project mission and contest context (1 paragraph)

`pact` is the operator-and-Claude lab for the comma.ai
[Video Compression Challenge](https://github.com/commaai/comma2k19) (the
"contest"). The mission is to minimize the contest's official
`evaluate.py` score on the pinned `upstream/` snapshot by training a
task-aware neural renderer + segmentation + pose chain end-to-end against
the frozen `SegNet` + `PoseNet` scorers, then compressing the trained
artifacts into a `submissions/<name>/archive.zip` + `inflate.sh` runtime
tree that the contest grader can replay. The contest scoring formula is
`25 * archive_bytes / 37,545,489 + seg_distortion + sqrt(10 *
pose_distortion)`. The official leaderboard ranks by `contest-CPU`
(GitHub Actions Linux x86_64). The CUDA axis is informational. Today's
public frontier is **PR101 gold at 0.193 [contest-CPU]**.

## 2. Current frontier state (one line per axis)

- **Public gold:** PR101 at `0.193 [contest-CPU]` (HNeRV-family,
  ~605 LOC).
- **Our internal apogee:** A1 substrate at `0.1928 [contest-CPU]`
  (verified on M5 Max macOS-CPU advisory + GHA Linux x86_64 within
  `6e-6` per `dual_eval_adjudicated.json`). A1 contest-CUDA is `0.2317`
  (paired re-eval after Z3 v2 phantom-score-directory incident
  2026-05-15).
- **CUDA-only apogee:** PR107 at `0.229 [contest-CUDA T4]`.
- **Plateau:** The `0.196-0.199` "saturated 90%" cluster across most
  substrates is the local minimum the `META-ASSUMPTION ADVERSARIAL
  REVIEW` non-negotiable was added to break out of (see CLAUDE.md).

## 3. Top 5-7 highest-impact lanes currently in flight

Snapshot 2026-05-16; re-run `python tools/lane_maturity.py audit | head
-40` for the live view.

1. **`lane_canonical_dispatch_optimization_protocol_20260515`** (L1) —
   Catalog #270 umbrella protocol gate. Strict-flip pending the
   substrate-trainer Tier 1 backfill sweep (26 of 32 trainers
   currently fail at least one Tier 1 signal).
2. **`lane_meta_layer_substrate_contract_auto_wire_20260515`** (L1) —
   The `@register_substrate(SubstrateContract(...))` decorator + 36-field
   contract schema. Catalog #241 + #242. 31 legacy substrate trainers
   are TAGGED-PENDING-MIGRATION.
3. **`lane_pr95_lesson_now_at_meta_level_20260515`** (L1) —
   `UNIQUE-AND-COMPLETE-PER-METHOD` operating-mode directive applied
   at the META infrastructure level (not just substrate-internals).
   This is the binding rule for every new substrate scaffold.
4. **`lane_nscs03_end_to_end_balle_joint_codec_20260515`** (L1) — END-TO-END
   Ballé 2018 joint codec; 548 LOC unique implementation; 76 tests;
   recipe still `research_only=true` until Phase 2 council λ_R sweep.
5. **`lane_nscs01_nullspace_split_renderer_20260515`** (L1) —
   Nullspace split-frame renderer exploiting SegNet's `x[:, -1, ...]`
   slice; awaits Modal smoke + paired Tier C + 5-PROCEED council.
6. **`lane_12_month_frustration_premortem_20260516`** (L1) — The
   premortem that motivated this `HANDOFF.md`. Reading order:
   `.omx/research/12_month_frustration_premortem_and_recommendations_20260516.md`.
7. **`lane_premortem_consolidation_wave_5_items_20260516`** (L1) — The
   wave landing this `HANDOFF.md` + `SYSTEM_MAP.md` + substrate
   retirement + gate quota + memory rotation + state archival.

## 4. Top 10 STRICT preflight catalogs to NOT violate

The full table is at `CLAUDE.md § Meta-bug class catalog (strict-mode
preflight)`. These 10 will bite hardest first:

1. **#117 / #157 / #174** — Commit serializer + `--expected-content-sha256`
   discipline. ALWAYS commit via `tools/subagent_commit_serializer.py
   --expected-content-sha256 <file>=<post-edit-sha256>`. Never
   bare `git commit`.
2. **#1** — `check_no_mps_fallback_default`. Never `device = "cuda" if
   torch.cuda.is_available() else "mps" else "cpu"`. Default to
   CUDA-REQUIRED; raise on no-CUDA.
3. **#5** — `eval_roundtrip=False` is FORBIDDEN. Every training path
   simulates `384 → 874 → uint8 → 384` in the proxy loss.
4. **#88** — EMA on every training path with decay=0.997 for weights;
   eval against EMA shadow with snapshot+restore.
5. **#117** — Subagent commits must go through the serializer lock.
6. **#127 / #221 / #226 / #249** — Authoritative tags require custody
   metadata. NEVER quote a score from a filename without checking the
   metadata says the same axis.
7. **#143** — Lightning + Modal dispatchers register a pending job
   BEFORE submit, promote to active POST-submit.
8. **#220 + #272** — Substrate L1+ scaffolds adding bytes must declare
   operational mechanism (#220) AND the distinguishing-feature contract
   (#272), OR be `research_only=true`.
9. **#244** — Every `scripts/remote_lane_substrate_*.sh` must carry the
   canonical 3-export NVML/CUDA env block (`DALI_DISABLE_NVML=1` +
   `CUBLAS_WORKSPACE_CONFIG=:4096:8` +
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`).
10. **#291 + #292** — META-ASSUMPTION ADVERSARIAL REVIEW recurring
    cadence (every 7 days OR every 50 subagent landings) + every grand
    council deliberation must carry per-member explicit
    "operating-within assumption" statements.

## 5. Firing a paid Modal dispatch (canonical operator-authorize wrapper)

The canonical entry point is `tools/operator_authorize.py`. It wires
**every** required pre-dispatch gate in the right order:

1. Catalog #152 input file validator
2. Catalog #243 local-pre-deploy harness
3. Catalog #271 codex pre-dispatch review (only if cost > $1)
4. Catalog #166 Modal HEAD-parity ledger
5. Catalog #167 smoke-before-full pattern
6. Catalog #143 register pending job before submit

```bash
# Always invoke through operator_authorize.py via the
# `scripts/operator_authorize_substrate_<id>_modal_a100_dispatch.sh`
# wrapper. Example for D4 substrate:
./scripts/operator_authorize_substrate_d4_wyner_ziv_frame_0_modal_a100_dispatch.sh

# For non-interactive runs (cron / subagent loop), use the paired-env
# bypass per Catalog #199:
export OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1
export OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=15.00
./scripts/operator_authorize_substrate_d4_wyner_ziv_frame_0_modal_a100_dispatch.sh
```

Never call `modal run` / `modal_train_lane.py` directly — Catalog
#243 + #271 are bypassed and you will burn paid GPU on an integration bug.

## 6. Reading the Modal call_id ledger

```bash
# Show the last 20 dispatches with status:
.venv/bin/python -c "
from tac.deploy.modal.call_id_ledger import load_call_ids
rows = load_call_ids()
for r in rows[-20:]:
    print(r['call_id'], r['lane_id'], r['status'])
"

# Query by lane_id:
.venv/bin/python -c "
from tac.deploy.modal.call_id_ledger import query_by_lane
for r in query_by_lane('lane_d4_wyner_ziv_frame_0_20260514'):
    print(r['call_id'], r['status'], r.get('score'))
"

# Find unharvested (older than 4 hours):
.venv/bin/python -c "
from tac.deploy.modal.call_id_ledger import query_unharvested
for r in query_unharvested(older_than_seconds=4*3600):
    print(r['call_id'], r['dispatched_at_utc'])
"
```

The ledger lives at `.omx/state/modal_call_id_ledger.jsonl`
(append-only, fcntl-locked per Catalog #245).

## 7. Harvesting a Modal call_id

```bash
# Crawl every experiments/results/lane_*_modal/modal_metadata.json
# and pull the artifact dict from the Modal FunctionCall return cache.
# The result-cache TTL is ~24h — harvest within 24h of dispatch or
# the artifact is GONE per CLAUDE.md "Modal .spawn() HARVEST OR LOSE":
.venv/bin/python tools/harvest_modal_calls.py

# Force-harvest a single call_id:
.venv/bin/python -c "
import modal
r = modal.functions.FunctionCall.from_id('fc-01XYZ...').get(timeout=2)
print(r.get('returncode'), r.get('elapsed_seconds'), len(r.get('artifacts', {})))
"
```

## 8. Consulting the META-ASSUMPTION audit before designing a new substrate

The 18-shared-assumption matrix lives at
`feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md`.
Read it BEFORE writing a substrate design memo. Every substrate
scaffold's design memo MUST include a `## Canonical-vs-unique decision
per layer` section (enforced by Catalog #290; WARN-only at landing
2026-05-15). For each canonical helper / META-layer field your substrate
adopts, document WHY the canonical serves vs forks.

The default operating question per CLAUDE.md
"UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable is:

> "What's the OPTIMAL ENGINEERING for THIS specific method to achieve
> the lowest score possible given the methods and techniques involved?"

NOT:

> "How do I share with the canonical?"

## 9. Five "this will bite you in week 1" gotchas

1. **`--expected-content-sha256` is post-edit content, not HEAD.**
   Per Catalog #157: snapshot the working-tree sha AFTER your edits, BEFORE
   the lock-acquire. Three subagents on 2026-05-13 declared HEAD shas and
   were refused with rc=4. The serializer hashes working-tree content
   to detect sibling edits during the lock-wait window.
2. **MPS is NOISE.** Per CLAUDE.md "MPS auth eval is NOISE": local
   Apple-Silicon MPS produces scores `2-23×` worse than CUDA. Tag every
   MPS-derived number `[MPS-PROXY]` and treat as advisory only.
   macOS-CPU is `[macOS-CPU advisory only]` until a paired
   `[contest-CPU GHA Linux x86_64]` lands per Catalog #192.
3. **Modal call_ids age out in 24h.** Per Catalog #245 + the
   `Modal .spawn() HARVEST OR LOSE` non-negotiable: every `.spawn()`
   call MUST register via `tac.deploy.modal.call_id_ledger.
   register_dispatched_call_id` AND be harvested within 24h. The
   FunctionCall return-value cache is the ONLY artifact custody.
4. **Premise verification is mandatory.** Per Catalog #229: every
   subagent landing memo dated `>= 2026-05-14` that claims `>= 3` bulk
   edits MUST include an empirical verdict table OR a reproducer-script
   path. The parent prompt's "7 trainers need backport" claim was
   falsified empirically on 2026-05-14 (0/7 actionable; 11+ OTHER
   trainers as the actual target).
5. **CPU-axis "wins" can lose CUDA.** Per the "Apples-to-apples
   evidence discipline" non-negotiable: the PR102 CUDA-CPU gap is
   `+0.033` and per-archive. Z3 v2 phantom CUDA score of 0.19869 was
   actually a CPU eval routed to a `*_cuda.json` filename per Catalog
   #249. Always verify the metadata axis matches the filename.

## 10. Contact + escalation

- **Operator:** `adpena@gmail.com` (handle: `adpena`)
- **For dispatch budget / Modal credit questions:** operator.
- **For council-grade design decisions:** invoke the sextet pact
  (Shannon + Dykstra + Yousfi + Fridrich + Contrarian +
  Assumption-Adversary) per CLAUDE.md "Council conduct" non-negotiable.
- **For codex adversarial review:** the codex CLI is wired into
  `tools/run_codex_review_for_dispatch.py` (Catalog #271). Use it for
  any code-changing landing that touches ≥2 canonical files.

---

**If something in this document is stale, the file `SYSTEM_MAP.md` and
`PROGRAM.md` should agree with this one. If they disagree, this file is
the most-recently-refreshed. Regenerate via `tools/regenerate_handoff_brief.py`
(deferred; not yet landed).**
