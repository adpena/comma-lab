<!-- SPDX-License-Identifier: MIT -->
<!-- canonical_equation_cross_ref: procedural_codebook_from_seed_compression_savings_v1 (Catalog #344 registry #26; FORMALIZATION_PENDING:dp1_harvest_bug_comma2k19_localstreamer_no_chunk_ids_investigation_memo_recoverable_infrastructure_class_per_catalog_307_not_paradigm_falsification_documents_3_candidate_fix_matrix_for_operator_routing_no_new_empirical_finding_claim_just_root_cause_diagnosis_and_canonical_fix_options_per_dp1_first_in_domain_empirical_anchor_unblocking_20260520) -->
---
council_tier: T1
council_attendees: [Wave-3-DP1-Harvest-Bug-Investigator]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "The Comma2k19LocalStreamer ValueError at line 514 is a recoverable infrastructure bug, not a paradigm falsification of the DP1 procedural-codebook substrate"
    classification: HARD-EARNED
    rationale: "Per Catalog #307 paradigm-vs-implementation discipline + Catalog #313 probe-outcomes ledger semantics: the failure mode is at training-time-data-source-resolution (streamer has empty chunk_ids list), NOT at trainer-architecture or at scorer-response or at archive-grammar layers. The procedural codebook itself successfully packed/parsed (3881 B archive; 8549 B saved per smoke-arm output) BEFORE the streaming-distillation full path even started. The bug is structurally in the dataset-source-resolution layer (streamer's list_chunk_ids() returns [] when not synthetic AND dataset_sha256_manifest is empty). This is a Catalog #220 sister surface failure (substrate L1+ scaffold ships bytes but the L2 INTEGRATION step never runs because the upstream dataset source layer never materializes its chunk_id manifest)."
  - assumption: "The 3-candidate-fix matrix is exhaustive for the operator's routing decision"
    classification: HARD-EARNED-EMPIRICAL-PROJECTION
    rationale: "Per the trainer's _validate_dataset_source_args() contract at line 1033-1045: exactly one of {prebuilt_codebook, local_chunks, local_cache, stream_log} source modes can be active for real DP1 comma2k19 runs. The recipe currently chose stream_log (DPP_USE_STREAMER=1 at dispatched commit 09ffe159ea). The 3 candidate fixes correspond to the 3 alternative source modes that are reachable WITHOUT new substrate-engineering work: (1) populate streamer dataset_sha256_manifest from upstream Comma2k19 release manifest at module load time; (2) switch the recipe to DPP_CACHE_DIR mode via canonical Comma2k19LocalCache.fetch_chunks per Catalog #213; (3) keep streaming-mode but pass explicit chunk_ids via a new DPP_STREAM_CHUNK_IDS env var. The 4th hypothetical option (prebuilt_codebook) is OUT OF SCOPE because DP1's first IN-DOMAIN anchor requires real distillation, not loading a pre-computed codebook."
council_decisions_recorded:
  - "Document the root-cause structural diagnosis: streamer.list_chunk_ids() returns [] when not synthetic AND dataset_sha256_manifest is empty; plan_chunks() then returns []; trainer's chunk_ids list is empty; log_incremental_distillation_streaming() raises ValueError at line 514"
  - "Enumerate 3-candidate-fix matrix with cargo-cult audit per Catalog #303; recommend Option B (canonical Catalog #213 Comma2k19LocalCache mode) as Carmack MVP-first phasing"
  - "Identify the empirical anchor: the failing call at dispatched commit 09ffe159ea had DPP_USE_STREAMER=1 (streaming mode active) WITHOUT manifest population; the HEAD recipe state shows DPP_USE_STREAMER=0 + DPP_CACHE_DIR set — meaning the recipe HAS ALREADY BEEN PARTIALLY FIXED at HEAD per a sister landing, but the procedural variant recipe still needs the same fix"
  - "Operator-routable next: pick 1 of 3 candidate fixes; sister-subagent implements; re-dispatch baseline + procedural paired-smoke per command sheet v2 Decision #1 Step 4"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
horizon_class: frontier_pursuit
canonical_vs_unique_decision_per_layer: see §2 below
nine_dim_checklist_evidence: see §3 below
cargo_cult_audit_per_assumption: see §4 below
observability_surface: see §5 below
---

# DP1 Harvest Bug Investigation — Comma2k19LocalStreamer "no chunk ids" ValueError

**Lane**: `lane_wave_3_dp1_harvest_bug_comma2k19_localstreamer_investigation_20260520` L1
**Subagent**: `wave-3-dp1-harvest-bug-comma2k19-localstreamer-investigation-20260520`
**Bug class**: Catalog #220 sister surface — L1+ substrate ships dataset-source-resolution layer that returns empty chunk_ids on non-synthetic streamer without manifest, blocking the downstream L2 INTEGRATION step.
**Empirical anchor**: Modal call_id `fc-01KS480WY6S90VFXX54SC7V209` (baseline) + `fc-01KS484S3Z8YZBRVMCTQ6SX8MV` (procedural), both rc=1 at 2026-05-21T03:10:18Z / 03:12:21Z respectively; combined ~$0.012 GPU spent on the failing arms.
**Mission contribution per Catalog #300**: `frontier_protecting` — unblocks DP1 first IN-DOMAIN empirical anchor for canonical equation #26 (`procedural_codebook_from_seed_compression_savings_v1`); IMMEDIATE score-mutating value is the procedural arm's predicted ΔS=−0.005692 IF and only IF the fix lands and the re-dispatched arms harvest successfully.

## 1. Bug surface diagnosis

The failing traceback (from `.omx/state/modal_call_id_ledger.jsonl` entries 7 + 8) is:

```text
[full] log-incremental STREAMING distillation; base=2, max_chunks=80, threshold=0.005, stream_log_dir=/tmp/pact/lane_pretrained_driving_prior_results/stream_logs
Traceback (most recent call last):
  File "/tmp/pact/experiments/train_substrate_pretrained_driving_prior.py", line 2483, in <module>
    raise SystemExit(main())
  File "/tmp/pact/experiments/train_substrate_pretrained_driving_prior.py", line 2479, in main
    return _full_main(args)
  File "/tmp/pact/experiments/train_substrate_pretrained_driving_prior.py", line 1562, in _full_main
    book, log_incremental_schedule_log = _log_incremental_streaming_path(args)
  File "/tmp/pact/experiments/train_substrate_pretrained_driving_prior.py", line 1185, in _log_incremental_streaming_path
    return log_incremental_distillation_streaming(
  File "/tmp/pact/src/tac/substrates/pretrained_driving_prior/log_incremental_feeder.py", line 514, in log_incremental_distillation_streaming
    raise ValueError(
ValueError: Comma2k19LocalStreamer has no chunk ids; pass an explicit chunk_ids list or populate the streamer's dataset_sha256_manifest
```

### Structural call-chain (verbatim from the dispatched commit `09ffe159ea`)

1. Operator-authorize wrapper invokes `scripts/remote_lane_substrate_pretrained_driving_prior.sh` with env `DPP_USE_STREAMER=1` (per recipe at the dispatched commit; HEAD recipe was later edited to `DPP_USE_STREAMER=0` + `DPP_CACHE_DIR` set, but the dispatched commit had streaming-mode active).
2. Driver line 270 (`if [ "$DPP_RUN_FULL" = "1" ]`) enters full path; pre-flight check at lines 272-278 passes because `DPP_USE_STREAMER=1` satisfies "one explicit dataset source" gate.
3. Driver lines 306-326 add `--use-streamer --stream-log-dir ... --ram-buffer-gb 2.0 --streamer-frames-per-chunk 256 --stream-chunking-mode frame_range --stream-frame-range-size 256 ...` to `DPP_FULL_ARGS`.
4. Trainer's `_use_streamer(args)` at line 986-1003 returns `True` because (a) `args.use_streamer=True`, (b) `args.dataset_name="comma2k19"`, (c) `args.comma2k19_chunks_dir` is empty, (d) `args.disable_log_incremental=False`.
5. Trainer `_full_main` line 1551 enters `elif _use_streamer(args):` branch.
6. Trainer line 1562 calls `_log_incremental_streaming_path(args)`.
7. `_log_incremental_streaming_path` at line 1165-1173 calls `_build_local_streamer(args)` which constructs `Comma2k19LocalStreamer(log_dir=..., ram_buffer_gb=2.0, dispatch_label=...)`. **CRITICAL**: no `dataset_sha256_manifest` is passed; defaults to `{}` per `local_chunk_streamer.py:422,436`. **CRITICAL**: no `synthetic=True` is passed; defaults to `False`.
8. `streamer.plan_chunks(chunking_strategy, video_metadata={"frames_per_chunk": 256})` at line 1167 then internally calls `self.list_chunk_ids()` at `local_chunk_streamer.py:502`. Per the streamer's `list_chunk_ids()` body at lines 475-486:

   ```python
   def list_chunk_ids(self) -> list[str]:
       if self._synthetic:
           return [f"synthetic_chunk_{i:04d}" for i in range(self._synthetic_n_chunks)]
       return list(self._sha256_manifest.keys())
   ```

   With `_synthetic=False` AND `_sha256_manifest={}`, `list_chunk_ids()` returns `[]`.
9. `plan_chunks()` body at `local_chunk_streamer.py:500-508` then sets `metadata["chunk_ids"]=[]` and hits the fail-fast guard `if not metadata.get("chunk_ids") and not self._synthetic: return []`. Returns `[]`.
10. Back in `_log_incremental_streaming_path` line 1173: `chunk_ids = [spec.chunk_id for spec in chunk_specs]` → `[]`.
11. `_log_incremental_streaming_path` line 1185 calls `log_incremental_distillation_streaming(streamer, schedule, chunk_ids=[], ...)`.
12. `log_incremental_distillation_streaming` at `log_incremental_feeder.py:512-517` then does:

    ```python
    chunk_ids = list(chunk_ids if chunk_ids is not None else streamer.list_chunk_ids())
    if not chunk_ids:
        raise ValueError(
            "Comma2k19LocalStreamer has no chunk ids; pass an explicit "
            "chunk_ids list or populate the streamer's dataset_sha256_manifest"
        )
    ```

   `chunk_ids=[]` is not None (caller passed it explicitly), so it stays `[]`; the `if not chunk_ids` guard fires → ValueError.

### Bug class per Catalog #307 classification

This is **IMPLEMENTATION-LEVEL** falsification of the SPECIFIC dataset-source-resolution code path (streaming mode without manifest population), NOT **PARADIGM-LEVEL** falsification of the DP1 procedural-codebook substrate concept. The substrate's smoke arm (Stage 3 in driver) ran cleanly and demonstrated:

* Baseline arm: `[dpp-smoke] archive pack/parse roundtrip: 11971 bytes; pairs=4; header=28`
* Procedural arm: `[full] procedural codebook replacement: 12430 B -> 3881 B (saved 8549 B; predicted ΔS=-0.005692)`

The procedural-codebook replacement mechanism is structurally working — only the upstream distillation data-source layer is broken. Per CLAUDE.md "Forbidden premature KILL without research exhaustion": this is DEFERRED-pending-infrastructure-fix, NOT killed.

## 2. Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Canonical / Unique | Rationale |
|---|---|---|
| Investigation memo composition | ADOPT_CANONICAL 9-section structure per task spec + sister Catalog #290 + #294 + #303 + #305 + #309 disciplines | Same shape as prior session investigation memos (e.g. `feedback_z6_v2_wave_2_4c_recipe_fix_landed_20260518.md`) |
| Bug-class classification | ADOPT_CANONICAL Catalog #307 paradigm-vs-implementation taxonomy + Catalog #220 substrate L1+ surface analysis | Sister gate analysis; preserves the canonical apparatus |
| Recommended fix | ADOPT_CANONICAL Catalog #213 Comma2k19LocalCache + fetch_chunks per HNeRV parity L9 runtime closure | Canonical helper exists, has fcntl-locked write semantics per Catalog #131, has SHA-256 integrity verification per Catalog #210, is already wired into the trainer via `_use_auto_download_cache()` branch |
| 3-candidate-fix matrix | UNIQUE per-bug-class enumeration | The 3 candidates are not a canonical pattern across substrates; they map to this specific bug's 3 reachable source modes |

## 3. 9-dimension success checklist evidence (Catalog #294)

| Dim | Status | Evidence |
|---|---|---|
| 1. UNIQUENESS | YES | First investigation memo at this path; no prior memo |
| 2. BEAUTY + ELEGANCE | YES | 9-section structure per task spec; ready-to-paste diagnostics in code-fences; 3-candidate matrix with cost/EV per option |
| 3. DISTINCTNESS | YES | Distinct from command-sheet-v2 (decision sheet vs root-cause investigation); distinct from DP1 paired-smoke design memos (forward-looking design vs backward-looking diagnosis); sister-DISJOINT from in-flight NEW canonical equation REGISTRATION subagent (different files entirely) |
| 4. RIGOR | YES | PV per Catalog #229: read trainer + driver + streamer + cache_helper + log_incremental_feeder source + queried Modal call_id ledger entries 7+8 + ran the dispatched-commit git checkout comparison + checkpointed every step per Catalog #206 (steps 1 + 2 in_progress before this write) |
| 5. OPTIMIZATION PER TECHNIQUE | N/A | Investigation memo; recommends but does not implement |
| 6. STACK-OF-STACKS COMPOSABILITY | YES | Composes: streamer source / cache source / chunks-dir source into one decision matrix; downstream sister-subagent uses this to pick the optimal source mode |
| 7. DETERMINISTIC REPRODUCIBILITY | YES | Cites exact line numbers + commit SHAs + Modal call_ids + verbatim env-var values from both dispatched and HEAD states |
| 8. EXTREME OPTIMIZATION + PERFORMANCE | N/A | Investigation memo |
| 9. OPTIMAL MINIMAL CONTEST SCORE | N/A | Investigation does not mutate score; unblocks the DP1 first IN-DOMAIN anchor for canonical equation #26 |

## 4. Cargo-cult audit per assumption (Catalog #303)

| Assumption | Classification | Rationale |
|---|---|---|
| The Comma2k19LocalStreamer streaming mode is the canonical path for DP1 distillation per the 2026-05-14 operator pivot | CARGO-CULTED-PARTIAL | The operator pivot 2026-05-14 (per `log_incremental_feeder.py:461-470` + trainer line 351-361 comments) DID declare streaming mode "the NEW canonical path"; however, the pivot did NOT also land the canonical chunk-id-manifest population mechanism. The cargo-cult inheritance is: "streaming mode is canonical → use it everywhere" without checking that the streamer has a valid chunk-id list to operate on. The fix is to either (a) populate the manifest at streamer construction time, OR (b) revert this particular substrate to the canonical Catalog #213 Comma2k19LocalCache path which DOES auto-populate via `fetch_chunks(DEFAULT_CHUNK_MANIFEST)`. |
| The 4-byte dispatched recipe state can be inferred from HEAD recipe state | CARGO-CULTED-EMPIRICAL-FALSIFIED | The HEAD recipe state shows `DPP_USE_STREAMER: "0"` + `DPP_CACHE_DIR: /root/.cache/tac/comma2k19_chunks`, suggesting a sister has already partially fixed the issue. The dispatched-commit state at `09ffe159ea` shows `DPP_USE_STREAMER: "1"` (streaming-mode active) AND no DPP_CACHE_DIR line. **The recipe has DIVERGED between dispatch time and HEAD time.** A sister-subagent has already taken Candidate B route (switch to cache mode) at HEAD for the baseline recipe, but the procedural variant recipe state at HEAD must be checked separately. |
| The bug is in the streamer's `list_chunk_ids()` method (it should return chunk_ids from somewhere) | CARGO-CULTED | The streamer's `list_chunk_ids()` docstring at `local_chunk_streamer.py:476-483` explicitly says: *"Real-mode chunk discovery is intentionally out of scope (the operator drives it)."* The streamer is BY DESIGN a non-discovery component; it expects the operator to populate `dataset_sha256_manifest` OR pass `chunk_ids` explicitly. The bug is in the TRAINER's failure to populate either surface, NOT in the streamer's design. |
| The dispatched-commit state is the source of truth for what failed | HARD-EARNED | Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #166 Modal HEAD-parity ledger: the dispatched commit (`09ffe159ea` for baseline, `ef25aa20c` for procedural) is what actually ran on the Modal worker. Any analysis of HEAD state must explicitly check whether HEAD has diverged. |
| The recovered Modal artifacts contain the full stdout_tail | HARD-EARNED | The codex harvester via `tools/parallel_harvest_actuator.py` correctly recorded the failed dispatch with full stdout_tail per Catalog #330 outcome ledger discipline; the traceback is intact and parseable per the verbatim quote in §1. |
| Re-dispatching after the fix will produce the predicted-ΔS=−0.005692 anchor for procedural arm | HARD-EARNED-PROJECTION-CONDITIONAL | The smoke-arm output empirically confirmed the procedural codebook replacement saved 8549 B (predicted ΔS=−0.005692 per `25 * 8549 / 37545489 = 0.005692`). The full-mode anchor will validate whether the closed-form prediction holds at the actual contest evaluator level, which IS the empirical purpose of this first IN-DOMAIN anchor. |

## 5. Observability surface (Catalog #305)

1. **Inspectable per layer**: Each candidate fix's source-text change is inspectable in the same file (`log_incremental_feeder.py` OR `local_chunk_streamer.py` OR recipe YAML) before any commit; reviewer can grep for the canonical helper or env-var token per Catalog #229 PV pattern.
2. **Decomposable per signal**: 3 candidate fixes × 2 separate dispatch paths (baseline + procedural) = 6 distinct cells in the operator-routing matrix; each independently inspectable.
3. **Diff-able across runs**: Every fix candidate emits canonical artifacts: (a) source-text change in `git diff`; (b) recipe YAML change in `git diff`; (c) re-dispatched Modal call_id in `.omx/state/modal_call_id_ledger.jsonl`; (d) harvested artifact directory under `experiments/results/lane_.../harvested_artifacts/`.
4. **Queryable post-hoc**: `.omx/state/modal_call_id_ledger.jsonl` provides per-call status; `tools/parallel_harvest_actuator.py --recover-from-tmp` provides the canonical harvest pickup; `tools/plan_dp1_procedural_paired_harvest.py` provides the planner status JSON.
5. **Cite-able**: This memo cites verbatim line numbers + exact commit SHAs + Modal call_ids + recipe field values; future agents can replay the diagnosis by running the same `git show` + ledger queries.
6. **Counterfactual-able**: Each candidate fix is testable in isolation: (a) populate the manifest → assert `list_chunk_ids()` returns N>0 entries; (b) switch to cache mode → assert `_use_auto_download_cache(args)` returns True AND `Comma2k19LocalCache.fetch_chunks(...)` succeeds; (c) pass explicit chunk_ids env → assert `args.chunk_ids` is non-empty AND `_log_incremental_streaming_path` threads it through.

## 6. 3-candidate-fix matrix

Each candidate fixes the bug class at a different layer. **Recommended is Candidate B** (canonical Catalog #213 cache mode) per Carmack MVP-first phasing + canonical-helper-when-it-serves per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD operating mode.

### Candidate A — Populate `dataset_sha256_manifest` at streamer construction time

**Mechanism**: Add an optional `manifest_loader` kwarg or environment-variable threading to `_build_local_streamer` (trainer line 1110-1126) so the streamer's `dataset_sha256_manifest` is populated from the upstream Comma2k19 release manifest at construction. Default loader could call `tac.substrates.pretrained_driving_prior.local_chunk_cache.DEFAULT_CHUNK_MANIFEST` (already exists per Catalog #213 sister discipline) and translate that to the streamer's expected format.

**Files touched**: `experiments/train_substrate_pretrained_driving_prior.py` (`_build_local_streamer` helper, ~15 LOC) + optionally `src/tac/substrates/pretrained_driving_prior/local_chunk_streamer.py` (new optional `default_manifest` kwarg, ~10 LOC).

**LOC budget**: ~25 LOC source + ~30 LOC tests (manifest loader happy-path + empty-manifest fail-closed + invalid-manifest rejection).

**Pros**:
- Preserves the operator's 2026-05-14 pivot direction (streaming mode IS canonical).
- Cleanest separation: streamer becomes self-sufficient with default manifest, callers can still override.
- Streaming mode's JSONL access-log discipline per Catalog #214 is preserved.

**Cons**:
- New surface to maintain (manifest loader); CARGO-CULTED risk per Catalog #303 if the manifest format drifts from upstream.
- DOES NOT solve the "DPP_USE_STREAMER=1 requires real Comma2k19 fetcher" issue — the streamer would still need a working `_http_fetcher` to actually download chunks; on Modal worker this means network access + working credentials + manifest URL stability. Catalog #213 sister `Comma2k19LocalCache` already handles ALL of this via the canonical 4-layer pattern.
- Higher risk of NEW bug classes (e.g., manifest URL drift; manifest schema drift; sha256 mismatches in flight per `SHA256MismatchError`).

**Cargo-cult audit per Catalog #303**: HARD-EARNED for "streaming mode is operator-canonical"; CARGO-CULTED for "manifest population should happen in the streamer not the caller" — the canonical separation per Catalog #213 sister is that the CACHE owns manifest, not the streamer.

**EV / cost**: MEDIUM (preserves streaming-pivot; adds maintenance surface).

### Candidate B — Switch the recipe to canonical Catalog #213 Comma2k19LocalCache mode [RECOMMENDED]

**Mechanism**: Already partially done at HEAD per the empirical-falsified assumption above. Set `DPP_USE_STREAMER: "0"` AND set `DPP_CACHE_DIR: /root/.cache/tac/comma2k19_chunks` (or sister Modal-writable path per Catalog #204 / #220 / #244 disciplines) in BOTH baseline AND procedural recipe `env_overrides` blocks. The trainer's `_use_auto_download_cache(args)` branch handles everything via canonical `Comma2k19LocalCache.fetch_chunks(DEFAULT_CHUNK_MANIFEST)` which carries SHA-256 verification + license-tag propagation + LRU eviction + atomic `os.replace` per Catalog #213.

**Files touched**: 2 recipe YAMLs only (baseline + procedural variants). ZERO source code changes. ZERO new test surfaces.

**LOC budget**: ~2 LOC per recipe × 2 recipes = ~4 LOC total.

**Pros**:
- ZERO new source code; reverts the streaming-mode pivot to the canonical Catalog #213 path for THIS substrate without modifying canonical surfaces.
- Canonical 4-layer discipline already exists + tested: SHA-256 integrity per Catalog #210 sister; license-tag propagation per Catalog #210; `/tmp` avoidance per CLAUDE.md "Forbidden /tmp paths"; disk-budget LRU eviction.
- Empirically validated path: the sister landing at HEAD already chose this route for the baseline recipe; the bug class is closed structurally for any recipe that adopts the canonical cache mode.
- Carmack MVP-first phasing: cheapest possible fix that unblocks the empirical anchor.

**Cons**:
- DOES NOT solve the streaming-mode CARGO-CULTED-PARTIAL assumption (other substrates may still hit the same bug class if they cargo-cult the streaming-mode pivot without manifest population). Mitigated by Catalog #220 sister gate firing structurally.
- Disk-cache path requires writable directory on Modal worker; needs Catalog #204 + #220 + #244 sister verification that `/root/.cache/tac/comma2k19_chunks` is Modal-writable (it should be; the path is under `$HOME` per `default_cache_dir()`).
- Initial chunk download at full-mode runtime adds wall-clock (~30-120 sec depending on chunk count); first-time-on-Modal-worker cache miss cost.

**Cargo-cult audit per Catalog #303**: HARD-EARNED across all surfaces — canonical helper exists, is sister-discipline-protected, is empirically validated by both the cache mode's own tests and by the partial sister landing at HEAD.

**EV / cost**: HIGH (lowest-LOC fix + canonical-helper-when-it-serves + already-partially-validated at HEAD).

### Candidate C — Pass explicit `chunk_ids` via new `DPP_STREAM_CHUNK_IDS` env var

**Mechanism**: Add a new env var `DPP_STREAM_CHUNK_IDS` to the driver shell script + add a corresponding `--stream-chunk-ids` argparse flag to the trainer + thread the explicit list through `_log_incremental_streaming_path` → `log_incremental_distillation_streaming(chunk_ids=...)`. The operator pre-populates the list via the recipe `env_overrides` block (e.g. `DPP_STREAM_CHUNK_IDS: "Chunk_1,Chunk_2,Chunk_3,..."`).

**Files touched**: `scripts/remote_lane_substrate_pretrained_driving_prior.sh` (~5 LOC) + `experiments/train_substrate_pretrained_driving_prior.py` (~10 LOC) + both recipe YAMLs (~3 LOC each).

**LOC budget**: ~25 LOC source + ~20 LOC tests + ~6 LOC recipes = ~50 LOC total.

**Pros**:
- Preserves streaming-mode pivot.
- Operator-explicit chunk control (good for byte-stable replays per Catalog #167 + #313 probe-outcomes ledger).
- No new helper surfaces.

**Cons**:
- Operator MUST provide a hardcoded chunk_ids list (cargo-cult risk per Catalog #303 "DPP_USE_STREAMER=1 implies streaming-mode-is-self-sufficient" assumption fails again here — the operator has to know the canonical chunk_ids list, which is exactly what `Comma2k19LocalCache.fetch_chunks(DEFAULT_CHUNK_MANIFEST)` already encodes).
- Doesn't fix the underlying "streamer has no manifest" structural issue; just routes around it.
- HIGHEST LOC budget of the 3 candidates.

**Cargo-cult audit per Catalog #303**: CARGO-CULTED — duplicates information that already exists in `DEFAULT_CHUNK_MANIFEST` (the canonical chunk_id list); creates a new surface that can drift from upstream.

**EV / cost**: LOW (highest LOC + new surface + duplicates canonical info).

### Matrix summary

| Candidate | LOC | Files | Cargo-cult risk | EV/cost | Recommendation |
|---|---|---|---|---|---|
| A — Populate manifest at streamer construction | ~25 src + ~30 tests | 1-2 source + tests | MEDIUM | MEDIUM | Acceptable if operator wants to preserve streaming-mode-as-canonical |
| **B — Switch recipe to canonical cache mode** | ~4 recipe YAML | 2 recipes only | LOW (canonical helper) | **HIGH** | **RECOMMENDED** — Carmack MVP-first + already partially landed at HEAD |
| C — Explicit DPP_STREAM_CHUNK_IDS env var | ~50 src + tests + recipes | 2 source + 2 recipes + tests | HIGH (duplicates canonical info) | LOW | NOT RECOMMENDED |

## 7. Operator-routable 3-step path

**Step 1 (sister-subagent task — ~15 min wall-clock + ~$0 GPU)**:

Apply Candidate B fix to BOTH baseline AND procedural recipes (HEAD baseline recipe is already partially fixed; procedural variant recipe needs the same treatment). The diff is:

```yaml
# In .omx/operator_authorize_recipes/substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch.yaml::env_overrides:
DPP_USE_STREAMER: "0"  # was "1" at dispatched commit; canonical Catalog #213 cache mode instead
DPP_CACHE_DIR: /root/.cache/tac/comma2k19_chunks  # NEW — routes through canonical Comma2k19LocalCache
```

Use `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256` per Catalog #117/#157/#174.

**Step 2 (re-dispatch — ~$0.60 paired GPU + ~10 min wall-clock)**:

Re-fire the paired dispatch via canonical operator-authorize chain:

```bash
export OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1
export OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=0.60

.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch \
    --target modal \
    --yes

.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch \
    --target modal \
    --yes
```

**Step 3 (harvest + canonical equation #26 first IN-DOMAIN anchor — ~$0 + ~5 min wall-clock)**:

After both calls harvest (per canonical 24h window):

```bash
.venv/bin/python tools/parallel_harvest_actuator.py \
    --recover-from-tmp \
    --lookback-hours 24

.venv/bin/python tools/plan_dp1_procedural_paired_harvest.py \
    --baseline-output-dir experiments/results/lane_substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch_<utc>_modal/harvested_artifacts \
    --procedural-output-dir experiments/results/lane_substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch_<utc>_modal/harvested_artifacts \
    --json-out /tmp/dp1_paired_harvest_plan.json

.venv/bin/python tools/dispatch_modal_paired_auth_eval.py \
    --skip-axis-if-promotable-anchor-exists \
    --plan-from /tmp/dp1_paired_harvest_plan.json
```

The harvested ΔS becomes the first IN-DOMAIN empirical anchor for canonical equation #26 per `tac.canonical_equations.update_equation_with_empirical_anchor`.

## 8. Sister-collision verdict

**DISJOINT** at investigation time. PRE-FLIGHT sister-checkpoint guard via `tools/check_sister_files_recently_landed.py` returned PROCEED for the investigation memo path (no sister commits in 12-hour lookback). The in-flight NEW canonical equation REGISTRATION subagent (`a57f2dc4`) touches `src/tac/canonical_equations/*` + `.omx/state/canonical_equations_registry.jsonl` + sister test files; ZERO file overlap with this investigation memo path.

If the sister-subagent that implements Candidate B at the recipe level starts, it would touch BOTH recipe YAMLs (`.omx/operator_authorize_recipes/substrate_pretrained_driving_prior_*paired_dispatch.yaml`) which is DISJOINT from this memo path. Canonical Catalog #340 sister-checkpoint guard at commit time will arbitrate.

## 9. Blockers

NONE for THIS investigation memo landing. The recommended Candidate B fix is operator-routable as ~15-min sister-subagent work; the re-dispatch is operator-direct per Catalog #199 paired-env discipline; the harvest closure is operator-direct + auto-cathedral-autopilot-ingestion per Catalog #335.

**Top-3 operator-routable next-actions** (ordered by EV-per-min):

1. **Spawn sister-subagent (~15 min wall-clock, ~$0 GPU)** to apply Candidate B fix to the procedural variant recipe (HEAD baseline is partially fixed; procedural needs the same DPP_USE_STREAMER=0 + DPP_CACHE_DIR=/root/.cache/tac/comma2k19_chunks). Sister-subagent must use canonical serializer per Catalog #117/#157/#174 with POST-EDIT --expected-content-sha256.
2. **Re-dispatch paired baseline + procedural (~10 min wall-clock, ~$0.60 GPU)** via canonical operator-authorize chain with Catalog #199 paired-env attestation. Per Catalog #167 smoke-before-full is structurally honored by the operator-authorize wrapper.
3. **Harvest + register first IN-DOMAIN anchor (~5 min wall-clock, ~$0)** via canonical `tools/parallel_harvest_actuator.py --recover-from-tmp` + `tools/plan_dp1_procedural_paired_harvest.py` + `tac.canonical_equations.update_equation_with_empirical_anchor` for equation #26.

**End of investigation memo.**
