# FEED-INFRA — V9 C0 governed storage and memory gate

**UTC:** 2026-07-15T09:44:26Z
**Lane:** `lane_infra_governed_launch_gate_20260715`
**Status:** `MEMORY_ADMIT_LOCAL_STORAGE_PASS_REAL_SPAWN_BLOCKED`
**verdict_scope:** INSTANCE — this isolated worktree, the live 128 GiB host snapshot, and the
`v9_cgauge_ideal_mod19` C0 ticket. Neither V9 nor micro-batching receives a family negative.
**Pointer:** submittable 0.19108 UNMOVED; borrowed-bank 0.18804 UNMOVED. No GPU or paid dispatch.

## STORES CONSULTED

- `docs/operating_manual_craft_handoff.md`, `CLAUDE.md`, `AGENTS.md`, the v7.5/v8 operating specs,
  `PROGRAM.md`, and `.omx/research/P0_campaign_queue_20260715.md`.
- `/Volumes/VertigoDataTier/pact`: mounted, `adpena:staff`, 771 GiB filesystem free at diagnosis;
  workspace write probe and child mkdir refused with `PermissionError`.
- `/Volumes/APDataStore/pact`: not mounted.
- `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` plus a depth-6 search of the connected SSD:
  no cache found from this worktree.
- `.omx/state/lane_registry.json`, `.omx/state/subagent_progress.jsonl`, the V9 typed compiler,
  trainer parser, lever registry, memory governor, and storage waterfall planner.
- Live inbox through broadcast cursor `2026-07-14T20:32:37Z`; no INFRA-specific later directive.

## DAG nodes

1. `#261_B1_B4_n8_memory_anchor` — **MEASURED historical diagnostic:** complete process peak
   5,907 MiB at B1 and 5,918 MiB at B4. It is not a current-V9 n600 observation.
2. `v9_micro_batch_conservative_launch_memory_guard_v1` — **DERIVED:**
   `G(B)=max(0,B-1)*(5918/1024) GiB`; the entire measured B4 process peak is charged for each
   extra pair. This is deliberately larger than the historical 11 MiB B4-B1 delta.
3. `v9_mod19_n600_memory_projection` — **DERIVED:** self-orient OFF, vbatch32 gives B1 24.48 GiB
   and B2 30.26 GiB. Realized-spike-adjusted values are 30.78 and 36.56 GiB, both below the
   108.8 GiB single-workload envelope.
4. `live_system_admission_20260715T0940Z` — **MEASURED host snapshot + DERIVED composition:** B1
   ADMIT with 51.3 GiB headroom; B2 ADMIT with 45.5 GiB headroom; zero active heavy jobs.
5. `micro_batch_waterfill_split_custody` — **DERIVED selection:** with target-n projected RSS and
   historical n24 timing kept separate, the B2/vbatch32 point is SAFE and wins x1.001. The speed
   number is `MEASURED_HISTORICAL_N24_DIAGNOSTIC_NOT_CURRENT_V9_AUTHORITY`.
6. `storage_waterfall` — preferred Vertigo child creation REFUSES in this sandbox; APDataStore is
   absent; explicit operator-authorized local tier selects a writable root with 596+ GiB usable
   after the 40 GiB reserve. Storage therefore has a local PASS, not a preferred-tier PASS.
7. `exact_c0_dry_run` — **MEASURED dry-run:** rc0, 224/224 flags, DSL/schedule provenance PASS,
   memory PASS, system ADMIT, no spawn. Real spawn remains NO-GO here because the host-specific
   safe-compile certificate and the custodied n600 GT cache are absent.

## Triality

- **DSL:** the named clean C0 compiler emits self-orient OFF, `--micro-batch-pairs 1`, and
  `--verdict-batch 32`; the lever registry reports `--micro-batch-pairs` MAPPED by `MicroBatch` and
  non-stale. B2 is an existing typed treatment, not an invented flag. It must not silently replace
  C0 because the canonical control pins B1.
- **DAG:** this FEED separates historical measurements, derived guard arithmetic, live system
  composition, storage placement, and real-spawn blockers.
- **Equations:** `src/tac/canonical_equations/micro_batch_memory_guard_20260715.py` exposes the
  canonical equation; dependency-free arithmetic lives in `src/tac/micro_batch_memory_guard.py` so
  the standalone governor does not import optional scientific packages. The preflight consumes that
  core, and the waterfill consumes the resulting target-n envelope rather than defaulting to B1.

## Exact disposition

`NO_GO_REAL_SPAWN_IN_THIS_WORKTREE`. The two requested infrastructure questions are resolved:
memory admits C0 B1 and the B2 treatment, and the storage waterfall has a passing explicit local
fallback. A real C0 spawn would nevertheless fail closed on two newly exposed custody prerequisites:

1. `SAFE_COMPILE_HOST_CERTIFICATE_ABSENT` — exact remediation printed by the launcher:
   `PYTHONPATH=src python -m tac.mlx_safe_compile --certify --out .omx/state/mlx_safe_compile_manifest.json`.
   This arm did not run it because certification is host/MLX work and the mission forbids GPU use.
2. `GT_N600_CACHE_ABSENT` — supply a custodied cache path/hash, or rebuild through
   `tools/build_shared_gt_cache_for_mlx_fleet.py` onto the selected storage tier after its storage
   preflight. This arm did not create a 5 GiB cache.

For the preferred SSD child only, observed ownership does not require sudo. Outside this restricted
workspace the exact remediation is:
`mkdir -p /Volumes/VertigoDataTier/pact/experiments/results/v9_cgauge_ideal_mod19_20260715 && chmod u+rwx /Volumes/VertigoDataTier/pact/experiments/results/v9_cgauge_ideal_mod19_20260715`.

Machine-readable custody: `infra_v9_cgauge_mod19_governed_gate_receipt_20260715.json`, both storage
plans, the B1/B2 point file, and the generated dry-run `launch.sh`/constants manifest.

## Round-1 recursive review and three-clean seal

Round 1 did **not** rubber-stamp the first patch. It found and fixed four class issues:

1. `POINT-CUSTODY-01`: a searchable DERIVED RSS row was setting `KnobStatus.measured=true`.
   `searchable` and `measured` are now separate; the committed B2 row is searchable but explicitly
   not measured.
2. `POINT-VALIDATION-01`: duplicate-B, non-finite, non-positive, or malformed-scale point rows could
   be ambiguous or fail during arithmetic. They now fail closed to B1 with a reason.
3. `B-INTEGER-01`: the preflight API silently truncated non-integral B through `int()`. It now uses
   integer-index validation and refuses floats/bools.
4. `STANDALONE-DEPS-01`: importing the equation through the canonical-equations package pulled
   optional SciPy into the standalone preflight. The dependency-free core/canonical-wrapper split
   restores raw system-Python operation.

After those fixes, three consecutive clean passes completed:

1. **Code/test pass:** 149 passed, 1 platform skip across equation, preflight, waterfill, launcher,
   lever-registry, and V9 config tests; changed-file Ruff and `git diff --check` clean.
2. **Governed dry-run pass:** rc0, 224/224 flags, DSL/schedule/memory/system gates green, emitted
   `launch.sh` byte hash stable, and the two real-spawn custody blockers reproduced exactly.
3. **Artifact consistency pass:** 16/16 assertions green across B1/B2 recomputation, point rows,
   storage PASS/REFUSE split, emitted C0 flags, artifact hashes, lever mapping, pointer, and score
   authority; changed-file Ruff and diff check clean.

Lane-local maturity now records `impl_complete`, `strict_preflight`, and `three_clean_review` at L1.
The global `lane_maturity.py validate` remains red on 110 pre-existing historical evidence paths
that are absent from this isolated worktree; none names this lane. This pass did not rewrite or
weaken those unrelated historical gates.
