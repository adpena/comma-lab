# ddm_cu1 — a clean clone could not reproduce our own frontier. Now it can.

**Date:** 2026-08-03 · **Arm:** `ddm_cu1` · **Axis:** apparatus (custody / deterministic
reproducibility). **Pointer UNMOVED — this is not a score claim.** No archive bytes changed, no
solve changed, no `d_seg` / `d_pose` / rate changed. Every number below is a SHA or a byte count.

---

## Answer first

**The frontier code was never missing from git. It was in git and NOT RUNNABLE in the target
environment** — and the only runnable copy lived, hand-edited, on the external SSD.

MEASURED: a runtime tree assembled from repo files at `a125c34b6c` raised
`ModuleNotFoundError: No module named 'tac'` under a simulated contest runtime. Only the SSD
snapshot decoded. Three top-level/lazy `from tac....` imports in modules that are VENDORED FLAT
into the runtime tree were the whole cause. Someone had de-`tac`-ified them by hand at deployment;
that hand-edit was never landed, so the tracked source silently stopped being the source of truth.

**After the fix, MEASURED end-to-end:**

| leg | result |
|---|---|
| container archive rebuilt from a **clean clone at HEAD** | `sha256 1d3ab694c337f3f7374fa42034664b0494d0dfda1be479b1d367e964da78701f`, **353,808 B** — byte-identical to `ddm_cx1` |
| full 600-pair decode, repo-only runtime tree, **`tac` unimportable** | `0.raw sha256 988785e7cadfd6137d918b53d020fe5f34e5735765433a4873241cb15f7e200e`, **3,662,409,600 B** — byte-identical to the `ddm_cx1` reference |

So `ddm_pj2` 0.8308905 and `ddm_cx1` 0.8264972 are now reproducible from tracked sources alone.

---

## CU1-1 — the true external dependency set (DERIVED, not inherited)

Walked from the real entry points (`stage_v4d_realized_gate.sh`, `inflate_runner_v4d.py`, the
`pfs1` receiver, `ddm_ix2_archive_container`, `ddm_tr1_runtime`, `ddm_r7_token_coder`).

**The frontier path has ZERO untracked module imports.** Every file is tracked:

| vendored name | repo path | tracked |
|---|---|---|
| `inflate_runner.py` | `experiments/inflate_runner_v4d.py` | ✓ |
| `ddm_r7_token_coder.py` | `experiments/ddm_r7_token_coder.py` | ✓ |
| `ddm_tr1_runtime.py` | `src/tac/optimization/ddm_tr1_runtime.py` | ✓ |
| `pfs1_warp_receiver.py` | `src/tac/optimization/pfs1_warp_receiver.py` | ✓ |
| `ddm_ix2_archive_container.py` | `src/tac/optimization/ddm_ix2_archive_container.py` | ✓ |
| `repair_entropy_coder_runtime_adapters.py` | `src/tac/optimization/repair_entropy_coder_runtime_adapters.py` | ✓ |
| builder | `tools/cx1_build_ix2_container_archive.py` | ✓ |
| pose solve | `tools/pj2_pose_scale_joint_solve.py` | ✓ |

**Two filed rows corrected by measurement:**

* *"`pfs1_warp_receiver.py` exists ONLY on the external volume, not in git"* — **already closed**
  by `19839b98b5`. It is tracked at `src/tac/optimization/pfs1_warp_receiver.py` and the SSD copy is
  byte-identical (`ddfb9f9c24e3`). Row is stale; the defect it names is real but was elsewhere.
* *"MAIN IS UNCLONABLE — 11 untracked modules imported by 16 tracked files"* — the **11 is exactly
  right** and independently reproduced two ways (see CU1-3). The importer count is **27 tracked
  files**, not 16; 16 is the importer count of the single worst module,
  `tac.witness_dsl.taskspace_outer_archive_codec`. **None of the 11 are on the frontier path** —
  they are the `witness_dsl` v9/v10 lineage.

---

## CU1-2 — reconcile per file (a) ours / (b) drifted vendored copy / (c) not-code

Staged files vs their SSD deployment snapshot at `…/ddm_pfs1_20260729/d1/eval_root/submissions/pfs1`:

| file | repo sha12 | ssd sha12 | verdict |
|---|---|---|---|
| `pfs1_warp_receiver.py` | `ddfb9f9c24e3` | `ddfb9f9c24e3` | **(a)** identical — no action |
| `ddm_r7_token_coder.py` | `5ac88a67d497` | `2a9c5a3a0b11` | **(b)** drifted BOTH ways |
| `ddm_tr1_runtime.py` | `6f4916f9a466` | `f3b93708b542` | **(b)** repo is a strict superset |
| `repair_entropy_coder_runtime_adapters.py` | `7b5820ed29ea` | `8ef8520d5ed5` | **(b)** drifted BOTH ways |
| `inflate.sh` | — | 139 B | **(c)** stock 4-line bare-`python` wrapper, no decode logic |

**"Differ textually but agree bit-for-bit" was true only under the dev host, and that caveat was
load-bearing.** The repo copies agree bit-for-bit *when a `tac` package happens to be installed*.
In a contest runtime tree there is none, and the repo copies do not import at all. The direction of
drift is not uniform, which is why neither naive reconciliation works:

* repo → SSD: the repo carries **newer decode capability** the snapshot lacks
  (`_encode_smevr_reference`, `_smevr_encode_contexts`, `_decode_smevr_reference`,
  `decode_token_codes(..., verify=)`, `_pack_frame`/`_unpack_frame`,
  `_encode_tokens(codes, selector=None)`). Copying the SSD over the repo would **regress** these.
* SSD → repo: the snapshot carries the **runtime-layout edits** the repo lacks — flat
  `from repair_entropy_coder_runtime_adapters import …` and a local `sha256_bytes`. Assuming the
  repo copy was fine leaves it **unrunnable**.

**Behavioural divergence, measured, on the paths this build touches: NONE.** Same 12-pair decode
`sha256 e5ef66a9c191…` from the SSD tree and from the fixed repo tree; same full-600 `0.raw`
`988785e7…`. The divergence is entirely (i) capability the frontier does not yet exercise and
(ii) import layout.

**Fix — one file serves both layouts**, replicating the try/except the receiver already uses at
`inflate_runner_v4d.py:76-88`, so the snapshot is no longer needed:

* `experiments/ddm_r7_token_coder.py` — flat `repair_entropy_coder_runtime_adapters` first,
  `tac.optimization.…` fallback.
* `src/tac/optimization/repair_entropy_coder_runtime_adapters.py` — `tac.repo_io.sha256_bytes`
  with a local `hashlib` fallback, byte-identical (`sha256(data).hexdigest()`, verified against
  both the repo helper and the snapshot's own copy).
* `src/tac/optimization/ddm_tr1_runtime.py` — same for the lazy `ddm_ll1_window_solve` import.
  Latent only (`window_solve` defaults OFF and is byte-identical when off, so it is not on the v4d
  path) — fixed anyway because it is the same class.

`experiments/stage_v4d_realized_gate.sh` now sources the **entire** runtime tree from the repo
(`ddm_cx1` had fixed only the receiver), **fails closed** rather than falling back to a snapshot —
a silent fallback is what produced the stale-receiver defect — prints a sha for every staged
module, and derives `REPO_ROOT` from `BASH_SOURCE` (honouring `PACT_REPO_ROOT`) instead of the
hardcoded `/Users/adpena/Projects/pact`, which was itself clone-hostile.

---

## CU1-3 — the review gate is not the cause. A tolerated RED is.

I set out to show the gate blocks correct code, and **the measurement refuted that.**
`tools/audit_untracked_source_artifacts.py` is already wired as Gate #10 of
`tools/all_lanes_preflight.py`, already runs `--strict` with the disposition manifest, and **today
returns rc=2 with 118 blockers, 71 of them the stranded `witness_*` modules, naming each by path.**

The detector works and has been naming these files. What failed is that its RED is a standing,
tolerated condition — §8.9 "the silent guard" and the second-identical-warning rule. So the cure is
**not** a new gate (that would be the built-instead-of-paid trap); it is to **drain the queue.**

What made draining infeasible was shape: 161,127 untracked source-like rows, 161,010 dispositioned
as rebuildable/private, leaving **117 undispositioned** presented as one undifferentiated wall with
no signal about which ones actually matter. So I paid the debt on the existing surface:

`find_clone_breaking_untracked_imports()` joins the import graph against the untracked set and
reports **only** the case a fresh clone provably cannot survive — a TRACKED module importing a
module that resolves to an UNTRACKED file that **exists on disk**. A dotted import naming nothing
is a dead import, a different class, deliberately not claimed. Vendored/pinned trees (`upstream/`,
`submissions/`, `reverse_engineering/`, `experiments/results/`) are skipped: their imports describe
their layout, and the upstream snapshot is immutable.

Measured on this repo: **11 clone-breaking targets, 31 import edges** — reproducing the filed 11
from a completely independent method. Those 11 rows now carry
`[CLONE-BREAKING: a tracked module imports it]` in the blocker text and are printed first, so the
117 becomes "11 that break the build, then 106 that are merely untracked." Default exit-code
behaviour is unchanged, so Gate #10 keeps its current contract. 4 dedicated tests.

**Deliberately NOT done:** landing the 11 `witness_dsl` modules (~590 KB). They belong to another
lineage, they are off the frontier path, and my own fixes are unreviewed new code — landing half a
megabyte of someone else's in-flight work behind two rubber-stamped review passes would be exactly
the failure this arm exists to fix. They exit **OWNED**, not deferred: see NEXT-IF-RESUMED.

On `REVIEW_GATE_OVERRIDE=1`: not used. All `.py` went through two recorded
`tools/review_tracker.py mark-file` passes.

---

## CU1-4 — proof

1. **Clean clone** of `a125c34b6c` at `/Volumes/VertigoDataTier/pact/ddm_cu1_cleanclone_20260803`.
2. **Isolation verified, not assumed** (the shared-venv hijack is a named trap): with
   `PYTHONPATH=<clone>/src`, `tac.__file__` resolves into the **clone**, not `~/Projects/pact/src`.
3. **Build:** `tools/cx1_build_ix2_container_archive.py` from the clone, on
   `v4d_composed_pj2_archive.zip` → `sha256 1d3ab694c337…`, **353,808 B**. Byte-identical to
   `ddm_cx1`. Receipt: all eight `verification.*_bit_identical` / `*_exact` true.
4. **Decode, dev layout:** full 600 pairs → `0.raw sha256 988785e7cadfd613…`, 3,662,409,600 B.
   Byte-identical to the `ddm_cx1` reference.
5. **Decode, simulated contest runtime** (`tac` blocked by a `MetaPathFinder`, which is what
   `inflate.sh`'s bare `python` actually sees): **before the fix `ModuleNotFoundError: 'tac'`;
   after the fix** full 600 pairs → the same `988785e7cadfd613…`, 3,662,409,600 B.

**Still external, and why.** The chain is reproducible **given the solved artifacts**, which is the
honest scope: `v4d_composed_pj2_archive.zip` is the input to `cx1`'s transform, and re-running
`tools/pj2_pose_scale_joint_solve.py` over 600 pairs is a fresh solve, not a rebuild. Both the
solver and the transform are tracked, and the transform is bit-exact. The remaining external inputs
are the custodied clip (`upstream/videos/0.mkv`) and the solved pose artifact — neither is code.

**Evidence (durable, on the SSD — no `/tmp`):** `/Volumes/VertigoDataTier/pact/ddm_cu1_repro_20260803/`
holds `cu1_cleanclone_ix2_archive.zip`, `cu1_cleanclone_receipt.json`, `cloneonly_raw.sha256`,
`full_decode_isolated.log`, the two probe scripts, and the four staged trees. The 3.5 GB `0.raw`
files are rebuildable evidence with recorded shas, not tracked files.

---

## Round-1 self-review (attacks run against my own fixes)

* **Flat-first import shadowing in dev?** No flat `repair_entropy_coder_runtime_adapters.py` is
  reachable on `sys.path` in the dev layout (`src/tac/optimization` is not a flat path entry), so
  the flat attempt fails and the `tac.` fallback runs. Verified: dev-layout imports OK, 16 r7 tests
  and 88 audit/ix2/cx1 tests pass.
* **Does `except ModuleNotFoundError` mask a real error?** It would if the flat module existed but
  itself raised `ModuleNotFoundError` for a sub-dependency — which is precisely what the repo copy
  did before this fix, and no longer does. The narrower `ModuleNotFoundError` (not `ImportError`)
  is deliberate.
* **Point-fix or class-fix?** Class. All three `from tac....` sites in the runtime-tree modules were
  found by exhaustive grep and all three fixed, including the latent lazy one that is not on the
  v4d path. The staging script now sources the whole tree, not one file.
* **Would my test pass if the code were broken?** `test_clone_breaking_ignores_tracked_and_absent_targets`
  and `test_clone_breaking_skips_vendored_and_pinned_trees` assert `== ()` — they fail on a helper
  that over-reports; `test_clone_breaking_import_edge_is_reported` fails on one that under-reports.
* **Fix-introduced lint:** RUF100 fired on two `noqa: PLC0415` I had added unnecessarily — removed.
  The one remaining ruff finding, RUF022 on `__all__` in `ddm_r7_token_coder.py`, is **pre-existing
  at HEAD** (verified against `git show HEAD:`) and out of scope.
* **My own worst error this arm, caught by re-deriving:** I first read the audit's exit logic as
  "text mode always returns 0" and was about to report the gate as structurally silent. Re-reading
  the CLI showed `--strict` gates the exit code in *both* formats and the gate *does* pass
  `--strict`. Had I not re-derived, this memo would have named the wrong cause and proposed a gate
  that already exists. Also mis-attributed `pj2_pose_scale_joint_solve.py` to `experiments/` from a
  glanced `ls` before checking — it is in `tools/` and tracked.
* **Not verified:** that the 106 non-clone-breaking undispositioned rows are individually safe. I
  measured that they do not break imports; I did not audit their content.

---

## NEXT-IF-RESUMED (every row OWNED)

1. **Drain Gate #10's 117 undispositioned rows, clone-breakers first.** The 11 belong to the
   `witness_dsl` lineage — route to the arm that owns `taskspace_*`/`witness_dsl`, worst-first by
   the new ranking (`taskspace_outer_archive_codec`, 16 importers, is #1). Command:
   `python tools/audit_untracked_source_artifacts.py --strict --disposition-manifest
   .omx/research/untracked_source_dispositions_20260505_codex.json`. Each row exits as **track** (2
   review passes) or **dispositioned with a reason** — no third option.
2. **`ddm_cu1` owns:** re-running the clean-clone proof after this commit lands (the clone is at
   the pre-fix HEAD; the fix is verified against main's working tree and against the simulated
   contest runtime, but the *clone-at-new-HEAD* leg is owed).
3. **`ddm_cu1` owns:** applying the same repo-is-source-of-truth staging to any sibling stager
   still copying from `${EVAL_ROOT}/submissions/*` — not yet enumerated.
4. **Unowned-but-named:** five dotted imports resolve to files absent from disk entirely
   (`tac.scoring`, `tac.video`, `tac.scene_embedding_distiller`,
   `tac.local_acceleration.bench_scorer_ops`, `tac.findings_lagrangian_pp`) — dead imports in
   tracked files, a different class from clone-breakage, deliberately excluded from the new
   ranking. Needs a triage pass.

**Pointer: UNMOVED.** Live best remains `ddm_cx1` S = 0.8264972 at 353,808 B. This arm changed no
archive byte and made no score claim; it changed whether that row survives a fresh checkout.
