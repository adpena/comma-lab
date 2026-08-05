# DDM EK2 Worktree Harvest Receipt

Generated UTC: 2026-08-05T16:04:30Z

Assigned branch: `codexwt/ddm_ek2_worktree_harvest`
Source worktree head before harvest: `6a78ee8209e4a335dde64eb512ddcd361f6d33b6`
Current Git state: branch creation blocked by managed-sandbox ref write permission; no serializer commit was made.

## Disposition Table

| Path | Initial state | Class | Action | Evidence |
|---|---:|---|---|---|
| `.omx/research/einstein_kolmogorov_crux_20260719T214408Z_codex.md` | modified | CODE-signal | keep for commit | EK crux memo; research-only, no score promotion authority. |
| `.omx/research/einstein_kolmogorov_crux_DAG_FEED_20260719.md` | modified | CODE-signal | keep for commit | DAG/equation feed for the scoped EK action-rate contract. |
| `.omx/research/einstein_kolmogorov_review_bugclass_scope_extensions_20260720_codex.md` | modified | CODE-signal | keep for commit | Review bug-class scope-extension memo and live guard counts. |
| `CLAUDE.md` | modified | CODE-signal | keep for commit | Governing Catalog #351 recursive-review amendment; no score claim. |
| `docs/meta_bug_class_catalog.md` | modified | CODE-signal | keep for commit | Consolidated Catalog #154/#344/#351 replacement semantics and mutation seal. |
| `src/tac/canonical_equations/einstein_kolmogorov_crux_20260719.py` | modified | CODE-signal | keep for commit | Research-only canonical equation leg with producer custody hardening. |
| `src/tac/canonical_equations/registry.py` | modified | CODE-signal | keep for commit | Anchor roundtrip strictness, physical ledger parsing, exact type drift reporting. |
| `src/tac/preflight.py` | modified | CODE-signal | keep for commit | Preflight scope-extension guards for cleanup identity, anchor roundtrip, and canonical producer identity. |
| `src/tac/tests/test_check_154_manifestless_cleanup_identity_scope_extension.py` | modified | CODE-signal | keep for commit | Mutation coverage for Catalog #154 scope extension. |
| `src/tac/tests/test_check_344_anchor_roundtrip_scope_extension.py` | modified | CODE-signal | keep for commit | Mutation coverage for Catalog #344 scope extension. |
| `src/tac/tests/test_check_351_canonical_producer_identity_scope_extension.py` | modified | CODE-signal | keep for commit | Mutation coverage for Catalog #351 scope extension; repaired during EK2. |
| `uv.lock` | modified | CODE-signal | keep for commit | Lockfile residue already present in the EK worktree; not bulk, not scratch. |
| `.omx/research/einstein_kolmogorov_xi_bridge_blocker_20260719.json` | untracked | CODE-signal | keep for commit | Small machine-readable execution-surface blocker receipt, SHA-256 `cb59f52aface149c94dc59bbfa05bed640dff9b9c2d099efbd9794792d6403c8`. |
| `.omx/research/einstein_kolmogorov_xi_bridge_full_preflight_blocker_20260719.json` | untracked | CODE-signal | keep for commit | Small governed-full preflight blocker receipt, SHA-256 `0a851a72ad6788cfc1d76fefade2e2db5428bcbfcde5e2f349df28cd95cfab96`. |
| `.omx/research/einstein_kolmogorov_xi_bridge_hash_receipt_20260719.json` | untracked | CODE-signal | keep for commit | Small input-custody hash receipt, SHA-256 `855ef64cab70865f97744bd1105c068ed3c84c281c194be650226dac5848ac8f`. |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719/` | untracked directory | rebuildable-bulk | certified and moved to SSD cold store | 1,476 files, 18,763,542 bytes, content tree SHA-256 `cc7f176d9163200d2a9b17e0dfb221150681a400552b5d4a2de3f53f7acd7778`. |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719_pre_review_invalid/` | untracked directory | rebuildable-bulk | certified and moved to SSD cold store | 81 files, 768,853 bytes, content tree SHA-256 `9649f5c249df985bc9695e287c21367715e8d126846082cbefeb59343d5ccb8a`. |

No stale-scratch file was deleted. The two run-output directories were moved, not destroyed, after machine-readable certification.

## Canonical Equation Verification

The EK equation module imports and builds:

```text
equation_id = einstein_kolmogorov_crux_action_rate_contract_v1
empirical_anchors = 2
canonical_producers = (
  .omx/research/einstein_kolmogorov_crux_measurement_20260719.json,
  .omx/research/einstein_kolmogorov_frontier_magnitude_chart_20260720.json
)
research_only = True
promotes_pointer = False
score_claim = False
```

Producer SHA-256 checks matched the constants in the module:

| Path | SHA-256 |
|---|---|
| `.omx/research/einstein_kolmogorov_crux_measurement_20260719.json` | `0b2e02e39601f863d07465bca66e006f7bad503b64c9ab3f901b44bed9637451` |
| `.omx/research/einstein_kolmogorov_frontier_magnitude_chart_20260720.json` | `1c5926d8e899b32a0ef46c13cfd32f0d6f1f9585cc7435cf52a7605720927ae6` |

Additional source hashes verified during recall:

| Path | SHA-256 |
|---|---|
| `.omx/research/einstein_kolmogorov_banked_n12_ab_20260720_v3.json` | `9c5d636a76a9ef77bb29dec64e4221b098e449510f5f04c2f7218da885c63f0a` |
| `.omx/research/constructive_solver_541_rung_e_n48_20260719.json` | `d966e066bfd24deb0f7ad1fda865337ed2f108c03c93244c24dd592ac69682a9` |
| `.omx/research/seg_secant_rd_curve_n24_20260719_v2.json` | `28940965904e9238668de6350785ef0e12348275b64fab83b22901726b0d1f85` |
| `.omx/research/joint_seg_pose_inverse_solve_receipt_n24_20260719.json` | `7a6fdbdfb8f6084a6fd79bb0a63490335b22ae308774032fff7471bb4281e3e9` |
| `/Volumes/VertigoDataTier/pact/evidence/v10_uint8_lattice_n600_20260719/aggregate_n600_receipt.json` | `1509431e2f06963a0d711819d6fcee131fe44599d997ef137dbf4b352f2f2e60` |
| `/Volumes/VertigoDataTier/pact/evidence/c1_two_plane_receiver_20260719/capstone_submission/archive.zip` | `e4cd154f79a30e2b1d759af0d26e54444d22807f81700565e475392eae064f42` |
| `/Volumes/VertigoDataTier/pact/evidence/c1_two_plane_receiver_20260719/modal_contest_cpu/harvest_fc01KXXRAR/contest_auth_eval.json` | `4ef77cc58c4232fc0bfa76f02c74ceb6c258e064707522a4b510e9fe06495e99` |
| `/Volumes/VertigoDataTier/pact/evidence/c1_two_plane_receiver_20260719/modal_contest_cpu/harvest_fc01KXXRAR/modal_cpu_auth_eval_validation.json` | `7230903c6fdad0474ccc4470160fa9dda36ddcd06af3ba5a8c454cfe4d414196` |

## Cold-Store Certification

Cold-store root:

```text
/Volumes/VertigoDataTier/pact/cold_store/ddm_ek2_worktree_harvest_20260810
```

| Directory | Files | Bytes | du KiB | Relative path-list SHA-256 | Content tree SHA-256 |
|---|---:|---:|---:|---|---|
| `einstein_kolmogorov_crux_runs_20260719` | 1,476 | 18,763,542 | 21,056 | `8a086f97d9fb40497c79d00e25dec448a533805e50a8c6514f6e18aa1df39a08` | `cc7f176d9163200d2a9b17e0dfb221150681a400552b5d4a2de3f53f7acd7778` |
| `einstein_kolmogorov_crux_runs_20260719_pre_review_invalid` | 81 | 768,853 | 904 | `19a4ec7b587f9fa273b34fb4ad9f7f7d4b19e27655db918a228aeba657a99551` | `9649f5c249df985bc9695e287c21367715e8d126846082cbefeb59343d5ccb8a` |

The same counts and hashes were measured before and after the move.

## Review And Tests

Two review-tracker passes were marked for each modified Python file:

```text
reviewer = codex-ek2, pass = ek2-review-pass-1
reviewer = codex-ek2-adversarial, pass = ek2-review-pass-2
```

Modified Python files covered:

```text
src/tac/canonical_equations/einstein_kolmogorov_crux_20260719.py
src/tac/canonical_equations/registry.py
src/tac/preflight.py
src/tac/tests/test_check_154_manifestless_cleanup_identity_scope_extension.py
src/tac/tests/test_check_344_anchor_roundtrip_scope_extension.py
src/tac/tests/test_check_351_canonical_producer_identity_scope_extension.py
```

Targeted test suite:

```text
PYTHONPATH=src .venv/bin/python -m pytest \
  src/tac/tests/test_check_154_manifestless_cleanup_identity_scope_extension.py \
  src/tac/tests/test_check_344_anchor_roundtrip_scope_extension.py \
  src/tac/tests/test_check_351_canonical_producer_identity_scope_extension.py

121 passed in 3.89s
```

Direct preflight guard checks after the fix:

```text
check_154 = 0
check_344 = 0
check_351 = 0
```

## RECALL EVIDENCE

Sources read before acting:

```text
PROGRAM.md
CLAUDE.md
AGENTS.md
docs/operating_manual_craft_handoff.md
.omx/state/main_hot_state.md
EK2 charter
common contract
```

Searches and commands used:

```text
git worktree list --porcelain
git status --porcelain=v1
rg -n "einstein|kolmogorov|Kolmogorov|#604|U2|canonical producer|anchor roundtrip|manifestless cleanup|canonical_producer"
tools/list_canonical_equations.py --json
```

Findings beyond the seed charter:

```text
The EK worktree was registered at head 6a78ee8209e4a335dde64eb512ddcd361f6d33b6.
The live equation registry already contains the EK research-only equation rows, including a later corrected frontier SHA row.
The canonical equation leg imports and builds but is explicitly research_only, promotes_pointer false, and score_claim false.
The untracked xi_bridge JSON files are small execution/custody receipts, not bulk.
The two untracked run directories are rebuildable run-output residue and were moved to SSD cold store.
The pre-existing #351 test residue initially failed; the guard and tests were repaired and the focused suite is now green.
```

What changed during EK2:

```text
Repaired Catalog #351 preflight discovery and diagnostics so unrelated advisory provenance modules are not falsely captured.
Preserved producer argument order for guard diagnostics.
Made final canonical producer symlink refusal final-specific while preserving ancestor-component symlink refusal.
Updated the EK canonical producer helper and #351 tests to match the final-path refusal semantics.
Cold-stored certified run-output directories under the SSD tier.
Added this receipt package.
```

## Score And Launch Boundary

No scorer launch, exact replay, Modal dispatch, CUDA dispatch, or paid job was run by EK2.

Own-vehicle frontier statement required by the common contract:

```text
S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]
```

Contest pointer state:

```text
borrowed/external pointer remains unmoved by EK2
no new archive.zip exact score
no contest-CPU or contest-CUDA promotion
```

## Branch And Commit Blocker

Required branch creation failed before any serializer attempt:

```text
git switch -c codexwt/ddm_ek2_worktree_harvest
fatal: cannot lock ref 'refs/heads/codexwt/ddm_ek2_worktree_harvest': Unable to create '.git/refs/heads/codexwt/ddm_ek2_worktree_harvest.lock': Operation not permitted
```

Because the assigned branch could not be created, no serializer commit was attempted on the predecessor branch. This is a Git ref-write blocker, not a code/test blocker.

## MAIN merge-debt adjudication (2026-08-05, boundary-window landing review)

The owed ek2 branch→main merge is CLOSED AS SUPERSEDED-BY-MAIN, not merged:
- main commit 57e4c4e52b ("einstein-kolmogorov: harden canonical producer identity
  (#876)") landed the SAME crux module + #154/#344/#351 hardening + all 3 scope-
  extension test files independently (parallel session), evolving past the branch
  base 6a78ee8209. `git merge-tree` confirms the only content conflicts are two
  co-descendant evolutions of identical work.
- The branch's sole unique CODE delta is a 6-line error-message taxonomy refinement
  in `_canonical_producer_reference` (leaf-symlink "must not be a symlink" vs main's
  "has a symlinked component") — semantically equivalent refusal both sides; main's
  reviewed variant KEPT (no strict-custody module edit for a message nicety).
- HAZARD RECORDED: `git merge-tree --write-tree` emitted a 2,495-byte preflight.py
  blob (vs 4,012,838 B real) for this branch pair — a whole-branch merge attempted
  naively could have clobbered preflight.py (m20 clobber class). Do NOT merge this
  branch; it is retained read-only for provenance.
- Harvested ADDITIVELY in this landing: this disposition dir (4 files) + the 3
  xi_bridge blocker/receipt JSONs. The 3 einstein_kolmogorov research memos were
  already on main (byte-identical lineage).
