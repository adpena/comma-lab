# ddm_gt2 — the DALI selector cure was already live; lineage closure and red control

**Task:** #1142 · **axis:** `[macOS-CPU advisory, scorer-free retained GT-vector analysis]`
· `score_claim=false` · `promotion_eligible=false` · **no Modal dispatch**.

## Answer first

The charter's selector premise was stale on the live checkout. There is one
`top_mass_pairs` producer and one selection call. The call loads
`qs1.GT_POSE`, which commit `809199d24f23b3298f8407574870ede35c0f7874`
already points at the registered DALI table. A complete n600 re-derivation
reproduces the historical PyAV-vs-DALI disagreement — Spearman
**0.1221645060**, top-30 overlap **1/30** — while proving that the **live**
selection equals the retained DALI ranking. This is the charter's explicit
"cure was already consumed" outcome, not a second repoint.

I made that inherited binding fail closed by routing the live read through
`tac.gt_lineage.assert_gt_lineage(..., required=AUTHORITY_LINEAGE)`. I also
declared the 11 real consumers after verifying their bytes, removed one
docstring false positive from the detector, exposed a strict standalone
refusal while leaving the aggregate host warn-only, and executed its red
positive control (`rc=1`). The live undeclared population is now **0/11**.

This arm did **not** move the exact pointer. The current own exact frontier was
already jg5 at `S 0.14839100138338618 @ 180,625 B [contest-CUDA T4, n600]`.

## Selector census and retained rankings

| role | site | live object | result |
|---|---|---|---|
| producer | `experiments/ddm_ps1u_uncapped_pose_solve.py:137` | `top_mass_pairs(n, base_pose, gt_pose)` | one producer found in the searched live Python scope |
| path binder | `experiments/ddm_ps1u_uncapped_pose_solve.py:125` | `qs1.GT_POSE` | content-addressed DALI refusal added |
| consumer | `experiments/ddm_ps1u_uncapped_pose_solve.py:428` | `run(... --top-mass N)` | one consumer; live selection byte-equals the retained DALI top-N set |
| sibling selectors | live `experiments/`, `tools/`, `src/` | pair-ranking functions used by this solve | did not find another top-mass sibling in this bounded scope |

Pinned inputs, all n600×6:

| input | bytes | SHA-256 | lineage/use |
|---|---:|---|---|
| `/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/direct_v6/full_n600_eval/retained/pose_vectors/cp135_base_first6_n600.npy` | 14,528 | `e64e8bd36c1a603da30c15fa581cdaeda409e8939cefe61c3d01d09ac0850386` | selector base |
| `/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/direct_v6/full_n600_eval/retained/pose_vectors/gt_first6_dali_n600.npy` | 14,528 | `8d5cfa83df55b89493ba43b1e5386d792c836c32791666192499a089068e7eff` | `DALI_NVDEC`, live authority target |
| `/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/direct_v6/full_n600_eval/retained/pose_vectors/gt_first6_n600.npy` | 14,528 | `82ed61ce6a11a6612502527fbb6864a22fe6c6099312e637d971214ab660fb27` | `PYAV_YUV420_TO_RGB`, historical comparison only |
| `/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_dali.pt` | 117,980,732 | `a91d98252fe377c51ff7f3380c2fc9d30d84093fc54ee89e5e5f5102e6354994` | producer-declared DALI ruler required by the charter |

The retained result is
`/Volumes/VertigoDataTier/pact/ddm_gt2/retained/DDM_GT2_SELECTOR_RESULT.json`
(5,149 B, SHA-256
`b99dc2300f452bc7026d3daf4df6e73356e6ed910496094fec0e6c598d93681e`).
All nine materialized mass/rank/top-set arrays are retained beside it with
individual bytes and SHA-256 records. An identical second execution reproduced
the result JSON SHA exactly.

**PyAV top 30, ranked:** `156, 270, 282, 425, 355, 263, 235, 343, 386, 365,
294, 237, 445, 592, 471, 350, 510, 275, 208, 6, 527, 410, 536, 71, 221, 540,
344, 429, 185, 357`.

**DALI top 30, ranked:** `88, 87, 73, 316, 89, 70, 448, 64, 63, 91, 66, 57,
67, 75, 365, 548, 488, 58, 566, 74, 65, 454, 68, 476, 141, 207, 447, 85, 61,
94`.

The sole common member is pair **365**. Membership changes at **29/30** slots.
The charter's prior-law prediction about the historical rankings is confirmed;
its prediction that the live selector remained PyAV is falsified.

## Historical selection provenance

The old `ps1u` 60-pair fleet was materially chosen by the PyAV-GT ranking. I did
not re-rank history. A lineage rider now sits next to the result in
`.omx/research/ddm_ps1u_r2_dual_axis_pose_verdict_20260816.md` and next to the
fleet table in `.omx/research/ddm_ps1u_uncapped_pose_solve_20260816.md`.
The original `verdict_scope: INSTANCE` remains unchanged.

## The 11 real consumers

The pre-declaration detector returned 12 rows. One was not a consumer:
`ddm_cpu1` mentions `gt_first6_n600.npy` only in a module docstring explaining
the basename collision. AST docstring ownership now excludes that prose. The
remaining denominator is 11 real load-site literals.

| site | lineage before | lineage after | byte verification | landing |
|---|---|---|---|---|
| `experiments/ddm_dt1_ans_decode_wallclock.py:63` | DALI, undeclared | DALI, declared | symlink resolves to `382d7dfe38b37c0c…` | working tree; serializer blocked by read-only Git objects |
| `experiments/ddm_hp3_hpac_section_and_zip_frame.py:72` | DALI, undeclared | DALI, declared | same registered SHA | same |
| `experiments/ddm_jg1_seg_solve.py:89` | PyAV, undeclared | PyAV, deliberately declared for paired DALI/AV differencing | exact AV ruler `837b5852dc71ded…` | same |
| `experiments/ddm_pk2_pose_carrier_representation.py:45` | DALI, undeclared | DALI, declared | xz expands to registered SHA `382d7dfe38b37c0c…` | same |
| `experiments/ddm_pz2_pose_representation_20260810/run_pose_representation.py:349` | DALI, undeclared | DALI, declared | registered SHA `382d7dfe38b37c0c…` | same |
| `experiments/ddm_rg1b_weight_space_gradient_cosine.py:453` | DALI, undeclared | DALI, declared | registered SHA `382d7dfe38b37c0c…` | same |
| `tools/build_mx2_pose_adapter_caches.py:38` | DALI, undeclared | DALI, declared | xz expands to registered SHA `382d7dfe38b37c0c…` | same |
| `tools/fit_ddm_cl1_hpac_capacity.py:72` | DALI, undeclared | DALI, declared | registered SHA `382d7dfe38b37c0c…` | same |
| `tools/run_ddm_ec2_sparse_event_hpac_conditioning.py:66` | DALI, undeclared | DALI, declared | registered SHA `382d7dfe38b37c0c…` | same |
| `tools/run_ddm_xi1_screw_conditioned_learned_prior.py:70` | DALI, undeclared | DALI, declared | registered SHA `382d7dfe38b37c0c…` | same |
| `tools/run_ddm_xi2_xi_context_full_scale.py:62` | DALI, undeclared | DALI, declared | registered SHA `382d7dfe38b37c0c…` | same |

The ten DALI consumers read the 117,981,301-byte registered cache content SHA
`382d7dfe38b37c0cc5017e5645032faa045af6924db66e0b67549cc96c840195`;
the intake `.xz` is 526,820 B, compressed SHA
`233884c672eff22258376cf9532bb69a52017980000a2615bbd917ba7a8ec3dc`,
and expands to those exact DALI bytes. No declaration changes the object read.

## Two-landing gate and executed red control

The existing Catalog #351 extension remains **WARN-ONLY in the aggregate host**.
This landing adds the strict standalone surface
`check_gt_lineage_objective_custody(..., strict=True)`. That split preserves the
two-landing contract: normal preflight reports the population during adoption,
while an explicit control/new-consumer check refuses.

Positive-control fixture:
`/Volumes/VertigoDataTier/pact/ddm_gt2/positive_control_repo/tools/undeclared_gt_consumer.py`
(338 B, SHA-256
`b931076347afdafe84e48ef5fd5b55a7dcaa88af5406363f45b2231070e1662f`).
Receipt:
`/Volumes/VertigoDataTier/pact/ddm_gt2/retained/DDM_GT2_POSITIVE_CONTROL.json`
(1,010 B, SHA-256
`24fc6bd61f0571763c3654fbb012b22b4fef82c6f845596dcc1f0258980ab04c`).

Executed command:

```bash
.venv/bin/python -c 'from pathlib import Path; from tac.preflight import check_gt_lineage_objective_custody; check_gt_lineage_objective_custody(Path("/Volumes/VertigoDataTier/pact/ddm_gt2/positive_control_repo"), strict=True)'
```

Executed red output:

```text
tac.preflight.PreflightError: check_gt_lineage_objective_custody refused 1 undeclared GT consumption(s):
  tools/undeclared_gt_consumer.py:7: GT artifact ['gt_first6_n600.npy'] consumed with UNDECLARED decode lineage.
rc=1
```

The live repo check after declarations returns **0 findings**. Tests pin all
three branches: host strict still warns/returns, standalone strict refuses, and
docstring prose is not a consumption.

## $0 reruns

| row | action in this arm | measured result | disposition |
|---|---|---|---|
| `qs3` unlock | re-executed the governed resume command against the existing retained store | `rc=0`; GT SHA `91d3ff11…` matched; identity-validated attribution `B=108, H=76, W=5`, `B-H=32`; complete 200-row screen still admits 0 toy-bracket rows | **FOLDED** into QS4, exactly as the refreshed result records |
| `sq2 R8` carrier re-solve | full-corpus recall found the requested run already completed by SL2; not overwritten or repeated | 32/32 stratified pairs, persisted edited frame bytes; `d_seg 0.0043002764 → 0.0010015170`; terminal carrier `d_pose 0.0580729881 → 0.0082654368`; 479-B terminal packet; advisory/non-promotable | **FOLDED** into `.omx/research/ddm_sl2_20260805/SL2_RECEIPT.md` |

QS3 command executed here:

```bash
.venv/bin/python experiments/ddm_qs3_saturation_compose.py --output /Volumes/VertigoDataTier/pact/ddm_qs3_20260813 --resume-from /Volumes/VertigoDataTier/pact/ddm_qs3_20260813
```

Its refreshed `FINAL_RESULT.json` is 22,245 B, SHA-256
`cedd8bdb9eaee726c51e2ade17d638b54a2ee61065e172d5f754fe720136562c`;
all materialized candidate/ranking payloads remain retained. No scorer forward
or Modal job ran.

The exact SL2 reproduction command already recorded by its receipt is:

```bash
.venv/bin/python experiments/ddm_sl2_sq2_persist_and_compose.py --out-dir /Volumes/VertigoDataTier/pact/ddm_sl2_20260805 --seg-resource-step-bound 150 --seg-eval-every 5 --seg-convergence-patience-evals 3 --pose-relinearizations 2
```

That four-hour object is already complete and SHA-custodied; rerunning it would
not answer a new question.

## ps135b / ps1u and the T4 fire order

The historical ps135b/ps1u rows remain wrong-object **instances** and are now
explicitly labeled. The resolving measurement was subsequently executed at
stronger scope on the live carrier vehicle by UP2/UP3: all 600 independent
12-DOF carrier solves targeted DALI, converged with 429 improved and 0 worsened,
and moved `d_pose 7.769484e-06 → 7.649247e-06`. UP3 byte-closed it at 176,420 B
and MAIN's T4 row measured
`S 0.15652626435208142 [contest-CUDA T4, n600]`. The later jg5 joint solve then
moved the exact pointer below 0.15.

Therefore the requested new `$0.16` ps1u T4 leg is **FOLDED**, not queued: its
DALI-targeted successor already ran at n600 and on T4. Re-running the older hv1
instance would duplicate the resolved mechanism on a superseded body and could
not move the live jg5 pointer. This arm fired no Modal dispatch and did not touch
the live rr8 call/store.

For custody, the exact sealed MAIN command that consumed this successor row is
preserved in `/Volumes/APDataStore/pact/ddm_up3/t4_row_r1/FIRE_MANIFEST.json`;
its call `fc-01M0CG6G4WTDFJAS0GDF8C62JK` is already harvested. It is historical
evidence, **not a command to fire again**.

## RECALL EVIDENCE

Sources and queries searched before adjudication:

- Full research corpus: `rg -n "qs3|sq2|R8|top[-_ ]mass|ps135b|ps1u" .omx/research --glob '*.md'`, then full reads of `ddm_dg1`, `ddm_na10`, the two ps1u memos, QS3, SL2, UP2, and UP3 receipts.
- Code producer/consumer census: `rg -n "def top_mass_pairs|top_mass_pairs\(|--top-mass|GT_POSE" experiments tools src --glob '*.py'`.
- Canonical equations: `.venv/bin/python tools/list_canonical_equations.py --json`, filtered for `DALI_NVDEC`, `PYAV_YUV420_TO_RGB`, `gt_lineage`, and `1.4061e-04`; found `cw1_gt_lineage_additive_pose_offset_v1`.
- Research index and DAG: searches for `top_mass_pairs|DALI|PyAV|GT lineage|ps1u|1142` in `.omx/research/CANONICAL_RESEARCH_INDEX*` and `sub015_DAG_*`; read FEED-up1/up2/up3/jg5/jg5t4.
- Design/SPEC and task surfaces: same content query over `SPEC*.md`, `.omx/state/canonical_task_status.jsonl`, `.omx/state/operator_p0_ledger.jsonl`, and the live `main_hot_state.md`.
- Artifact identity: direct size/SHA checks of both pose tables, both #906 ruler caches, all `382d7dfe…` consumer targets, the symlink target, and the decompressed `.xz` content.

Findings beyond the charter's seeds changed the plan materially: (1) live
`top_mass_pairs` already inherited DALI, so the code action became a fail-closed
content guard plus proof; (2) `ddm_cpu1` was documentation, not a twelfth read;
(3) SL2 had already executed the requested SQ2 carrier re-solve; (4) UP2/UP3 had
already executed and T4-measured the stronger all-pair DALI carrier successor;
and (5) the live own exact frontier is now jg5 sub-0.15, superseding the common
contract's stale frontier paragraph.

## Verification and landing boundary

- `py_compile` passed for every edited Python file.
- `ruff` passed on the new selector/guard surfaces; the all-file invocation
  surfaced two pre-existing findings outside these hunks (`B905` in the old
  test helper and import ordering in `build_mx2_pose_adapter_caches.py`).
- `85 passed` across the complete sp2 protection file plus GT-lineage suite.
- Retained-output audit reloaded all nine arrays, rehashed every record, and
  reproduced the one-pair overlap.
- Two genuine review-tracker passes marked every edited Python file.
- `git diff --check` passed for this arm's paths.

The mandated serializer was invoked with per-file post-edit SHA pins,
`[no-triality] [p0-ledger-ok]`, and no co-author trailer. It failed before
staging because this managed checkout cannot write Git objects:

```text
error: unable to create temporary file: Operation not permitted
error: .omx/research/ddm_ps1u_r2_dual_axis_pose_verdict_20260816.md: failed to insert into database
fatal: updating files failed
```

The staged index remains empty. The implementation and receipt are present in
the working tree; no commit is claimed on the shared checkout.

A portable patch containing only this arm's 16 tracked edits and two new files
is retained at
`/Volumes/VertigoDataTier/pact/ddm_gt2/retained/DDM_GT2_WORKTREE.patch`.
It deliberately excludes every concurrent unrelated worktree edit; a reverse
apply check against the current working tree passed. Its byte count and SHA-256
are recorded in the adjacent `DDM_GT2_WORKTREE_PATCH_MANIFEST.json`.

## Boundaries

No `upstream/` or protected runtime file changed. No Modal dispatch, full-n600
scorer job, contest-CPU row, contest-CUDA row, archive promotion, or rr8 access
occurred in this arm. Selector/ranking results are scorer-free advisory facts,
not scores. The SQ2 result remains n32 advisory. Negative claims above are
`verdict_scope: INSTANCE` for the historical ps1u/hv1 candidate and
`verdict_scope: FORMULATION` only where the cited QS3/SL2 receipts say so.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — disposition: land the already-verified ddm_gt2 source and receipt only; owner: MAIN/operator with a Git-writable checkout; consumer store: this working tree plus `/Volumes/VertigoDataTier/pact/ddm_gt2/retained/`; fire trigger: the Git object/index store becomes writable, then rerun the SHA-pinned serializer without changing experimental state.

Own-vehicle frontier: **S 0.14839100138338618 @ 180,625 B [contest-CUDA T4, n600], UNMOVED by ddm_gt2.**
