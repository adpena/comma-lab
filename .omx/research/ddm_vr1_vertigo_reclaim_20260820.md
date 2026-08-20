# ddm_vr1 Vertigo certify-and-move space reclaim (second pass)

**Disposition:** `MOVE_COMPLETE_VERIFIED__TARGET_MET`
**Axis:** `[byte/custody apparatus, scorer-free]`
**Frontier effect:** none. No scorer, archive mutation, Modal dispatch, or exact evaluation ran.

## Result

| Measure | Before | After |
|---|---:|---:|
| Vertigo available | **0.87 GiB** | **76.05 GiB** |
| Vertigo used | 1861.67 GiB | 1786.49 GiB |
| APDataStore available | 129.02 GiB | **50.02 GiB** |
| APDataStore used | 1737.57 GiB | 1812.94 GiB |

**75.18 GiB freed** (78,836,860 KiB), against a ≥50 GiB target and a ≥25 GiB APDataStore floor.
Nothing was deleted except two source trees whose destination copies were hash-verified equal first.

| # | Original | Destination | Data files | Logical bytes | Source+dest manifest SHA-256 | Freed |
|---:|---|---|---:|---:|---|---:|
| 1 | `pact/ddm_pfs1_20260729` | `vertigo_coldstore/pact/ddm_pfs1_20260729` | 307 | 51,420,282,443 | `b70760cd1c486cb0e2af03edc41c511788b5e1d1437b0afa34228301f8136ce2` | 47.89 GiB |
| 2 | `pact/experiments/results/dqs1_local_first` | `vertigo_coldstore/pact/experiments/results/dqs1_local_first` | 238 | 29,307,503,277 | `806dafb29e66905b7ff722431caae0b4d40107d6f7d3de0995091ed8ae1149aa` | 27.30 GiB |

Both manifest digests are the value computed **twice**: once from the source stream during the copy,
once from an independent re-read of the destination. Both original paths are now absolute symlinks
and both resolve. An independent spot-check read `d1/eval_root/submissions/pfs1/archive.zip` through
the symlink and got `624ffe57000c6fe4a6802a6d8b9a5d6002617f29b0bbb9e186d1273fa996600c`, matching its
manifest row exactly — so the 728 citations of pfs1 still resolve to byte-identical content.

Ledger: `.omx/research/ddm_vr1_move_cert_ledger.jsonl`. Per-tree manifests, both `source.sha256` and
`destination.sha256`, live under `/Volumes/APDataStore/pact/vertigo_coldstore/_manifests/`.

## Why this pass exists

`ddm_sr2` (2026-08-11) freed 218.667 GiB and left a standing fire order: *"fire trigger: Vertigo
available space falls below 150 GiB."* Nine days later Vertigo was at **890 MiB free of 1.8 TiB
(100% full)**. That is a live ALWAYS-KEEP-THE-PAYLOAD hazard, not a tidiness problem: a retained
payload write to that volume fails outright, and the non-negotiable forbids running anything that
cannot persist its payload.

The 242 GiB that sr2 freed was consumed almost entirely by one tree that postdates its census:
`pact/ddm_pk4_20260813` at **164.89 GiB**.

## Method

Certify-or-block, never delete. For each selected tree:

1. Census the source excluding AppleDouble `._*` and `.DS_Store` sidecars (ExFAT destination
   materialises those; the ai1/sr2 precedent established data-fork-only comparison as the valid
   cross-filesystem test).
2. Full per-file SHA-256 source manifest; the manifest file itself is digested.
3. `rsync -a` to `/Volumes/APDataStore/pact/vertigo_coldstore/<original-relative-path>`.
4. Destination path-set equality, then full per-file SHA-256 destination manifest.
5. Manifest digests must be equal. Unequal, missing, or extra paths = BLOCK, source untouched.
6. Cert row appended and `fsync`ed to the JSONL ledger **before** any source byte is retired.
7. Source renamed aside, absolute symlink installed at the original path, symlink probed by
   resolving the largest known file through it, and only then is the renamed original removed.

The single permitted deletion class is step 7: a source whose destination copy is hash-verified
equal, whose cert row is already durable, and whose original path still resolves through a symlink.

Tool: `tools/vertigo_certify_move.py`. Ledger: `.omx/research/ddm_vr1_move_cert_ledger.jsonl`.
Ledger rows are appended at every phase transition (`PLAN` → `SOURCE_MANIFEST` → `COPIED` →
`VERIFIED` → `MOVED_SYMLINKED`), so the ledger is truthful at any interruption point.

## What was selected, and why

A volume-wide `find -prune` for `inflated` / `inflated_frames` / `eval_root` / `extracted`
directories returned **257 trees totalling 379.66 GiB** with **zero nesting overlap**. That is the
certified-rebuildable class: deterministic decode output whose generator sits beside it.

The pfs1 rebuildability proof is exact: **14 of 14** submission directories carry `archive.zip`,
`inflate.sh`, and a runtime `.py` set, and each `inflated/` holds a single `0.raw` of 3.4 GB.
Deterministic decode is a contest non-negotiable, so those raws are reproducible from bytes that
travel with them. The dqs1 tree carries the project's own prior classification in-tree:
`raw_artifact_retention_plan.json` (`schema: comma_lab.artifact_retention_plan.v1`, 2026-05-23)
lists kinds `locality_inflated_raw` and `local_cpu_advisory_inflated_raw` with
`promotion_eligible: false`, `score_claim: false`, `ready_for_exact_eval_dispatch: false`.

Quiescence and custody, measured before any byte moved:

| Tree | Allocated | Files newer than 14 d | Symlinks | Active lane claims | In hot state / current focus | Corpus refs |
|---|---:|---:|---:|---:|---|---:|
| `pact/ddm_pfs1_20260729` | 47.89 GiB | 0 | 0 | 0 | no | 728 |
| `pact/experiments/results/dqs1_local_first` | 27.30 GiB | 0 | 0 | 0 | no | many |

Zero symlinks in both means the SHA-256 manifest equality is a **complete** proof for these two
moves — `rsync -a` copies symlinks, but symlinks are excluded from the data-fork manifest, so a
nonzero count would have left a gap. It did not arise here; it is recorded as a hardening item
below. High corpus reference counts are exactly why the original path is preserved as a symlink
rather than removed.

## Deliberately left in place

| Tree | Allocated | Reason |
|---|---:|---|
| `pact/ddm_pk4_20260813` | 164.89 GiB | Retained scientific payload (`cpu_authority_run/retained` 108.15 GiB + `retained/jacobian_bank` 56.74 GiB, 64 `pair_*` banks). Seven days old, inside the 14-day live boundary. ALWAYS-KEEP-THE-PAYLOAD custody, not rebuildable decode output. |
| `pact/evidence` | 328.47 GiB | Broad live/historical evidence namespace; sr2 left it UNKNOWN and no whole-tree quiescence proof was completed. Holds 78.72 GiB of inflated bulk across 31 trees — partitionable next pass. |
| `pact/experiments` | 274.56 GiB | Same; holds 68.22 GiB of inflated bulk across 28 trees beyond the dqs1 subtree taken here. |
| `mac-lab/images` | 123.78 GiB | Non-pact sibling root. Ownership and consumers outside this charter's proof surface. |
| `pact/pr135_joint_solve_20260810` | 31.46 GiB | Explicitly protected candidate custody. Untouched. |
| `pact/ddm_cb1_perclass_carrier_byteclose_20260725T203310Z` | 30.71 GiB | Certified quiescent (0 files newer than 14 d, 0 active lane claims) but moving it would drop APDataStore below the 25 GiB floor. Fire-ordered below. |
| `pact/hprc_projection_gap_repairs` | 23.76 GiB | Old and quiescent by mtime, but carries **1 active lane claim reference**. Certify-or-block: reference present, so not moved. |
| `pact/ddm_dx1/retained` | 184 KiB | Protected by charter. Also negligible. |
| `pact/ddm_jg5_custody` | 1.3 MiB | Current sub-0.15 frontier custody. Untouched, and negligible. |

Destination budget governs the cut: APDataStore held **129.02 GiB** free, and the tool refuses any
move projected to leave it below **25 GiB**. That caps this pass at roughly 104 GiB and is why
`ddm_cb1` is deferred rather than taken.

## Operating conditions, and one design correction made mid-pass

The first implementation used the sr2-style ordering: hash the source, then `rsync`, then hash the
destination. That reads the source **twice**. After 85 minutes it had hashed roughly 41% of one
tree, so I measured the volume directly instead of assuming:

| Measurement | Result |
|---|---:|
| Vertigo single-stream uncached read (`dd`, 320 MiB) | 6.25 MiB/s |
| Vertigo 6 parallel streams, aggregate | 15.7 MiB/s |
| Single-pass copy+hash at 8 workers, sustained | **32.5 MiB/s** |

Both volumes are USB Samsung T7 Shield SSDs, so 6 MiB/s is contention and 100%-full fragmentation,
not device capability. Two other live arms were competing: `ddm_cpu1_gt_lineage_attribution.py` at
393% CPU and a `ddm_cd1` `inflate.py` writing into APDataStore.

At those speeds the second source read was worth roughly four hours, so I aborted the run before any
byte was retired, recorded an `ABORTED_RESTART` ledger row stating the source was intact and the
destination empty, and rewrote the copy to **hash the source during the copy**. This does not weaken
the proof: the destination is still re-read from disk and hashed independently, so equality is still
established between two separately-read byte streams. Measured effect: 6.25 → 32.5 MiB/s, about 5×.

AP free space was sampled repeatedly and drifted only 129.02 → 125.04 GiB from the concurrent arm,
so the destination headroom held. No active writer touched Vertigo in the preceding two hours, and
no `No space left on device` failure appears in any log from the last day — the hazard was latent,
not yet biting.

## RECALL EVIDENCE

**Stores searched:** `CLAUDE.md`, `MEMORY.md`, `.omx/research/ddm_sr2_vertigo_space_reclaim_20260811.md`,
`.omx/research/ddm_oc2_origin_consolidation_20260820/`, `.omx/state/` (lane registry, active lane
dispatch claims, current focus, durable daemons), the APDataStore custody mirror and its
`_manifests/` tree, and the repo-wide corpus for each candidate name.

**Beyond the charter seeds:** sr2's own fire order named the trigger and the method (partition
`evidence` and `experiments` by newest descendant and current consumer). sr2's four DEAD-ENDS were
adopted directly rather than re-derived: AppleDouble sidecars inflate raw destination counts, root
mtime alone is not whole-tree coldness, an AP mirror subtree is not proof of a mirrored census
directory, and rsync completion is not deletion authority. The oc2 consolidation (144 files,
83.8 MB) is too small to matter here and was not re-walked.

## Boundaries and score status

- No scorer, Modal dispatch, training, archive mutation, or exact evaluation ran. Pointer unmoved.
- No file under `ddm_dx1/retained`, `ddm_jg5_custody`, `ddm_ps135_20260810`,
  `pr135_joint_solve_20260810`, `upstream/`, or any tree with an active lane claim was changed.
- No source byte was retired before a destination copy, a full destination manifest, digest
  equality, a durable cert row, and a resolving symlink probe.
- Census figures are **allocated KiB** from `du -x -k`; ledger `logical_data_bytes` is data-fork
  logical bytes. They are different denominators and are not conflated.
- The `--min-dest-avail-gib` gate compares HFS+ allocated KiB against ExFAT free space; cluster-size
  differences make it approximate, and it is deliberately conservative.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN storage custody. Fire trigger: APDataStore regains
  ≥ 60 GiB free, or a third destination tier is attached. Action: move
  `pact/ddm_cb1_perclass_carrier_byteclose_20260725T203310Z` (30.71 GiB, already certified quiescent:
  0 descendants newer than 14 d, 0 active lane claims) with the same tool and ledger.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN storage custody. Fire trigger: Vertigo falls below
  50 GiB free again. Action: partition `pact/evidence` (78.72 GiB of inflated bulk in 31 trees) and
  `pact/experiments` (68.22 GiB in 28 trees) by newest descendant and current consumer, then certify
  and move whole subtrees. The census is committed at `.omx/research/ddm_vr1_census_20260820/`
  (`inflated_dirs.txt` — the 257 `find`-derived trees; `inflated_sizes.tsv` — allocated KiB each;
  `census_ge5gib.tsv` — the ≥5 GiB rows; `vertigo_du_d2.txt` — the full depth-2 walk) and mirrored to
  `/Volumes/APDataStore/pact/vertigo_coldstore/_manifests/_vr1_census_20260820/`. It is committed
  rather than left in `.omx/tmp/`, which is ephemeral scratch and not a durable evidence path.
- **HARDENING OWED** — `tools/vertigo_certify_move.py` (landed at `ef46ed13a4`): (a) write and
  compare a `symlinks.tsv` of link targets so a tree containing symlinks still gets a complete
  equality proof — both trees here had zero symlinks, so the gap did not bite, but the next tree may
  not be so clean; (b) record a ledger row when `shutil.rmtree` of the retired original partially
  fails, so a `.RETIRING` remnant is never silent. The third item — removing the wasted second
  source read — was **fixed during this pass**, not deferred: the copy now carries the source hash.

## LIVE-HYPOTHESES

- The 379.66 GiB inflated/eval_root census is a lower bound on the rebuildable class. Directories
  named for their candidate rather than for `inflated` were not matched by the `find` predicate.
- `pact/ddm_pk4_20260813`'s 56.74 GiB `jacobian_bank` becomes a legitimate cold-move candidate once
  it crosses the 14-day boundary on 2026-08-27, provided its rung consumers are quiescent by then.

## DEAD-ENDS

- Deleting inflated raws instead of moving them is closed by charter and by ALWAYS-KEEP-THE-PAYLOAD.
  Rebuildability justifies a **move**, never a discard.
- Moving `pact/hprc_projection_gap_repairs` on mtime evidence alone is closed: it carries an active
  lane claim reference, and certify-or-block treats a live reference as a block.
- Treating `du` allocated KiB and manifest logical bytes as the same number is closed; they differ
  and both are reported separately.
