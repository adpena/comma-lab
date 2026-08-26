---
arm: ddm_w96b_aligned_loss_implementation
date: 2026-08-26
axis: "[build + APDataStore source-backed exact SHA-256 inventory; no scorer, Metal, training, Modal, n600, or contest evaluation]"
score_claim: false
promotion_eligible: false
verdict_scope: "IMPLEMENTATION GREEN; STORAGE BLOCKED; NO ALIGNED-CONFIG VERDICT"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_w96b — the exact aligned loss is built; real dedup does not clear the two-seed storage gate

**Disposition: IMPLEMENTATION GREEN / BLOCKED_STORAGE / QUEUED-WITH-A-FIRE-ORDER.** The exact CE1
target-vs-best-other expected-flip law is now a reviewed, selectable WD3/S1A loss. The legacy WD3 law
remains the default and preserves its original operation sequence when the treatment is absent. The
aligned branch binds the full 65-epoch tau schedule, the one-percent cosine floor, pose at step zero,
strict resume identity, DSL selection, and lossless content-addressed evaluation retention.

The storage prediction was falsified by the real retained bytes. Across the matched two-seed
65-epoch evaluation schedule, exact content-addressed dedup reduces evaluation allocation from
27,525,120,000 B to 15,572,930,560 B. Adding the measured 4,703,256,576 B non-evaluation allocation
per seed gives **24,979,443,712 B**, which is **2,660,372,480 B above** w96a's recorded
22,319,071,232 B available denominator. With the binding 8 GiB reserve, the required free space is
**33,569,378,304 B**, a shortfall of **11,250,307,072 B** against that denominator. The only live
blocker is therefore storage. No aligned checkpoint or score row exists.

## Measured storage result

Axis: **[APDataStore source-backed exact SHA-256 byte inventory; no scorer]**. The inventory hashed
each of the 35 retained OFF evaluation trees once in place. It did not copy, move, compact, delete,
or symlink any OFF payload. The 35-tree cohort includes seed 20260816's crash-resume-extra epoch 31
screen and its post-65 continuation through epoch 95. The two-seed demand cohort uses the matched
sealed schedule from seed 20260815 — epochs 1, 5, 10, ..., 65 — for 14 rows per seed and 28 trees.

| cohort / term | pre-dedup allocated B | post-dedup allocated B | saved B |
|---|---:|---:|---:|
| all 35 observed evaluation trees | 34,406,400,000 | 19,420,168,192 | 14,986,231,808 |
| matched 28-tree two-seed 65-epoch evaluations | 27,525,120,000 | 15,572,930,560 | 11,952,189,440 |
| non-evaluation allocation, per seed | — | 4,703,256,576 | — |
| complete two-seed demand | — | **24,979,443,712** | — |
| complete demand plus 8 GiB reserve | — | **33,569,378,304** | — |

The 28-tree CAS inventory contains 2,229 unique objects and 15,563,519,143 logical object bytes;
4 KiB object allocation plus 1,835,008 B of projected manifests yields the 15,572,930,560 B figure.
The non-evaluation term is not guessed: w96a measured 18,465,816,576 B for one complete seed, and
the exact seed-20260815 evaluation allocation in this inventory is 13,762,560,000 B; their
difference is 4,703,256,576 B. Two copies are charged because this implementation deduplicates the
evaluation trees, not checkpoints or the stage-controller population.

The retained receipt recorded 12,943,491,072 live AP free bytes after the cached-inventory reseal.
That value is drift-prone and is not substituted for w96a's chartered 22,319,071,232 B comparison
denominator. Both make the same disposition BLOCKED_STORAGE.

## Exact aligned implementation

The ON loss executed on WD3's real selected scorer cells is

`100 * mean(sigmoid(-(z_target - max(z_other)) / tau))`.

`z_target` is gathered at the original SegNet argmax class; the target channel is replaced by
`-1e9` before the best-other reduction. The implementation test calls CE1's original
`target_margin` directly (source SHA
`ffdf098801863ff8bffe8bd818ce101928dd75b4937cbbffb2e225bddbc12f4b`) and requires exact tensor
equality after removing CE1's retained singleton channel. WD3's prior calibrated
`100 * stage_scale * mean(1 - p_target)` branch is not renamed or reused as the aligned law.

The complete aligned contract is:

- loss selector: `expected_flip_margin`; missing selector means legacy OFF;
- tau: linear `0.15 -> 0.05` over the global 65-epoch, 39,000-step window at batch size 1, with
  exact first/last endpoints and the same global step after resume;
- optimizer: AdamW at the transferred `2e-5` peak and
  `CosineAnnealingLR(..., eta_min=0.01*lr)`; OFF remains at its original `0.02*lr` floor;
- pose: exact nonlinear pose score is active from step zero; any nonzero `pose_start_step` is
  rejected;
- resume: `resume_from` and the trainer's self-pin remain the only masked self-referential fields;
  loss law, both tau endpoints, full-window length, pose-start step, retention contract, and every
  other config field remain strict;
- terminal epoch: aligned resume ends at sealed epoch 65 rather than adding another 65 epochs;
- DSL: two real nilary factories emit the WD3 trainer's existing `--compiled-config` flag, one for
  each sealed seed config; the config validator owns the scientific law instead of invented shadow
  flags.

Default-OFF identity is tested against an independent reproduction of the original softmax/gather,
calibration, pose, and total-expression order. The OFF telemetry keys are unchanged; ON-only tau and
expected-flip quantities are emitted only by the aligned branch.

## Lossless content-addressed retention

`tac.content_addressed_retention` stores immutable SHA-256 objects and one manifest record per
logical file: relative path, logical byte count, complete-file SHA-256, mode, chunking law, and the
ordered object SHA/byte sequence. The exact repetition boundaries are:

- `receiver_pairs.rgb.u8`: one 3,052,008 B camera frame per object, exposing the repeated fixed
  frame 0 without changing layout;
- NPZ/ZIP: exact ZIP local-member byte ranges plus the central-directory suffix, exposing repeated
  scorer arrays and byte-identical archive members;
- every other payload: one whole-file object.

Future aligned evaluations first materialize every ordinary payload, atomically materialize CAS
objects, write the manifest, deep-hash every referenced object, and only then remove manifest-covered
physical duplicates. No symlink is created. A crash after the manifest lands resumes from the
manifest's retained `EVALUATION_RESULT.json` instead of rerunning the scorer. The API restores either
the complete tree or any individual arbitrarily large payload as a regular file and verifies its
complete-file SHA before rename. Corrupt objects, unsafe paths, CAS-inside-logical-tree layouts,
symlinks, identity drift, and overwrite attempts fail closed. Failed object writes remove their
partial temporary; source bytes remain until the complete manifest verifies.

The OFF inventory is source-backed measurement only. It did not materialize a second 19.4 GB CAS
copy of the existing OFF corpus, because doing so could not clear the gate and would consume the
very capacity being measured. The 35 OFF trees remain the retained source of every hashed byte.

## Tests, reviews, and preflight

- `.venv/bin/python -m pytest -q src/tac/tests/test_w96b_aligned_loss_and_retention.py
  src/tac/tests/test_wd3_resume_config_identity.py src/tac/tests/test_ddm_ds1_variation_gate.py`:
  **49 passed**.
- `.venv/bin/ruff check` on the five edited Python surfaces: **PASS**.
- `git diff --check` on the same surfaces: **PASS**.
- `.venv/bin/python -m tac.preflight --scope dev --timeout-s 30`: **PASS, PARTIAL — 25/27
  declared gates examined**. This is a developer/build gate, not a release/custody or score claim.
- `.venv/bin/python -m tac.preflight --scope all --allow-slow-preflight`: **ENVIRONMENT-BLOCKED,
  not PASS**. It passed upstream-pin, always-keep-payload, codebase-drift, supply-chain, provider,
  runtime-closure, remote-custody, and MCP-config-disabled gates, then the managed sandbox refused
  the live-MCP gate's `ps -axo` subprocess with `PermissionError: Operation not permitted`. No
  release, dispatch, or frontier claim is made.
- Review pass 1, `pass1-correctness`: exact CE1 transfer, legacy identity, resume schedule, atomic
  retention, and source-backed denominator; tracker contains 147 review events (some files were
  re-marked after the host interrupted the first marking loop).
- Review pass 2, `pass2-adversarial`: surrogate/tau/pose drift, corrupt object, symlink, partial
  object, store nesting, individual restore, and crash-resume attacks; all 130 current entities
  across the five files are marked reviewed.

Pass 1 found and fixed partial-object cleanup, store nesting, and a missing direct comparison to the
original CE1 function. Pass 2 added independent large-file restore and corrupt-object negative
controls. No review override was used.

## Sealed fire order

The retained fire order is `BLOCKED_STORAGE`; both configs remain launch-disabled and fail WD3's
launch gate only on the expected unclaimed scorer lane, unclaimed Metal lane, false launch authority,
and unverified r5 exit. Their live trainer self-pins match the reviewed source.

1. **QUEUED-WITH-A-FIRE-ORDER — storage reclaim.** Owner: MAIN/operator. Consumer store:
   `/Volumes/APDataStore/pact/ddm_w96a_aligned_window/`. Fire trigger: #1165 Vertigo reclaim completes
   after the pk4 cold-move boundary on 2026-08-27, every moved byte has its certify-or-block receipt,
   and APDataStore free space reaches **33,569,378,304 B** (24,979,443,712 B measured demand plus the
   binding 8 GiB reserve). Nothing is deleted or moved by this arm.
2. **QUEUED-WITH-A-FIRE-ORDER — seed 20260815.** Owner: MAIN. Consumer store: the same AP root.
   Fire trigger: step 1 is green; fresh distinct scorer and Metal claims exist; MAIN reseals
   `launch_authorized=true` and `r5_exit_verified=true`; WD3 config validation passes. Run to retained
   stage-end epoch 65 before the next seed.
3. **QUEUED-WITH-A-FIRE-ORDER — seed 20260816.** Owner: MAIN. Consumer store: the same AP root.
   Fire trigger: seed 20260815 has ended with its checkpoint, CAS manifests, and result receipts
   intact; then repeat the same fresh-lane/config validation.
4. **QUEUED-WITH-A-FIRE-ORDER — per-checkpoint S1E n60 screen.** Owner: MAIN. Consumer store: the
   same AP root. Fire trigger: each aligned checkpoint result lands. Compare each seed separately to
   its matched OFF checkpoint on fixed IDs `0,10,...,590`.
5. **QUEUED-WITH-A-FIRE-ORDER — n600 sampled leg.** Owner: MAIN, holder of the global scorer slot.
   Consumer store: the same AP root. Fire trigger: one aligned checkpoint reaches
   `composed_delta <= matched OFF composed_delta / 5`, the n600 slot is idle and freshly claimed,
   and selected checkpoint/config/CAS payload SHAs are sealed. If both seed-65 screens remain worse
   than matched OFF/2, do not fire n600; close only that two-seed aligned instance.

## Ledger receipt

`tools/canonical_task_status.py` appended two actor-`ddm_w96b` rows to
`ddm_w96a_aligned_config_renderer_window`: `blocked -> in_progress` records the paid implementation
gate, then `in_progress -> blocked` supersedes the prior two-blocker state with the measured storage
blocker alone. Test status is green. No malformed direct `blocked -> blocked` row was appended; the
CLI refused that attempted invalid transition before any write.

## RECALL EVIDENCE

Searched `.omx/research/` memos and receipts by content for `W96`,
`film_amortized_flat_w96`, `expected_flip`, `target_margin`, `ce_fraction`,
`softplus_fraction`, `aligned objective`, `S1E`, `pose gate`, `#1089`, `#1091`, `#1251`, and
`#1273`; searched `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks, design/SPEC files, WD3/S1A
source and tests, canonical task history, the canonical-equations JSON, and the retained AP evidence.

Beyond the charter seeds, this recovered:

- CE1's live source law and exact tau/cosine behavior, rather than relying on a memo equation;
- CW1's re-derivation that objective allocation removed 92.651% of the ancestor surcharge and that
  the allocation is scale-invariant, while the tau and aligned learning-rate optimum remained
  unswept;
- EF3000/EF6000's ancestor descent evidence, scoped to a different semantic-renderer vehicle;
- S1A's strict resume/seed apparatus and the crash-induced seed-2 epoch-31 extra screen;
- all 35 live OFF evaluation trees, rather than w96a's original 14-row snapshot;
- no compiled canonical-equation entry in the searched registry scope that already bridges CE1's
  law into WD3; the transfer therefore had to be executable source plus tests, not a claimed config
  alias.

This changed the plan in three concrete ways: the test binds directly to CE1 source; the storage
measurement reports both the all-35 corpus and the matched 28-tree schedule; and the fire order routes
the falsified storage prediction to #1165 instead of declaring the gate paid.

## Retained payloads and source identities

| path | bytes | SHA-256 | disposition |
|---|---:|---|---|
| `/Volumes/APDataStore/pact/ddm_w96a_aligned_window/W96B_BUILD_AND_STORAGE_RECEIPT.json` | 33,416 | `6fac1bd8c8c40f9fc97df57b341b0b99c010105ecbdf12c2495304e7491cd42b` | KEEP; exact inventory + build receipt |
| `/Volumes/APDataStore/pact/ddm_w96a_aligned_window/SEALED_FIRE_ORDER_W96B.json` | 2,723 | `92d9047563a3c9da7c1250a97b475f4e14c5c93ea2eb2994151c13200301b8f2` | KEEP; BLOCKED_STORAGE handoff |
| `/Volumes/APDataStore/pact/ddm_w96a_aligned_window/launch_requests/aligned_seed_20260815.json` | 7,146 | `a46eb336b5294d0b84a849dadc3a82617089d0e91fc45cfb8dac80c949967ba2` | KEEP; launch-disabled seed config |
| `/Volumes/APDataStore/pact/ddm_w96a_aligned_window/launch_requests/aligned_seed_20260816.json` | 7,146 | `f088866828d6e2c690f3ea52556a5feafc2b36c91ff5788e761797101cf45af9` | KEEP; launch-disabled seed config |
| `/Volumes/APDataStore/pact/ddm_w96a_aligned_window/off_baseline_s1e_rerun.json` | 296,698 | `3037d264f097cd1b239cd96fc2302f5d812e0f3384eea7813fdc5cb074b60b18` | READ-ONLY; 35-row source replay |
| `/Volumes/APDataStore/pact/ddm_w96a_aligned_window/STORAGE_PREFLIGHT_BLOCKER.json` | 1,870 | `4e62de066d3304e970a20002aa08400789015f9d26d69f143d82f4227f3e1056` | READ-ONLY; superseded reference-form demand |

Reviewed source SHAs before serialization:

- WD3 trainer: `6a567db93c9947e63b5fb022411dd583ce848ccb22e3fe0e2393fe58c94a86df`
- receipt/config sealer: `fa4a854d5fae2c5361374ea3e4ab6684a6e36cca0274a36d5fb7800774f20d6b`
- content-addressed retention: `0e9048a1e53d5835ea61dcb7454c07d69eaca23a509fa4bb9b621db0494e6003`
- DSL registration: `053bd12e198bb74a44036e497a1277d9d36638c96acdabba278a2c72f2234923`
- focused tests: `b1d4f6bf935bf8b21214300d7e20feb24a772f833e95a2a039620c5f6122c0be`

## GESTALT-DELTA

GESTALT-DELTA: w96a had a 45,521,567,744 B reference-form storage blocker plus a missing executable
loss; w96b pays the loss gate and measures a 24,979,443,712 B content-addressed demand, but narrows —
does not clear — storage to a 2,660,372,480 B shortfall before reserve and 11,250,307,072 B with reserve.

No training, scorer, Metal, Modal, n600, contest evaluation, archive score, or pointer mutation was
performed. The scientific aligned-W96 hypothesis remains untested; this is not a formulation or
family negative.

**Own-vehicle frontier: S = 0.14811799921260607 @ 180,215 B [contest-CUDA T4 n600], GB1 groupbin8 archive SHA `ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4` — UNMOVED by ddm_w96b.**
