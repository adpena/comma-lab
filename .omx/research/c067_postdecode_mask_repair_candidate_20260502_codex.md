# C067 Charged Postdecode Mask Repair Candidate - 2026-05-02

## Scope

Implemented a contest-faithful, deterministic candidate builder for charged
postdecode AMR1 mask repair over C067 legacy `masks.mkv` streams.

The builder is:

- `experiments/build_c067_postdecode_mask_repair_candidate.py`
- schema: `c067_postdecode_mask_repair_candidate_v1`
- output contract: `renderer.bin`, lossy `masks.mkv`,
  `alpha4_residual_repair.amr1[.zlib|.xz|.br]`, `optimized_poses.bin`
- score status: `score_claim=false`, `promotion_eligible=false`
- runtime hook: existing `submissions/robust_current/inflate_renderer.py`
  applies AMR1 after decoding legacy `masks.mkv` when `grayscale.mkv` is absent

No remote dispatch or CUDA eval was launched.

## Candidate Built

Command:

```bash
.venv/bin/python experiments/build_c067_postdecode_mask_repair_candidate.py \
  --base-archive experiments/results/c067_fixedslice_unpacked_runtime_20260502/archive.zip \
  --lossy-archive experiments/results/c067_micro_mask_reencode_plan_20260502/builds/c067_micro_av1_mask_reencode_save12k/archive.zip \
  --output-archive experiments/results/c067_postdecode_mask_repair_candidate_20260502/save12k_top256_pairclass/archive.zip \
  --manifest-json experiments/results/c067_postdecode_mask_repair_candidate_20260502/save12k_top256_pairclass/c067_postdecode_mask_repair_manifest.json \
  --policy top_pixels \
  --atom-granularity pair_class \
  --max-atoms 256 \
  --repair-compressor lzma_xz \
  --label c067_save12k_top256_pairclass_postdecode_repair
```

Artifact:

- archive: `experiments/results/c067_postdecode_mask_repair_candidate_20260502/save12k_top256_pairclass/archive.zip`
- archive bytes: `412436`
- archive SHA-256:
  `6573a968cbc5e088336ca30c8788f509adf8e6303e097c6d52c1111ce8a066b0`
- manifest:
  `experiments/results/c067_postdecode_mask_repair_candidate_20260502/save12k_top256_pairclass/c067_postdecode_mask_repair_manifest.json`
- base runtime archive bytes: `281303`
- base runtime archive SHA-256:
  `886562a9f25f207c2004161fbc9d01881f28f125123e68a8582c330467e54f80`
- source C067 score archive bytes: `276214` per unpack manifest, SHA-256
  `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`

Member accounting:

```text
renderer.bin                         59288 bytes
masks.mkv                           185143 bytes
alpha4_residual_repair.amr1.xz      172736 bytes
optimized_poses.bin                   7200 bytes
```

Repair accounting:

- source mask stream SHA-256:
  `a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb`
- source decoded class SHA-256:
  `0344fcfc39e683f21a71db1085a8697a94c4606f91f883362e9acc02fc7b5b45`
- lossy mask stream SHA-256:
  `002be9cb1a535e8ac3e519747ec79fbdd0306bdc3613667afc59b526f2e001c8`
- lossy decoded class SHA-256:
  `46c33d28505be0eae99ca26719a164fbc54005805614d5e77282b58dbc72b627`
- total residual pixels: `359711`
- total pair/class atoms: `1500`
- selected atoms: `256`
- selected repair pixels: `141478`
- selected repair runs: `74600`
- raw AMR1 bytes: `847958`
- raw AMR1 SHA-256:
  `708e4d45b8d8fc382c9370982b4371067da84b92625cf98aee76075ca66094c0`
- compressed AMR1 bytes: `172736`
- compressed AMR1 SHA-256:
  `42d3eaa19eb87dca1db51382b0b767565eededfaa55af5af503e56e9857e23e0`

## Interpretation

This is a successful charged-path scaffold and concrete archive build, not a
promotable score artifact. The top256 pair/class policy is too expensive:
`masks.mkv` saves `38242` member bytes relative to C067 source masks, while the
selected repair adds `172736` charged member bytes before outer ZIP effects.
The policy is useful as byte/provenance calibration for the next pass, not as a
dispatch target.

The next patch should add a byte-budgeted atom selector that estimates AMR1
compressed contribution per atom, then searches for policies under explicit
repair payload budgets such as 4k, 8k, and 12k compressed bytes. That selector
should keep the current manifest fields and add per-atom compressed byte cost
or marginal bytes-per-pixel before any exact CUDA dispatch is considered.

## Verification

Focused tests:

```bash
.venv/bin/python -m pytest src/tac/tests/test_build_c067_postdecode_mask_repair_candidate.py
```

Result: `4 passed`.

The tests cover deterministic ZIP packaging, zip-slip rejection before decode,
duplicate lossy-mask member rejection, selected-atom manifest fields, and proof
that the repair payload is charged inside the candidate archive.

## 2026-05-02T21:05Z - compressed-byte budget selector

The builder now supports a compressed-byte budget for the AMR1 repair member.
This is a stricter selector than top-k atoms because the archive pays
compressed bytes, not atom count or repaired-pixel count. The policy builds
the deterministic atom prefix once, then binary-searches the largest prefix
whose compressed `alpha4_residual_repair.amr1.xz` payload fits the requested
budget.

New policy field:

- `repair_selector.compressed_byte_budget`

New CLI option:

- `--max-repair-payload-bytes`

Concrete C067/save12k candidates:

- `budget8000`
  - archive:
    `experiments/results/c067_postdecode_mask_repair_candidate_20260502/save12k_budget_sweep/budget8000/archive.zip`
  - archive bytes: `247414`
  - archive SHA-256:
    `97a25794da10e2d778b13fd44a6a45c623981a5619ec1ed8ea6a88798ed267f3`
  - compressed repair bytes: `7764`
  - raw repair bytes: `35202`
  - repair pixels: `7565`
  - repair runs: `2969`
  - selected atoms: `16`
  - formula-only rate delta vs C067: `-0.019176737849918534`
  - exact eval:
    `exact_eval_c067_postdecode_repair_save12k_budget8000_l40sdiag_20260502T2101Z`
    submitted as L40S diagnostic only.
- `budget12000`
  - archive:
    `experiments/results/c067_postdecode_mask_repair_candidate_20260502/save12k_budget_sweep/budget12000/archive.zip`
  - archive bytes: `251402`
  - archive SHA-256:
    `f22e2d2c6372cb6ab74031ce4a565cdddc40ea46e122c098b61680812fe4eaf1`
  - compressed repair bytes: `11752`
  - raw repair bytes: `55712`
  - repair pixels: `11856`
  - selected atoms: `26`
  - formula-only rate delta vs C067: `-0.016521292344867315`
  - not dispatched yet; hold until `top10` and `budget8000` exact traces tell
    whether additional charged atoms have positive marginal value.

The `top10` trace-frame candidate remains an exact diagnostic in flight:

- archive bytes: `251122`
- archive SHA-256:
  `447dc22798c287fbb26ba1a9bf63d925fc328d281e51988fc07148b325e0c4fd`
- compressed repair bytes: `11472`
- selected atoms: `50`
- exact eval:
  `exact_eval_c067_postdecode_repair_save12k_top10_l40sdiag_20260502T2054Z`
  harvested on L40S.
- exact CUDA score: `1.4680100510237182`
- PoseNet distance: `0.10732614`
- SegNet distance: `0.00264815`
- component trace:
  `experiments/results/lightning_batch/exact_eval_c067_postdecode_repair_save12k_top10_l40sdiag_20260502T2054Z/component_trace.json`
- component trace SHA-256:
  `5b4d28329c60cfc29e955bca70d33dd136c2ec36264dbe9eae05d818c09434f1`
- adjudication:
  `scientific_score_eligible=true`, `promotion_eligible=false`,
  `hardware_promotion_gate_triggered=true`.

This is not near promotion. It is useful because it is the first charged
postdecode repair response point: top10 repairs improve over un-repaired
save12k, but the score remains dominated by PoseNet. The next optimizer should
consume this component trace and rerank atoms by marginal benefit density
instead of carrying the same static top-trace prefix forward.

The `budget8000` compressed-byte prefix candidate was also harvested:

- archive bytes: `247414`
- archive SHA-256:
  `97a25794da10e2d778b13fd44a6a45c623981a5619ec1ed8ea6a88798ed267f3`
- compressed repair bytes: `7764`
- selected atoms: `16`
- exact CUDA score: `1.5492577842107278`
- PoseNet distance: `0.125453`
- SegNet distance: `0.00264457`
- component trace:
  `experiments/results/lightning_batch/exact_eval_c067_postdecode_repair_save12k_budget8000_l40sdiag_20260502T2101Z/component_trace.json`
- component trace SHA-256:
  `5406f8bc4b05592064031bdc0f3e616e9bd9ae3ff6bc0f50c67e271c31790a8b`
- adjudication:
  `scientific_score_eligible=true`, `promotion_eligible=false`,
  `hardware_promotion_gate_triggered=true`.

This is a worse score than top10, so the static compressed-byte prefix is not
the current promotion route.

An offline water-fill planner now consumes exact diagnostic component traces
and postdecode repair manifests:

- tool:
  `experiments/plan_c067_postdecode_mask_repair_waterfill.py`
- tests:
  `src/tac/tests/test_plan_c067_postdecode_mask_repair_waterfill.py`
- frame-class plan:
  `experiments/results/c067_postdecode_mask_repair_candidate_20260502/c067_postdecode_mask_repair_waterfill_frame_class_plan.json`
- pair-class plan:
  `experiments/results/c067_postdecode_mask_repair_candidate_20260502/c067_postdecode_mask_repair_waterfill_pair_class_plan.json`

The first trace-driven exact diagnostic candidate is pair-waterfill4k:

- archive:
  `experiments/results/c067_postdecode_mask_repair_candidate_20260502/save12k_waterfill_pair_sweep/budget4000/archive.zip`
- archive bytes: `243422`
- archive SHA-256:
  `25f890f449796f79c0da246758d05684c40ccce8b29a3bc2322b8958fa7ae489`
- compressed repair bytes: `3772`
- selected atoms: `4`
- selected repair pixels: `2362`
- formula-only rate delta vs C067: `-0.025223403003220974`
- exact eval:
  `exact_eval_c067_postdecode_repair_save12k_pairwaterfill4k_l40sdiag_20260502T2114Z`
  submitted as L40S diagnostic only.

Verification:

```bash
.venv/bin/python -m py_compile \
  experiments/build_c067_postdecode_mask_repair_candidate.py \
  src/tac/tests/test_build_c067_postdecode_mask_repair_candidate.py

.venv/bin/python -m pytest \
  src/tac/tests/test_build_c067_postdecode_mask_repair_candidate.py -q
```

Result: `5 passed, 1 warning`.
