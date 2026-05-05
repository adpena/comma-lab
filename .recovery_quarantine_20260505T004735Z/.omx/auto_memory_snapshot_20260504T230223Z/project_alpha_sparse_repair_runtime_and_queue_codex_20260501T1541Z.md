# Codex Memory - Alpha Sparse Repair Runtime And Exact-Eval Queue - 2026-05-01T15:41Z

Context:

- Active frontier is C-044, OWV3 0120 plus PFP16, T4 score
  `0.9975385870574276` to `0.9975405870574277`, archive bytes `609963`, SHA
  `1e9195cb6e0e08fc98ee393590770e2b22905a2ee2718edb8b737cada125f279`.
- The high-EV next path remains Alpha mask-payload collapse plus exact CUDA
  component protection, not further PFP16 microsteps.

Landed:

- Added optional Alpha AMR1 sparse-repair support to
  `submissions/robust_current/inflate_renderer_grayscale.py`.
- Supported repair members:
  `alpha4_residual_repair.amr1`, `.amr1.xz`, `.amr1.zlib`, `.amr1.br`.
- Runtime verifies AMR1 schema, shape, bounds, class ids, decoded-candidate
  SHA, and full-repair source SHA when declared non-partial.
- Extended `experiments/build_alpha_mask_replacement_archive.py` to build
  `repair_policy=full` or `class_prefix_<ids>` archives with raw/zlib/xz/brotli
  repair payloads. It records selected runs/pixels, raw/compressed SHA/bytes,
  and anchor mask SHA match.
- Focused verification passed:
  - py_compile on touched repair files/tests.
  - pytest `src/tac/tests/test_build_alpha_mask_replacement_archive.py`
    `src/tac/tests/test_inflate_renderer_grayscale_repair.py` -> `6 passed`.
  - `bash -n submissions/robust_current/inflate.sh`.

Concrete archives:

- CRF63 class-prefix-2 sparse repair on C-044:
  `experiments/results/alpha_mask_repair_c044_crf63_class2_lzma_20260501/archive.zip`
  bytes `549148`, SHA
  `fc2721050dd3cca77c84ddb0604d91701a7c371de079bbd29ede61d7299dae03`,
  byte delta vs C-044 `-60815`, rate delta `-0.04049421223412485`.
- CRF62 class-prefix-2 sparse repair on C-044:
  `experiments/results/alpha_mask_repair_c044_crf62_class2_lzma_20260501/archive.zip`
  bytes `606572`, SHA
  `3cd592c53056585944dcf270fcc39532c01750ebe7b0ded24ffe695715d907b0`,
  byte delta vs C-044 `-3391`, rate delta `-0.002257927710037283`.
- CRF63 top-residual-frame-group sparse repair on C-044:
  `experiments/results/alpha_mask_repair_c044_crf63_topgroup1_lzma_20260501/archive.zip`
  bytes `467747`, SHA
  `e2d0548ee63d0df4c6ab3c9e3a2ce9c23a8062cdf1274bc3ea29a3179dbab6d9`,
  byte delta vs C-044 `-142216`, rate delta `-0.09469579687722272`.
  It selects frame group `20` (`frames 1000-1049`), `93336` repair pixels,
  and `43074` runs.

Lightning:

- Staged source/artifacts to `scratch-studio-devbox` with manifest
  `.omx/state/alpha_repair_exact_20260501T1540Z_manifest.json`.
- Lightning doctor passed local supply chain, SSH, remote supply chain, and
  machine inventory.
- Queued exact CUDA/T4 jobs:
  - `exact_eval_alpha_repair_c044_crf63_class2_t4_20260501T1540Z`
  - `exact_eval_alpha_repair_c044_crf62_class2_t4_20260501T1540Z`
  - `exact_eval_alpha_repair_c044_crf63_topgroup1_t4_20260501T1546Z`
- Baseline for adjudication: C-044 score `0.9975385870574276`, bytes `609963`,
  PoseNet `0.00357302`, SegNet `0.00402365`.
- Component gates are forensic fail-open at `1.25` relative to preserve exact
  negative evidence if the sparse repair collapses.

Next:

- Poll via Lightning SDK or harvest once artifacts appear. If local wrapper
  `refresh-status` hits missing teamspace inference, direct SDK with
  `Job(name, teamspace="comma-lab", user="adpena")` works.
- Harvest through `scripts/launch_lightning_batch_job.py harvest-ssh
  --state-path .omx/state/lightning_batch_jobs.json --job-name <job>
  --ssh-target scratch-studio-devbox --require-adjudication --overwrite`.
- Add claim matrix rows only after exact CUDA JSON/adjudication exists.

Update 2026-05-01T15:53Z:

- Built additional byte-screened C-044 sparse-repair archives:
  - CRF63 topgroup2:
    `experiments/results/alpha_mask_repair_c044_crf63_topgroup2_lzma_20260501/archive.zip`,
    bytes `547416`, SHA
    `c41be612114b05189038d782f9795469f9eb75f0a7380ac1939b27c746986fea`,
    rate delta vs C-044 `-0.04164747994093245`.
  - CRF62 topgroup1:
    `experiments/results/alpha_mask_repair_c044_crf62_topgroup1_lzma_20260501/archive.zip`,
    bytes `533251`, SHA
    `98766ecc68e10ed446ce9fe9987baef9c293c06f68569ba0a4aa01730cf74b96`,
    rate delta vs C-044 `-0.05107937201190801`.
- Staged them through reproducible Lightning manifest
  `.omx/state/alpha_repair_next_exact_20260501T1600Z_manifest.json`
  with remote verify OK (`1165` files, `20719121` bytes).
- Queued exact CUDA/T4 jobs:
  - `exact_eval_alpha_repair_c044_crf63_topgroup2_t4_20260501T1600Z`
  - `exact_eval_alpha_repair_c044_crf62_topgroup1_t4_20260501T1600Z`
- Byte-regressive comparison archives were retained locally but not dispatched:
  CRF63 topgroup4/topgroup8/class2_1, CRF62 topgroup4, and CRF60 topgroup1.

Update 2026-05-01T15:59Z:

- The first CRF63/CRF62 class2 Lightning jobs failed before scoring because
  `contest_auth_eval.py` rejected `alpha4_residual_repair.amr1.xz` as an
  unknown archive suffix. This is a harness allowlist bug only; no score JSON
  or component evidence was produced.
- Preserved failure logs at
  `.omx/state/lightning_alpha_repair_amr1_allowlist_failures_20260501T1600Z/`.
- Fixed the bug class:
  - `experiments/contest_auth_eval.py` allows `.amrc`, `.amr1`,
    `.amr1.xz`, `.amr1.zlib`, `.amr1.br`.
  - `experiments/canonical_local_auth_eval_smoke.py` whitelist updated for
    parity.
  - Focused tests passed: `11 passed`.
  - `AGENTS.md` now records that AMR1 suffixes must stay allowed in both exact
    auth-eval and local smoke whitelists.
- Stopped stale pre-fix submitted jobs and restaged patched source plus all five
  sparse-repair archives with
  `.omx/state/alpha_repair_amr1_allowlist_fix_20260501T1605Z_manifest.json`.
- Relaunched exact CUDA/T4 jobs with `fix1` names for all five candidates.
- Also launched L40S diagnostic hedges for CRF63 topgroup1 and CRF62 topgroup1
  while T4 was pending. Treat these as triage only; T4/equivalent exact eval is
  still required for promotion.

Update 2026-05-01T16:06Z:

- Added deterministic `pair_indices_<ids>` Alpha repair policy. It repairs
  AMR1 residual runs whose `frame_index // 2` is in the absolute contest pair
  set; this maps pair `i` to frames `2*i` and `2*i+1`.
- Used Lane W v2 hard-pair metadata to build two C-044 hard-pair archives:
  - CRF63 hardpairs30: bytes `473571`, SHA
    `96e233deda811c34b8db44e3fbe4db250aa22c02df6f02b891859528362e5ff2`,
    rate delta `-0.09081783433423919`.
  - CRF62 hardpairs30: bytes `537520`, SHA
    `d845c934f5cdbd2e4fca5968b50d972a4139200a8c83783aa56922ca7be0da55`,
    rate delta `-0.04823682014102946`.
- Staged with
  `.omx/state/alpha_repair_hardpairs_exact_20260501T1615Z_manifest.json`
  and queued T4 exact plus L40S diagnostic jobs for both.
