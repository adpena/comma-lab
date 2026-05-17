# L5 v2 TT5L side-info effect curve Lightning paired-axis plan

Generated: 2026-05-17T09:36:33Z

This memo supersedes the CUDA-only Lightning alternate-provider memo for the TT5L side-info effect curve. It does not launch provider work and does not claim score movement. It records ten dry-run exact-eval cells: five side-info variants times `[contest-CPU]` and `[contest-CUDA]`.

## Status

- Source commit: `d8df9ecc5c4b9769645bee7951a0bfa210d81b84`
- Source variant manifest: `.omx/research/l5_v2_tt5l_current_code_fullshape_sideinfo_variant_packets_20260517_codex.json`
- Source dispatch plan: `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_20260517_codex.json`
- Modal blocker still recorded at: `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_modal_billing_blocker_20260517_codex.json`
- Raw dry-run artifact root: `experiments/results/lightning_batch/l5_v2_tt5l_sideinfo_effect_curve_paired_axes` (ignored result-tree state; hashes recorded below)
- Dispatch attempted: `false`
- Score claim: `false`
- Promotion eligible: `false`
- Required axes: `[contest-CPU]`, `[contest-CUDA]`
- Required variants: `zero`, `random_lsb`, `shuffled`, `trained`, `ablated`

## Paired-Axis Dry-Run Cells

| variant | axis | role | required device | bytes | archive SHA-256 | command SHA-256 | state SHA-256 |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| `zero` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 34373 | `b444cc91f102c9807a865ed59f182ca5c83f3239a49ec2aa400b497d7dea37a3` | `60b76e129ee3c866d761579d03e42bb0f67503c0d10995c79aca217c7fde4efd` | `f92f537a8d30a13b64a9e3cf88910a6620878f19afcf62424a859d6f14cefd1a` |
| `zero` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 34373 | `b444cc91f102c9807a865ed59f182ca5c83f3239a49ec2aa400b497d7dea37a3` | `62e734227ea8f14e9924a2a92a03933af83bdb10d7f719c0d24c76b087764619` | `edcd3e914afe943e3a84a0ed8c7250d211a29794e54162f618fcedd3fea2cee1` |
| `random_lsb` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 38681 | `ccce77aaf1907d6e70d8cba498261708b241ac7d78a9bf22978aa459cb6b7fd1` | `b7ec13b3419b64a7d25f7eb30f771d6accd1da7e46f62b8f8e9d1bc7bc7d140a` | `c1665fa5b8e30b798eb0b441aed91cbb9f62d55ac3151eae564fe72ff2bd5505` |
| `random_lsb` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 38681 | `ccce77aaf1907d6e70d8cba498261708b241ac7d78a9bf22978aa459cb6b7fd1` | `292effd1a921bde6cc3b20097b84bec95e829a25d6814fec17e8b03ecec28890` | `62a757518c70c3955c8e9c20771a618a17441fb2df4246e374c192d676b40be5` |
| `shuffled` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 43284 | `c235e5cb91f4122c3bb642354b424dd21f8b37cc0d2ac7b3e03c0b84dcc49bc3` | `8e5940c356ca6b9be2907b76b9cc0803b61d5bf0f7f111dc250aa1cd116cd7f8` | `813f265ebaedd7fd62367f355a5099eccf20057e17390c1511983a015813f29d` |
| `shuffled` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 43284 | `c235e5cb91f4122c3bb642354b424dd21f8b37cc0d2ac7b3e03c0b84dcc49bc3` | `0d2f821b914d86286ebbe9728373aed67bffa2069b9a583e971a856554b5d4e7` | `f5e8418b7a0ef3a22670ef4a8c0ea751b5b14cb3fce4c09a84c0ca8a2e0fee8d` |
| `trained` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 43323 | `f08299c5e77908eeeb82cc9948e530fd5a790894902a726ebd8a258596b4bf1a` | `d0598c86f2bf9479629137b813ceed8bf02721b12dd3b61ed9d9d050d42c6a29` | `f08b56e8b218bd0df2c2ee9a31f5f798de76bdaea32b7339e7147015ea059a58` |
| `trained` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 43323 | `f08299c5e77908eeeb82cc9948e530fd5a790894902a726ebd8a258596b4bf1a` | `42fddc0a2a8070752d444777e880066eb7f09b6a1350ee24c7c8c751c949bada` | `2a601196870ff4c3d490e2e1d9fdf9ed228cf153f16848b382e8ff957506b880` |
| `ablated` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 42419 | `ec343265899859495fca1d2874c1b85d211210ea13a2de5bb7e52c9816ba6b39` | `abc81b15cc4166a51388e826ff9e5c9f8e6bc0aa19f8f000b706b73fa7a958fa` | `7fcb1f267fadfc486c16430d27c6e3e92410cd2034fde8f09c932ba23df44bdc` |
| `ablated` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 42419 | `ec343265899859495fca1d2874c1b85d211210ea13a2de5bb7e52c9816ba6b39` | `c487d87e0138db5e2d9fa658a0b3e330fb653fc103b44d0b4fc86e3c2ffdad15` | `7fb57feba3ca6253abb74e4dff51143d35f3fd1c038ff459ca2019c148e4ac67` |

## Axis Invariants Verified

- CPU cells are `exact_cpu_eval`, adjudicate with `required_device=cpu`, contain `--device cpu`, omit `--device cuda`, and omit `INFLATE_REQUIRE_CUDA=1`.
- CUDA cells are `exact_cuda_eval`, adjudicate with `required_device=cuda`, contain `--device cuda`, and retain `INFLATE_REQUIRE_CUDA=1`.
- All ten `dry_run.stderr` files are zero bytes when the plan is ready.
- Queue metadata carries the expected `axis`, `variant`, `pair_group_id`, `run_id`, `archive_sha256`, and source dispatch-plan pointer for every cell.

## Reactivation Criteria

Before any non-dry-run submit, configure Lightning identity/workspace variables, stage a fresh source manifest, run the remote preflight, and claim the per-axis lane. The effect-curve predicate remains blocked until all ten cells are harvested and the aggregate side-info effect artifact passes its paired-axis validation.
