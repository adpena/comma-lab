# L5 v2 TT5L side-info effect curve Lightning paired-axis plan

Generated: 2026-05-17T15:19:12Z

This memo supersedes the CUDA-only Lightning alternate-provider memo for the TT5L side-info effect curve. It does not launch provider work and does not claim score movement. It records ten dry-run exact-eval cells: five side-info variants times `[contest-CPU]` and `[contest-CUDA]`.

## Status

- Source commit: `730b52ee3118b3ec6331a600d760f4fe44897d4e`
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
| `zero` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 34373 | `b444cc91f102c9807a865ed59f182ca5c83f3239a49ec2aa400b497d7dea37a3` | `60b76e129ee3c866d761579d03e42bb0f67503c0d10995c79aca217c7fde4efd` | `d3967a4e102fdf719d631c15c0c75ac5e562f07ec3816ac710d0337186a3f6d5` |
| `zero` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 34373 | `b444cc91f102c9807a865ed59f182ca5c83f3239a49ec2aa400b497d7dea37a3` | `62e734227ea8f14e9924a2a92a03933af83bdb10d7f719c0d24c76b087764619` | `6b2eab45f9c20e1a0e2329ed53e18a489d0cb58cf91627af12aa12fa7f63c9d4` |
| `random_lsb` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 38681 | `ccce77aaf1907d6e70d8cba498261708b241ac7d78a9bf22978aa459cb6b7fd1` | `b7ec13b3419b64a7d25f7eb30f771d6accd1da7e46f62b8f8e9d1bc7bc7d140a` | `97911300cca57438ccc3788fe9d51677ad500347c4238da13c7930373822d4af` |
| `random_lsb` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 38681 | `ccce77aaf1907d6e70d8cba498261708b241ac7d78a9bf22978aa459cb6b7fd1` | `292effd1a921bde6cc3b20097b84bec95e829a25d6814fec17e8b03ecec28890` | `35a83b6c00d8c557b67fccbc7ad9853344759514d38f3b5df52a02f46af3c016` |
| `shuffled` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 43284 | `c235e5cb91f4122c3bb642354b424dd21f8b37cc0d2ac7b3e03c0b84dcc49bc3` | `8e5940c356ca6b9be2907b76b9cc0803b61d5bf0f7f111dc250aa1cd116cd7f8` | `cfe0f465611cc147837e7fbb1817df65e10c91503a730205042bff8456867182` |
| `shuffled` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 43284 | `c235e5cb91f4122c3bb642354b424dd21f8b37cc0d2ac7b3e03c0b84dcc49bc3` | `0d2f821b914d86286ebbe9728373aed67bffa2069b9a583e971a856554b5d4e7` | `967326523323536f7c9192ee05f2fe3d5e013a3100c9bd6c42c8149998196433` |
| `trained` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 43323 | `f08299c5e77908eeeb82cc9948e530fd5a790894902a726ebd8a258596b4bf1a` | `d0598c86f2bf9479629137b813ceed8bf02721b12dd3b61ed9d9d050d42c6a29` | `cf10bace8e1fe7499a2669f9a05f88d5bc3302fab8c6785c7f5a7c37cd712f45` |
| `trained` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 43323 | `f08299c5e77908eeeb82cc9948e530fd5a790894902a726ebd8a258596b4bf1a` | `42fddc0a2a8070752d444777e880066eb7f09b6a1350ee24c7c8c751c949bada` | `7c2d8c01aa3aeaea58e4213ba1eac95dd8d2c28a9b1d064e0dfd40aed95eb24f` |
| `ablated` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 42419 | `ec343265899859495fca1d2874c1b85d211210ea13a2de5bb7e52c9816ba6b39` | `abc81b15cc4166a51388e826ff9e5c9f8e6bc0aa19f8f000b706b73fa7a958fa` | `17d08a6ad3024cf7dbc0306493488f146cdcf8ec66a82613e510c0f22d2e3cd4` |
| `ablated` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 42419 | `ec343265899859495fca1d2874c1b85d211210ea13a2de5bb7e52c9816ba6b39` | `c487d87e0138db5e2d9fa658a0b3e330fb653fc103b44d0b4fc86e3c2ffdad15` | `58cfa4dfcbc988936c8e2f028e362a87fa609f84b97b3a37c8e250211b505589` |

## Axis Invariants Verified

- CPU cells are `exact_cpu_eval`, adjudicate with `required_device=cpu`, contain `--device cpu`, omit `--device cuda`, and omit `INFLATE_REQUIRE_CUDA=1`.
- CUDA cells are `exact_cuda_eval`, adjudicate with `required_device=cuda`, contain `--device cuda`, and retain `INFLATE_REQUIRE_CUDA=1`.
- All ten `dry_run.stderr` files are zero bytes when the plan is ready.
- Queue metadata carries the expected `axis`, `variant`, `pair_group_id`, `run_id`, `archive_sha256`, and source dispatch-plan pointer for every cell.

## Reactivation Criteria

Before any non-dry-run submit, configure Lightning identity/workspace variables, stage a fresh source manifest, run the remote preflight, and claim the per-axis lane. The effect-curve predicate remains blocked until all ten cells are harvested and the aggregate side-info effect artifact passes its paired-axis validation.
