# L5 v2 TT5L side-info effect curve Lightning paired-axis plan

Generated: 2026-05-17T14:00:15Z

This memo supersedes the CUDA-only Lightning alternate-provider memo for the TT5L side-info effect curve. It does not launch provider work and does not claim score movement. It records ten dry-run exact-eval cells: five side-info variants times `[contest-CPU]` and `[contest-CUDA]`.

## Status

- Source commit: `4b6ff1a6110fd78a0a79e6817a2d64369780f322`
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
| `zero` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 34373 | `b444cc91f102c9807a865ed59f182ca5c83f3239a49ec2aa400b497d7dea37a3` | `60b76e129ee3c866d761579d03e42bb0f67503c0d10995c79aca217c7fde4efd` | `ecba8583aafeeb0a506fd8846bb500bfcfec57db17d4524cd5ba87b2b35b3801` |
| `zero` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 34373 | `b444cc91f102c9807a865ed59f182ca5c83f3239a49ec2aa400b497d7dea37a3` | `62e734227ea8f14e9924a2a92a03933af83bdb10d7f719c0d24c76b087764619` | `dadcb765c43beb7cd20c9fca19f1b2e831d727b1adb63c074421147bea7c1ffc` |
| `random_lsb` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 38681 | `ccce77aaf1907d6e70d8cba498261708b241ac7d78a9bf22978aa459cb6b7fd1` | `b7ec13b3419b64a7d25f7eb30f771d6accd1da7e46f62b8f8e9d1bc7bc7d140a` | `d113d25632872cfc8372b728c629c9f6b176f028813b6b94c8d1f5719f4e1a03` |
| `random_lsb` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 38681 | `ccce77aaf1907d6e70d8cba498261708b241ac7d78a9bf22978aa459cb6b7fd1` | `292effd1a921bde6cc3b20097b84bec95e829a25d6814fec17e8b03ecec28890` | `677c2aaa991d647a78dd366d781d1b38ea3a76a8dc43dcb696a0700cf24543cd` |
| `shuffled` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 43284 | `c235e5cb91f4122c3bb642354b424dd21f8b37cc0d2ac7b3e03c0b84dcc49bc3` | `8e5940c356ca6b9be2907b76b9cc0803b61d5bf0f7f111dc250aa1cd116cd7f8` | `a32074b95c1d4e4c01d0b1e92d51b9a8900d25d024c9e3e669fbb1a140716686` |
| `shuffled` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 43284 | `c235e5cb91f4122c3bb642354b424dd21f8b37cc0d2ac7b3e03c0b84dcc49bc3` | `0d2f821b914d86286ebbe9728373aed67bffa2069b9a583e971a856554b5d4e7` | `ab08aec956ebf54b3026626cbea79c7c8076655957444844a20f21af4ff60d98` |
| `trained` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 43323 | `f08299c5e77908eeeb82cc9948e530fd5a790894902a726ebd8a258596b4bf1a` | `d0598c86f2bf9479629137b813ceed8bf02721b12dd3b61ed9d9d050d42c6a29` | `9f325ce2ba3d83fab729daad3cd7b3cbea05887243d04591899b01e117b0dbff` |
| `trained` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 43323 | `f08299c5e77908eeeb82cc9948e530fd5a790894902a726ebd8a258596b4bf1a` | `42fddc0a2a8070752d444777e880066eb7f09b6a1350ee24c7c8c751c949bada` | `736752720447e36d898abcd6f5269f542a1dd2b8bc57943f44a8a7e9467a657f` |
| `ablated` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 42419 | `ec343265899859495fca1d2874c1b85d211210ea13a2de5bb7e52c9816ba6b39` | `abc81b15cc4166a51388e826ff9e5c9f8e6bc0aa19f8f000b706b73fa7a958fa` | `f3d2112ade3ab0d88214c83a56d97d9754040e4f4f11a1c771c3511cdd7a44e9` |
| `ablated` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 42419 | `ec343265899859495fca1d2874c1b85d211210ea13a2de5bb7e52c9816ba6b39` | `c487d87e0138db5e2d9fa658a0b3e330fb653fc103b44d0b4fc86e3c2ffdad15` | `ce269ee87f2c311d210b281b472441fd1fbb31406773f818a54b2898f025e787` |

## Axis Invariants Verified

- CPU cells are `exact_cpu_eval`, adjudicate with `required_device=cpu`, contain `--device cpu`, omit `--device cuda`, and omit `INFLATE_REQUIRE_CUDA=1`.
- CUDA cells are `exact_cuda_eval`, adjudicate with `required_device=cuda`, contain `--device cuda`, and retain `INFLATE_REQUIRE_CUDA=1`.
- All ten `dry_run.stderr` files are zero bytes when the plan is ready.
- Queue metadata carries the expected `axis`, `variant`, `pair_group_id`, `run_id`, `archive_sha256`, and source dispatch-plan pointer for every cell.

## Reactivation Criteria

Before any non-dry-run submit, configure Lightning identity/workspace variables, stage a fresh source manifest, run the remote preflight, and claim the per-axis lane. The effect-curve predicate remains blocked until all ten cells are harvested and the aggregate side-info effect artifact passes its paired-axis validation.
