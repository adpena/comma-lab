# L5 v2 TT5L side-info effect curve Lightning paired-axis plan

Generated: 2026-05-17T06:42:15Z

This memo supersedes the CUDA-only Lightning alternate-provider memo for the TT5L side-info effect curve. It does not launch provider work and does not claim score movement. It records ten dry-run exact-eval cells: five side-info variants times `[contest-CPU]` and `[contest-CUDA]`.

## Status

- Source commit: `649f44f7179216cc372fd18ad64f8a55b3d0aad0`
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
| `zero` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 34373 | `b444cc91f102c9807a865ed59f182ca5c83f3239a49ec2aa400b497d7dea37a3` | `fee398bbb3a8dc5e769aacab44625a98e5940ec0b402063ea10129ad2fe8863e` | `e9470d2878b21e4eee289928846b7530d6137de2440496b220681a912eacf2fe` |
| `zero` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 34373 | `b444cc91f102c9807a865ed59f182ca5c83f3239a49ec2aa400b497d7dea37a3` | `a01d9f49bbd86c27613efa5f347bbed0c687e143c174a8225d4bfed55f38f816` | `1d9aaeadd3ff2785eb6c0fb4cd5afdc8ddbab841edea0150fe4296a0ecb96219` |
| `random_lsb` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 38681 | `ccce77aaf1907d6e70d8cba498261708b241ac7d78a9bf22978aa459cb6b7fd1` | `dca7754cad178dcb44820f818db9bbbf30687b76786ba0454a75a8cf0cd8d69c` | `2df930cccaad25e9747bb9b76c01e31a13c48a0695cfc3b4dc45013ae0dddf6c` |
| `random_lsb` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 38681 | `ccce77aaf1907d6e70d8cba498261708b241ac7d78a9bf22978aa459cb6b7fd1` | `bb731c4cff7928edd590de7f37812c97409a5d9064fd3e3d9545f524b6cea6bb` | `e96cec6dd4b29e13dc94360b14bf15d248cb5c67f0e662bb451c4dbd244a4d70` |
| `shuffled` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 43284 | `c235e5cb91f4122c3bb642354b424dd21f8b37cc0d2ac7b3e03c0b84dcc49bc3` | `27b5d520edf8e3a464f39f7d8cd07ba3be895754acdc020b97a517f32ff4cb71` | `7c5177c75b42eb456439c0e70fd74477df647c996305150d3d03f2194f621f0a` |
| `shuffled` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 43284 | `c235e5cb91f4122c3bb642354b424dd21f8b37cc0d2ac7b3e03c0b84dcc49bc3` | `2abbb5612bf28468354d0f12c74364fc7f43658279fb6832c5e5f06a30142364` | `79162971a1cc2bf76dd0ca95063a74d9a936ef00c5072e287180a771d7a0ce6d` |
| `trained` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 43323 | `f08299c5e77908eeeb82cc9948e530fd5a790894902a726ebd8a258596b4bf1a` | `ba5da288d8297589de1c8b156cde53c95d7cbac520630e3b7a05fb8a05991a19` | `8d3fa198e3cb87ac1fd79451a4aa0c1be220527a9b2292eb0064fa54dd2db211` |
| `trained` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 43323 | `f08299c5e77908eeeb82cc9948e530fd5a790894902a726ebd8a258596b4bf1a` | `e14bcd4760fbce291d79f5db0e50621f1064ac7e1ef82d2e56885bc69d03d28d` | `02513c83972182e2fb8e98ce9ba2f45b1fafa918bf5a1b7330c6e2ffe538798d` |
| `ablated` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 42419 | `ec343265899859495fca1d2874c1b85d211210ea13a2de5bb7e52c9816ba6b39` | `65a19e50570efb8aa25bdfb4685fe5ad3b84d6caa93f1c4b59f0e7303f0bc5eb` | `a6115253b157b518a4b529668f2a966a5aacc201a1e80f11901983868ce94ac2` |
| `ablated` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 42419 | `ec343265899859495fca1d2874c1b85d211210ea13a2de5bb7e52c9816ba6b39` | `1b256a884140d25753e73e123e3e80d82f144bad062548a0bbc3fe0449c40c99` | `7a75063636de6bd533b5578ecdf251f78d79e719cf4a596e4850ee3d61875deb` |

## Axis Invariants Verified

- CPU cells are `exact_cpu_eval`, adjudicate with `required_device=cpu`, contain `--device cpu`, omit `--device cuda`, and omit `INFLATE_REQUIRE_CUDA=1`.
- CUDA cells are `exact_cuda_eval`, adjudicate with `required_device=cuda`, contain `--device cuda`, and retain `INFLATE_REQUIRE_CUDA=1`.
- All ten `dry_run.stderr` files are zero bytes when the plan is ready.
- Queue metadata carries the expected `axis`, `variant`, `pair_group_id`, `archive_sha256`, and source dispatch-plan pointer for every cell.

## Reactivation Criteria

Before any non-dry-run submit, configure Lightning identity/workspace variables, stage a fresh source manifest, run the remote preflight, and claim the per-axis lane. The effect-curve predicate remains blocked until all ten cells are harvested and the aggregate side-info effect artifact passes its paired-axis validation.
