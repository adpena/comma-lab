# L5 v2 TT5L side-info effect curve Lightning paired-axis plan

Generated: 2026-05-17T07:05:00Z

This memo supersedes the CUDA-only Lightning alternate-provider memo for the TT5L side-info effect curve. It does not launch provider work and does not claim score movement. It records ten dry-run exact-eval cells: five side-info variants times `[contest-CPU]` and `[contest-CUDA]`.

## Status

- Source commit: `a23606c9b1a81aacf154d6c2c12aca5408203d46`
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
| `zero` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 34373 | `b444cc91f102c9807a865ed59f182ca5c83f3239a49ec2aa400b497d7dea37a3` | `fee398bbb3a8dc5e769aacab44625a98e5940ec0b402063ea10129ad2fe8863e` | `e8786131705410d61cfcdfd29df2917333c14206a4308e59d3076c0b269419c2` |
| `zero` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 34373 | `b444cc91f102c9807a865ed59f182ca5c83f3239a49ec2aa400b497d7dea37a3` | `a01d9f49bbd86c27613efa5f347bbed0c687e143c174a8225d4bfed55f38f816` | `cb6b934878a48e4977da68bf6cdb590bab5cce03a2e8039cd2461d3efde57f8e` |
| `random_lsb` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 38681 | `ccce77aaf1907d6e70d8cba498261708b241ac7d78a9bf22978aa459cb6b7fd1` | `dca7754cad178dcb44820f818db9bbbf30687b76786ba0454a75a8cf0cd8d69c` | `1468c64491db78c3db6cf50b6f642998f23074891e43f12dbcb646b3d63ed1e7` |
| `random_lsb` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 38681 | `ccce77aaf1907d6e70d8cba498261708b241ac7d78a9bf22978aa459cb6b7fd1` | `bb731c4cff7928edd590de7f37812c97409a5d9064fd3e3d9545f524b6cea6bb` | `9edb07c37d47366918e9ed588f4efa612ed87253c794a5c92d5ad3a42957c770` |
| `shuffled` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 43284 | `c235e5cb91f4122c3bb642354b424dd21f8b37cc0d2ac7b3e03c0b84dcc49bc3` | `27b5d520edf8e3a464f39f7d8cd07ba3be895754acdc020b97a517f32ff4cb71` | `d2545f79dcfcba248e39a8221d1cc32bba1a357f060c2111cccb8de31a72c14a` |
| `shuffled` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 43284 | `c235e5cb91f4122c3bb642354b424dd21f8b37cc0d2ac7b3e03c0b84dcc49bc3` | `2abbb5612bf28468354d0f12c74364fc7f43658279fb6832c5e5f06a30142364` | `37f3ee218037c7f50b2955e5f935634a335036f22dbb7d60cf22e612d62ea0d3` |
| `trained` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 43323 | `f08299c5e77908eeeb82cc9948e530fd5a790894902a726ebd8a258596b4bf1a` | `ba5da288d8297589de1c8b156cde53c95d7cbac520630e3b7a05fb8a05991a19` | `9d23b1fa559cff1856492d4f1c1aabc757b7736fa1a290ee0bac3e6c4de4e3bb` |
| `trained` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 43323 | `f08299c5e77908eeeb82cc9948e530fd5a790894902a726ebd8a258596b4bf1a` | `e14bcd4760fbce291d79f5db0e50621f1064ac7e1ef82d2e56885bc69d03d28d` | `aeeed227c4bc48781e15907557e4b38ed842bcdc492b445c63d6f1cf31377ee6` |
| `ablated` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 42419 | `ec343265899859495fca1d2874c1b85d211210ea13a2de5bb7e52c9816ba6b39` | `65a19e50570efb8aa25bdfb4685fe5ad3b84d6caa93f1c4b59f0e7303f0bc5eb` | `6f68c8a46fb08b117723587c02f51bf90e0998500e76573a09597e0256af6874` |
| `ablated` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 42419 | `ec343265899859495fca1d2874c1b85d211210ea13a2de5bb7e52c9816ba6b39` | `1b256a884140d25753e73e123e3e80d82f144bad062548a0bbc3fe0449c40c99` | `e67ce52c7053318df146b17d98af24bb0a73d72e3256fb2feb40cf2472a6b60c` |

## Axis Invariants Verified

- CPU cells are `exact_cpu_eval`, adjudicate with `required_device=cpu`, contain `--device cpu`, omit `--device cuda`, and omit `INFLATE_REQUIRE_CUDA=1`.
- CUDA cells are `exact_cuda_eval`, adjudicate with `required_device=cuda`, contain `--device cuda`, and retain `INFLATE_REQUIRE_CUDA=1`.
- All ten `dry_run.stderr` files are zero bytes when the plan is ready.
- Queue metadata carries the expected `axis`, `variant`, `pair_group_id`, `archive_sha256`, and source dispatch-plan pointer for every cell.

## Reactivation Criteria

Before any non-dry-run submit, configure Lightning identity/workspace variables, stage a fresh source manifest, run the remote preflight, and claim the per-axis lane. The effect-curve predicate remains blocked until all ten cells are harvested and the aggregate side-info effect artifact passes its paired-axis validation.
