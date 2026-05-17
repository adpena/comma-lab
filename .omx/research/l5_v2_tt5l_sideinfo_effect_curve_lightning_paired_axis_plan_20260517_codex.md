# L5 v2 TT5L side-info effect curve Lightning paired-axis plan

Generated: 2026-05-17T09:29:14Z

This memo supersedes the CUDA-only Lightning alternate-provider memo for the TT5L side-info effect curve. It does not launch provider work and does not claim score movement. It records ten dry-run exact-eval cells: five side-info variants times `[contest-CPU]` and `[contest-CUDA]`.

## Status

- Source commit: `fe65c9ec71ef568ea42f4c0edc38b34408489f83`
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
| `zero` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 34373 | `b444cc91f102c9807a865ed59f182ca5c83f3239a49ec2aa400b497d7dea37a3` | `fee398bbb3a8dc5e769aacab44625a98e5940ec0b402063ea10129ad2fe8863e` | `6188893c7d7f6f8a8efa04da91c976fd29bc68812f923f3eb71cbfd14e51850b` |
| `zero` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 34373 | `b444cc91f102c9807a865ed59f182ca5c83f3239a49ec2aa400b497d7dea37a3` | `a01d9f49bbd86c27613efa5f347bbed0c687e143c174a8225d4bfed55f38f816` | `8688cea76bc12657216d0e8d77bbf047e4b2d551c47f214ca43554e8c1481fdb` |
| `random_lsb` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 38681 | `ccce77aaf1907d6e70d8cba498261708b241ac7d78a9bf22978aa459cb6b7fd1` | `dca7754cad178dcb44820f818db9bbbf30687b76786ba0454a75a8cf0cd8d69c` | `7be8417f672baaf3b2619b765aa728e9041d0f9634d6fa71aadbeab86d083082` |
| `random_lsb` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 38681 | `ccce77aaf1907d6e70d8cba498261708b241ac7d78a9bf22978aa459cb6b7fd1` | `bb731c4cff7928edd590de7f37812c97409a5d9064fd3e3d9545f524b6cea6bb` | `75716f7b8ed3dabbe12041bd83ca6661976058b869d0c6c034c67ae7db4165e0` |
| `shuffled` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 43284 | `c235e5cb91f4122c3bb642354b424dd21f8b37cc0d2ac7b3e03c0b84dcc49bc3` | `27b5d520edf8e3a464f39f7d8cd07ba3be895754acdc020b97a517f32ff4cb71` | `4af817f96e27b2cc73f50979e1da2243b279e467ac7a87c0d197db53fd830713` |
| `shuffled` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 43284 | `c235e5cb91f4122c3bb642354b424dd21f8b37cc0d2ac7b3e03c0b84dcc49bc3` | `2abbb5612bf28468354d0f12c74364fc7f43658279fb6832c5e5f06a30142364` | `f2472a75ec43c5ffdf6544c33f46598bcf26a3e3559a02f01e6824f1f64e2904` |
| `trained` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 43323 | `f08299c5e77908eeeb82cc9948e530fd5a790894902a726ebd8a258596b4bf1a` | `ba5da288d8297589de1c8b156cde53c95d7cbac520630e3b7a05fb8a05991a19` | `f16d0b493ea96a6fa53b33533086a71966fbb3a237413e7378159e5614c9a65f` |
| `trained` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 43323 | `f08299c5e77908eeeb82cc9948e530fd5a790894902a726ebd8a258596b4bf1a` | `e14bcd4760fbce291d79f5db0e50621f1064ac7e1ef82d2e56885bc69d03d28d` | `24dc1a35e53c02472351aa80435994cc42ee591372531198d936ddecbb357516` |
| `ablated` | `contest_cpu` | `exact_cpu_eval` | `cpu` | 42419 | `ec343265899859495fca1d2874c1b85d211210ea13a2de5bb7e52c9816ba6b39` | `65a19e50570efb8aa25bdfb4685fe5ad3b84d6caa93f1c4b59f0e7303f0bc5eb` | `54816eafce74955b370fae29bc887268cd427d3ad89dd4a9abbefd7540b3dd04` |
| `ablated` | `contest_cuda` | `exact_cuda_eval` | `cuda` | 42419 | `ec343265899859495fca1d2874c1b85d211210ea13a2de5bb7e52c9816ba6b39` | `1b256a884140d25753e73e123e3e80d82f144bad062548a0bbc3fe0449c40c99` | `e8381582369a98fe4f58fb5ee7c1d6e9d71175bcc939894b1c6fb6c9d74e5c08` |

## Axis Invariants Verified

- CPU cells are `exact_cpu_eval`, adjudicate with `required_device=cpu`, contain `--device cpu`, omit `--device cuda`, and omit `INFLATE_REQUIRE_CUDA=1`.
- CUDA cells are `exact_cuda_eval`, adjudicate with `required_device=cuda`, contain `--device cuda`, and retain `INFLATE_REQUIRE_CUDA=1`.
- All ten `dry_run.stderr` files are zero bytes when the plan is ready.
- Queue metadata carries the expected `axis`, `variant`, `pair_group_id`, `archive_sha256`, and source dispatch-plan pointer for every cell.

## Reactivation Criteria

Before any non-dry-run submit, configure Lightning identity/workspace variables, stage a fresh source manifest, run the remote preflight, and claim the per-axis lane. The effect-curve predicate remains blocked until all ten cells are harvested and the aggregate side-info effect artifact passes its paired-axis validation.
