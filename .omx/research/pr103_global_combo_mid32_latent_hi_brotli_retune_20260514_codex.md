# PR103 Global Combo Mid32 + Latent-Hi Brotli Retune Candidate - 2026-05-14

## Classification

- lane_id: `pr103_global_combo_mid32_latent_hi_brotli_retune`
- status: byte-closed local packet candidate; not dispatched
- score_claim: false
- scorers_invoked: false
- axis_label: local CPU inflate-shell parity only; exact CUDA missing
- research_only: true until an operator-approved exact-CUDA dispatch claim exists

## Candidate Packet

- source archive: `experiments/results/public_pr_intake_full/public_pr103_intake_20260505_auto/archive.zip`
- source bytes: 178223
- source SHA-256: `31881b2d23d027e6619f2d8df2fe35d4d207d08882ec673d6c1b7ff119f18c30`
- packet archive: `experiments/results/pr103_global_combo_mid32_latent_hi_brotli_retune_20260514_codex/packet/archive.zip`
- packet bytes: 178205
- packet SHA-256: `7d1e46331a048abeeb40a59e95eb87970bc93f070a2f51f3bf9af8e107ec2c10`
- packet member: stored `x`, 178105 bytes, SHA-256 `c57dbca04d4aa31db32f842f5ce2de482eea3d0b9d550abf8bc27c06d76e170f`
- charged archive delta versus public PR103 source: -18 bytes
- retune-only delta versus the prior mid32+latent-hi materialization path: -2 bytes

## Charged Section Accounting

- `ac_histograms_brotli`: 895 -> 880 bytes (-15)
- `latent_hi_histogram_brotli`: 15 -> 13 bytes (-2), retuned to Brotli quality 3 / lgwin 16
- `latent_low_bytes_brotli`: 15537 -> 15536 bytes (-1), retuned to Brotli quality 4 / lgwin 16
- `merged_range_coded_weights_and_hi_latents`: 153856 -> 153856 bytes (0), changed SHA only due histogram model; semantic stream parity passed
- net archive delta: -18 bytes after ZIP/member accounting

## Verification Artifacts

- materialization manifest: `experiments/results/pr103_global_combo_mid32_latent_hi_brotli_retune_20260514_codex/manifest.json`
- runtime adapter manifest: `experiments/results/pr103_global_combo_mid32_latent_hi_brotli_retune_20260514_codex/runtime_adapter_manifest.json`
- packet manifest: `experiments/results/pr103_global_combo_mid32_latent_hi_brotli_retune_20260514_codex/packet_manifest.json`
- local full inflate-shell parity: `experiments/results/pr103_global_combo_mid32_latent_hi_brotli_retune_20260514_codex/inflate_shell_output_parity_full_cpu.json`
- strict compliance report: `experiments/results/pr103_global_combo_mid32_latent_hi_brotli_retune_20260514_codex/pre_submission_compliance.strict.json`

Local shell parity passed through `inflate.sh archive_dir output_dir file_list` for `0.mkv`.
Source and candidate both emitted `0.raw` with 3662409600 bytes and SHA-256
`074f834f14ba4611f9358bb0a3f8e729bb43e4ea673be23e2acf85e7448dd1e5`.

Strict pre-submission compliance passed 38 checks against the packet archive.

## Dispatch And Blockers

No remote dispatch was attempted. The packet manifest still blocks exact-eval
readiness on:

- `lane_dispatch_claim_missing`
- `exact_cuda_auth_eval_missing`

This is rate-positive only under exact same-output local shell parity. It is not
a public-frontier score claim until a claimed exact CUDA artifact exists.

## Six-Hook Disposition

- sensitivity map: charged byte sections and byte deltas are recorded in `manifest.json`
- Pareto constraint: rate-only transform with local full output parity; exact CUDA still required
- bit allocator hook: `byte_accounting` records archive/member deltas and estimated rate-only term with `estimate_is_score_claim=false`
- autopilot dispatch hook: `packet_manifest.json` carries explicit readiness blockers and refuses exact eval readiness
- continual-learning posterior: update as a +18 charged-byte local parity candidate, not as score movement
- probe-disambiguator: Brotli retune is opt-in per section; default materialization path remains unchanged
