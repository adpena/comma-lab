# Codex Findings: Z8 RD Waterfill Rate Takeover

UTC: 2026-05-31T20:33:19Z

## Finding

The `.omx` Z8/hierarchical-PC memos agree on one current binding fact: the
architecture is distortion-faithful enough to keep working, but the archive is
rate-dominated. The actionable Z8 takeover item is not a new entropy coder. The
range-coder surface is already a local optimum at the measured subband modes;
the missing executable control is the per-subband delta operating point feeding
the existing joint P18/P19 coefficient actuator.

## Change

- Hardened `per_subband_rd_waterfill_solver` so fixed-lambda choices use the
  correct local objective `n_i * (D_i + lambda * R_i)` without size-biased
  lambda drift.
- Added exact finite multiple-choice search for hard byte-budget and
  weighted-MSE ceilings over nondominated measured RD points. The live six
  subband Z8 report now solves exactly instead of hoping a hull/bisection point
  happens to satisfy the contest acceptance constraint.
- Kept the output as the existing actuator input:
  `entropy_detail_quantization_steps` keyed
  `frame_{0,1}_details:level:{lh,hl,hh}`.
- Made incomplete keep-raw maps fail closed by default because the current
  materializer requires a step for every detail subband when per-subband
  storage quantization is enabled.
- Extended `tools/build_z8_entropy_delta_schedule.py` with
  `--strategy rd-waterfill` while preserving the legacy max-subband-MSE mode.
- Added `comma_lab.local_exact_auth_gate` plus
  `tools/gate_local_candidate_for_exact_auth.py` so MLX/local replay output can
  recommend exact auth only after a local advisory win, while keeping every
  false-authority field false.

## Verified Against Roadmap

Relevant `.omx` surfaces inspected:

- `z8_per_subband_rd_waterfill_solver_landed_20260531.md`
- `council_grand_council_negative_findings_extreme_rigor_audit_20260531.md`
- `z8_m12a_modal_t4_l2_long_training_recipe_authoring_per_catalog_325_symposium_proceed_with_revisions_landed_20260530.md`
- `codex_findings_z8_true_p19_codec_runner_integration_20260531T195828Z_codex.md`
- `codex_findings_z8_true_p19_variational_contract_alignment_20260531T200645Z_codex.md`

The code path now matches the mandate: rate-first Z8 work, exact/full-video
replay before score authority, true P19 custody preserved, no reactivation
theater around a better entropy coder.

## Orphan-Signal Takeover

Meitner and Descartes' read-only `.omx` passes found the next codec-adjacent
work should stay in this order:

1. Regenerate the full 600-pair true-P19 bundle, run strict codec search,
   materialize the archive, and benchmark `inflate.sh`. This is the direct path
   from scorer-sensitive gradients to byte reduction.
2. Feed the live measured P18/P19 surface into the Z8 coefficient deadzone
   materializer. The current actuator can rewrite bytes; the missing step is
   using the full measured surface instead of toy/null weights.
3. Add a Z8 in-loop archive replay selector: export `Z8HPC1`, parse, inflate or
   replay, emit archive SHA and blocker labels, and select checkpoints by the
   archive path instead of proxy loss.
4. Add `z8_hpc1` archive-family adapters so wavelet, Wyner-Ziv, metadata,
   header, and safe coder-boundary transforms can enter the shared
   materializer/receiver-proof contract.
5. Compose P11 selector payloads on top of P18/P19 coefficient waterfill,
   keyed by pair, region, and subband, with entropy-coded selector bytes.
6. Promote Z8 entropy/range coding from probe output into export contract only
   when dependency closure, receiver proof, and inflate benchmark are present.
7. Keep null-byte codebook, section-wise codec portfolio, richer Wyner-Ziv
   side-info, scorer-region cache handoff, direct-transform Mamba/Dreamer codec
   MVP, and Z8 artifact-consumer work as executable backlog, not prose.

## Verification

- `ruff check` on the solver, test, and schedule CLI: passed.
- Focused solver/schedule/gate tests: 43 passed.
- `tools/build_z8_entropy_delta_schedule.py --strategy rd-waterfill
  --max-weighted-mse 5e-5` emitted a v2 materializer-ready schedule with 12
  actuator keys and no blockers.
- The emitted v2 schedule materialized through the byte-closed Z8 actuator:
  archive ZIP `df32840163a48b632f0e5d171a3caee3910e56f94dbce86fb236bc4e038521bb`,
  24,573,973 bytes; raw candidate bin
  `a725297c92f164857c6e76f04bd28e268e06888c86b781610ed129d3e77601e3`,
  24,475,266 bytes. Receiver proof executed; exact-auth dispatch stayed false
  pending MLX/local scorer replay.

## Next

Run full-video MLX replay on the materialized candidate, CPU replay only if MLX
clears the local gate, and exact CPU/CUDA only after the local gate recommends a
claim-and-dispatch path. In parallel, lower the full-archive quantized-detail
packer hotspot; this materialization is CPU-bound enough to justify vectorized
or native Rust work.
