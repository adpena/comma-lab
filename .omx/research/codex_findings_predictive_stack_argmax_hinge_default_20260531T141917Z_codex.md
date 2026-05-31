# Codex Findings: Predictive Stack Argmax-Hinge Default

## Scope

Adversarial review target: predictive hierarchical coding stack with DreamerV3
RSSM, Z7 Mamba-2, and Z8 hierarchical predictive coding under MLX-local
training, archive/runtime custody, and false-authority contest discipline.

## Finding

The 2026-05-31 boundary-loss research selected the Crammer-Singer multiclass
hinge as the mathematically faithful SegNet boundary objective because it is
zero exactly when the target class wins by margin. DreamerV3 exposed that
objective, but the default still used the softer `kl_t2` baseline. Z7 and Z8
full MLX trainers did not expose the objective at all, so the predictive stack
could keep training new candidates on an inferior soft objective unless the
operator manually remembered a one-off Dreamer flag.

This was a dangling-helper integration failure, not a research failure.

## Landing

Code now makes `boundary_argmax_hinge` the default SegNet distillation objective
for DreamerV3, Z7 Mamba-2, and Z8 full MLX training. `kl_t2` remains available
only as an explicit legacy baseline replay. Z7 also now threads the selected
objective, tau, hinge margin, and replay argv into both `0.bin` archive metadata
and the shared archive-bound runtime manifest, keeping MLX advisory while
preserving the training functional used to generate the candidate.

Follow-up adversarial review found two custody/provenance hazards and folded
them into the same landing:

- DreamerV3 partial archives now fail closed for exact-handoff custody by
  requiring the full 600-pair contest receiver byte contract until deterministic
  cycling or full-archive emission is implemented.
- Z8 M10 provenance now names the only proven pixel-consuming surface:
  Mallat wavelet archive bytes. Mamba/Dreamer/Wyner-Ziv sections remain
  parsed custody until byte-mutation receiver proofs show pixel effects.

## Authority

No score claim is made here. These rows remain `[macOS-MLX research-signal]`
until byte-closed candidate archives pass receiver proof and exact CPU/CUDA
authority. The change improves the default acquisition/training functional and
metadata custody; it does not promote any candidate by itself.

## Verification

- `ruff` on touched files
- `pytest src/tac/tests/test_predictive_stack_seg_objective_wiring.py -q`
- `pytest src/tac/tests/test_z7_mamba2_mlx_module_smoke.py::test_z7_mamba2_mlx_canonical_ssd_backend_uses_helper_and_exports_bridge -q`
- `pytest src/tac/substrates/dreamer_v3_rssm/tests/test_dreamer_v3_seg_distill_objective_flag.py::test_seg_distill_objective_default_is_boundary_argmax_hinge -q`
- `pytest src/tac/substrates/dreamer_v3_rssm/tests/test_basic.py::test_archive_bound_export_requires_contest_receiver_byte_contract -q`
- `pytest src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_archive_candidate_bridge.py -q`
- `pytest src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_inflate_mallat_wavelet_archive_consumption.py::{test_inflate_imports_mallat_wavelet_reconstruction_helpers,test_build_progress_m10_landed_status} -q`
- `pytest src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_m11_l1_macos_cpu_smoke.py::TestZ8M11BuildProgressConsistency::test_m11_predecessor_chain_canonical -q`
- `git diff --check`
- `tools/review_gate_hook.py`

## Remaining Work

Z8 still needs its MLX full-training export path promoted from advisory
TrainingArtifact metadata into byte-closed Z8HPC1 archive emission. Separately,
Mamba/Dreamer/Wyner-Ziv Z8 sections must earn pixel-consuming status through
byte-mutation receiver proofs or remain custody-only. DreamerV3 needs a real
600-pair or deterministic-cycling runtime bridge before partial MLX archives can
be exact-handoff eligible.
