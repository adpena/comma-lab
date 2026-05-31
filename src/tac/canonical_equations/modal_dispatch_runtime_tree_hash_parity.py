# SPDX-License-Identifier: MIT
"""Canonical equation: Modal dispatch runtime-tree-hash LOCAL vs worker parity (v1).

Builder for
``modal_dispatch_runtime_tree_hash_local_vs_worker_parity_v1`` —
the formal invariant *the LOCAL projector ``runtime_tree_sha256`` computed by
``tools/dispatch_modal_paired_auth_eval.py::_modal_uploaded_runtime_hashes_for_axis``
MUST equal the Modal worker actual ``runtime_tree_sha256`` computed by
``experiments/contest_auth_eval.py::_runtime_dependency_manifest`` on the
extracted ``--submission-dir`` tree*. Sister of canonical equations
#146 (contest-compliant inflate runtime template) + #205 (canonical
``select_inflate_device``) at the dispatch-infrastructure surface.

Per CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable: the
LOCAL projector and Modal worker MUST agree on every component that feeds
the ``_canonical_json_sha256`` of the ``tree_payload`` — file list +
per-file ``repo_relative_path`` + ``repo_local_tac_import_manifest`` +
``external_dependency_roots`` + ``upstream_evaluate_py``. Drift in ANY
component cascades into a runtime_tree_sha256 mismatch that fails dispatch
pre-validation at the ``--expected-runtime-tree-sha256`` check
(``experiments/contest_auth_eval.py::_validate_expected_runtime_tree``).

Empirical anchor (the bug class this equation extincts):

  PR111-candidate paired-CUDA RATIFICATION 4× DEFER 2026-05-28
  (commit ``6bc74e074``; landing memo
  ``.omx/research/pr111_composite_paired_cuda_ratification_infrastructure_deferred_landed_20260528.md``).
  Composite NSCS06 v8 chroma_lut + Compound C heterogeneous bit (sha
  ``dfff1358638ef7f7``, 1,917,982 B) paired-CUDA + paired-CPU dispatch
  attempts:
    Attempt 1 CUDA fc-01KSQVET4YGSWHJB2PTBHFQ8R2 rc=1
      expected efa31c12... actual 1e9bf123...
    Attempt 1 CPU  fc-01KSQVFGNSJ4VYFKKSA9PDV0AE rc=1
      expected c886528c... actual 60256159...
    Attempt 2 CUDA fc-01KSQVM7T6D2YN4Z40D8R1RH29 rc=1 (--expected ... auto)
    Attempt 2 CPU  fc-01KSQVMYN7HQG8V6KH4NGZE6DG rc=1 (--expected ... auto)
  Cumulative paid Modal spend: ~$0.06; cumulative score evidence: ZERO.
  Predicted [contest-CPU] band [0.163, 0.167] per Compound F first-order
  Volterra α=0.85; UNTESTED EMPIRICALLY until the infrastructure fix.

Root cause: ``experiments/contest_auth_eval.py::_module_exists`` and
``::_module_paths`` used ``Path.exists()`` which case-folds on macOS
HFS+/APFS / Windows NTFS. The LOCAL projector therefore picked up a
phantom ``tac.dykstra_pareto_solver.Polytope`` (capital-P) module via
``polytope.py`` case-fold match; Linux Modal worker (case-sensitive)
correctly returned False. ``module_count`` diverged 37 (LOCAL) vs 36
(WORKER); ``file_count`` diverged 40 vs 39; the ``repo_local_tac_import_manifest``
dict-payload diverged, cascading into a ``runtime_tree_sha256`` mismatch.

Canonical fix: ``_path_exists_case_sensitive`` helper in
``experiments/contest_auth_eval.py`` walks ``parent.iterdir()`` and
requires exact basename match for every component. ``_module_exists`` and
``_module_paths`` both route through it. Empirical proof: post-fix the
LOCAL projector produces:
  CUDA: ``1e9bf123e8eac353591c2fa57af96d3eb330855d34d375faa886dc9c32026afb``
  CPU:  ``60256159c7d65405fca5139d8ffd4a81a8444c13ed0dd804fd0aecb6097b55e6``
= EXACT MATCH with the Modal worker actual hashes (composite sha
``dfff1358638ef7f7``).

Producer/consumer wiring per CLAUDE.md "Subagent coherence-by-default"
6-hook discipline:

- Producer #1: ``tools/dispatch_modal_paired_auth_eval.py::_modal_uploaded_runtime_hashes_for_axis``
  (LOCAL projector — emits expected hashes per axis)
- Producer #2: ``experiments/contest_auth_eval.py::_runtime_dependency_manifest``
  (canonical hash computer — used by both LOCAL projector path and
  Modal worker live computation)
- Consumer #1: ``experiments/modal_auth_eval.py::_validate_uploaded_runtime_tree_expectation``
  (Modal CUDA worker pre-validation gate)
- Consumer #2: ``experiments/modal_auth_eval_cpu.py::_validate_uploaded_runtime_tree_expectation``
  (Modal CPU worker pre-validation gate)
- Consumer #3: ``experiments/contest_auth_eval.py::_validate_expected_runtime_tree``
  (post-extraction subprocess gate inside the worker that ultimately
  raises ``RuntimeError("inflate runtime tree hash mismatch")``)

Sister catalog #s: #146 (contest-compliant inflate runtime template) +
#205 (canonical ``select_inflate_device``) + #229 (premise verification
before edit) + #270 (canonical dispatch optimization protocol) + #287
(empirical-claim-evidence-tag) + #307 (paradigm-vs-implementation
falsification) + #313 (probe-outcomes ledger) + #323 (canonical Provenance
umbrella) + #344 (canonical equations registry) + #348 (retroactive sweep
for new gate) + #377 (NEW STRICT gate: refuses
``experiments/contest_auth_eval.py`` state that drops the canonical
``_path_exists_case_sensitive`` helper).
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)


def build_modal_dispatch_runtime_tree_hash_local_vs_worker_parity_v1() -> CanonicalEquation:
    """Equation: LOCAL projector hash == Modal worker actual hash (per axis).

    Mathematical invariant:

        ``_modal_uploaded_runtime_hashes_for_axis(submission_dir, inflate_sh_rel,
        remote_submission_dir=<axis>)['runtime_tree_sha256']``
        ==
        ``_runtime_dependency_manifest(<extracted submission_dir>, upstream_dir,
        repo_root=<worker_repo>)['runtime_tree_sha256']``

    where the equality holds when ``_module_exists`` / ``_module_paths``
    perform CASE-SENSITIVE filesystem checks even on macOS / Windows. Per
    CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable:
    drift in this invariant produces a deterministic dispatch
    pre-validation failure with cumulative paid spend at $0 score
    evidence.

    The PR111 anchor is the canonical first empirical observation: pre-fix,
    LOCAL projected ``efa31c12... / c886528c...`` and worker actual
    ``1e9bf123... / 60256159...`` diverged for the composite NSCS06 v8 +
    Compound C submission (sha ``dfff1358638ef7f7``).
    """
    anchor = EmpiricalAnchor(
        anchor_id=(
            "pr111_composite_nscs06_v8_plus_compound_c_local_vs_worker_runtime_tree_hash_parity_20260528"
        ),
        measurement_utc="2026-05-28T18:30:00Z",
        inputs={
            "composite_archive_sha256": (
                "dfff1358638ef7f7bad4596958cddb62215ed06c5b850a8501e3ad42a2c13402"
            ),
            "composite_archive_bytes": 1917982,
            "submission_dir_path": (
                "experiments/results/composite_nscs06_v8_plus_compound_c_pr111_candidate_20260528/submission"
            ),
            "inflate_sh_rel": "inflate.sh",
            "n_dispatches_failed_pre_fix": 4,
            "cumulative_paid_modal_spend_usd_pre_fix": 0.06,
            "measurement_axis": "[infrastructure-parity]",
            "measurement_hardware": "macos_local_vs_linux_x86_64_t4_modal",
        },
        predicted_output={
            "invariant_form": (
                "local_projector_runtime_tree_sha256 == modal_worker_actual_runtime_tree_sha256"
            ),
            "predicted_outcome": "EQUAL (deterministic)",
        },
        empirical_output={
            "pre_fix_cuda_projector_hash_first_16": "efa31c12b82ebf55",
            "pre_fix_cuda_worker_actual_hash_first_16": "1e9bf123e8eac353",
            "pre_fix_cpu_projector_hash_first_16": "c886528ccc311ae2",
            "pre_fix_cpu_worker_actual_hash_first_16": "60256159c7d65405",
            "pre_fix_invariant_holds": False,
            "post_fix_cuda_projector_hash_first_16": "1e9bf123e8eac353",
            "post_fix_cpu_projector_hash_first_16": "60256159c7d65405",
            "post_fix_invariant_holds": True,
            "root_cause": (
                "experiments/contest_auth_eval.py::_module_exists used "
                "Path.exists() which case-folds on macOS HFS+/APFS; LOCAL "
                "projector picked up phantom tac.dykstra_pareto_solver.Polytope "
                "(capital P) via polytope.py case-fold; Linux Modal worker "
                "correctly returned False. module_count diverged 37 vs 36."
            ),
            "canonical_fix": (
                "_path_exists_case_sensitive helper walks parent.iterdir() "
                "and requires exact basename match; _module_exists and "
                "_module_paths both route through it."
            ),
            "fix_commit": "pending in landing batch",
            "verification_artifact": (
                "src/tac/tests/test_modal_runtime_tree_hash_local_vs_worker_parity.py "
                "(16/16 tests pass)"
            ),
        },
        residual=0.0,  # Post-fix invariant holds exactly; no normalized residual.
        source_artifact=(
            ".omx/research/pr111_composite_paired_cuda_ratification_infrastructure_deferred_landed_20260528.md"
        ),
        measurement_method=(
            "empirical_paired_dispatch_4x_failure_then_local_simulation_verification_then_canonical_fix"
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=(
                ".omx/research/pr111_composite_paired_cuda_ratification_infrastructure_deferred_landed_20260528.md"
            ),
            reactivation_criteria=(
                "future Modal dispatch failure with rc=1 + 'inflate runtime tree hash mismatch' "
                "must traceback to this equation's invariant violation; the canonical fix has "
                "extincted the macOS case-fold sub-class structurally."
            ),
            measurement_axis="[infrastructure-parity]",
            hardware_substrate="macos_local_vs_linux_x86_64_t4_modal",
        ),
    )
    return CanonicalEquation(
        equation_id=(
            "modal_dispatch_runtime_tree_hash_local_vs_worker_parity_v1"
        ),
        name=(
            "Modal dispatch runtime-tree-hash LOCAL projector vs worker parity invariant"
        ),
        one_line_summary=(
            "LOCAL projector runtime_tree_sha256 MUST equal Modal worker actual "
            "runtime_tree_sha256 for paired dispatch pre-validation to pass."
        ),
        latex_form=(
            r"\forall \text{axis} \in \{\text{cuda},\text{cpu}\}, "
            r"\forall \text{submission\_dir} \in \mathcal{S}: "
            r"H_{\text{local-proj}}(\text{submission\_dir}, \text{axis}) = "
            r"H_{\text{modal-worker}}(\text{extract}(\text{submission\_dir}), \text{axis})"
        ),
        python_callable_module_path=(
            "tools.dispatch_modal_paired_auth_eval:_modal_uploaded_runtime_hashes_for_axis"
        ),
        domain_of_validity={
            "axes": ["contest_cuda", "contest_cpu"],
            "remote_submission_dirs": [
                "/tmp/modal_auth_eval/submission_dir",
                "/tmp/modal_auth_eval_cpu/submission_dir",
            ],
            "supported_local_filesystems": [
                "macOS_HFS_plus",
                "macOS_APFS",
                "Linux_ext4",
                "Linux_xfs",
                "Windows_NTFS",
            ],
            "supported_worker_filesystem": "Linux_modal_overlay_fs_case_sensitive",
            "extraction_root_pattern": "/tmp/modal_auth_eval*/submission_dir",
        },
        units_in={
            "submission_dir": "absolute_filesystem_path",
            "inflate_sh_rel": "relative_path_inside_submission_dir",
            "axis": "enum_contest_cuda_contest_cpu",
        },
        units_out={
            "runtime_tree_sha256": "hex_string_64_chars",
            "runtime_content_tree_sha256": "hex_string_64_chars",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "pr111_composite_nscs06_v8_plus_compound_c_post_fix": 0.0,
        },
        last_calibration_utc="2026-05-28T18:30:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "experiments.modal_auth_eval",
            "experiments.modal_auth_eval_cpu",
            "experiments.contest_auth_eval",
            "tools.dispatch_modal_paired_auth_eval",
        ),
        canonical_producers=(
            "experiments/contest_auth_eval.py",
            "src/tac/deploy/modal/auth_eval.py",
            "tools/dispatch_modal_paired_auth_eval.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="modal_dispatch_runtime_tree_hash_local_vs_worker_parity.v1",
            inputs_sha256=(
                "dfff1358638ef7f7bad4596958cddb62215ed06c5b850a8501e3ad42a2c13402"
            ),
            measurement_axis="[infrastructure-parity]",
            hardware_substrate="macos_local_vs_linux_x86_64_t4_modal",
        ),
    )


__all__ = ["build_modal_dispatch_runtime_tree_hash_local_vs_worker_parity_v1"]
