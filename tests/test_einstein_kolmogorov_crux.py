from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tac.optimization.einstein_kolmogorov_crux import (
    RATE_SCORE_PER_BYTE,
    DSPSAState,
    admit_candidate,
    admit_seg_only_component_diagnostic,
    coordinate_candidates,
    dspsa_perturbation,
    marginal_beats_waterline,
    middle_point,
    project_theta,
    project_uint8,
    score,
    wang_corners,
    wang_dspsa_step,
)
from tac.witness_dsl import einstein_kolmogorov_crux_20260719 as crux_config_module
from tac.witness_dsl.einstein_kolmogorov_crux_20260719 import (
    EinsteinKolmogorovConfigError,
    EinsteinKolmogorovCruxConfig,
)


def _probe_module():
    path = Path(__file__).parents[1] / "tools" / "probe_einstein_kolmogorov_crux.py"
    spec = importlib.util.spec_from_file_location("einstein_kolmogorov_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(**changes: object) -> EinsteinKolmogorovCruxConfig:
    values: dict[str, object] = {
        "packet_path": "/inputs/payload.bin",
        "packet_sha256": "a" * 64,
        "gt_path": "/inputs/gt_n24.npz",
        "gt_sha256": "b" * 64,
        "segnet_path": "/inputs/upstream/models/segnet.safetensors",
        "segnet_sha256": "c" * 64,
        "upstream_path": "/inputs/upstream",
        "output_dir": str(Path(__file__).parents[1] / ".omx" / "research" / "test-run-a"),
        "pair_indices": (0, 1),
        "family": "dspsa",
        "seed": 7,
        "iterations": 2,
    }
    values.update(changes)
    return EinsteinKolmogorovCruxConfig(**values)  # type: ignore[arg-type]


def _fake_closure(probe) -> dict[str, object]:
    return {
        "schema": probe.CLOSURE_SCHEMA,
        "sources": {name: {"path": f"/source/{name}.py", "sha256": "d" * 64} for name in sorted(probe._SOURCE_NAMES)},
        "runtime": {
            "python": {
                "version": "3.test",
                "implementation": "CPython",
                "executable": "/python",
                "executable_sha256": "a" * 64,
                "cache_tag": "cpython-test",
            },
            "numpy": {"version": "test", "module_path": "/numpy/__init__.py", "module_sha256": "b" * 64},
            "torch": {"version": "test", "module_path": "/torch/__init__.py", "module_sha256": "c" * 64},
            "brotli": {"version": "test", "module_path": "/brotli.py", "module_sha256": "d" * 64},
            "safetensors": {
                "version": "test",
                "module_path": "/safetensors/__init__.py",
                "module_sha256": "e" * 64,
            },
            "platform": {
                "platform": "test",
                "system": "test",
                "release": "test",
                "version": "test",
                "machine": "test",
                "processor": "test",
            },
            "threads": {
                "torch_num_threads": 1,
                "torch_num_interop_threads": 1,
                "environment": dict.fromkeys(probe._THREAD_ENV_NAMES),
            },
        },
        "base_git_head": "e" * 40,
    }


def test_config_round_trip_has_stable_fingerprint() -> None:
    config = _config()
    restored = EinsteinKolmogorovCruxConfig.from_json(config.to_json())
    assert restored == config
    assert restored.fingerprint == config.fingerprint
    assert json.loads(config.to_json())["pair_indices"] == [0, 1]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"packet_path": "relative.bin"}, "absolute"),
        ({"family": "fourier"}, "unknown family"),
        ({"seed": -1}, "seed"),
        ({"pair_indices": (1, 0)}, "sorted"),
        ({"scorer_authority": "many_gpu_scorers"}, "singleton"),
        ({"iterations": 0}, "require iterations"),
        ({"checkpoint_every_pairs": 2}, "exactly 1"),
        ({"target_first_displacement": 2.0}, "frozen at exactly 1.0"),
        ({"label_min_run": 2}, "active only"),
        ({"pair_indices": (24,)}, "diagnostic-only"),
        ({"iterations": 33}, "iterations must be"),
        ({"output_dir": "/outputs/run-a"}, "diagnostic output_dir"),
    ],
)
def test_config_refuses_implicit_or_nondeterministic_contract(changes: dict[str, object], message: str) -> None:
    with pytest.raises(EinsteinKolmogorovConfigError, match=message):
        _config(**changes)


def test_config_refuses_unknown_json_key() -> None:
    payload = _config().to_dict()
    payload["invented_flag"] = True
    with pytest.raises(EinsteinKolmogorovConfigError, match="unknown"):
        EinsteinKolmogorovCruxConfig.from_dict(payload)


def test_coordinate_warm_start_requires_iterations() -> None:
    with pytest.raises(EinsteinKolmogorovConfigError, match="require iterations"):
        _config(family="coordinate_warm_start", iterations=0)


def test_label_run_simplify_has_typed_non_inert_control() -> None:
    config = _config(family="label_run_simplify", iterations=0, label_min_run=3)
    assert config.label_min_run == 3
    with pytest.raises(EinsteinKolmogorovConfigError, match="label_run_simplify"):
        _config(family="label_run_simplify", iterations=1, label_min_run=3)
    with pytest.raises(EinsteinKolmogorovConfigError, match="label_run_simplify"):
        _config(family="label_run_simplify", iterations=0, label_min_run=1)


def test_label_run_simplify_is_simultaneous_and_deterministic() -> None:
    import numpy as np

    probe = _probe_module()
    labels = np.asarray([[[1, 1, 2, 3, 3, 3, 4, 5, 5]]], dtype=np.uint8)
    expected = np.asarray([[[1, 1, 3, 3, 3, 3, 3, 5, 5]]], dtype=np.uint8)
    first = probe._simplify_short_horizontal_runs(labels, minimum_run=2)
    second = probe._simplify_short_horizontal_runs(labels, minimum_run=2)
    assert np.array_equal(first, expected)
    assert np.array_equal(second, expected)
    assert np.array_equal(labels, np.asarray([[[1, 1, 2, 3, 3, 3, 4, 5, 5]]], dtype=np.uint8))


def test_coordinate_warm_start_retains_the_config_bound_source_palette() -> None:
    import numpy as np

    probe = _probe_module()
    source = np.arange(2 * 5 * 3, dtype=np.uint8).reshape(2, 5, 3)
    labels = np.zeros((2, 1, 1), dtype=np.uint8)
    result = probe._palette_for_family(_config(family="coordinate_warm_start"), source, labels, gt=None, segnet=None)
    assert np.array_equal(result, source)
    assert result is not source


def test_coordinate_warm_start_projects_source_palette_to_narrow_bounds() -> None:
    import numpy as np

    probe = _probe_module()
    source = np.asarray([[[0, 100, 255]] * 5], dtype=np.uint8)
    labels = np.zeros((1, 1, 1), dtype=np.uint8)
    result = probe._palette_for_family(
        _config(family="coordinate_warm_start", pair_indices=(0,), lower_bound=101, upper_bound=103),
        source,
        labels,
        gt=None,
        segnet=None,
    )
    assert result.min() == 101
    assert result.max() == 103


def test_perturbations_are_deterministic_and_not_constant() -> None:
    first = dspsa_perturbation(seed=9, iteration=2, dimension=15)
    assert first == dspsa_perturbation(seed=9, iteration=2, dimension=15)
    assert first != dspsa_perturbation(seed=9, iteration=3, dimension=15)
    assert set(first) <= {-1, 1}


def test_projection_and_coordinate_candidates_respect_bounds() -> None:
    assert project_uint8([-4, 4.6, 900]) == (0, 5, 255)
    candidates = coordinate_candidates([0, 255], lower=0, upper=255)
    assert candidates == ((1, 255), (0, 254))
    assert all(all(0 <= value <= 255 for value in item) for item in candidates)


def test_full_admission_never_lets_seg_mismatches_override_pose() -> None:
    before = (0.10, 0.0, 100)
    assert not admit_candidate(
        before=before,
        after=(0.09, 1.0, 100),
        before_mismatches=10,
        after_mismatches=9,
    )
    assert admit_candidate(
        before=before,
        after=(0.09, 0.0, 100),
        before_mismatches=10,
        after_mismatches=9,
    )


def test_seg_only_component_diagnostic_is_explicit_and_fixed_byte() -> None:
    assert admit_seg_only_component_diagnostic(
        before_mismatches=10,
        after_mismatches=9,
        before_component_bytes=100,
        after_component_bytes=100,
    )
    with pytest.raises(ValueError, match="identical component bytes"):
        admit_seg_only_component_diagnostic(
            before_mismatches=10,
            after_mismatches=9,
            before_component_bytes=100,
            after_component_bytes=99,
        )


def test_score_waterline_is_exactly_the_rate_term() -> None:
    assert score(d_seg=0.0, d_pose=0.0, archive_bytes=1) == RATE_SCORE_PER_BYTE
    assert marginal_beats_waterline(non_rate_score_improvement=RATE_SCORE_PER_BYTE * 2, added_bytes=1)
    assert not marginal_beats_waterline(non_rate_score_improvement=RATE_SCORE_PER_BYTE, added_bytes=1)


def test_projected_middle_point_uses_two_opposite_corners_and_calibrates_once() -> None:
    state = DSPSAState(
        theta=(20.0, 30.0, 10.2),
        best=(20, 30, 10),
        best_objective=3.0,
        iteration=0,
        seed=4,
        config_fingerprint="cfg",
    )
    assert middle_point(state.theta) == (20.5, 30.5, 10.5)
    signs = dspsa_perturbation(seed=4, iteration=0, dimension=3)
    plus, minus = wang_corners(state.theta, signs)
    assert all(abs(left - right) == 1 for left, right in zip(plus, minus, strict=True))
    next_state = wang_dspsa_step(
        state, objective_plus=1.0, objective_minus=3.0, target_first_displacement=1.0, gain_alpha=0.602, A=1
    )
    assert next_state.calibrated_a is not None
    assert next_state.best_objective == 1.0
    assert next_state.best == plus
    assert max(abs(after - before) for after, before in zip(next_state.theta, state.theta, strict=True)) == 1.0


def test_projected_middle_point_zero_gradient_keeps_uncalibrated_theta_and_best() -> None:
    state = DSPSAState(theta=(3.5, 4.5), best=(3, 4), best_objective=7.0, iteration=0, seed=1, config_fingerprint="cfg")
    next_state = wang_dspsa_step(
        state, objective_plus=5.0, objective_minus=5.0, target_first_displacement=1.0, gain_alpha=0.602, A=1
    )
    assert next_state.theta == project_theta((3.5, 4.5))
    assert next_state.calibrated_a is None
    assert next_state.best_objective == 5.0


def test_projected_middle_point_dspsa_refuses_out_of_bound_incumbent() -> None:
    state = DSPSAState(
        theta=(101.5, 102.5), best=(100, 102), best_objective=7.0, iteration=0, seed=1, config_fingerprint="cfg"
    )
    with pytest.raises(ValueError, match="incumbent"):
        wang_dspsa_step(
            state,
            objective_plus=5.0,
            objective_minus=6.0,
            target_first_displacement=1.0,
            gain_alpha=0.602,
            A=1,
            lower=101,
            upper=103,
        )


def test_checkpoint_round_trip_and_fingerprint_refusal() -> None:
    state = DSPSAState(
        theta=(1.5, 2.5),
        best=(1, 2),
        best_objective=4.0,
        iteration=3,
        seed=2,
        config_fingerprint="fingerprint",
        calibrated_a=1.0,
    )
    assert DSPSAState.from_json(state.to_json(), config_fingerprint="fingerprint") == state
    with pytest.raises(ValueError, match="fingerprint"):
        DSPSAState.from_json(state.to_json(), config_fingerprint="other")


def test_stage_paths_are_per_pair_and_immutable(tmp_path: Path) -> None:
    probe = _probe_module()
    first = probe._stage_checkpoint_path(tmp_path, 0)
    later = probe._stage_checkpoint_path(tmp_path, 12)
    assert first == tmp_path / "checkpoints" / "pair_0000.json"
    assert later == tmp_path / "checkpoints" / "pair_0012.json"
    assert probe._iteration_checkpoint_path(tmp_path, 12, 3) == tmp_path / "checkpoints" / "pair_0012_iter_0003.json"
    first.parent.mkdir()
    probe._atomic_immutable_bytes(first, b"stage-a")
    probe._atomic_immutable_bytes(first, b"stage-a")
    with pytest.raises(ValueError, match="immutable"):
        probe._atomic_immutable_bytes(first, b"stage-b")


def test_atomic_candidate_bytes_are_persisted_and_refuse_mutation(tmp_path: Path) -> None:
    probe = _probe_module()
    candidate = tmp_path / "candidate.pdw1p.bin"
    probe._atomic_immutable_bytes(candidate, b"strict-pdw1p")
    assert candidate.read_bytes() == b"strict-pdw1p"
    with pytest.raises(ValueError, match="immutable"):
        probe._atomic_immutable_bytes(candidate, b"mutated")


def test_implementation_fingerprints_are_complete_sha256_values(tmp_path: Path) -> None:
    probe = _probe_module()
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (upstream / "modules.py").write_text("class SegNet: pass\n", encoding="utf-8")
    config = _config(
        upstream_path=str(upstream),
        segnet_path=str(upstream / "models" / "segnet.safetensors"),
    )
    fingerprints = probe._implementation_fingerprints(config)
    assert set(fingerprints) == {
        "probe",
        "typed_dsl",
        "optimization",
        "canonical_action",
        "canonical_equation_schema",
        "provenance_builders",
        "pdw1_codec",
        "uint8_lattice_realization",
        "frozen_segnet_loader",
        "realization_necessity_equation",
        "segnet_head_equation",
        "upstream_modules",
    }
    assert all(len(value) == 64 and set(value) <= set("0123456789abcdef") for value in fingerprints.values())
    assert len(probe._base_git_head()) == 40


def test_probe_refuses_subset_candidate_custody_and_malformed_resume() -> None:
    probe = _probe_module()
    import numpy as np

    with pytest.raises(ValueError, match="complete packet"):
        probe._require_complete_packet_pairs(_config(), 3)
    with pytest.raises(ValueError, match="malformed checkpoint schema"):
        probe._validate_checkpoint_payload(
            _config(), {}, np.zeros((2, 5, 3), dtype="uint8"), expected_closure=_fake_closure(probe)
        )
    config = _config(family="baseline", iterations=0)
    payload = probe._checkpoint_payload(
        config,
        fills=np.zeros((2, 5, 3), dtype="uint8"),
        completed_pairs=[],
        rows=[],
        reproducibility_closure=_fake_closure(probe),
    )
    payload["fills"][0][0][0] = 1.5
    with pytest.raises(ValueError, match="integral uint8"):
        probe._validate_checkpoint_payload(
            config, payload, np.zeros((2, 5, 3), dtype="uint8"), expected_closure=_fake_closure(probe)
        )


def test_receiver_path_measures_synthetic_pdw1_with_fake_segnet() -> None:
    import numpy as np
    import torch

    from tac.codec.pdw1_plane_codec import Pdw1PlanePayload, decode_pdw1p, encode_pdw1p
    from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator

    probe = _probe_module()

    class FakeSegNet:
        def preprocess_input(self, value):
            return value

        def __call__(self, value):
            return torch.zeros((1, 5, 384, 512), dtype=torch.float32)

    payload = Pdw1PlanePayload(
        labels=np.zeros((1, 384, 512), dtype=np.uint8), fills=np.zeros((1, 5, 3), dtype=np.uint8)
    )
    blob = encode_pdw1p(payload)
    assert encode_pdw1p(decode_pdw1p(blob)) == blob
    operator = DisjointResizeOperator.build(camera_h=874, camera_w=1164, scorer_h=384, scorer_w=512)
    row = probe._measure_pair(
        payload=payload, pair=0, lstar=np.zeros((384, 512), dtype=np.uint8), operator=operator, segnet=FakeSegNet()
    )
    assert row["mismatch_px"] == 0
    assert row["factor2_certified_exact"]


def test_dspsa_has_only_two_hard_corner_evaluations_per_iteration() -> None:
    import numpy as np

    probe = _probe_module()
    config = _config(pair_indices=(0,), iterations=3)
    evaluations: list[tuple[int, ...]] = []
    checkpoints: list[int] = []

    def fake_measure(*, fills, pair, **_kwargs):
        palette = tuple(int(value) for value in fills[pair].reshape(-1))
        evaluations.append(palette)
        return {
            "pair": pair,
            "mismatch_px": sum(palette),
            "d_seg": float(sum(palette)),
            "factor2_certified_exact": True,
            "factor2_numerator_exact": True,
        }

    probe._evaluate_fill = fake_measure
    fills, row = probe._dspsa_search(
        config,
        labels=np.zeros((1, 1, 1), dtype=np.uint8),
        fills=np.zeros((1, 5, 3), dtype=np.uint8),
        pair=0,
        lstar=np.zeros((1, 1), dtype=np.uint8),
        operator=None,
        segnet=None,
        in_progress=None,
        save_iteration=lambda state, _row, _fills: checkpoints.append(state.iteration),
    )
    assert len(evaluations) == 1 + 2 * config.iterations  # one initial incumbent, two corners each iteration
    assert checkpoints == [1, 2, 3]
    assert row["dspsa_iterations"] == 3
    assert fills.shape == (1, 5, 3)


def test_dspsa_projects_warm_start_and_all_incumbents_to_narrow_bounds() -> None:
    import numpy as np

    probe = _probe_module()
    config = _config(pair_indices=(0,), iterations=3, lower_bound=101, upper_bound=103)
    evaluations: list[tuple[int, ...]] = []
    checkpoint_incumbents: list[tuple[int, ...]] = []

    def fake_measure(*, fills, pair, **_kwargs):
        palette = tuple(int(value) for value in fills[pair].reshape(-1))
        evaluations.append(palette)
        objective = sum((value - 102) ** 2 for value in palette)
        return {
            "pair": pair,
            "mismatch_px": objective,
            "d_seg": float(objective),
            "factor2_certified_exact": True,
            "factor2_numerator_exact": True,
        }

    probe._evaluate_fill = fake_measure
    fills, _row = probe._dspsa_search(
        config,
        labels=np.zeros((1, 1, 1), dtype=np.uint8),
        fills=np.zeros((1, 5, 3), dtype=np.uint8),
        pair=0,
        lstar=np.zeros((1, 1), dtype=np.uint8),
        operator=None,
        segnet=None,
        in_progress=None,
        save_iteration=lambda state, _row, _fills: checkpoint_incumbents.append(state.best),
    )
    assert evaluations
    assert all(all(101 <= value <= 103 for value in palette) for palette in evaluations)
    assert all(all(101 <= value <= 103 for value in best) for best in checkpoint_incumbents)
    assert int(fills.min()) >= 101
    assert int(fills.max()) <= 103


def test_terminal_dspsa_iteration_checkpoint_resumes_without_re_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    probe = _probe_module()
    monkeypatch.setattr(crux_config_module, "_DIAGNOSTIC_OUTPUT_ROOT", tmp_path)
    config = _config(pair_indices=(0,), iterations=2, output_dir=str(tmp_path))
    source = np.zeros((1, 5, 3), dtype=np.uint8)
    best = tuple([3] * 15)
    state = DSPSAState(
        theta=tuple([3.5] * 15),
        best=best,
        best_objective=9.0,
        iteration=2,
        seed=config.seed,
        config_fingerprint=config.fingerprint,
        calibrated_a=1.0,
        last_objective_plus=9.0,
        last_objective_minus=10.0,
    )
    best_row = {
        "pair": 0,
        "mismatch_px": 9,
        "d_seg": 9.0,
        "factor2_certified_exact": True,
        "factor2_numerator_exact": True,
    }
    fills = np.asarray(best, dtype=np.uint8).reshape(1, 5, 3)
    progress = {"pair": 0, "state_json": state.to_json(), "best_row": best_row}
    closure = _fake_closure(probe)
    payload = probe._checkpoint_payload(
        config, fills=fills, completed_pairs=[], rows=[], reproducibility_closure=closure, in_progress=progress
    )
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    terminal = probe._iteration_checkpoint_path(tmp_path, 0, 2)
    probe._atomic_immutable_bytes(terminal, probe._json_bytes(payload))
    latest = tmp_path / "checkpoint.json"
    probe._atomic_json(latest, payload)

    resumed_fills, completed, rows, resumed_progress = probe._resume_or_initial(
        config, latest, source, expected_closure=closure
    )
    evaluations: list[object] = []
    probe._evaluate_fill = lambda **_kwargs: evaluations.append(object())  # must never be called
    finished_fills, row = probe._dspsa_search(
        config,
        labels=np.zeros((1, 1, 1), dtype=np.uint8),
        fills=resumed_fills,
        pair=0,
        lstar=np.zeros((1, 1), dtype=np.uint8),
        operator=None,
        segnet=None,
        in_progress=resumed_progress,
        save_iteration=lambda *_args: pytest.fail("terminal resume must not checkpoint again"),
    )
    assert completed == [] and rows == []
    assert evaluations == []
    assert tuple(finished_fills.reshape(-1)) == best
    assert row["dspsa_best_objective"] == 9.0

    over_limit = dict(progress)
    over_limit["state_json"] = DSPSAState(
        theta=tuple([3.5] * 15),
        best=best,
        best_objective=9.0,
        iteration=3,
        seed=config.seed,
        config_fingerprint=config.fingerprint,
        calibrated_a=1.0,
        last_objective_plus=9.0,
        last_objective_minus=10.0,
    ).to_json()
    bad_payload = probe._checkpoint_payload(
        config, fills=fills, completed_pairs=[], rows=[], reproducibility_closure=closure, in_progress=over_limit
    )
    probe._atomic_json(latest, bad_payload)
    with pytest.raises(ValueError, match="unreachable"):
        probe._resume_or_initial(config, latest, source, expected_closure=closure)


def test_resume_refuses_source_mutation_without_checkpoint_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np
    import torch

    probe = _probe_module()
    monkeypatch.setattr(crux_config_module, "_DIAGNOSTIC_OUTPUT_ROOT", tmp_path)
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    modules = upstream / "modules.py"
    modules.write_text("class SegNet: pass\n", encoding="utf-8")
    config = _config(
        pair_indices=(0,),
        output_dir=str(tmp_path / "run"),
        upstream_path=str(upstream),
        segnet_path=str(upstream / "models" / "segnet.safetensors"),
    )
    original_closure = probe._reproducibility_closure(config, torch_module=torch)
    source = np.zeros((1, 5, 3), dtype=np.uint8)
    payload = probe._checkpoint_payload(
        config,
        fills=source,
        completed_pairs=[],
        rows=[],
        reproducibility_closure=original_closure,
    )
    checkpoint = tmp_path / "checkpoint.json"
    probe._atomic_json(checkpoint, payload)
    checkpoint_before = checkpoint.read_bytes()

    modules.write_text("class SegNet:\n    changed = True\n", encoding="utf-8")
    mutated_closure = probe._reproducibility_closure(config, torch_module=torch)
    assert mutated_closure != original_closure
    with pytest.raises(ValueError, match="closure mismatch"):
        probe._resume_or_initial(config, checkpoint, source, expected_closure=mutated_closure)
    assert checkpoint.read_bytes() == checkpoint_before


def test_final_receipt_fields_bind_complete_reproducibility_closure(tmp_path: Path) -> None:
    import torch

    probe = _probe_module()
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (upstream / "modules.py").write_text("class SegNet: pass\n", encoding="utf-8")
    config = _config(
        upstream_path=str(upstream),
        segnet_path=str(upstream / "models" / "segnet.safetensors"),
    )
    closure = probe._reproducibility_closure(config, torch_module=torch)
    fields = probe._receipt_closure_fields(closure)
    assert fields["reproducibility_closure"] == closure
    assert set(closure["sources"]) == {
        "probe",
        "optimization",
        "typed_dsl",
        "canonical_action",
        "canonical_equation_schema",
        "provenance_builders",
        "pdw1_codec",
        "uint8_lattice_realization",
        "frozen_segnet_loader",
        "realization_necessity_equation",
        "segnet_head_equation",
        "upstream_modules",
    }
    assert set(closure["runtime"]) == {
        "python",
        "numpy",
        "torch",
        "brotli",
        "safetensors",
        "platform",
        "threads",
    }
    assert fields["runtime"] == closure["runtime"]
    assert fields["base_git_head"] == closure["base_git_head"]
    assert all("checkpoint" not in name and "receipt" not in name for name in closure["sources"])


def test_runtime_distribution_mutation_refuses_resume(tmp_path: Path) -> None:
    import copy

    import torch

    probe = _probe_module()
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (upstream / "modules.py").write_text("class SegNet: pass\n", encoding="utf-8")
    config = _config(
        upstream_path=str(upstream),
        segnet_path=str(upstream / "models" / "segnet.safetensors"),
    )
    closure = probe._reproducibility_closure(config, torch_module=torch)
    mutated = copy.deepcopy(closure)
    mutated["runtime"]["brotli"]["module_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="closure changed"):
        probe._require_current_closure(config, torch_module=torch, expected=mutated)


def test_finalization_reuses_checkpoint_source_runtime_package_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types

    import numpy as np
    import torch

    from tac.codec.pdw1_plane_codec import Pdw1PlanePayload, encode_pdw1p

    probe = _probe_module()
    monkeypatch.setattr(crux_config_module, "_DIAGNOSTIC_OUTPUT_ROOT", tmp_path)
    upstream = tmp_path / "upstream"
    models = upstream / "models"
    models.mkdir(parents=True)
    modules_path = upstream / "modules.py"
    modules_path.write_text("class SegNet: pass\n", encoding="utf-8")
    segnet_path = models / "segnet.safetensors"
    segnet_path.write_bytes(b"test-segnet-custody")
    packet_path = tmp_path / "source.pdw1p.bin"
    packet_path.write_bytes(
        encode_pdw1p(
            Pdw1PlanePayload(
                labels=np.zeros((1, 384, 512), dtype=np.uint8),
                fills=np.zeros((1, 5, 3), dtype=np.uint8),
            )
        )
    )
    gt_path = tmp_path / "gt.npz"
    np.savez(
        gt_path,
        gt_f1=np.zeros((1, 1), dtype=np.uint8),
        lstars=np.zeros((1, 384, 512), dtype=np.uint8),
    )
    config = _config(
        packet_path=str(packet_path),
        packet_sha256=probe._sha256_file(packet_path),
        gt_path=str(gt_path),
        gt_sha256=probe._sha256_file(gt_path),
        segnet_path=str(segnet_path),
        segnet_sha256=probe._sha256_file(segnet_path),
        upstream_path=str(upstream),
        output_dir=str(tmp_path / "run"),
        pair_indices=(0,),
        family="baseline",
        iterations=0,
    )

    class FakeSegNet:
        def preprocess_input(self, value):
            return value

        def __call__(self, _value):
            return torch.zeros((1, 5, 384, 512), dtype=torch.float32)

    upstream_module = types.ModuleType("modules")
    upstream_module.__file__ = str(modules_path)
    monkeypatch.setitem(sys.modules, "modules", upstream_module)
    monkeypatch.setattr(probe, "load_frozen_segnet_cpu", lambda _path: FakeSegNet())

    receipt = probe.run(config)
    checkpoint = json.loads((tmp_path / "run" / "checkpoint.json").read_text())
    stage = json.loads((tmp_path / "run" / "checkpoints" / "pair_0000.json").read_text())
    closure = receipt["reproducibility_closure"]
    assert receipt["schema"] == "einstein_kolmogorov_crux_receipt.v2"
    assert checkpoint["schema"] == "einstein_kolmogorov_crux_checkpoint.v2"
    assert checkpoint["reproducibility_closure"] == closure
    assert stage["reproducibility_closure"] == closure
    assert receipt["implementation_fingerprints"] == {name: item["sha256"] for name, item in closure["sources"].items()}
