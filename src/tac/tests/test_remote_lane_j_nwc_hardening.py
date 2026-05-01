"""Static guards for Lane J-NWC/NWCS remote-script hardening."""
from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
REMOTE_SCRIPTS = [
    REPO / "scripts" / "remote_lane_j_nwc_neural_weight_compression.sh",
    REPO / "scripts" / "remote_lane_j_nwcs_sensitivity_aware_codec.sh",
    REPO / "scripts" / "remote_lane_j_nwcs_ec_stack.sh",
]
NWCS_SCRIPTS = REMOTE_SCRIPTS[1:]


def _text(path: Path) -> str:
    return path.read_text()


def test_j_nwc_remote_scripts_do_not_use_extractall() -> None:
    for script in REMOTE_SCRIPTS:
        text = _text(script)
        assert ".extractall(" not in text
        assert "unsafe archive member" in text
        assert "duplicate archive member" in text
        assert "zf.open(info" in text


def test_j_nwc_remote_scripts_fail_closed_on_non_cuda_eval() -> None:
    for script in REMOTE_SCRIPTS:
        text = _text(script)
        assert 'AUTH_EVAL_DEVICE="${AUTH_EVAL_DEVICE:-cuda}"' in text
        assert 'if [ "$AUTH_EVAL_DEVICE" != "cuda" ]; then' in text
        assert "--device cuda" in text
        assert '--device "$AUTH_EVAL_DEVICE"' not in text
        assert '--device "${AUTH_EVAL_DEVICE:-cuda}"' not in text
        assert "contest-CUDA" in text


def test_j_nwc_remote_scripts_adjudicate_exact_eval_with_component_gates() -> None:
    for script in REMOTE_SCRIPTS:
        text = _text(script)
        eval_pos = text.index("experiments/contest_auth_eval.py \\")
        adj_pos = text.index("scripts/adjudicate_contest_auth_eval.py", eval_pos)
        assert eval_pos < adj_pos
        adjudication_block = text[adj_pos:text.index("2>&1 | tee \"$LOG_DIR/adjudication.log\"", adj_pos)]
        assert "--contest-json \"$RESULT_JSON\"" in adjudication_block
        assert "--archive \"$ARCHIVE\"" in adjudication_block
        assert "--result-copy \"$ADJUDICATED_RESULT_JSON\"" in adjudication_block
        assert "--baseline-score 1.043987524793892" in adjudication_block
        assert "--baseline-archive-bytes 686635" in adjudication_block
        assert "--baseline-posenet-dist 0.00346442" in adjudication_block
        assert "--baseline-segnet-dist 0.00400656" in adjudication_block
        assert "--max-posenet-relative 1.01" in adjudication_block
        assert "--max-segnet-relative 1.01" in adjudication_block
        assert "--required-device cuda" in adjudication_block
        assert "--required-samples 600" in adjudication_block


def test_j_nwc_remote_scripts_record_artifact_custody() -> None:
    for script in REMOTE_SCRIPTS:
        text = _text(script)
        assert "def file_meta(path):" in text
        assert '"artifact_custody": {' in text
        assert '"archive": file_meta("$ARCHIVE")' in text
        assert '"result_json": file_meta("$RESULT_JSON")' in text
        assert '"adjudicated_result_json": file_meta("$ADJUDICATED_RESULT_JSON")' in text
        assert '"adjudication_provenance": file_meta("$ADJUDICATION_PROVENANCE")' in text


def test_j_nwc_remote_scripts_surface_adjudication_in_final_records() -> None:
    for script in REMOTE_SCRIPTS:
        text = _text(script)
        assert '"adjudicated_result_json": "$ADJUDICATED_RESULT_JSON"' in text
        assert '"adjudication_provenance": "$ADJUDICATION_PROVENANCE"' in text
        assert (
            '"score_source": '
            '"contest_auth_eval.adjudicated.json:score_recomputed_from_components"'
        ) in text
        assert '"adjudication_required": True' in text
        assert '"component_gates_required": True' in text


def test_nwc_train_cli_seeds_before_codec_construction() -> None:
    text = (REPO / "experiments" / "train_neural_weight_codec.py").read_text()
    seed_pos = text.index("torch.manual_seed(int(args.seed))")
    construct_pos = text.index("codec = WeightCodec(cfg)")
    assert seed_pos < construct_pos


def test_nwc_train_cli_replays_manifest_with_replay_root_and_exact_manifest_copy() -> None:
    text = (REPO / "experiments" / "train_neural_weight_codec.py").read_text()
    assert '"--corpus-manifest"' in text
    assert '"--corpus-replay-root"' in text
    assert "FATAL: --corpus-replay-root requires --corpus-manifest" in text
    assert "manifest_out.write_bytes(args.corpus_manifest.read_bytes())" in text
    assert (
        "build_corpus_from_manifest(\n"
        "        manifest,\n"
        "        replay_root=args.corpus_replay_root,"
    ) in text
    assert '"corpus_manifest_sha256": manifest_sha256' in text
    assert '"corpus_replay_root":' in text


def test_j_nwc_remote_scripts_support_prebuilt_corpus_manifest_replay_root() -> None:
    for script in REMOTE_SCRIPTS:
        text = _text(script)
        assert 'PREBUILT_CORPUS_MANIFEST="${PREBUILT_CORPUS_MANIFEST:-}"' in text
        assert 'CORPUS_REPLAY_ROOT="${CORPUS_REPLAY_ROOT:-}"' in text
        assert "missing PREBUILT_CORPUS_MANIFEST" in text
        assert "missing CORPUS_REPLAY_ROOT" in text
        assert '--corpus-manifest "$PREBUILT_CORPUS_MANIFEST"' in text
        assert '--corpus-replay-root "$CORPUS_REPLAY_ROOT"' in text
        assert '"prebuilt_corpus_manifest": "$PREBUILT_CORPUS_MANIFEST"' in text
        assert '"corpus_replay_root": "$CORPUS_REPLAY_ROOT"' in text
        assert (
            '"prebuilt_corpus_manifest": file_meta("$PREBUILT_CORPUS_MANIFEST")'
            in text
        )


def test_nwcs_remote_scripts_seed_before_sensitivity_codec_construction() -> None:
    for script in NWCS_SCRIPTS:
        text = _text(script)
        for needle in (
            "codec = SensitivityAwareWeightCodec(cfg)\ncodec, losses =",
            "codec = SensitivityAwareWeightCodec(cfg)\ncodec.load_state_dict",
        ):
            construct_pos = text.index(needle)
            prefix = text[:construct_pos]
            assert "torch.manual_seed(1234)" in prefix


def test_nwcs_remote_scripts_require_sensitivity_provenance() -> None:
    for script in NWCS_SCRIPTS:
        text = _text(script)
        assert "promotable ANCHOR_SENSITIVITY_PT must be a dict" in text
        assert "format=tac.nwcs_anchor_sensitivity_inputs.v1" in text
        assert "ANCHOR_SENSITIVITY_PT source must be component_sensitivity_v1.combined" in text
        assert "ANCHOR_SENSITIVITY_PT promotion_eligible must be true" in text
        assert "anchor_archive_sha256" in text
        assert "anchor_renderer_sha256" in text
        assert "metadata.parameters" in text
        assert 'COMPONENT_SENSITIVITY_MANIFEST="${COMPONENT_SENSITIVITY_MANIFEST:-}"' in text
        assert "requires COMPONENT_SENSITIVITY_MANIFEST" in text
        assert "requires CORPUS_SENSITIVITY_PT" in text
        assert "requires PREBUILT_CORPUS_MANIFEST matching CORPUS_SENSITIVITY_PT" in text
        assert "validate_component_sensitivity_manifest" in text
        assert "component_sensitivity_manifest_sha256" in text
        assert "stale ANCHOR_SENSITIVITY_PT component_sensitivity_manifest_sha256" in text
        assert "stale CORPUS_SENSITIVITY_PT component_sensitivity_manifest_sha256" in text
        assert "format=tac.nwcs_corpus_sensitivity_inputs.v1" in text
        assert "CORPUS_SENSITIVITY_PT source must be" in text
        assert "anchor_parameter_sensitivity_projected_to_corpus_manifest" in text
        assert "CORPUS_SENSITIVITY_PT promotion_eligible must be true" in text
        assert '"component_sensitivity_manifest": file_meta("$COMPONENT_SENSITIVITY_MANIFEST")' in text
        assert "corpus_manifest_sha256" in text
        assert "CORPUS_SENSITIVITY_PT num_blocks mismatch" in text
        assert "contains negative values" in text
        assert 'corpus_replay_root = Path("$CORPUS_REPLAY_ROOT")' in text
        assert "build_corpus_from_manifest(corpus_manifest, replay_root=corpus_replay_root)" in text
        assert 'NWCS_BUILD_ONLY="${NWCS_BUILD_ONLY:-0}"' in text


def test_nwcs_component_sensitivity_manifest_gate_precedes_auth_eval() -> None:
    for script in NWCS_SCRIPTS:
        text = _text(script)
        gate_pos = text.index("requires COMPONENT_SENSITIVITY_MANIFEST")
        eval_pos = text.index("experiments/contest_auth_eval.py \\")
        assert gate_pos < eval_pos


def test_nwcs_debug_and_build_only_paths_do_not_emit_auth_eval_json() -> None:
    for script in NWCS_SCRIPTS:
        text = _text(script)
        guard = 'if [ "$NWCS_BUILD_ONLY" = "1" ] || [ "$PROMOTION_ELIGIBLE" != "true" ]; then'
        guard_pos = text.index(guard)
        eval_pos = text.index("experiments/contest_auth_eval.py \\", guard_pos)
        assert guard_pos < eval_pos
        guard_block = text[guard_pos:eval_pos]
        assert '"score_claim": False' in guard_block
        assert '"auth_eval_skipped": True' in guard_block
        assert '"result_json": None' in guard_block
        assert "BUILD_ONLY_NON_PROMOTABLE" in guard_block


def test_j_nwc_build_only_path_does_not_emit_auth_eval_json() -> None:
    text = _text(REPO / "scripts" / "remote_lane_j_nwc_neural_weight_compression.sh")
    assert 'NWC_BUILD_ONLY="${NWC_BUILD_ONLY:-0}"' in text
    guard = 'if [ "$NWC_BUILD_ONLY" = "1" ]; then'
    guard_pos = text.index(guard)
    eval_pos = text.index("experiments/contest_auth_eval.py \\", guard_pos)
    assert guard_pos < eval_pos
    guard_block = text[guard_pos:eval_pos]
    assert '"build_only": True' in guard_block
    assert '"score_claim": False' in guard_block
    assert '"promotion_eligible": False' in guard_block
    assert '"auth_eval_skipped": True' in guard_block
    assert '"result_json": None' in guard_block
    assert '"corpus_manifest": file_meta("$CORPUS_MANIFEST")' in guard_block
    assert "LANE_J_NWC_BUILD_ONLY_NON_PROMOTABLE" in guard_block


def test_nwcs_remote_scripts_export_magic_container_not_ad_hoc_blob() -> None:
    for script in NWCS_SCRIPTS:
        text = _text(script)
        assert "export_nwcs_renderer_container" in text
        assert "NWCSRendererTensorEntry.from_tensor_blob" in text
        assert "is_nwcs_renderer_container(out_bin)" in text
        assert "_infer_asymmetric_config(model)" in text
        assert '"config": arch_config' in text
        assert "out_buf = bytearray()" not in text
        assert 'struct.pack("<H"' not in text
        assert 'struct.pack("<I"' not in text


def test_nwcs_remote_scripts_fail_closed_on_arch_config_fallback() -> None:
    for script in NWCS_SCRIPTS:
        text = _text(script)
        assert "except Exception as exc:" in text
        assert "refusing tensor_only fallback on promotion-eligible export" in text
        assert '"promotion_eligible": "$PROMOTION_ELIGIBLE" == "true"' in text
        assert '"promotion_eligible": $PROMOTION_ELIGIBLE' not in text


def test_nwcs_export_heredoc_imports_asymmetric_config() -> None:
    for script in NWCS_SCRIPTS:
        text = _text(script)
        block_start = text.index("export_nwcs_renderer_container(")
        heredoc_start = text.rfind("<<PY", 0, block_start)
        assert heredoc_start != -1
        heredoc_end = text.index("\nPY\n", block_start)
        heredoc = text[heredoc_start:heredoc_end]
        assert "from tac.renderer_export import _infer_asymmetric_config" in heredoc
