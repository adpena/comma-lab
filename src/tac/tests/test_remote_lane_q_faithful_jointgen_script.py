from pathlib import Path

SCRIPT = Path("scripts/remote_lane_q_faithful_jointgen.sh")


def _script_text() -> str:
    return SCRIPT.read_text()


def test_qfaithful_builds_half_frame_mask_archive() -> None:
    text = _script_text()

    assert "experiments/build_baseline_archive.py" in text
    assert "--half-frame" in text
    assert "masks.mkv extracted" in text


def test_qfaithful_deploys_raw_qfai_renderer_bin() -> None:
    text = _script_text()

    assert "qfai_path = Path('$LOG_DIR/train/renderer.bin')" in text
    assert "br_path = Path('$LOG_DIR/train/renderer.qfai.bin.br')" in text
    assert "cp \"$EXPORT_BIN\" \"$LOG_DIR/iter_0/renderer.bin\"" in text
    assert "save_qfai(gen, qfai_path, extra_meta=" in text
    assert "training_pose_contract" in text
    assert "inflate_renderer.py dispatches QFAI/QZS3 by file" in text


def test_qfaithful_auto_resume_is_pipefail_safe() -> None:
    text = _script_text()

    assert 'RESUME_FROM="${RESUME_FROM:-}"' in text
    assert 'RESUME_ARGS=(--resume-from "$RESUME_FROM")' in text
    assert 'if [ -f "$LOG_DIR/train/training_state_q_faithful_modal.pt" ]; then' in text
    assert "AUTO_RESUME=$(ls" not in text
    assert "NO resume checkpoint detected" in text
