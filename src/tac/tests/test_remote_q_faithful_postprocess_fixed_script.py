from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_q_faithful_postprocess_fixed.sh"


def test_q_faithful_postprocess_uses_top_submission_packaging_contract() -> None:
    text = SCRIPT.read_text()

    assert "masks_half_odd_crf50.mkv" in text
    assert "optimized_poses.bin" in text
    assert "RENDERER_COMPACT_MANIFEST" in text
    assert "repack_quantizr_faithful_qzs3_archive.py" in text
    assert "--payload-format rp2_fixed3" in text
    assert "--payload-member-name p" in text
    assert "--pose-codec pose_qpose14_col_delta_v1" in text
    assert "--pose-codec pose_qp1_v1" in text
    assert "score_claim" in text
    assert 'ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt.get("model", ckpt)))' in text
    assert "from dataclasses import asdict" in text
    assert 'validation["archive_path"] = str(validation["archive_path"])' in text
