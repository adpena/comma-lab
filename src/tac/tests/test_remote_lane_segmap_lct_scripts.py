"""Remote-script plumbing tests for SegMap LCT archive custody."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text()


def test_lane_sa_lct_mode_trains_copies_and_configures_charged_payload() -> None:
    src = _read("scripts/remote_lane_sa_segmap_clone.sh")

    assert 'SEGMAP_ENABLE_LCT="${SEGMAP_ENABLE_LCT:-0}"' in src
    assert "--learnable-class-targets --class-targets-filename" in src
    assert 'cp "$LCT_PAYLOAD" "$LOG_DIR/archive_src/$SEGMAP_CLASS_TARGETS_FILENAME"' in src
    assert "members.append('$SEGMAP_CLASS_TARGETS_FILENAME')" in src
    assert "SEGMAP_CLASS_TARGETS_FILENAME=$SEGMAP_CLASS_TARGETS_FILENAME" in src


def test_lane_sc_lct_mode_trains_copies_and_configures_charged_payload() -> None:
    src = _read("scripts/remote_lane_sc_plus_plus_kl_distill.sh")

    assert 'SEGMAP_ENABLE_LCT="${SEGMAP_ENABLE_LCT:-0}"' in src
    assert "--learnable-class-targets --class-targets-filename" in src
    assert 'cp "$LCT_PAYLOAD" "$LOG_DIR/archive_src/$SEGMAP_CLASS_TARGETS_FILENAME"' in src
    assert "members.append('$SEGMAP_CLASS_TARGETS_FILENAME')" in src
    assert "SEGMAP_CLASS_TARGETS_FILENAME=$SEGMAP_CLASS_TARGETS_FILENAME" in src


def test_lane_sh_preserves_lct_payload_and_uses_deterministic_archive_members() -> None:
    src = _read("scripts/remote_lane_sh_shannon_arithmetic.sh")

    assert 'UPSTREAM_CLASS_TARGETS="$EXTRACT_DIR/$SEGMAP_CLASS_TARGETS_FILENAME"' in src
    assert 'cp "$UPSTREAM_CLASS_TARGETS" "$LOG_DIR/archive_src/$SEGMAP_CLASS_TARGETS_FILENAME"' in src
    assert "ZipInfo(filename=n, date_time=(1980, 1, 1, 0, 0, 0))" in src
    assert "SEGMAP_ARCH=$SEGMAP_ARCH" in src
    assert "SEGMAP_CLASS_TARGETS_FILENAME=$SEGMAP_CLASS_TARGETS_FILENAME" in src
