"""Tests for tac.checkpoint_retention — the whole-run checkpoint archiver (operator 2026-06-30).

Covers the drop-in decorator + context-manager + direct API + the never-overwrite/dedup/manifest +
refuse-/tmp invariants. No GPU/MLX: pure numpy npz + filesystem."""
from __future__ import annotations

import json

import numpy as np
import pytest

from tac.checkpoint_retention import (
    CheckpointArchiver, archive_run_once, checkpoint_history, keep_checkpoint_history,
)


def _write_npz(path, epoch):
    np.savez(path, w=np.zeros((2, 2), np.float32), __epoch=np.asarray(epoch))


def _write_run(run, epoch, d_seg):
    run.mkdir(parents=True, exist_ok=True)
    _write_npz(run / "levelset_witness_ema_BEST.npz", epoch)
    _write_npz(run / "levelset_witness_ema_mlx.npz", epoch)
    (run / "levelset_best.json").write_text(json.dumps({"epoch": epoch, "d_seg": d_seg}))


class TestKeepDirect:
    def test_keep_best_epoch_dseg_encoded(self, tmp_path):
        src = tmp_path / "BEST.npz"; _write_npz(src, 900)
        arch = CheckpointArchiver(tmp_path / "arch")
        row = arch.keep(src, kind="best", epoch=900, d_seg=0.003988)
        assert row is not None and row["kind"] == "best" and row["epoch"] == 900
        assert (tmp_path / "arch" / "best_ep900_dseg0.003988.npz").exists()
        assert row["sha256"] and row["bytes"] > 0

    def test_dedup_same_best_returns_none(self, tmp_path):
        src = tmp_path / "BEST.npz"; _write_npz(src, 900)
        arch = CheckpointArchiver(tmp_path / "arch")
        assert arch.keep(src, kind="best", epoch=900, d_seg=0.004) is not None
        assert arch.keep(src, kind="best", epoch=900, d_seg=0.004) is None  # duplicate -> skip

    def test_new_best_does_not_overwrite_prior(self, tmp_path):
        src = tmp_path / "BEST.npz"
        arch = CheckpointArchiver(tmp_path / "arch")
        _write_npz(src, 850); arch.keep(src, kind="best", epoch=850, d_seg=0.0042)
        _write_npz(src, 900); arch.keep(src, kind="best", epoch=900, d_seg=0.0039)
        kept = sorted(p.name for p in (tmp_path / "arch").glob("best_ep*.npz"))
        assert kept == ["best_ep850_dseg0.004200.npz", "best_ep900_dseg0.003900.npz"]  # both preserved

    def test_manifest_is_appendonly_jsonl(self, tmp_path):
        src = tmp_path / "BEST.npz"; _write_npz(src, 1)
        arch = CheckpointArchiver(tmp_path / "arch")
        arch.keep(src, kind="best", epoch=1, d_seg=0.5)
        rows = [json.loads(ln) for ln in (tmp_path / "arch" / "manifest.jsonl").read_text().splitlines() if ln.strip()]
        assert len(rows) == 1 and rows[0]["location"] == "local"


class TestScanAndOnce:
    def test_scan_picks_up_best_and_latest(self, tmp_path):
        run = tmp_path / "run"; _write_run(run, 900, 0.003988)
        added = archive_run_once(run)
        assert added["best"] == 1 and added["latest"] == 1

    def test_scan_idempotent(self, tmp_path):
        run = tmp_path / "run"; _write_run(run, 900, 0.003988)
        archive_run_once(run)
        again = archive_run_once(run)
        assert again["best"] == 0 and again["latest"] == 0  # nothing new


class TestDecoratorAndContext:
    def test_decorator_archives_after_save(self, tmp_path):
        run = tmp_path / "run"; run.mkdir()

        @keep_checkpoint_history(run_dir=run)
        def _save(epoch, d_seg):
            _write_run(run, epoch, d_seg)
            return {"saved": True}

        _save(900, 0.0039)
        assert (run / "ckpt_archive" / "best_ep900_dseg0.003900.npz").exists()

    def test_context_manager_scans_on_exit(self, tmp_path):
        run = tmp_path / "run"; _write_run(run, 700, 0.0045)
        with checkpoint_history(run):
            pass
        assert (run / "ckpt_archive" / "best_ep700_dseg0.004500.npz").exists()


class TestHygiene:
    def test_refuses_tmp_archive_dir(self):
        with pytest.raises(ValueError):
            CheckpointArchiver("/tmp/should_refuse_ckpt_archive")

    def test_keep_window_keeps_all_when_zero(self, tmp_path):
        arch = CheckpointArchiver(tmp_path / "arch", keep_window=0)
        for ep in (10, 20, 30, 40):
            src = tmp_path / f"l{ep}.npz"; _write_npz(src, ep)
            arch.keep(src, kind="latest", epoch=ep)
        assert len(list((tmp_path / "arch").glob("latest_ep*.npz"))) == 4  # all kept


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
