"""Tests for the reusable HNeRV trainer parity guard."""

from __future__ import annotations

from pathlib import Path

from tac.hnerv_training_parity_guard import (
    assert_hnerv_training_parity_file,
    inspect_hnerv_training_parity_source,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_guard_accepts_primary_sane_hnerv_trainer() -> None:
    assert_hnerv_training_parity_file(
        REPO_ROOT / "experiments" / "train_substrate_sane_hnerv.py"
    )


def test_guard_accepts_wave3_tc_nerv_trainer() -> None:
    assert_hnerv_training_parity_file(
        REPO_ROOT / "experiments" / "train_substrate_tc_nerv.py"
    )


def test_guard_rejects_runtime_without_exact_file_list_signature() -> None:
    text = _GOOD_TRAINER_SOURCE.replace(
        '        "FILE_LIST=\\"$3\\"\\n"\n',
        "",
    ).replace(
        '        "\\"$DATA_DIR\\" \\"$OUTPUT_DIR\\" \\"$FILE_LIST\\"\\n"\n',
        '        "\\"$DATA_DIR\\" \\"$OUTPUT_DIR\\"\\n"\n',
    )
    report = inspect_hnerv_training_parity_source(text, path_label="bad_runtime.py")
    assert not report.passed
    assert any("file_list" in violation for violation in report.violations)


def test_guard_rejects_scorer_load_before_yuv6_patch() -> None:
    text = _GOOD_TRAINER_SOURCE.replace(
        "    yuv6_token = patch_upstream_yuv6_globally()\n"
        "    posenet, segnet = load_differentiable_scorers(args.upstream_dir)\n",
        "    posenet, segnet = load_differentiable_scorers(args.upstream_dir)\n"
        "    yuv6_token = patch_upstream_yuv6_globally()\n",
    )
    report = inspect_hnerv_training_parity_source(text, path_label="bad_yuv6.py")
    assert not report.passed
    assert any("before patching" in violation for violation in report.violations)


def test_guard_rejects_missing_archive_build_loop_and_ema_shadow() -> None:
    text = _GOOD_TRAINER_SOURCE.replace("    ema.apply(model)\n", "").replace(
        "    ema_state = ema.state_dict()\n",
        "",
    ).replace(
        "    bin_bytes = pack_archive(decoder_sd, latents, meta)\n",
        "",
    ).replace(
        "    _build_archive_zip(\n"
        "        args.output_dir / 'archive.zip',\n"
        "        bin_bytes=bin_bytes,\n"
        "        submission_dir=args.output_dir / 'submission',\n"
        "    )\n",
        "",
    )
    report = inspect_hnerv_training_parity_source(text, path_label="bad_archive.py")
    assert not report.passed
    joined = "\n".join(report.violations)
    assert "ema.apply" in joined
    assert "ema.state_dict()" in joined
    assert "pack_archive" in joined
    assert "_build_archive_zip" in joined


_GOOD_TRAINER_SOURCE = '''
def _write_runtime(submission_dir):
    inflate_sh = (
        "#!/usr/bin/env bash\\n"
        "set -euo pipefail\\n"
        "DATA_DIR=\\"$1\\"\\n"
        "OUTPUT_DIR=\\"$2\\"\\n"
        "FILE_LIST=\\"$3\\"\\n"
        "exec \\"${PYTHON:-python3}\\" \\"$HERE/inflate.py\\" "
        "\\"$DATA_DIR\\" \\"$OUTPUT_DIR\\" \\"$FILE_LIST\\"\\n"
    )
    inflate_py = (
        "import sys\\n"
        "from pathlib import Path\\n"
        "def main():\\n"
        "    if len(sys.argv) != 4:\\n"
        "        return 2\\n"
        "    archive_dir = Path(sys.argv[1])\\n"
        "    output_dir = Path(sys.argv[2])\\n"
        "    file_list_path = Path(sys.argv[3])\\n"
        "    archive_bytes = (archive_dir / '0.bin').read_bytes()\\n"
        "    for line in file_list_path.read_text(encoding='utf-8').splitlines():\\n"
        "        line = line.strip()\\n"
        "    return 0\\n"
    )


def _full_main(args):
    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally
    from tac.scorer import load_differentiable_scorers
    from tac.training import EMA

    yuv6_token = patch_upstream_yuv6_globally()
    posenet, segnet = load_differentiable_scorers(args.upstream_dir)
    ema = EMA(model, decay=args.ema_decay)
    loss, parts = loss_fn(
        rgb_0, rgb_1, gt_0, gt_1, archive_bytes_proxy,
        apply_eval_roundtrip=True,
    )
    ema.update(model)
    ema.apply(model)
    ema_state = ema.state_dict()
    torch.save({"state_dict": ema_state}, ckpt_best_path)
    bin_bytes = pack_archive(decoder_sd, latents, meta)
    _write_runtime(args.output_dir / 'submission')
    _build_archive_zip(
        args.output_dir / 'archive.zip',
        bin_bytes=bin_bytes,
        submission_dir=args.output_dir / 'submission',
    )
    provenance = {"ready_for_exact_eval_dispatch": False}
'''
