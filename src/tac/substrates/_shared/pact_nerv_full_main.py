# SPDX-License-Identifier: MIT
"""Canonical shared score-aware training loop for the PACT-NeRV substrate family.

# AUTOCAST_FP16_WAIVED:score-aware-scorer-path-pending-canonical-autocast-backport-per-ds_nerv-sister-precedent
# TORCH_COMPILE_WAIVED:defer-until-per-substrate-canary-validates-Inductor-graph-breaks-per-ds_nerv-sister-precedent

PACT-NeRV-FULL-MAIN-IMPLEMENTATION-WAVE 2026-05-27. The 18 PACT-NeRV
substrate trainers (``experiments/train_substrate_pact_nerv_*.py``) each
shipped an L0 SCAFFOLD ``_full_main`` raising ``NotImplementedError`` per
the PACT-NERV-DESIGN-SYMPOSIUM HYBRID Stage 1 verdict (commit ``5371d4dd4``).
This helper extinguishes that ``NotImplementedError`` by providing the
substrate-AGNOSTIC training loop while every variant keeps its UNIQUE
distinguishing feature (IA3 γ-only / VQ codebook / Mamba SSM / MoE routing /
diffusion distillation / selector menu / neural-codec-E2E / Bayesian
posterior / cross-codec / multi-modal / distilled-scorer surrogate).

## Canonical-vs-unique decision per layer (Catalog #290)

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" the
canonical-vs-unique split is:

- ADOPT_CANONICAL_BECAUSE_SERVES (this helper):
  * train/val pair split, AdamW + cosine-annealing schedule, NaN watchdog
    (Council D), EMA shadow update after every ``optimizer.step`` + EMA-at-
    eval snapshot/restore (CLAUDE.md "EMA — NON-NEGOTIABLE"), best-EMA-
    checkpoint selection by validation Lagrangian. These are substrate-
    AGNOSTIC and identical to the canonical ``ds_nerv`` sister
    (``experiments/train_substrate_ds_nerv.py::_full_main``).
  * ``decode_real_pairs`` / ``device_or_die`` / ``EMA`` / scorer load via
    the canonical ``tac.substrates._shared.trainer_skeleton`` + ``tac.scorer``
    + ``tac.training`` (already shared across all substrates).
- FORK_BECAUSE_PRINCIPLED_MISMATCH (stays in each substrate package):
  * architecture (``architecture.py``) — the distinguishing primitive.
  * archive grammar (``archive.py``) — each variant's ``pack_archive``
    signature differs (ia3 ships ego_poses + pose_dim; vq ships codebook +
    indices; selector_v2 ships selector_bytes + palette_size; mamba ships
    ssm_state; etc.).
  * numpy/PIL-portable inflate (``inflate.py``) — already landed per HNeRV
    parity L4 (≤200 LOC, no scorer imports, no MLX dep).
  * score-aware loss (``score_aware_loss.py``) — variant-specific extra
    terms (vq commitment loss; distilled_scorer surrogate; etc.) are wired
    via the ``compute_loss`` callback so the canonical loop never assumes a
    fixed loss signature.

The variant passes a ``compute_loss`` callback so each substrate's UNIQUE
forward + loss path stays in its own package; only the AGNOSTIC scaffold is
shared.

## MLX-first directive note (8th standing directive)

Per CLAUDE.md "MLX-FIRST NUMPY-PORTABLE INDIVIDUALLY-FRACTAL STANDING
DIRECTIVE": NEW substrates should train MLX-first. The PACT-NeRV family is
an EXISTING PyTorch architecture family (all 18 ``architecture.py`` modules
are ``torch.nn.Module`` subclasses; the canonical score-aware loss routes
through PyTorch SegNet/PoseNet via ``load_differentiable_scorers`` +
Catalog #164 ``score_pair_components_dispatch``). Migrating 18 architectures
to MLX renderers is an architecture migration, NOT a ``_full_main``
implementation; the OPTIMAL ENGINEERING for these PyTorch substrates is the
canonical PyTorch training loop (the implemented sister ``ds_nerv`` uses it).
The INFLATE path is already numpy/PIL-portable (no MLX dep) per the 8th
directive's portability requirement. The full training path is CUDA-required
(``device_or_die`` rejects MPS per Catalog #1); it is paid-GPU-gated by
``dispatch_enabled: false`` on every recipe per Catalog #325 until each
substrate clears its per-substrate symposium.

## Non-promotable by construction

Per CLAUDE.md "MPS portable-local-substrate authority" + Catalog #127/#192:
this helper does NOT promote, rank, or claim any score. Promotion requires a
paired ``[contest-CUDA]`` + ``[contest-CPU]`` anchor on 1:1 contest-compliant
hardware via the operator-authorize harness. The trainer's post-loop
auth-eval + posterior-update tail (per-variant) carries the canonical
custody discipline.

[verified-against: experiments/train_substrate_ds_nerv.py::_full_main canonical PyTorch pattern]
[verified-against: src/tac/substrates/_shared/trainer_skeleton.py decode_real_pairs/device_or_die/EMA]
[verified-against: src/tac/substrates/score_aware_common.score_pair_components_dispatch Catalog #164]
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

CONTEST_NORMALIZER: float = 37_545_489.0
EVAL_HW: tuple[int, int] = (384, 512)
N_PAIRS_FULL: int = 600


@dataclass
class PactNervTrainingResult:
    """Result of a canonical PACT-NeRV score-aware training run.

    All fields are substrate-AGNOSTIC. The variant-specific archive packing
    consumes ``best_ema_state_dict`` (the EMA shadow, NEVER live weights per
    CLAUDE.md "EMA — NON-NEGOTIABLE").
    """

    best_ema_state_dict: dict[str, Any]
    best_val_lagrangian: float
    best_epoch: int
    train_elapsed_sec: float
    n_pairs: int
    n_train_pairs: int
    n_val_pairs: int
    epochs_completed: int
    stage_log: list[dict[str, Any]] = field(default_factory=list)
    used_end_of_training_fallback: bool = False


def decode_pairs_for_training(
    video_path: Any,
    *,
    substrate_tag: str,
    n_pairs: int = N_PAIRS_FULL,
    max_pairs: int | None = None,
    repo_root: Any | None = None,
):
    """Thin canonical wrapper over ``trainer_skeleton.decode_real_pairs``.

    Returns ``torch.Tensor`` shape ``(N, 2, 3, 384, 512)`` float32 in
    ``[0, 255]`` per Catalog #114 (real contest video; synthetic FORBIDDEN
    outside ``--smoke``).
    """
    from tac.substrates._shared.trainer_skeleton import decode_real_pairs

    return decode_real_pairs(
        video_path,
        n_pairs=n_pairs,
        substrate_tag=substrate_tag,
        max_pairs=max_pairs,
        repo_root=repo_root,
    )


def run_pact_nerv_score_aware_training(
    *,
    model: Any,
    pair_tensor: Any,
    compute_loss: Callable[..., tuple[Any, dict[str, Any]]],
    archive_bytes_proxy: Any,
    device: Any,
    output_dir: Any,
    substrate_tag: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    ema_decay: float = 0.997,
    val_pair_count: int = 64,
    val_every_epochs: int = 10,
    max_nan_strikes: int = 3,
    gt_cache: Any | None = None,
    stage_log: list[dict[str, Any]] | None = None,
    config_asdict: dict[str, Any] | None = None,
    extra_checkpoint_fields: dict[str, Any] | None = None,
) -> PactNervTrainingResult:
    """Run the canonical PACT-NeRV score-aware training loop.

    The loop is substrate-AGNOSTIC; the ``compute_loss`` callback carries the
    variant-specific forward + score-aware Lagrangian. The callback signature
    is::

        compute_loss(
            model, idx, gt_0, gt_1, archive_bytes_proxy,
            gt_pose_batch=..., gt_seg_batch=..., gt_seg_already_probs=...,
        ) -> (loss_tensor, parts_dict)

    where ``idx`` is the (B,) long pair-index batch, ``gt_0``/``gt_1`` are the
    (B, 3, H, W) ground-truth frames in ``[0, 255]``, and the callback is
    responsible for calling ``model(idx)``, scaling outputs to ``[0, 255]``,
    applying eval_roundtrip, routing through the variant's score-aware loss
    (which itself routes through ``score_pair_components_dispatch`` per
    Catalog #164), and adding any variant-specific terms (e.g. VQ commitment
    loss read off ``model.last_commitment_loss``).

    Args:
        model: the (already-built, already-``.to(device)``) substrate model.
            MUST expose ``.state_dict()`` + ``.parameters()`` (torch.nn.Module).
        pair_tensor: (N, 2, 3, H, W) float32 GT pairs in ``[0, 255]`` on
            ``device``.
        compute_loss: variant-specific forward+loss callback (see above).
        archive_bytes_proxy: scalar torch.Tensor (closed-form weight-byte
            proxy) on ``device``; passed through to ``compute_loss``.
        device: torch device (CUDA-required for non-smoke; MPS rejected
            upstream via ``device_or_die``).
        output_dir: pathlib.Path; ``best.pt`` is written here.
        substrate_tag: canonical substrate id (for logging).
        epochs / batch_size / lr / weight_decay / grad_clip / ema_decay /
        val_pair_count / val_every_epochs / max_nan_strikes: training hparams.
        gt_cache: optional F3 GTScorerCache (Catalog #228); when present the
            loop looks up per-pair-index batched scorer GT and threads it into
            ``compute_loss`` via ``gt_pose_batch`` / ``gt_seg_batch`` /
            ``gt_seg_already_probs``.
        stage_log: optional list accumulating ``{"stage", "at"}`` markers.
        config_asdict: optional config dict serialized into the checkpoint.
        extra_checkpoint_fields: optional extra fields merged into the
            best.pt dict (e.g. variant-specific quantization metadata).

    Returns:
        PactNervTrainingResult with the best EMA shadow state_dict.

    Raises:
        RuntimeError: NaN watchdog tripped (``max_nan_strikes`` consecutive
            non-finite losses) — aborts to preserve the EMA shadow.
    """
    import torch

    from tac.substrates._shared.trainer_skeleton import utc_now_iso
    from tac.training import EMA

    if stage_log is None:
        stage_log = []

    def _stage(name: str) -> None:
        stage_log.append({"stage": name, "at": utc_now_iso()})

    n_pairs = int(pair_tensor.shape[0])
    val_count = max(1, min(val_pair_count, max(1, n_pairs // 8)))
    val_idx_start = n_pairs - val_count
    train_indices = torch.arange(0, val_idx_start, device=device, dtype=torch.long)
    val_indices = torch.arange(val_idx_start, n_pairs, device=device, dtype=torch.long)

    ema = EMA(model, decay=ema_decay)
    _stage(f"ema_wired_decay_{ema_decay}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best_path = output_dir / "best.pt"

    n_train = int(train_indices.shape[0])
    bs = max(1, batch_size)
    best_val_lag = math.inf
    best_epoch = -1
    nan_strike = 0
    train_started_at = time.time()

    def _gt_cache_lookup(idx: Any):
        if gt_cache is None:
            return None, None, None
        gt_pose_batch, gt_seg_batch = gt_cache.lookup(idx, device=device)
        return gt_pose_batch, gt_seg_batch, gt_cache.seg_already_probs

    for epoch in range(epochs):
        model.train()
        perm = train_indices[torch.randperm(n_train, device=device)]
        epoch_loss_sum = 0.0
        epoch_batches = 0
        for start in range(0, n_train, bs):
            idx = perm[start : start + bs]
            if idx.numel() == 0:
                continue
            gt = pair_tensor[idx]
            gt_0 = gt[:, 0]
            gt_1 = gt[:, 1]
            gt_pose_batch, gt_seg_batch, gt_seg_already_probs = _gt_cache_lookup(idx)
            loss, _parts = compute_loss(
                model,
                idx,
                gt_0,
                gt_1,
                archive_bytes_proxy,
                gt_pose_batch=gt_pose_batch,
                gt_seg_batch=gt_seg_batch,
                gt_seg_already_probs=gt_seg_already_probs,
            )
            if not torch.isfinite(loss):
                nan_strike += 1
                print(
                    f"[full:{substrate_tag}] WARN non-finite loss epoch {epoch} "
                    f"batch {start}; strike {nan_strike}/{max_nan_strikes}"
                )
                if nan_strike >= max_nan_strikes:
                    raise RuntimeError(
                        f"NaN watchdog: {nan_strike} consecutive non-finite losses; aborting to preserve EMA shadow."
                    )
                optimizer.zero_grad(set_to_none=True)
                continue
            nan_strike = 0
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            ema.update(model)
            epoch_loss_sum += float(loss.detach().item())
            epoch_batches += 1

        scheduler.step()
        avg_loss = epoch_loss_sum / max(1, epoch_batches)

        is_val_epoch = (epoch + 1) % val_every_epochs == 0 or epoch == epochs - 1
        if is_val_epoch:
            orig_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.apply(model)
            model.eval()
            with torch.no_grad():
                vgt = pair_tensor[val_indices]
                v_pose, v_seg, v_probs = _gt_cache_lookup(val_indices)
                val_loss, _vp = compute_loss(
                    model,
                    val_indices,
                    vgt[:, 0],
                    vgt[:, 1],
                    archive_bytes_proxy,
                    gt_pose_batch=v_pose,
                    gt_seg_batch=v_seg,
                    gt_seg_already_probs=v_probs,
                )
            val_lag = float(val_loss.detach().item())
            model.load_state_dict(orig_state)
            model.train()
            print(
                f"[full:{substrate_tag}] epoch {epoch + 1}/{epochs} "
                f"train_avg_loss={avg_loss:.6f} val_lagrangian={val_lag:.6f} "
                f"(best={best_val_lag:.6f} @ ep{best_epoch + 1})"
            )
            if val_lag < best_val_lag and math.isfinite(val_lag):
                best_val_lag = val_lag
                best_epoch = epoch
                ema_state = ema.state_dict()
                ckpt = {
                    "state_dict": {k: v.detach().cpu() for k, v in ema_state.items()},
                    "config": config_asdict,
                    "ema_decay": ema_decay,
                    "best_val_lagrangian": val_lag,
                    "best_epoch": int(epoch),
                    "saved_at_utc": utc_now_iso(),
                    "substrate_tag": substrate_tag,
                    "training_axis_note": ("[contest-CUDA] for promotion; auth eval still required"),
                }
                if extra_checkpoint_fields:
                    ckpt.update(extra_checkpoint_fields)
                torch.save(ckpt, ckpt_best_path)

    train_elapsed_sec = time.time() - train_started_at
    _stage(f"train_complete_elapsed_{int(train_elapsed_sec)}s")

    used_fallback = False
    if not ckpt_best_path.is_file():
        used_fallback = True
        print(f"[full:{substrate_tag}] WARN no improving val checkpoint; saving EMA shadow at end-of-training.")
        ema_state = ema.state_dict()
        ckpt = {
            "state_dict": {k: v.detach().cpu() for k, v in ema_state.items()},
            "config": config_asdict,
            "ema_decay": ema_decay,
            "best_val_lagrangian": best_val_lag,
            "best_epoch": int(epochs - 1),
            "saved_at_utc": utc_now_iso(),
            "substrate_tag": substrate_tag,
            "fallback_end_of_training_save": True,
        }
        if extra_checkpoint_fields:
            ckpt.update(extra_checkpoint_fields)
        torch.save(ckpt, ckpt_best_path)

    best_blob = torch.load(ckpt_best_path, map_location="cpu", weights_only=False)
    return PactNervTrainingResult(
        best_ema_state_dict=best_blob["state_dict"],
        best_val_lagrangian=(best_val_lag if math.isfinite(best_val_lag) else float("nan")),
        best_epoch=int(best_epoch),
        train_elapsed_sec=float(train_elapsed_sec),
        n_pairs=n_pairs,
        n_train_pairs=int(train_indices.shape[0]),
        n_val_pairs=int(val_indices.shape[0]),
        epochs_completed=int(epochs),
        stage_log=stage_log,
        used_end_of_training_fallback=used_fallback,
    )


def write_contest_runtime(
    submission_dir: Any,
    *,
    substrate_pkg_name: str,
    repo_root: Any,
    runtime_module_files: Sequence[str] = ("architecture.py", "archive.py", "inflate.py"),
    inflate_import_line: str | None = None,
) -> None:
    """Emit the contest-compliant ``inflate.sh`` + ``inflate.py`` pair.

    Canonical per-variant runtime emission (mirrors ds_nerv ``_write_runtime``)
    parameterized by substrate package name. Per Catalog #146 semantics:

    * 3-positional-arg ``inflate.sh`` ($1=archive_dir $2=output_dir $3=file_list)
    * ``set -euo pipefail``
    * NO runtime network/dependency fetches
    * NO scorer-network imports in ``inflate.py`` (strict-scorer-rule)
    * per-video loop in ``inflate.py``
    * vendors the substrate package modules into ``submission/src/tac/...`` so
      the inflate path is self-contained per Catalog #295 (no PYTHONPATH shim
      depending on the dev repo); numpy/PIL-portable (no MLX dep) per the 8th
      standing directive.

    Args:
        submission_dir: pathlib.Path; the contest submission directory.
        substrate_pkg_name: e.g. ``"pact_nerv_ia3"`` (the substrates/ subdir).
        repo_root: pathlib.Path of the repo root (to locate source modules).
        runtime_module_files: substrate package modules to vendor.
        inflate_import_line: the ``from tac.substrates.<pkg>.inflate import
            inflate_one_video`` line; defaults to the canonical form.
    """
    import shutil

    submission_dir.mkdir(parents=True, exist_ok=True)
    runtime_pkg = submission_dir / "src" / "tac" / "substrates" / substrate_pkg_name
    runtime_pkg.mkdir(parents=True, exist_ok=True)
    for pkg_init in (
        submission_dir / "src" / "tac" / "__init__.py",
        submission_dir / "src" / "tac" / "substrates" / "__init__.py",
        runtime_pkg / "__init__.py",
    ):
        pkg_init.write_text("", encoding="utf-8")
    substrate_src = repo_root / "src" / "tac" / "substrates" / substrate_pkg_name
    for name in runtime_module_files:
        src_file = substrate_src / name
        if src_file.is_file():
            shutil.copy2(src_file, runtime_pkg / name)

    if inflate_import_line is None:
        inflate_import_line = f"from tac.substrates.{substrate_pkg_name}.inflate import inflate_one_video"

    inflate_sh = (
        "#!/usr/bin/env bash\n"
        f"# {substrate_pkg_name} contest-compliant inflate "
        "(PACT-NERV-FULL-MAIN-WAVE 2026-05-27)\n"
        "# Contract: $1=archive_dir $2=output_dir $3=file_list\n"
        "set -euo pipefail\n"
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'DATA_DIR="$1"\n'
        'OUTPUT_DIR="$2"\n'
        'FILE_LIST="$3"\n'
        'mkdir -p "$OUTPUT_DIR"\n'
        'export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"\n'
        'exec "${PYTHON:-python3}" "$HERE/inflate.py" '
        '"$DATA_DIR" "$OUTPUT_DIR" "$FILE_LIST"\n'
    )
    (submission_dir / "inflate.sh").write_text(inflate_sh, encoding="utf-8")
    (submission_dir / "inflate.sh").chmod(0o755)

    inflate_py = (
        "#!/usr/bin/env python\n"
        f'"""{substrate_pkg_name} contest-compliant inflate runtime.\n'
        "\n"
        "Reads archive_dir/0.bin via the packaged substrate parser, then for\n"
        "each base in file_list writes per-frame .png under output_dir/<base>/.\n"
        "No scorer-network imports (strict-scorer-rule contract).\n"
        '"""\n'
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "HERE = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(HERE / 'src'))\n"
        f"{inflate_import_line}\n"
        "\n"
        "def main() -> int:\n"
        "    if len(sys.argv) != 4:\n"
        "        print('usage: inflate.py <archive_dir> <output_dir> <file_list>',\n"
        "              file=sys.stderr)\n"
        "        return 2\n"
        "    archive_dir = Path(sys.argv[1])\n"
        "    output_dir = Path(sys.argv[2])\n"
        "    file_list_path = Path(sys.argv[3])\n"
        "    archive_bytes = (archive_dir / '0.bin').read_bytes()\n"
        "    for line in file_list_path.read_text(encoding='utf-8').splitlines():\n"
        "        line = line.strip()\n"
        "        if not line:\n"
        "            continue\n"
        "        base = line.rsplit('.', 1)[0]\n"
        "        inflate_one_video(archive_bytes, output_dir / base, device='cpu')\n"
        "    return 0\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    sys.exit(main())\n"
    )
    (submission_dir / "inflate.py").write_text(inflate_py, encoding="utf-8")


def build_archive_zip(
    archive_zip_path: Any,
    *,
    bin_bytes: bytes,
    submission_dir: Any,
) -> None:
    """Deterministic ``archive.zip`` (0.bin + inflate.sh + inflate.py + src/).

    Per Catalog #19 ``check_archive_builders_use_deterministic_zip``: ZipInfo
    + writestr with fixed timestamp + DEFLATE. Mirrors ds_nerv
    ``_build_archive_zip``.
    """
    import zipfile

    archive_zip_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_ts = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(archive_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zi = zipfile.ZipInfo("0.bin", date_time=fixed_ts)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zi, bin_bytes)
        for name in ("inflate.sh", "inflate.py"):
            src = submission_dir / name
            if not src.is_file():
                continue
            zi = zipfile.ZipInfo(name, date_time=fixed_ts)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, src.read_bytes())
        runtime_root = submission_dir / "src"
        if runtime_root.is_dir():
            for src in sorted(runtime_root.rglob("*.py")):
                rel = src.relative_to(submission_dir).as_posix()
                zi = zipfile.ZipInfo(rel, date_time=fixed_ts)
                zi.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(zi, src.read_bytes())


def closed_form_weight_byte_proxy(
    model: Any,
    *,
    extra_param_names: Sequence[str] = (),
    fp16: bool = True,
):
    """Closed-form archive-byte proxy: 2 bytes/param (fp16) for all params.

    Mirrors ``ds_nerv`` ``_archive_bytes_proxy_closed_form``. The proxy is a
    differentiable-constant scalar fed into the rate term of the score-aware
    Lagrangian. ``extra_param_names`` is accepted for API symmetry but the
    proxy counts ALL parameters uniformly (the per-variant archive grammar's
    actual byte cost is measured at archive-build time, not training time).

    Args:
        model: torch.nn.Module.
        extra_param_names: unused (API symmetry); documented per
            CLAUDE.md "Comment-only contracts are FORBIDDEN".
        fp16: 2 bytes/param when True; 4 bytes/param when False.

    Returns:
        scalar torch.Tensor on the model's device.
    """
    import torch

    del extra_param_names
    bytes_per = 2.0 if fp16 else 4.0
    total = sum(p.numel() for p in model.parameters())
    device = next(model.parameters()).device
    return torch.tensor(float(total) * bytes_per, dtype=torch.float32, device=device)


__all__ = [
    "CONTEST_NORMALIZER",
    "EVAL_HW",
    "N_PAIRS_FULL",
    "PactNervTrainingResult",
    "build_archive_zip",
    "closed_form_weight_byte_proxy",
    "decode_pairs_for_training",
    "run_pact_nerv_score_aware_training",
    "write_contest_runtime",
]
