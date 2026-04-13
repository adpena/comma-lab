#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Load tac deploy_config so kernel specs stay in parity with Modal + Lightning
_REPO_ROOT_PROBE = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT_PROBE / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_PROBE / "src"))
from tac.deploy.deploy_config import ALL_VARIANTS  # noqa: E402

from kaggle_kernel_builder import KaggleKernelSpec, write_bundle


REPO_ROOT = Path(__file__).resolve().parents[4]
KAGGLE_ROOT = REPO_ROOT / "experiments" / "kaggle_kernels"
KAGGLE_CREDS = Path.home() / ".kaggle" / "kaggle.json"
ASSET_DATASET_REF = "adpena/comma-lab-private-assets"


def load_tac_bootstrap_renderer():
    module_path = REPO_ROOT / "src" / "tac" / "bootstrap_codegen.py"
    spec = importlib.util.spec_from_file_location("tac_bootstrap_codegen", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.render_bootstrap


def kaggle_username() -> str:
    payload = json.loads(KAGGLE_CREDS.read_text())
    username = payload.get("username")
    if not isinstance(username, str) or not username:
        raise ValueError(f"Missing username in {KAGGLE_CREDS}")
    return username


def _asym_warp_variant_preamble(variant: str, resume_from: str | None = None) -> str:
    """Bootstrap preamble that sets ASYM_VARIANT (and optionally RESUME_FROM).

    Uses semicolons on a single line so write_bundle can safely prepend it to
    any Python script regardless of its import structure.

    Args:
        variant: one of ALL_VARIANTS
        resume_from: if set, path to a .pt checkpoint inside the Kaggle dataset
                     mount, e.g. /kaggle/input/comma-lab-private-assets/renderer_best_v3.pt
    """
    # Use __import__('os') to avoid duplicate 'import os' when prepended to a
    # launcher file that already imports os at module level.
    parts = [
        f'__import__("os").environ.setdefault("ASYM_VARIANT", "{variant}")',
    ]
    if resume_from:
        parts.append(f'__import__("os").environ.setdefault("RESUME_FROM", "{resume_from}")')
    return "; ".join(parts)


#: Checkpoint from asym_v3_longer_tight (ep ~12400) uploaded to the dataset.
#: Both supervised and raft_only resume from the same starting point for a
#: clean A/B comparison of the two supervision strategies.
_V3_RESUME_PATH = (
    "/kaggle/input/comma-lab-private-assets/renderer_best_v3.pt"
)

#: Variants that should resume from the v3 checkpoint rather than train from scratch.
_RESUME_VARIANTS: frozenset[str] = frozenset({"supervised", "raft_only"})


def kernel_specs() -> dict[str, KaggleKernelSpec]:
    render_bootstrap = load_tac_bootstrap_renderer()

    # --- Asymmetric warp renderer variants (in strict parity with Modal + Lightning) ---
    # One kernel spec per variant. Variant is controlled by the ASYM_VARIANT env var
    # injected via bootstrap_preamble. Flags come from tac.deploy.deploy_config at runtime.
    # NOTE: launch_policy fields (bounded, checkpoint_priority, etc.) are silently
    # ignored by the Kaggle server — they are metadata-only documentation.
    asym_warp_specs = {}
    for variant in ALL_VARIANTS:
        resume_from = _V3_RESUME_PATH if variant in _RESUME_VARIANTS else None
        asym_warp_specs[f"asym_warp_{variant}"] = KaggleKernelSpec(
            slug=f"comma-lab-asym-warp-{variant.replace('_', '-')}",
            title=f"comma-lab asym warp {variant}",
            code_source=REPO_ROOT / "experiments" / "kaggle_asym_warp_launcher.py",
            code_file="kaggle_asym_warp_launcher.py",
            include_paths=(
                REPO_ROOT / "experiments" / "train_renderer_fridrich.py",
            ),
            dataset_sources=(ASSET_DATASET_REF,),
            bootstrap_preamble=_asym_warp_variant_preamble(variant, resume_from=resume_from),
        )

    return {
        **asym_warp_specs,
        "dilated_h64_long1000": KaggleKernelSpec(
            slug="comma-lab-dilated-h64-long1000",
            title="comma-lab dilated h64 long1000",
            code_source=REPO_ROOT / "experiments" / "train_postfilter_dilated_h64.py",
            code_file="train_postfilter_dilated_h64.py",
            dataset_sources=(ASSET_DATASET_REF,),
            bootstrap_preamble=render_bootstrap(
                required_symbols=(
                    "build_postfilter_meta",
                    "make_dilated_default_tag",
                    "resolve_cloud_asset_bundle",
                    "resolve_cloud_output_dir",
                    "save_best_checkpoint",
                    "save_final_artifacts",
                    "normalize_archive_source_path",
                ),
                dataset_hint="comma-lab-private-assets",
            ),
        ),
        "segnet_attack_fixed_h32": KaggleKernelSpec(
            slug="comma-lab-segnet-attack-fixed-h32",
            title="comma-lab segnet attack fixed h32",
            code_source=REPO_ROOT / "experiments" / "cloud_segnet_attack_h32_trainer.py",
            code_file="cloud_segnet_attack_h32_trainer.py",
            dataset_sources=(ASSET_DATASET_REF,),
            bootstrap_preamble=render_bootstrap(
                required_symbols=(
                    "build_postfilter_meta",
                    "make_fixed_h32_segnet_tag",
                    "resolve_cloud_asset_bundle",
                    "resolve_cloud_base_dir",
                    "resolve_cloud_output_dir",
                    "save_best_checkpoint",
                    "save_final_artifacts",
                    "normalize_archive_source_path",
                ),
                dataset_hint="comma-lab-private-assets",
            ),
        ),
        "pairaware_smoke": KaggleKernelSpec(
            slug="comma-lab-pairaware-smoke",
            title="comma-lab pairaware smoke",
            code_source=REPO_ROOT / "experiments" / "train_postfilter_pairaware.py",
            code_file="train_postfilter_pairaware.py",
            dataset_sources=(ASSET_DATASET_REF,),
        ),
        "constrained_gen_smoke": KaggleKernelSpec(
            slug="comma-lab-constrained-gen-smoke",
            title="comma-lab constrained gen smoke",
            code_source=REPO_ROOT / "experiments" / "kaggle_constrained_gen_launcher.py",
            code_file="kaggle_constrained_gen_launcher.py",
            dataset_sources=(ASSET_DATASET_REF,),
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build Kaggle kernel bundles for the next experiment lanes.")
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset of kernel spec keys to build.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=KAGGLE_ROOT,
        help="Directory where bundle folders will be written.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    username = kaggle_username()
    specs = kernel_specs()
    selected = args.only or list(specs.keys())
    for key in selected:
        spec = specs[key]
        bundle_dir = args.output_root / key
        write_bundle(bundle_dir=bundle_dir, username=username, spec=spec, repo_root=REPO_ROOT)
        print(f"built {key} -> {bundle_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
