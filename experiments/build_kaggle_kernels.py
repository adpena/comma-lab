#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from kaggle_kernel_builder import KaggleKernelSpec, write_bundle


REPO_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_ROOT = REPO_ROOT / "experiments" / "kaggle_kernels"
KAGGLE_CREDS = Path.home() / ".kaggle" / "kaggle.json"
ASSET_DATASET_REF = "adpena/comma-lab-private-assets"


def kaggle_username() -> str:
    payload = json.loads(KAGGLE_CREDS.read_text())
    username = payload.get("username")
    if not isinstance(username, str) or not username:
        raise ValueError(f"Missing username in {KAGGLE_CREDS}")
    return username


def kernel_specs() -> dict[str, KaggleKernelSpec]:
    return {
        "dilated_h64_long1000": KaggleKernelSpec(
            slug="comma-lab-dilated-h64-long1000",
            title="comma-lab dilated h64 long1000",
            code_source=REPO_ROOT / "experiments" / "train_postfilter_dilated_h64.py",
            code_file="train_postfilter_dilated_h64.py",
            dataset_sources=(ASSET_DATASET_REF,),
            include_paths=(
                REPO_ROOT / "experiments" / "masks" / "posenet_saliency.npy",
                REPO_ROOT / "reports" / "raw" / "2026-04-06-av1-roi-experiments" / "decode_base_archive.zip",
            ),
        ),
        "segnet_attack_fixed_h32": KaggleKernelSpec(
            slug="comma-lab-segnet-attack-fixed-h32",
            title="comma-lab segnet attack fixed h32",
            code_source=REPO_ROOT / "experiments" / "cloud_segnet_attack_h32_trainer.py",
            code_file="cloud_segnet_attack_h32_trainer.py",
            dataset_sources=(ASSET_DATASET_REF,),
            include_paths=(
                REPO_ROOT / "reports" / "raw" / "2026-04-06-av1-roi-experiments" / "decode_base_archive.zip",
            ),
        ),
        "pairaware_smoke": KaggleKernelSpec(
            slug="comma-lab-pairaware-smoke",
            title="comma-lab pairaware smoke",
            code_source=REPO_ROOT / "experiments" / "train_postfilter_pairaware.py",
            code_file="train_postfilter_pairaware.py",
            include_paths=(
                REPO_ROOT / "experiments" / "train_postfilter_pairaware.py",
            ),
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
