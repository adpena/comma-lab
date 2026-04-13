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
    """Full Kaggle bootstrap preamble injected into train_renderer_fridrich.py.

    Kaggle script kernels only upload the single code_file — no other bundle
    files reach the kernel environment.  The solution: make train_renderer_fridrich.py
    the code_file and inject a preamble that:

      1. Installs the tac wheel from the Kaggle dataset (stdlib only at this point)
      2. Verifies via cloud_bootstrap
      3. Installs runtime deps + clones the upstream scorer repo
      4. Resolves supervision assets from the dataset
      5. Builds the correct CLI flags via runner.build_kaggle_command()
      6. Injects the flags into sys.argv so Click reads them when main() runs

    All logic is delegated to existing canonical runner.py infrastructure — no
    duplication, no ad-hoc paths.

    Args:
        variant: one of ALL_VARIANTS ('base', 'supervised', 'raft_only')
        resume_from: optional path to a .pt checkpoint inside the Kaggle dataset
                     mount, e.g. /kaggle/input/comma-lab-private-assets/renderer_best_v3.pt
    """
    resume_line = f"_RESUME_FROM = {repr(resume_from)}"
    return f"""\
# --- Kaggle bootstrap (injected by build_kaggle_kernels.py — do not edit) ---
import os as _os, sys as _sys, shutil as _shutil, subprocess as _subprocess
from pathlib import Path as _Path

_ASYM_VARIANT = {repr(variant)}
{resume_line}


def _kaggle_setup() -> None:
    \"\"\"Install tac, deps, upstream scorer; inject CLI argv before main() runs.\"\"\"
    _os.environ.setdefault("ASYM_VARIANT", _ASYM_VARIANT)
    if _RESUME_FROM:
        _os.environ.setdefault("RESUME_FROM", _RESUME_FROM)
    _os.environ.setdefault("PYTHONUNBUFFERED", "1")
    _input_root = _Path(_os.environ.get("CLOUD_INPUT_ROOT", "/kaggle/input"))

    # Stage 1: install tac wheel (stdlib only — tac not yet importable)
    try:
        import tac  # noqa: F401
    except ImportError:
        _hits = sorted(_input_root.rglob("tac-*.whl"))
        if not _hits:
            raise ImportError(f"tac wheel not found in {{_input_root}}")
        _wheel = _hits[-1]
        _uv = _shutil.which("uv") or next(
            (str(_p) for _p in (
                _Path.home() / ".local" / "bin" / "uv",
                _Path.home() / ".cargo" / "bin" / "uv",
            ) if _Path(_p).exists()),
            None,
        )
        _cmd_base = (
            [_uv, "pip", "install", "--system", "-q", "--no-deps"]
            if _uv else [_sys.executable, "-m", "pip", "install", "-q", "--no-deps"]
        )
        _subprocess.check_call(_cmd_base + [str(_wheel)])

    # Stage 2: full post-install verification via cloud_bootstrap
    from tac.deploy.cloud_bootstrap import bootstrap as _cb
    _cb(_input_root, verify_submodule="tac.deploy.kaggle.runner")

    # Stage 3: install runtime deps + clone upstream scorer repo
    from tac.deploy.kaggle.runner import (
        ensure_deps as _ensure_deps,
        ensure_upstream as _ensure_upstream,
        set_training_env as _set_training_env,
        build_kaggle_command as _build_cmd,
    )
    _ensure_deps()
    _upstream = _ensure_upstream()
    _set_training_env(_upstream)

    # Stage 4: resolve assets and build CLI flags via canonical runner logic
    _asset_root = _input_root / "comma-lab-private-assets"
    _resume = _os.environ.get("RESUME_FROM") or None
    if _resume and not _Path(_resume).exists():
        print(f"  WARNING: RESUME_FROM not found: {{_resume}} — starting from scratch")
        _resume = None

    _cmd = _build_cmd(
        variant=_ASYM_VARIANT,
        script_path=_Path(__file__),
        asset_root=_asset_root,
        resume_from=_resume,
    )
    # _cmd = [sys.executable, script_path, flag1, value1, ...]
    # Inject training flags into sys.argv; Click reads them when main() runs below.
    _sys.argv = [_sys.argv[0]] + list(_cmd[2:])
    print(f"  [Kaggle] variant={{_ASYM_VARIANT}} | argv={{_sys.argv[1:]}}")


if __name__ == "__main__":
    _kaggle_setup()
# --- end Kaggle bootstrap ---"""


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
            # The training script IS the code_file. Kaggle script kernels only
            # upload code_file — include_paths are local-only build artifacts
            # that never reach the Kaggle kernel environment.  The bootstrap
            # preamble handles all setup inline before main() runs.
            code_source=REPO_ROOT / "experiments" / "train_renderer_fridrich.py",
            code_file="train_renderer_fridrich.py",
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
