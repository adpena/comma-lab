#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
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

    # Stage 3.5: Preflight — verify all required assets exist ON DISK with nonzero size.
    # Kaggle mounts may show files before they're fully synced from object storage.
    # Retry up to 5 times with 60s sleep to handle mount propagation delay.
    import time as _time
    # Kaggle mount path varies: older kernels use /kaggle/input/<slug>,
    # newer kernels use /kaggle/input/datasets/<owner>/<slug>.
    # Search both to be robust across Kaggle platform changes.
    _asset_root = _input_root / "comma-lab-private-assets"
    if not _asset_root.exists():
        _asset_root = _input_root / "datasets" / "adpena" / "comma-lab-private-assets"
    _required = ["raft_flow.pt", "renderer_best_v3.pt", "posenet_targets.bin", "0.mkv"]
    for _attempt in range(1, 6):
        _missing = []
        for _f in _required:
            _p = _asset_root / _f
            if not _p.exists():
                _missing.append(_f)
            elif _p.stat().st_size == 0:
                _missing.append(f"{{_f}} (0 bytes — mount incomplete)")
        if not _missing:
            print(f"  Preflight OK: {{len(_required)}} required assets verified on disk (attempt {{_attempt}})")
            break
        if _attempt < 5:
            print(f"  Preflight attempt {{_attempt}}: {{len(_missing)}} missing — {{_missing}}. Retrying in 60s...")
            _time.sleep(60)
    else:
        _available = sorted(str(p.name) for p in _asset_root.iterdir()) if _asset_root.exists() else []
        raise FileNotFoundError(
            f"Preflight FAILED after 5 attempts: {{len(_missing)}} asset(s) missing/incomplete:\\n"
            f"  Missing: {{_missing}}\\n"
            f"  Mount: {{_asset_root}}\\n"
            f"  Available: {{_available}}\\n"
            f"  Dataset mount never propagated. Re-push after longer wait."
        )

    # Stage 4.5: Redirect results to /kaggle/working/ (writable).
    # The training script sets RESULTS_DIR = Path(__file__).parent / "results" / ...
    # On Kaggle, __file__ = /kaggle/src/script.py, so results lands in /kaggle/src/results
    # which is READ-ONLY. Create a symlink so writes go to /kaggle/working/.
    _results_parent = _Path(__file__).resolve().parent / "results"
    if not _results_parent.exists():
        _working_results = _Path("/kaggle/working/results")
        _working_results.mkdir(parents=True, exist_ok=True)
        try:
            _results_parent.symlink_to(_working_results)
            print(f"  Results redirected: {{_results_parent}} -> {{_working_results}}")
        except OSError:
            # /kaggle/src might truly be read-only even for symlinks
            # Fallback: override RESULTS_DIR via environment variable
            _os.environ["TAC_RESULTS_DIR"] = str(_working_results / "fridrich_renderer")
            print(f"  Results env override: TAC_RESULTS_DIR={{_os.environ['TAC_RESULTS_DIR']}}")

    # Stage 5: resolve assets and build CLI flags via canonical runner logic
    _resume = _os.environ.get("RESUME_FROM") or None
    if _resume and not _Path(_resume).exists():
        # Try the new Kaggle mount path (datasets/owner/slug/)
        _alt_resume = _asset_root / _Path(_resume).name
        if _alt_resume.exists():
            _resume = str(_alt_resume)
            print(f"  RESUME_FROM resolved to new path: {{_resume}}")
        else:
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


# Execute setup at module scope — MUST run before the training script's
# imports (torch, click, etc.) are parsed. Guarding with __main__ is wrong:
# Python parses ALL imports before executing __main__, so torch/click would
# fail before _kaggle_setup() installs them.
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


# Required assets and their expected byte sizes.
# Used by wait_for_dataset_ready() to confirm all files are fully uploaded
# before pushing kernels. Sizes are checked with 1% tolerance to handle
# minor variations across dataset versions.
REQUIRED_DATASET_ASSETS: dict[str, int] = {
    "raft_flow.pt": 943_735_027,
    "renderer_best_v3.pt": 3_527_290,
    "posenet_targets.bin": 6_794,
    "0.mkv": 37_545_489,
    "tac-1.0.5-py3-none-any.whl": 0,  # size varies; 0 = existence check only
}


def _kaggle_bin() -> str:
    """Return the kaggle CLI binary path, preferring the venv-local install."""
    # Try venv .venv/bin/kaggle relative to repo root, then PATH
    venv_kaggle = REPO_ROOT / ".venv" / "bin" / "kaggle"
    if venv_kaggle.exists():
        return str(venv_kaggle)
    import shutil
    found = shutil.which("kaggle")
    if found:
        return found
    raise FileNotFoundError(
        "kaggle CLI not found. Install with: uv pip install kaggle"
    )


def wait_for_dataset_ready(
    dataset_ref: str = ASSET_DATASET_REF,
    required: dict[str, int] | None = None,
    poll_interval: int = 60,
    max_attempts: int = 30,
) -> None:
    """Block until all required files are present in the Kaggle dataset at full size.

    This is the atomicity guard for the dataset→kernel deployment pipeline.
    Kaggle's `datasets version` command acknowledges the upload but files may
    not be accessible inside kernel mounts for several minutes (observed: up to
    10 min for 944MB files). Pushing a kernel before the dataset is ready causes
    a FileNotFoundError deep inside the kernel bootstrap — hours into a run.

    Args:
        dataset_ref:   Kaggle dataset ref (owner/slug).
        required:      Dict of filename→expected_size. Defaults to REQUIRED_DATASET_ASSETS.
        poll_interval: Seconds between API polls (default 60).
        max_attempts:  Max polls before giving up (default 30 = 30 min).
    """
    if required is None:
        required = REQUIRED_DATASET_ASSETS
    kaggle = _kaggle_bin()

    print(f"\n=== Waiting for dataset {dataset_ref!r} to be fully ready ===")
    print(f"  Required files: {list(required.keys())}")
    print(f"  Poll interval: {poll_interval}s, max wait: {max_attempts * poll_interval}s\n")

    for attempt in range(1, max_attempts + 1):
        result = subprocess.run(
            [kaggle, "datasets", "files", dataset_ref, "--csv"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  [attempt {attempt}] kaggle datasets files failed: {result.stderr.strip()}")
        else:
            # CSV format: name,size,creationDate
            present: dict[str, int] = {}
            for line in result.stdout.splitlines()[1:]:  # skip header
                parts = line.split(",")
                if len(parts) >= 2:
                    fname = parts[0].strip()
                    try:
                        size = int(parts[1].strip())
                    except ValueError:
                        size = 0
                    present[fname] = size

            missing = []
            wrong_size = []
            for fname, expected_size in required.items():
                if fname not in present:
                    missing.append(fname)
                elif expected_size > 0 and abs(present[fname] - expected_size) > expected_size * 0.01:
                    wrong_size.append(
                        f"{fname}: got {present[fname]:,} expected ~{expected_size:,}"
                    )

            if not missing and not wrong_size:
                print(f"  [attempt {attempt}] All {len(required)} required files present. Dataset ready.")
                return

            if missing:
                print(f"  [attempt {attempt}] Missing: {missing}")
            if wrong_size:
                print(f"  [attempt {attempt}] Wrong size (still uploading?): {wrong_size}")

        if attempt < max_attempts:
            print(f"  Retrying in {poll_interval}s ...")
            time.sleep(poll_interval)

    raise TimeoutError(
        f"Dataset {dataset_ref!r} not ready after {max_attempts * poll_interval}s. "
        "Check the Kaggle dataset page for upload errors."
    )


def push_kernels(bundle_dirs: list[Path]) -> None:
    """Push each kernel bundle to Kaggle, failing fast if any push fails."""
    kaggle = _kaggle_bin()
    for bundle_dir in bundle_dirs:
        print(f"  Pushing kernel: {bundle_dir.name} ...")
        result = subprocess.run(
            [kaggle, "kernels", "push", "-p", str(bundle_dir)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"kaggle kernels push failed for {bundle_dir.name}:\n{result.stderr}"
            )
        print(f"  {result.stdout.strip()}")
    print(f"\n  {len(bundle_dirs)} kernel(s) pushed.")


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
    p.add_argument(
        "--push",
        action="store_true",
        default=False,
        help=(
            "After building, wait for the Kaggle dataset to be fully available "
            "(all required assets at expected size), then push the kernels. "
            "This is the atomic deploy path — never push without this flag "
            "after uploading a new dataset version."
        ),
    )
    p.add_argument(
        "--skip-dataset-wait",
        action="store_true",
        default=False,
        help="With --push: skip the dataset readiness wait (use only if dataset was already verified).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    username = kaggle_username()
    specs = kernel_specs()
    selected = args.only or list(specs.keys())
    bundle_dirs = []
    for key in selected:
        spec = specs[key]
        bundle_dir = args.output_root / key
        write_bundle(bundle_dir=bundle_dir, username=username, spec=spec, repo_root=REPO_ROOT)
        print(f"built {key} -> {bundle_dir}")
        bundle_dirs.append(bundle_dir)

    if args.push:
        if not args.skip_dataset_wait:
            wait_for_dataset_ready()
        push_kernels(bundle_dirs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
