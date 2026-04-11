from __future__ import annotations


def render_bootstrap(*, required_symbols: tuple[str, ...], dataset_hint: str) -> str:
    quoted = ", ".join(repr(name) for name in required_symbols)
    return f"""
def tac_has_required_entrypoints(module: object) -> bool:
    required = {{{quoted}}}
    return all(hasattr(module, name) for name in required)


def find_tac_wheel_candidates(*, input_root: Path = Path("/kaggle/input"), script_dir: Path = SCRIPT_PATH.parent) -> list[Path]:
    candidates = [*sorted(script_dir.glob("comma_video_lab_ball_pack-*.whl"))]
    exact_root = input_root / "{dataset_hint}"
    candidates.extend(sorted(exact_root.glob("comma_video_lab_ball_pack-*.whl")))
    if input_root.exists():
        candidates.extend(sorted(input_root.rglob("comma_video_lab_ball_pack-*.whl")))
    return candidates


def ensure_tac_importable() -> None:
    try:
        import tac  # noqa: F401
        from tac import entrypoints as tac_entrypoints  # type: ignore
        if tac_has_required_entrypoints(tac_entrypoints):
            return
    except ImportError:
        pass

    wheel_candidates = find_tac_wheel_candidates()
    if not wheel_candidates:
        input_root = Path("/kaggle/input")
        visible = sorted(str(path) for path in input_root.glob("*")) if input_root.exists() else []
        raise ImportError(
            f"tac is not importable and no bundled wheel was found for Kaggle bootstrap; "
            f"visible input roots={{visible}}"
        )
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(wheel_candidates[0])])
""".strip()
