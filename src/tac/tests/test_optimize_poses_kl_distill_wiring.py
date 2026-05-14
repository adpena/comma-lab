# SPDX-License-Identifier: MIT
"""Lane G regression: verify optimize_poses.py KL-distill SegNet-only wiring is REAL.

Lane G (KL-distill pose TTO) was blocked because the canonical
`tac.losses.kl_distill_segnet_only` helper was not wired into
`experiments/optimize_poses.py`. The Lane G subagent correctly aborted
launch rather than ship dead-flag wiring (per CLAUDE.md non-negotiable
"NEVER invent CLI flags"). These tests pin the wiring once it lands.

What the wiring must guarantee:
  A. `--kl-distill-weight` and `--kl-distill-temperature` flags exist on
     optimize_poses.py argparse (visible in --help). Default to
     0.0 / 2.0 respectively (off by default — zero overhead when unused).
  B. `optimize_poses_batch` accepts `gt_frames_pair`,
     `kl_distill_weight`, `kl_distill_temperature` parameters with sane
     defaults so existing callers (which do not pass them) keep working.
  C. The body imports the canonical helper
     `from tac.losses import kl_distill_segnet_only` — NOT the full
     `kl_distill_scorer_loss` (the latter double-counts the SegNet term
     200x and adds extra PoseNet pressure, historically causing PoseNet
     collapse per CLAUDE.md "Critical Lessons").
  D. The caller in main() builds `batch_gt_frames_pair` from `gt_frames`
     and threads it (with the two new args) through to
     `optimize_poses_batch`.
  E. The KL block is gated on `kl_distill_weight > 0` so the default
     behaviour is byte-identical to the pre-Lane-G call path.

Mirrors test_train_renderer_auth_eval_wiring.py — same dead-flag-detection
methodology applied to optimize_poses.py.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "optimize_poses.py"


# ───────────────────────────────────────────────────────────────────────
# Test 1: --help surfaces both flags (real argparse, not source-grep)
# ───────────────────────────────────────────────────────────────────────
def test_optimize_poses_argparse_includes_kl_distill_flags():
    """Run --help in a subprocess (real argparse — what the operator sees)
    and assert both flags appear. This is the canonical dead-flag check
    per CLAUDE.md: don't trust source comments, ask argparse itself."""
    env_pythonpath = f"{REPO / 'src'}:{REPO / 'upstream'}"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": env_pythonpath, "PATH": "/usr/bin:/bin"},
        timeout=30,
    )
    # --help should exit 0
    assert result.returncode == 0, (
        f"optimize_poses.py --help failed: stderr={result.stderr[:500]}"
    )
    out = result.stdout
    assert "--kl-distill-weight" in out, (
        "Lane G wiring missing: --kl-distill-weight not in --help. "
        "Without it operators cannot enable SegNet KL-distill auxiliary "
        "loss for pose TTO."
    )
    assert "--kl-distill-temperature" in out, (
        "Lane G wiring missing: --kl-distill-temperature not in --help."
    )


# ───────────────────────────────────────────────────────────────────────
# Test 2: argparse defaults preserve baseline call signature
# ───────────────────────────────────────────────────────────────────────
def test_kl_distill_default_off_matches_baseline_call_signature():
    """When --kl-distill-weight is omitted, the value must default to 0.0
    so the KL block is gated off and existing experiments produce the
    identical loss surface they did pre-Lane-G. Source-grep the argparse
    block + the call site in main() to catch silent default flips."""
    src = SCRIPT.read_text()

    # Argparse defaults: kl-distill-weight=0.0, kl-distill-temperature=2.0
    weight_default = re.search(
        r'add_argument\(\s*"--kl-distill-weight"[^)]*default\s*=\s*([0-9.]+)',
        src,
        re.DOTALL,
    )
    assert weight_default is not None, (
        "--kl-distill-weight argparse declaration not found"
    )
    assert float(weight_default.group(1)) == 0.0, (
        f"--kl-distill-weight default must be 0.0 (off), got "
        f"{weight_default.group(1)} — this would silently enable Lane G "
        f"for ALL existing pose-TTO callers."
    )

    temp_default = re.search(
        r'add_argument\(\s*"--kl-distill-temperature"[^)]*default\s*=\s*([0-9.]+)',
        src,
        re.DOTALL,
    )
    assert temp_default is not None, (
        "--kl-distill-temperature argparse declaration not found"
    )
    assert float(temp_default.group(1)) == 2.0, (
        f"--kl-distill-temperature default must be 2.0 (Hinton 2015), got "
        f"{temp_default.group(1)}"
    )

    # Call-site: optimize_poses_batch must receive kl_distill_weight=
    # args.kl_distill_weight. If the default flips and someone hardcoded
    # a non-zero value here, this catches it. Walk balanced parens to
    # extract complete arg blocks (a regex can't handle nested `(`/`)`
    # like `torch.arange(...)` inside the call). Skip the def-site.
    def _extract_calls(text: str, fn_name: str) -> list[str]:
        out = []
        idx = 0
        token = fn_name + "("
        while True:
            i = text.find(token, idx)
            if i < 0:
                break
            start = i + len(token)
            depth = 1
            j = start
            while j < len(text) and depth > 0:
                ch = text[j]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                j += 1
            if depth == 0:
                out.append(text[start:j - 1])
            idx = j
        return out

    call_blocks = _extract_calls(src, "optimize_poses_batch")
    assert call_blocks, "optimize_poses_batch call site not found"
    # Drop the def-site (contains type annotations like `torch.nn.Module`);
    # keep call sites that pass real values.
    call_sites = [
        block for block in call_blocks
        if "torch.Tensor | None" not in block and "torch.nn.Module" not in block
    ]
    assert call_sites, (
        "Could not find a non-def-site optimize_poses_batch( call. "
        "Did the call-site signature change?"
    )
    call_block = call_sites[-1]
    assert "kl_distill_weight=args.kl_distill_weight" in call_block, (
        "main() must pass kl_distill_weight=args.kl_distill_weight to "
        "optimize_poses_batch — without this the CLI flag is dead."
    )
    assert "kl_distill_temperature=args.kl_distill_temperature" in call_block, (
        "main() must pass kl_distill_temperature=args.kl_distill_temperature."
    )
    assert "gt_frames_pair=" in call_block, (
        "main() must pass gt_frames_pair= so the KL block has GT logits "
        "to distill from. Without it kl_distill_segnet_only no-ops."
    )


# ───────────────────────────────────────────────────────────────────────
# Test 3: helper imported from canonical location (no double-count trap)
# ───────────────────────────────────────────────────────────────────────
def test_kl_distill_helper_imported_correctly():
    """The wiring MUST import `kl_distill_segnet_only` (the canonical
    SegNet-only helper). It MUST NOT import `kl_distill_scorer_loss`
    (which returns 100*seg_kl + sqrt(10*pose_dist) and stacking it on
    top of scorer_loss double-counts the SegNet term 200x — this is
    the historical "KL distill caused PoseNet collapse" failure mode
    documented in CLAUDE.md "Critical Lessons")."""
    src = SCRIPT.read_text()
    assert "from tac.losses import kl_distill_segnet_only" in src, (
        "optimize_poses.py must import kl_distill_segnet_only (the canonical "
        "SegNet-only helper). Without it the KL block can't compute its "
        "loss, or worse, would import the full kl_distill_scorer_loss and "
        "double-count the SegNet term 200x."
    )
    # Negative check: the full scorer-loss helper would produce the
    # PoseNet-collapse failure mode if used as an aux on top of the
    # standard loss. Make sure we did not accidentally reach for it.
    forbidden_imports = [
        "from tac.losses import kl_distill_scorer_loss",
        "from tac.losses import (kl_distill_scorer_loss",
    ]
    for forbidden in forbidden_imports:
        assert forbidden not in src, (
            f"optimize_poses.py imports the FORBIDDEN double-count helper: "
            f"{forbidden!r}. Use kl_distill_segnet_only instead — see "
            f"CLAUDE.md 'KL distill caused PoseNet collapse'."
        )


# ───────────────────────────────────────────────────────────────────────
# Test 4 (bonus): default-off path is byte-identical to baseline
# ───────────────────────────────────────────────────────────────────────
def test_kl_distill_does_not_break_existing_call_path():
    """When kl_distill_weight=0.0, the KL block must not run at all.
    This guarantees the default call path is byte-identical to
    pre-Lane-G optimize_poses_batch behaviour. We verify two ways:
      (a) the gating condition exists in source (cheap structural check)
      (b) the function signature exposes the new parameters with safe
          defaults, so older callers that pass positional+keyword args
          without the new ones still work.
    A full end-to-end smoke would require loading scorers + a renderer
    fixture, which is heavy for a wiring regression — the structural
    check is sufficient and matches the pattern used in
    test_train_renderer_auth_eval_wiring.py."""
    src = SCRIPT.read_text()

    # (a) The gate `if <effective>weight > 0 and gt_frames_pair is not None`
    # — this is the explicit zero-overhead path. After Lane G V3-V2 the
    # effective weight is computed via `_effective_kl_weight = (
    # kl_distill_snr_controller.weight if ... else kl_distill_weight)`, so
    # accept either the original direct gate OR the post-V2 indirect gate.
    gate_re = re.compile(
        r"if\s+(?:_effective_kl_weight|kl_distill_weight)\s*>\s*0"
        r"\s+and\s+gt_frames_pair\s+is\s+not\s+None\s*:",
    )
    assert gate_re.search(src) is not None, (
        "KL-distill block must be gated on "
        "`if (_effective_kl_weight | kl_distill_weight) > 0 and "
        "gt_frames_pair is not None:` — without the gate, default callers "
        "(weight=0.0) would hit the import + branch and pay overhead, and "
        "without the `gt_frames_pair is not None` half a misconfiguration "
        "would crash with an unhelpful AttributeError."
    )

    # (b) Function signature exposes safe defaults (= 0.0, = 2.0, = None).
    sig_re = re.compile(
        r"def optimize_poses_batch\((?P<sig>.*?)\)\s*->\s*tuple",
        re.DOTALL,
    )
    sig_match = sig_re.search(src)
    assert sig_match is not None, "optimize_poses_batch signature not found"
    sig = sig_match.group("sig")
    assert "gt_frames_pair: torch.Tensor | None = None" in sig, (
        "optimize_poses_batch must accept `gt_frames_pair: torch.Tensor | "
        "None = None` so existing callers don't need to pass it."
    )
    assert "kl_distill_weight: float = 0.0" in sig, (
        "optimize_poses_batch must accept `kl_distill_weight: float = 0.0` "
        "so existing callers default to disabled."
    )
    assert "kl_distill_temperature: float = 2.0" in sig, (
        "optimize_poses_batch must accept `kl_distill_temperature: float = "
        "2.0` (Hinton 2015 default) so existing callers default to a "
        "scientifically meaningful temperature."
    )


# ───────────────────────────────────────────────────────────────────────
# Test 5 (extra paranoia): per-step log surfaces kl when active
# ───────────────────────────────────────────────────────────────────────
def test_kl_distill_log_line_includes_kl_when_active():
    """When kl_distill_weight > 0, the per-step log line must surface
    the KL value so operators can monitor it during long pose-TTO runs.
    Silent KL = wasted GPU per CLAUDE.md 'no wasted resources' rule."""
    src = SCRIPT.read_text()
    # The log block formats `kl=...` in the f-string when the weight is on.
    assert 'kl_str = f" kl={kl_loss_val:.6f}" if _effective_kl_weight > 0 else ""' in src, (
        "Per-step log must include `kl={value:.6f}` when "
        "the effective KL weight is active so the operator can see the auxiliary "
        "loss component in real time. CLAUDE.md no-wasted-resources rule."
    )


# ───────────────────────────────────────────────────────────────────────
# Test 6 (codex R5-r6 #2): KL block must NOT pass raw `pairs` to the
# helper — the SegNet scoring path uses simulate_eval_roundtrip first,
# so the KL auxiliary must operate on the SAME roundtripped distribution.
# ───────────────────────────────────────────────────────────────────────
def test_kl_distill_uses_roundtripped_frames_not_raw_pairs():
    """codex R5-r6 #2 regression test: feed roundtripped frames to the
    KL helper, NOT raw renderer output `pairs`. Lane G burned $0.85
    chasing a 350x proxy-auth gap that this fix closes.
    """
    src = SCRIPT.read_text()
    # The KL block must derive `rendered_pair_hwc_rt` (or any *_rt name)
    # from frames_chw — NOT pass `pairs` directly.
    assert "rendered_pair_hwc_rt" in src, (
        "codex R5-r6 #2 fix missing: optimize_poses.py KL block must "
        "construct `rendered_pair_hwc_rt` from the SegNet path's "
        "roundtripped frames_chw. Without this, KL gradients optimise "
        "for a different rendered distribution than the scored loss path."
    )
    # AND the call site must pass that variable to kl_distill_segnet_only.
    call_re = re.compile(
        r"kl_distill_segnet_only\s*\(\s*rendered_pair_hwc_rt",
    )
    assert call_re.search(src) is not None, (
        "codex R5-r6 #2 fix missing: kl_distill_segnet_only(...) must "
        "be called with the roundtripped HWC tensor as the first "
        "positional arg, not `pairs`."
    )
    # Defensive: the OLD wiring `kl_distill_segnet_only(pairs, ...)` (or
    # the more obvious `rendered_pair_hwc = pairs`) must be GONE.
    forbidden_aliases = ("rendered_pair_hwc = pairs", "rendered_pair_hwc=pairs")
    for alias in forbidden_aliases:
        assert alias not in src, (
            f"codex R5-r6 #2 regression: forbidden alias `{alias}` "
            f"remains in optimize_poses.py — KL block reverted to raw "
            f"renderer output."
        )


def test_snr_controller_only_mode_materializes_gt_pairs():
    """If the SNR controller is active, KL is active even when static
    --kl-distill-weight is omitted. The main loop must still build
    gt_frames_pair; otherwise the controller's positive effective weight
    silently no-ops because optimize_poses_batch gates on gt_frames_pair.
    """
    src = SCRIPT.read_text()
    assert (
        "kl_distill_active = bool(args.kl_distill_weight > 0 or "
        "kl_distill_snr_controller is not None)"
    ) in src
    assert re.search(r"if\s+kl_distill_active\s*:\s*\n\s*pair_t = torch\.stack", src), (
        "main() must materialize batch_gt_frames_pair when either static KL "
        "or the SNR controller is active. Gating only on args.kl_distill_weight "
        "makes controller-only KL a silent no-op."
    )
    assert "kl_distill_snr_controller=kl_distill_snr_controller" in src
