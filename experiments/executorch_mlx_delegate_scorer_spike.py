# SPDX-License-Identifier: MIT
"""De-risk spike: can the OFFICIAL ExecuTorch MLX delegate export+run our exact
contest scorer (SegNet smp.Unet-efficientnet_b2 + PoseNet FastViT-T12) on the
Apple GPU at FP32 fidelity?

NOT a production integration. A bounded GO/NO-GO feasibility probe per the
operator 2026-06-11 directive. torch-CPU is the ONLY authority (CLAUDE.md); the
MLX/ExecuTorch path is [macOS-MLX/ExecuTorch research-signal] and non-promotable.

Run inside the throwaway venv:
    .venv_executorch_spike/bin/python experiments/executorch_mlx_delegate_scorer_spike.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
UPSTREAM = REPO / "upstream"
for p in (str(REPO), str(REPO / "src"), str(UPSTREAM)):
    if p not in sys.path:
        sys.path.insert(0, p)

RESULTS: dict = {
    "spike": "executorch_mlx_delegate_scorer",
    "axis_tag": "[macOS-MLX/ExecuTorch research-signal]",
    "authority": "torch-CPU (this run compares against it; never promotable)",
    "torch_version": torch.__version__,
}


def _load_frozen_scorer() -> torch.nn.Module:
    """Frozen upstream DistortionNet with real contest weights (CPU, FP32)."""
    # Patch upstream rgb_to_yuv6 to be export-traceable (it is @torch.no_grad /
    # in-place upstream which both severs gradients AND can trip torch.export).
    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally

    patch_upstream_yuv6_globally()
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    net = DistortionNet().eval()
    net.load_state_dicts(posenet_sd_path, segnet_sd_path, "cpu")
    for prm in net.parameters():
        prm.requires_grad = False
    return net.to(torch.float32)


class SegNetExportable(torch.nn.Module):
    """SegNet forward on a SINGLE last-frame RGB tensor already at model input
    size (384x512). We export the conv body only (preprocess interpolate/slice
    is done in torch-CPU host code) so the spike isolates the heavy net."""

    def __init__(self, segnet: torch.nn.Module):
        super().__init__()
        self.segnet = segnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,3,384,512)
        return self.segnet(x)  # (B,5,384,512) logits


class PoseNetExportable(torch.nn.Module):
    """PoseNet forward on the already-preprocessed 12-channel YUV6 tensor
    (B,12,384,512). Returns the 6 scored pose dims."""

    def __init__(self, posenet: torch.nn.Module):
        super().__init__()
        self.posenet = posenet

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,12,384,512)
        out = self.posenet(x)
        return out["pose"][..., :6]


def _short_op(tgt: str) -> str:
    """Compact aten op name from a verbose EdgeOpOverload repr."""
    import re

    m = re.search(r"(aten\.[a-z_0-9]+\.[A-Za-z0-9_]+)", tgt)
    return m.group(1) if m else tgt[:60]


def _enumerate_unsupported_plain_edge(exported, label: str) -> dict:
    """Support-check on the DEFAULT to_edge decomposition (conv2d -> convolution).

    This is the naive path; it mis-reports forward-conv as unsupported because
    the default decomposition rewrites conv2d into convolution.default which the
    MLX handler only accepts for transposed=True. Kept for diagnostic contrast.
    """
    from executorch.exir import to_edge

    edge = to_edge(exported)
    return _support_of_program(edge.exported_program(), label, "plain_to_edge")


def _enumerate_unsupported_partitioner_path(exported, label: str) -> dict:
    """Support-check on the PARTITIONER path that preserves conv2d.default.

    Uses ops_to_not_decompose so conv2d is NOT rewritten to convolution.default,
    matching how MLXPartitioner actually delegates. This is the honest measure
    of what MLX can take."""
    from executorch.backends.mlx import MLXPartitioner
    from executorch.exir import to_edge_transform_and_lower

    lowered = to_edge_transform_and_lower(exported, partitioner=[MLXPartitioner()])
    ep = lowered.exported_program()
    # Count nodes that ended up inside delegate (lowered_module) vs left on CPU.
    delegated = 0
    cpu_fallback_ops: dict[str, int] = {}
    for node in ep.graph.nodes:
        if node.op == "get_attr" and "lowered_module" in node.name:
            delegated += 1
        elif node.op == "call_function":
            t = _short_op(str(node.target))
            if t.startswith("aten."):
                cpu_fallback_ops[t] = cpu_fallback_ops.get(t, 0) + 1
    return {
        "label": label,
        "path": "partitioner_to_edge_transform_and_lower",
        "n_delegate_partitions": delegated,
        "n_cpu_fallback_op_kinds": len(cpu_fallback_ops),
        "cpu_fallback_ops": dict(sorted(cpu_fallback_ops.items())),
        "fully_delegatable": delegated >= 1 and len(cpu_fallback_ops) == 0,
        "has_gpu_partition": delegated >= 1,
    }


def _support_of_program(edge_program, label: str, path: str) -> dict:
    from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder

    builder = MLXProgramBuilder(edge_program)
    builder.check_support_only()
    supported_ops: dict[str, int] = {}
    unsupported_ops: dict[str, int] = {}
    for node, info in builder.node_info.items():
        if getattr(node, "op", None) != "call_function":
            continue
        tgt = _short_op(str(getattr(node, "target", node)))
        bucket = supported_ops if info.supported else unsupported_ops
        bucket[tgt] = bucket.get(tgt, 0) + 1
    return {
        "label": label,
        "path": path,
        "n_supported_nodes": sum(supported_ops.values()),
        "n_unsupported_nodes": sum(unsupported_ops.values()),
        "unsupported_ops": dict(sorted(unsupported_ops.items())),
        "fully_delegatable": len(unsupported_ops) == 0,
    }


def _try_full_lower_and_run(
    exported, example_inputs, ref_out: torch.Tensor, label: str
) -> dict:
    """Attempt full MLX lowering + GPU run; compare to torch-CPU ref."""
    from executorch.backends.mlx import MLXPartitioner
    from executorch.exir import to_edge_transform_and_lower
    from executorch.runtime import Runtime

    rec: dict = {"label": label}
    try:
        lowered = to_edge_transform_and_lower(
            exported, partitioner=[MLXPartitioner()]
        )
        et_prog = lowered.to_executorch()
        pte_path = REPO / f".omx/tmp/spike_{label}.pte"
        pte_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pte_path, "wb") as fh:
            fh.write(et_prog.buffer)
        rec["lowered_ok"] = True
        rec["pte_bytes"] = pte_path.stat().st_size
    except Exception as exc:  # noqa: BLE001
        rec["lowered_ok"] = False
        rec["lower_error"] = f"{type(exc).__name__}: {exc}"[:600]
        return rec

    try:
        rt = Runtime.get()
        program = rt.load_program(pte_path)
        method = program.load_method("forward")
        t0 = time.time()
        mlx_out = method.execute(list(example_inputs))[0]
        rec["gpu_run_ok"] = True
        rec["gpu_run_seconds"] = round(time.time() - t0, 4)
        rec.update(_fidelity(label, ref_out, mlx_out))
    except Exception as exc:  # noqa: BLE001
        rec["gpu_run_ok"] = False
        rec["run_error"] = f"{type(exc).__name__}: {exc}"[:600]
    return rec


def _fidelity(label: str, ref: torch.Tensor, got: torch.Tensor) -> dict:
    got = got.to(torch.float32)
    ref = ref.to(torch.float32)
    out: dict = {}
    if label == "segnet":
        # d_seg authority surface = per-pixel argmax FLIP count over 384x512.
        ref_arg = ref.argmax(dim=1)
        got_arg = got.argmax(dim=1)
        flips = (ref_arg != got_arg).sum().item()
        total = ref_arg.numel()
        out["seg_argmax_flips"] = int(flips)
        out["seg_total_pixels"] = int(total)
        out["seg_flip_rate"] = round(flips / total, 8)
        out["seg_logit_max_abs_err"] = round((ref - got).abs().max().item(), 6)
    else:
        num = (ref - got).pow(2).mean().item()
        den = ref.pow(2).mean().item() + 1e-12
        out["pose_mse"] = round(num, 10)
        out["pose_rel_mse"] = round(num / den, 8)
        out["pose_max_abs_err"] = round((ref - got).abs().max().item(), 8)
    return out


def main() -> None:
    t_start = time.time()
    print("[spike] loading frozen scorer (real contest weights, CPU FP32)...")
    net = _load_frozen_scorer()

    def _probe(label: str, mod: torch.nn.Module, example_in) -> dict:
        rec: dict = {}
        try:
            with torch.no_grad():
                ref = mod(*example_in)
            exported = torch.export.export(mod.eval(), example_in)
            rec["torch_export_ok"] = True
            # Diagnostic: naive plain-to_edge support (mis-reports conv).
            rec["plain_edge_support"] = _enumerate_unsupported_plain_edge(
                exported, label
            )
            # Honest: partitioner path that preserves conv2d.default.
            part = _enumerate_unsupported_partitioner_path(exported, label)
            rec["partitioner_support"] = part
            # Attempt the full lower + GPU run whenever a delegate partition
            # exists (mixed delegate/CPU is the realistic ExecuTorch mode).
            if part.get("has_gpu_partition"):
                rec["full_lower_run"] = _try_full_lower_and_run(
                    exported, example_in, ref, label
                )
        except Exception as exc:  # noqa: BLE001
            rec["torch_export_ok"] = False
            rec["export_error"] = f"{type(exc).__name__}: {exc}"[:1000]
        return rec

    print("[spike] SegNet: export + MLX support (plain-edge + partitioner)...")
    seg_mod = SegNetExportable(net.segnet)
    seg_in = (torch.randn(1, 3, 384, 512, dtype=torch.float32),)
    RESULTS["segnet"] = _probe("segnet", seg_mod, seg_in)

    print("[spike] PoseNet: export + MLX support (plain-edge + partitioner)...")
    pose_mod = PoseNetExportable(net.posenet)
    pose_in = (torch.randn(1, 12, 384, 512, dtype=torch.float32),)
    RESULTS["posenet"] = _probe("posenet", pose_mod, pose_in)

    RESULTS["wall_seconds"] = round(time.time() - t_start, 2)
    out_path = REPO / ".omx/tmp/executorch_mlx_delegate_scorer_spike_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(RESULTS, indent=2))
    print("\n===== SPIKE RESULT =====")
    print(json.dumps(RESULTS, indent=2))
    print(f"\n[spike] wrote {out_path}")


if __name__ == "__main__":
    main()
