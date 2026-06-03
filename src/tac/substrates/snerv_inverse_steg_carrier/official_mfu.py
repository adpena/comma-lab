# SPDX-License-Identifier: MIT
"""Official SNeRV MFU graph and numeric primitives.

This module models the source-backed MFU contract from the official SNeRV
checkout without importing torch. It is intended for trainer/export plumbing:
ConvTranspose2d output shapes, skip-concat compatibility, RB channel interfaces,
and small numeric reference forwards are executable here. Full source-forward
parity remains blocked until real official weights are mapped into these
portable kernels and replayed against the official torch graph.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

import numpy as np

from tac.substrates.snerv_inverse_steg_carrier.official_hfr import (
    OfficialConv2dNchw,
    leaky_relu01,
)

OFFICIAL_SNERV_MFU_SOURCE: Final[str] = (
    "SNeRV/model/snerv.py lines 68-71 and 104-109 at "
    "0844a08f9591eea9625f8b961ed91d08030e06d1"
)
OFFICIAL_SNERV_RB_SOURCE: Final[str] = (
    "SNeRV/model/residual_block.py ResidualBlocksWithInputConv"
)
OFFICIAL_SNERV_MFU_NUMERIC_PARITY_BLOCKERS: Final[tuple[str, ...]] = (
    "official_weight_tensor_mapping_not_loaded",
    "full_official_mfu_forward_artifact_not_emitted",
)


class OfficialSnervMfuError(ValueError):
    """Raised when official SNeRV MFU graph/shape contracts are violated."""


@dataclass(frozen=True)
class NchwShape:
    """A concrete NCHW tensor shape for graph-level execution."""

    n: int
    c: int
    h: int
    w: int

    def __post_init__(self) -> None:
        for name, value in zip(("n", "c", "h", "w"), self.as_tuple(), strict=True):
            if int(value) <= 0:
                raise OfficialSnervMfuError(f"NCHW dimension {name} must be positive")

    @classmethod
    def from_shape(cls, shape: Iterable[int]) -> NchwShape:
        values = tuple(int(v) for v in shape)
        if len(values) != 4:
            raise OfficialSnervMfuError(f"expected NCHW rank 4 shape, got {values!r}")
        return cls(*values)

    @classmethod
    def from_array(cls, array: np.ndarray) -> NchwShape:
        return cls.from_shape(np.asarray(array).shape)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (int(self.n), int(self.c), int(self.h), int(self.w))


@dataclass(frozen=True)
class TensorSpec:
    """A named NCHW shape with dtype metadata for export manifests."""

    shape: NchwShape
    name: str = ""
    dtype: str = "float32"

    @classmethod
    def from_shape(
        cls,
        shape: Iterable[int],
        *,
        name: str = "",
        dtype: str = "float32",
    ) -> TensorSpec:
        return cls(shape=NchwShape.from_shape(shape), name=str(name), dtype=str(dtype))

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        *,
        name: str = "",
    ) -> TensorSpec:
        arr = np.asarray(array)
        return cls(shape=NchwShape.from_array(arr), name=str(name), dtype=str(arr.dtype))

    def with_shape(self, shape: NchwShape, *, name: str | None = None) -> TensorSpec:
        return TensorSpec(shape=shape, name=self.name if name is None else str(name), dtype=self.dtype)

    @property
    def nchw(self) -> tuple[int, int, int, int]:
        return self.shape.as_tuple()


@dataclass(frozen=True)
class ConvTranspose2dShapeSpec:
    """Torch ``nn.ConvTranspose2d`` shape contract for NCHW inputs."""

    in_channels: int
    out_channels: int
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int] = (0, 0)
    output_padding: tuple[int, int] = (0, 0)
    dilation: tuple[int, int] = (1, 1)
    groups: int = 1
    bias: bool = True
    source: str = OFFICIAL_SNERV_MFU_SOURCE

    def __post_init__(self) -> None:
        object.__setattr__(self, "in_channels", int(self.in_channels))
        object.__setattr__(self, "out_channels", int(self.out_channels))
        object.__setattr__(self, "kernel_size", _pair_int(self.kernel_size, "kernel_size"))
        object.__setattr__(self, "stride", _pair_int(self.stride, "stride"))
        object.__setattr__(self, "padding", _pair_int(self.padding, "padding", minimum=0))
        object.__setattr__(
            self,
            "output_padding",
            _pair_int(self.output_padding, "output_padding", minimum=0),
        )
        object.__setattr__(self, "dilation", _pair_int(self.dilation, "dilation"))
        object.__setattr__(self, "groups", int(self.groups))
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise OfficialSnervMfuError("ConvTranspose2d channels must be positive")
        if self.groups <= 0:
            raise OfficialSnervMfuError("ConvTranspose2d groups must be positive")
        if self.in_channels % self.groups or self.out_channels % self.groups:
            raise OfficialSnervMfuError("ConvTranspose2d channels must be divisible by groups")
        for out_pad, stride, dilation in zip(self.output_padding, self.stride, self.dilation, strict=True):
            if out_pad >= stride and out_pad >= dilation:
                raise OfficialSnervMfuError(
                    "ConvTranspose2d output_padding must be smaller than stride or dilation"
                )

    @classmethod
    def official(cls, *, channels: int, stride: int) -> ConvTranspose2dShapeSpec:
        """Build the official MFU upsampler: kernel=stride, padding=0."""

        return cls(
            in_channels=int(channels),
            out_channels=int(channels),
            kernel_size=(int(stride), int(stride)),
            stride=(int(stride), int(stride)),
            padding=(0, 0),
            output_padding=(0, 0),
            dilation=(1, 1),
        )

    def forward_spec(self, x: TensorSpec, *, name: str) -> TensorSpec:
        if x.shape.c != self.in_channels:
            raise OfficialSnervMfuError(
                f"ConvTranspose2d expected {self.in_channels} input channels, got {x.shape.c}"
            )
        out_h = _conv_transpose2d_axis(
            x.shape.h,
            kernel=self.kernel_size[0],
            stride=self.stride[0],
            padding=self.padding[0],
            output_padding=self.output_padding[0],
            dilation=self.dilation[0],
        )
        out_w = _conv_transpose2d_axis(
            x.shape.w,
            kernel=self.kernel_size[1],
            stride=self.stride[1],
            padding=self.padding[1],
            output_padding=self.output_padding[1],
            dilation=self.dilation[1],
        )
        return x.with_shape(NchwShape(x.shape.n, self.out_channels, out_h, out_w), name=name)

    def torch_weight_shape(self) -> tuple[int, int, int, int]:
        return (
            int(self.in_channels),
            int(self.out_channels // self.groups),
            int(self.kernel_size[0]),
            int(self.kernel_size[1]),
        )

    def torch_bias_shape(self) -> tuple[int] | None:
        return (int(self.out_channels),) if self.bias else None


@dataclass(frozen=True)
class OfficialConvTranspose2dNchw:
    """PyTorch-style ``nn.ConvTranspose2d`` weights in NCHW/IOHW layout."""

    weight: np.ndarray
    bias: np.ndarray | None = None
    stride: int | tuple[int, int] = 1
    padding: int | tuple[int, int] = 0
    output_padding: int | tuple[int, int] = 0
    dilation: int | tuple[int, int] = 1
    groups: int = 1

    def __post_init__(self) -> None:
        weight = np.asarray(self.weight, dtype=np.float64)
        bias = None if self.bias is None else np.asarray(self.bias, dtype=np.float64)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "stride", _pair_int(self.stride, "stride"))
        object.__setattr__(self, "padding", _pair_int(self.padding, "padding", minimum=0))
        object.__setattr__(
            self,
            "output_padding",
            _pair_int(self.output_padding, "output_padding", minimum=0),
        )
        object.__setattr__(self, "dilation", _pair_int(self.dilation, "dilation"))
        object.__setattr__(self, "groups", int(self.groups))
        if weight.ndim != 4:
            raise OfficialSnervMfuError(
                f"convtranspose weight must be IOHW, got {weight.shape}"
            )
        if self.groups <= 0:
            raise OfficialSnervMfuError("ConvTranspose2d groups must be positive")
        in_channels = int(weight.shape[0])
        out_channels = int(weight.shape[1]) * int(self.groups)
        if in_channels % int(self.groups):
            raise OfficialSnervMfuError(
                "ConvTranspose2d input channels must be divisible by groups"
            )
        if bias is not None and bias.shape != (out_channels,):
            raise OfficialSnervMfuError(
                f"convtranspose bias shape {bias.shape} does not match out channels {out_channels}"
            )

    @property
    def in_channels(self) -> int:
        return int(self.weight.shape[0])

    @property
    def out_channels(self) -> int:
        return int(self.weight.shape[1]) * int(self.groups)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return conv_transpose2d_nchw(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
        )


@dataclass(frozen=True)
class OfficialResidualBlockNoBN:
    """Official ``ResidualBlockNoBN``: conv, LeakyReLU(0.1), conv, skip add."""

    conv1: OfficialConv2dNchw
    conv2: OfficialConv2dNchw

    def __post_init__(self) -> None:
        if self.conv1.weight.shape[2:] != (3, 3) or self.conv1.padding != 1:
            raise OfficialSnervMfuError("ResidualBlockNoBN conv1 must be 3x3 padding=1")
        if self.conv2.weight.shape[2:] != (3, 3) or self.conv2.padding != 1:
            raise OfficialSnervMfuError("ResidualBlockNoBN conv2 must be 3x3 padding=1")
        if self.conv1.in_channels != self.conv1.out_channels:
            raise OfficialSnervMfuError("ResidualBlockNoBN conv1 must preserve channels")
        if self.conv2.in_channels != self.conv1.out_channels:
            raise OfficialSnervMfuError("ResidualBlockNoBN conv2 channels mismatch")
        if self.conv2.out_channels != self.conv1.out_channels:
            raise OfficialSnervMfuError("ResidualBlockNoBN conv2 must preserve channels")

    @property
    def channels(self) -> int:
        return int(self.conv1.in_channels)

    def forward(self, x: np.ndarray) -> np.ndarray:
        arr = _ensure_nchw_array(x)
        if arr.shape[1] != self.channels:
            raise OfficialSnervMfuError(
                f"ResidualBlockNoBN expected {self.channels} channels, got {arr.shape[1]}"
            )
        out = leaky_relu01(self.conv1.forward(arr))
        return (arr + self.conv2.forward(out)).astype(np.float64, copy=False)


@dataclass(frozen=True)
class OfficialResidualBlocksWithInputConv:
    """Official ``ResidualBlocksWithInputConv`` numeric reference."""

    input_conv: OfficialConv2dNchw
    residual_blocks: tuple[OfficialResidualBlockNoBN, ...]

    def __post_init__(self) -> None:
        if self.input_conv.weight.shape[2:] != (3, 3) or self.input_conv.padding != 1:
            raise OfficialSnervMfuError("RB input conv must be 3x3 padding=1")
        blocks = tuple(self.residual_blocks)
        object.__setattr__(self, "residual_blocks", blocks)
        for block in blocks:
            if block.channels != self.input_conv.out_channels:
                raise OfficialSnervMfuError(
                    "RB residual block channels must match input conv output"
                )

    @property
    def in_channels(self) -> int:
        return int(self.input_conv.in_channels)

    @property
    def out_channels(self) -> int:
        return int(self.input_conv.out_channels)

    def forward(self, x: np.ndarray) -> np.ndarray:
        arr = _ensure_nchw_array(x)
        if arr.shape[1] != self.in_channels:
            raise OfficialSnervMfuError(
                f"RB expected {self.in_channels} input channels, got {arr.shape[1]}"
            )
        out = self.input_conv.forward(arr)
        for block in self.residual_blocks:
            out = block.forward(out)
        return out.astype(np.float64, copy=False)


@dataclass(frozen=True)
class OfficialSnervMfuForwardOutput:
    """Concrete official MFU forward outputs and false-authority metadata."""

    up1: np.ndarray
    cat_mid: np.ndarray
    unet1: np.ndarray
    unet1_up: np.ndarray
    cat_high: np.ndarray
    pyr_out: np.ndarray

    def as_jsonable_metadata(self) -> dict[str, object]:
        return {
            "schema": "official_snerv_mfu_forward_output.v1",
            "up1_shape": list(self.up1.shape),
            "cat_mid_shape": list(self.cat_mid.shape),
            "unet1_shape": list(self.unet1.shape),
            "unet1_up_shape": list(self.unet1_up.shape),
            "cat_high_shape": list(self.cat_high.shape),
            "pyr_out_shape": list(self.pyr_out.shape),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "source_forward_replay_authority": False,
        }


@dataclass(frozen=True)
class OfficialSnervMfu:
    """Executable NumPy reference for official SNeRV MFU.

    This is the source-graph numeric primitive:
    ``ConvTranspose2d -> cat -> ResidualBlocksWithInputConv -> ConvTranspose2d
    -> cat -> ResidualBlocksWithInputConv``.  It proves portable math against
    torch when supplied with explicit weights; loading the actual upstream
    SNeRV checkpoint and emitting a full official source-forward replay remains
    a separate promotion gate.
    """

    spec: OfficialSnervMfuSpec
    upsample_mid: OfficialConvTranspose2dNchw
    rb_mid: OfficialResidualBlocksWithInputConv
    upsample_high: OfficialConvTranspose2dNchw
    rb_high: OfficialResidualBlocksWithInputConv

    def __post_init__(self) -> None:
        if self.upsample_mid.in_channels != self.spec.low_channels:
            raise OfficialSnervMfuError("MFU mid upsample input channels mismatch spec")
        if self.upsample_mid.out_channels != self.spec.low_channels:
            raise OfficialSnervMfuError("MFU mid upsample output channels mismatch spec")
        if self.upsample_mid.stride != (self.spec.mid_stride, self.spec.mid_stride):
            raise OfficialSnervMfuError("MFU mid upsample stride mismatch spec")
        if self.rb_mid.in_channels != self.spec.low_channels + self.spec.mid_channels:
            raise OfficialSnervMfuError("MFU mid RB input channels mismatch spec")
        if self.rb_mid.out_channels != self.spec.mid_channels:
            raise OfficialSnervMfuError("MFU mid RB output channels mismatch spec")
        if self.upsample_high.in_channels != self.spec.mid_channels:
            raise OfficialSnervMfuError("MFU high upsample input channels mismatch spec")
        if self.upsample_high.out_channels != self.spec.mid_channels:
            raise OfficialSnervMfuError("MFU high upsample output channels mismatch spec")
        if self.upsample_high.stride != (self.spec.high_stride, self.spec.high_stride):
            raise OfficialSnervMfuError("MFU high upsample stride mismatch spec")
        if self.rb_high.in_channels != self.spec.mid_channels + self.spec.high_channels:
            raise OfficialSnervMfuError("MFU high RB input channels mismatch spec")
        if self.rb_high.out_channels != self.spec.high_channels:
            raise OfficialSnervMfuError("MFU high RB output channels mismatch spec")

    def forward(
        self,
        low: np.ndarray,
        skip_mid: np.ndarray,
        skip_high: np.ndarray,
    ) -> OfficialSnervMfuForwardOutput:
        low_arr = _ensure_nchw_array(low)
        skip_mid_arr = _ensure_nchw_array(skip_mid)
        skip_high_arr = _ensure_nchw_array(skip_high)
        if low_arr.shape[1] != self.spec.low_channels:
            raise OfficialSnervMfuError(
                f"low channels {low_arr.shape[1]} do not match spec {self.spec.low_channels}"
            )
        if skip_mid_arr.shape[1] != self.spec.mid_channels:
            raise OfficialSnervMfuError(
                f"skip_mid channels {skip_mid_arr.shape[1]} do not match spec {self.spec.mid_channels}"
            )
        if skip_high_arr.shape[1] != self.spec.high_channels:
            raise OfficialSnervMfuError(
                f"skip_high channels {skip_high_arr.shape[1]} do not match spec {self.spec.high_channels}"
            )
        up1 = self.upsample_mid.forward(low_arr)
        cat_mid = concat_nchw_arrays((up1, skip_mid_arr))
        unet1 = self.rb_mid.forward(cat_mid)
        unet1_up = self.upsample_high.forward(unet1)
        cat_high = concat_nchw_arrays((unet1_up, skip_high_arr))
        pyr_out = self.rb_high.forward(cat_high)
        return OfficialSnervMfuForwardOutput(
            up1=up1,
            cat_mid=cat_mid,
            unet1=unet1,
            unet1_up=unet1_up,
            cat_high=cat_high,
            pyr_out=pyr_out,
        )


@dataclass(frozen=True)
class ResidualBlocksWithInputConvSpec:
    """Official RB interface: input conv maps channels, residual blocks preserve."""

    in_channels: int
    out_channels: int
    num_blocks: int
    source: str = OFFICIAL_SNERV_RB_SOURCE

    def __post_init__(self) -> None:
        object.__setattr__(self, "in_channels", int(self.in_channels))
        object.__setattr__(self, "out_channels", int(self.out_channels))
        object.__setattr__(self, "num_blocks", int(self.num_blocks))
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise OfficialSnervMfuError("RB channels must be positive")
        if self.num_blocks < 0:
            raise OfficialSnervMfuError("RB num_blocks must be non-negative")

    def forward_spec(self, x: TensorSpec, *, name: str) -> TensorSpec:
        if x.shape.c != self.in_channels:
            raise OfficialSnervMfuError(
                f"RB expected {self.in_channels} input channels after skip concat, got {x.shape.c}"
            )
        return x.with_shape(NchwShape(x.shape.n, self.out_channels, x.shape.h, x.shape.w), name=name)

    def torch_parameter_shapes(self, *, prefix: str) -> dict[str, tuple[int, ...]]:
        shapes: dict[str, tuple[int, ...]] = {
            f"{prefix}.main.0.weight": (self.out_channels, self.in_channels, 3, 3),
            f"{prefix}.main.0.bias": (self.out_channels,),
        }
        for idx in range(self.num_blocks):
            base = f"{prefix}.main.1.{idx}"
            shapes[f"{base}.conv1.weight"] = (self.out_channels, self.out_channels, 3, 3)
            shapes[f"{base}.conv1.bias"] = (self.out_channels,)
            shapes[f"{base}.conv2.weight"] = (self.out_channels, self.out_channels, 3, 3)
            shapes[f"{base}.conv2.bias"] = (self.out_channels,)
        return shapes


@dataclass(frozen=True)
class OfficialMfuGraphNode:
    """One graph-level operation in the official MFU trace."""

    name: str
    op: str
    inputs: tuple[str, ...]
    output: TensorSpec
    source: str


@dataclass(frozen=True)
class OfficialMfuShapeTrace:
    """Executable graph/shape trace for the official SNeRV MFU."""

    schema: str
    source: str
    nodes: tuple[OfficialMfuGraphNode, ...]
    output: TensorSpec
    parameter_shapes: dict[str, tuple[int, ...]]
    numeric_parity_blockers: tuple[str, ...]
    score_claim: bool
    promotion_eligible: bool
    ready_for_exact_eval_dispatch: bool

    def as_jsonable(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source": self.source,
            "nodes": [
                {
                    "name": node.name,
                    "op": node.op,
                    "inputs": list(node.inputs),
                    "output_shape": list(node.output.nchw),
                    "source": node.source,
                }
                for node in self.nodes
            ],
            "output_shape": list(self.output.nchw),
            "parameter_shapes": {key: list(value) for key, value in self.parameter_shapes.items()},
            "numeric_parity_blockers": list(self.numeric_parity_blockers),
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
            "ready_for_exact_eval_dispatch": self.ready_for_exact_eval_dispatch,
        }


@dataclass(frozen=True)
class OfficialSnervMfuSpec:
    """Official SNeRV MFU shape graph.

    ``low`` is upstream ``embed_list[-3]``, ``skip_mid`` is ``embed_list[-2]``,
    and ``skip_high`` is ``embed_list[-1]`` from ``model/snerv.py``.
    """

    low_channels: int
    mid_channels: int
    high_channels: int
    mid_stride: int
    high_stride: int
    num_blocks: int

    def __post_init__(self) -> None:
        for name in ("low_channels", "mid_channels", "high_channels", "mid_stride", "high_stride"):
            value = int(getattr(self, name))
            object.__setattr__(self, name, value)
            if value <= 0:
                raise OfficialSnervMfuError(f"{name} must be positive")
        object.__setattr__(self, "num_blocks", int(self.num_blocks))
        if self.num_blocks < 0:
            raise OfficialSnervMfuError("num_blocks must be non-negative")

    @classmethod
    def from_official_lists(
        cls,
        *,
        ngf_list: Iterable[int],
        dec_strds: Iterable[int],
        num_blocks: int,
    ) -> OfficialSnervMfuSpec:
        ngf = tuple(int(v) for v in ngf_list)
        strds = tuple(int(v) for v in dec_strds)
        if len(ngf) < 3:
            raise OfficialSnervMfuError("official MFU needs at least three decoder channel stages")
        if len(strds) < 2:
            raise OfficialSnervMfuError("official MFU needs at least two decoder strides")
        return cls(
            low_channels=ngf[-3],
            mid_channels=ngf[-2],
            high_channels=ngf[-1],
            mid_stride=strds[-2],
            high_stride=strds[-1],
            num_blocks=int(num_blocks),
        )

    @property
    def upsample_mid(self) -> ConvTranspose2dShapeSpec:
        return ConvTranspose2dShapeSpec.official(
            channels=self.low_channels,
            stride=self.mid_stride,
        )

    @property
    def rb_mid(self) -> ResidualBlocksWithInputConvSpec:
        return ResidualBlocksWithInputConvSpec(
            in_channels=self.low_channels + self.mid_channels,
            out_channels=self.mid_channels,
            num_blocks=self.num_blocks,
        )

    @property
    def upsample_high(self) -> ConvTranspose2dShapeSpec:
        return ConvTranspose2dShapeSpec.official(
            channels=self.mid_channels,
            stride=self.high_stride,
        )

    @property
    def rb_high(self) -> ResidualBlocksWithInputConvSpec:
        return ResidualBlocksWithInputConvSpec(
            in_channels=self.mid_channels + self.high_channels,
            out_channels=self.high_channels,
            num_blocks=self.num_blocks,
        )

    def forward_shape(
        self,
        low: TensorSpec | np.ndarray | Iterable[int],
        skip_mid: TensorSpec | np.ndarray | Iterable[int],
        skip_high: TensorSpec | np.ndarray | Iterable[int],
    ) -> OfficialMfuShapeTrace:
        low_spec = _tensor_spec(low, name="embed_list[-3]")
        skip_mid_spec = _tensor_spec(skip_mid, name="embed_list[-2]")
        skip_high_spec = _tensor_spec(skip_high, name="embed_list[-1]")

        nodes: list[OfficialMfuGraphNode] = []

        up1 = self.upsample_mid.forward_spec(low_spec, name="up1")
        nodes.append(
            OfficialMfuGraphNode(
                name="up1",
                op="ConvTranspose2d(kernel=stride,padding=0)",
                inputs=(low_spec.name,),
                output=up1,
                source=OFFICIAL_SNERV_MFU_SOURCE,
            )
        )

        concat_mid = concat_nchw_specs(
            (up1, skip_mid_spec),
            name="cat(up1, embed_list[-2])",
        )
        nodes.append(
            OfficialMfuGraphNode(
                name="cat_mid",
                op="torch.cat(dim=1)",
                inputs=(up1.name, skip_mid_spec.name),
                output=concat_mid,
                source=OFFICIAL_SNERV_MFU_SOURCE,
            )
        )

        unet1 = self.rb_mid.forward_spec(concat_mid, name="unet1")
        nodes.append(
            OfficialMfuGraphNode(
                name="unet1",
                op="ResidualBlocksWithInputConv",
                inputs=(concat_mid.name,),
                output=unet1,
                source=OFFICIAL_SNERV_RB_SOURCE,
            )
        )

        unet1_up = self.upsample_high.forward_spec(unet1, name="unet1_up")
        nodes.append(
            OfficialMfuGraphNode(
                name="unet1_up",
                op="ConvTranspose2d(kernel=stride,padding=0)",
                inputs=(unet1.name,),
                output=unet1_up,
                source=OFFICIAL_SNERV_MFU_SOURCE,
            )
        )

        concat_high = concat_nchw_specs(
            (unet1_up, skip_high_spec),
            name="cat(unet1_up, embed_list[-1])",
        )
        nodes.append(
            OfficialMfuGraphNode(
                name="cat_high",
                op="torch.cat(dim=1)",
                inputs=(unet1_up.name, skip_high_spec.name),
                output=concat_high,
                source=OFFICIAL_SNERV_MFU_SOURCE,
            )
        )

        pyr_out = self.rb_high.forward_spec(concat_high, name="pyr_out")
        nodes.append(
            OfficialMfuGraphNode(
                name="pyr_out",
                op="ResidualBlocksWithInputConv",
                inputs=(concat_high.name,),
                output=pyr_out,
                source=OFFICIAL_SNERV_RB_SOURCE,
            )
        )

        parameter_shapes = {
            "decoder_len+3.weight": self.upsample_mid.torch_weight_shape(),
            "decoder_len+3.bias": self.upsample_mid.torch_bias_shape() or (),
            "decoder_len+5.weight": self.upsample_high.torch_weight_shape(),
            "decoder_len+5.bias": self.upsample_high.torch_bias_shape() or (),
        }
        parameter_shapes.update(self.rb_mid.torch_parameter_shapes(prefix="decoder_len+4"))
        parameter_shapes.update(self.rb_high.torch_parameter_shapes(prefix="decoder_len+6"))
        return OfficialMfuShapeTrace(
            schema="official_snerv_mfu_shape_trace.v1",
            source=OFFICIAL_SNERV_MFU_SOURCE,
            nodes=tuple(nodes),
            output=pyr_out,
            parameter_shapes=parameter_shapes,
            numeric_parity_blockers=OFFICIAL_SNERV_MFU_NUMERIC_PARITY_BLOCKERS,
            score_claim=False,
            promotion_eligible=False,
            ready_for_exact_eval_dispatch=False,
        )


def concat_nchw_specs(specs: Iterable[TensorSpec], *, name: str) -> TensorSpec:
    """Model ``torch.cat(..., dim=1)`` for SNeRV skip-concat sites."""

    parts = tuple(specs)
    if len(parts) < 2:
        raise OfficialSnervMfuError("skip concat needs at least two tensors")
    first = parts[0]
    n, _c, h, w = first.nchw
    total_channels = 0
    for part in parts:
        if (part.shape.n, part.shape.h, part.shape.w) != (n, h, w):
            raise OfficialSnervMfuError(
                "skip concat requires matching N/H/W; "
                f"got {first.nchw} and {part.nchw}"
            )
        total_channels += int(part.shape.c)
    return TensorSpec(
        shape=NchwShape(n, total_channels, h, w),
        name=str(name),
        dtype=first.dtype,
    )


def concat_nchw_arrays(arrays: Iterable[np.ndarray]) -> np.ndarray:
    """Execute official ``torch.cat(..., dim=1)`` for NCHW NumPy tensors."""

    parts = tuple(_ensure_nchw_array(array) for array in arrays)
    if len(parts) < 2:
        raise OfficialSnervMfuError("skip concat needs at least two tensors")
    first = parts[0]
    n, _c, h, w = first.shape
    for part in parts:
        if (part.shape[0], part.shape[2], part.shape[3]) != (n, h, w):
            raise OfficialSnervMfuError(
                "skip concat requires matching N/H/W; "
                f"got {first.shape} and {part.shape}"
            )
    return np.concatenate(parts, axis=1).astype(np.float64, copy=False)


def conv_transpose2d_nchw(
    x: np.ndarray,
    weight: np.ndarray,
    *,
    bias: np.ndarray | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    output_padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> np.ndarray:
    """NumPy reference for PyTorch ``F.conv_transpose2d``.

    PyTorch stores ConvTranspose2d weights as ``(in_channels,
    out_channels/groups, kH, kW)``. This reference intentionally favors
    exactness and auditability over speed; the official SNeRV MFU kernels are
    small enough for parity probes, and training paths can lower hot loops
    after a measured profile.
    """

    arr = _ensure_nchw_array(x)
    w64 = np.asarray(weight, dtype=np.float64)
    if w64.ndim != 4:
        raise OfficialSnervMfuError(f"weight must be IOHW, got {w64.shape}")
    stride_pair = _pair_int(stride, "stride")
    padding_pair = _pair_int(padding, "padding", minimum=0)
    output_padding_pair = _pair_int(output_padding, "output_padding", minimum=0)
    dilation_pair = _pair_int(dilation, "dilation")
    for out_pad, step, dil in zip(
        output_padding_pair,
        stride_pair,
        dilation_pair,
        strict=True,
    ):
        if out_pad >= step and out_pad >= dil:
            raise OfficialSnervMfuError(
                "ConvTranspose2d output_padding must be smaller than stride or dilation"
            )
    groups_i = int(groups)
    if groups_i <= 0:
        raise OfficialSnervMfuError("groups must be positive")
    n, in_channels, h, w = arr.shape
    weight_in, out_per_group, kh, kw = w64.shape
    if in_channels != weight_in:
        raise OfficialSnervMfuError(
            f"input channels {in_channels} do not match weight channels {weight_in}"
        )
    if in_channels % groups_i:
        raise OfficialSnervMfuError("input channels must be divisible by groups")
    out_channels = int(out_per_group) * groups_i
    if bias is not None:
        b64 = np.asarray(bias, dtype=np.float64)
        if b64.shape != (out_channels,):
            raise OfficialSnervMfuError(
                f"bias shape {b64.shape} does not match out channels {out_channels}"
            )
    else:
        b64 = None

    out_h = _conv_transpose2d_axis(
        h,
        kernel=kh,
        stride=stride_pair[0],
        padding=padding_pair[0],
        output_padding=output_padding_pair[0],
        dilation=dilation_pair[0],
    )
    out_w = _conv_transpose2d_axis(
        w,
        kernel=kw,
        stride=stride_pair[1],
        padding=padding_pair[1],
        output_padding=output_padding_pair[1],
        dilation=dilation_pair[1],
    )
    out = np.zeros((n, out_channels, out_h, out_w), dtype=np.float64)
    in_per_group = in_channels // groups_i
    for batch in range(n):
        for group in range(groups_i):
            in_start = group * in_per_group
            out_start = group * out_per_group
            for cin_group in range(in_per_group):
                cin = in_start + cin_group
                for ih in range(h):
                    base_y = ih * stride_pair[0] - padding_pair[0]
                    for iw in range(w):
                        value = arr[batch, cin, ih, iw]
                        if value == 0.0:
                            continue
                        base_x = iw * stride_pair[1] - padding_pair[1]
                        for ky in range(kh):
                            oy = base_y + ky * dilation_pair[0]
                            if oy < 0 or oy >= out_h:
                                continue
                            for kx in range(kw):
                                ox = base_x + kx * dilation_pair[1]
                                if ox < 0 or ox >= out_w:
                                    continue
                                out[
                                    batch,
                                    out_start : out_start + out_per_group,
                                    oy,
                                    ox,
                                ] += value * w64[cin, :, ky, kx]
    if b64 is not None:
        out += b64[None, :, None, None]
    return out.astype(np.float64, copy=False)


def _conv_transpose2d_axis(
    size: int,
    *,
    kernel: int,
    stride: int,
    padding: int,
    output_padding: int,
    dilation: int,
) -> int:
    out = (int(size) - 1) * int(stride)
    out -= 2 * int(padding)
    out += int(dilation) * (int(kernel) - 1)
    out += int(output_padding)
    out += 1
    if out <= 0:
        raise OfficialSnervMfuError(f"ConvTranspose2d output axis must be positive, got {out}")
    return int(out)


def _ensure_nchw_array(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 4:
        raise OfficialSnervMfuError(f"expected NCHW tensor, got {arr.shape}")
    return arr


def _pair_int(
    value: int | tuple[int, int],
    name: str,
    *,
    minimum: int = 1,
) -> tuple[int, int]:
    pair = tuple(int(v) for v in value) if isinstance(value, tuple) else (int(value), int(value))
    if len(pair) != 2:
        raise OfficialSnervMfuError(f"{name} must be an int or length-2 tuple")
    if pair[0] < minimum or pair[1] < minimum:
        raise OfficialSnervMfuError(f"{name} values must be >= {minimum}")
    return pair  # type: ignore[return-value]


def _tensor_spec(
    value: TensorSpec | np.ndarray | Iterable[int],
    *,
    name: str,
) -> TensorSpec:
    if isinstance(value, TensorSpec):
        if value.name:
            return value
        return TensorSpec(shape=value.shape, name=name, dtype=value.dtype)
    if isinstance(value, np.ndarray):
        return TensorSpec.from_array(value, name=name)
    return TensorSpec.from_shape(value, name=name)


__all__ = [
    "OFFICIAL_SNERV_MFU_NUMERIC_PARITY_BLOCKERS",
    "OFFICIAL_SNERV_MFU_SOURCE",
    "OFFICIAL_SNERV_RB_SOURCE",
    "ConvTranspose2dShapeSpec",
    "NchwShape",
    "OfficialConvTranspose2dNchw",
    "OfficialMfuGraphNode",
    "OfficialMfuShapeTrace",
    "OfficialResidualBlockNoBN",
    "OfficialResidualBlocksWithInputConv",
    "OfficialSnervMfu",
    "OfficialSnervMfuError",
    "OfficialSnervMfuForwardOutput",
    "OfficialSnervMfuSpec",
    "ResidualBlocksWithInputConvSpec",
    "TensorSpec",
    "concat_nchw_arrays",
    "concat_nchw_specs",
    "conv_transpose2d_nchw",
]
