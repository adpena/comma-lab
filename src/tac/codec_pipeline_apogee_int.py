"""Op 3: apogee_intN substrate-transform CodecOp.

Per the canonical four-way stack composition contract
(`.omx/research/four_way_stack_composition_contract_20260507_claude.md`),
the apogee_intN family lands in the pipeline as a **substrate-transform**
op:

  - INPUT:  fp32 ``state_dict`` (HNeRV-shape tensors)
  - OUTPUT: a CPL1 blob carrying the intN-quantized blob (one
    :func:`encode_intN_block_fp` per tensor, length-prefixed) +
    per-tensor scales/bits in ``op_state``.
  - The dequantized state_dict produced at decode time is what the
    downstream op (Op 1 / Op 2) sees when this op is composed BEFORE
    them in the pipeline. The opt-in ``transforms_state_dict=True`` contract
    makes later encoders see lower-entropy input (intN-quantized weights
    round-tripped through fp32), so their BROTLI/AC streams compress harder.

Composition modes (per the contract memo):

  - **substitutional**: Op 3 alone (just the int-N substrate).
  - **substrate-transform**: ``[Op3, Op1]`` or ``[Op3, Op2]`` -
    Op 3 quantizes; Op 1/2 encodes the dequantized state_dict; the
    final blob carries BOTH ops' artifacts (per CPL1 wire format).
    Empirical byte-savings recorded in
    ``experiments/results/lane_codec_pipeline_apogee_int_<UTC>/manifest.json``.
  - **decorator**: not used here (decorator mode is decode-side only,
    e.g. PR102 inference tuning).

Strict-scorer-rule: pure CPU codec; no scorer load; no MPS/CUDA
dependency. Determinism: encode is bit-faithful for a given
state_dict + bits; the substrate-transform composition with Op 1/Op 2
is also byte-deterministic (state_dict -> dequantized -> encoder is a
pure function).

Invariants enforced at ``validate``-time:

  1. Schema match - state_dict must contain every name in
     :data:`tac.pr101_split_brotli_codec.FIXED_STATE_SCHEMA` with the
     correct shape (Op 3 is HNeRV-state-dict-shaped, like Op 1/Op 2).
  2. ``bits`` must be in ``{3, 4, 6, 7, 8}``. ``bits=5`` is REFUSED
     - apogee_int5 is **DEFERRED-pending-research** in the lane
     registry (basin-parity FAIL: pose_dist_delta 2.26x threshold per
     ``project_apogee_intN_basin_parity_safety_boundary_20260507.md``).
     The refusal is a Contrarian gate, NOT a kill - when basin-parity
     research lands, the bits=5 path can be unlocked.
  3. ``bits=4`` is permitted but FALSIFIED at score 1.4287
     ``[contest-CUDA T4]`` per
     ``project_apogee_int4_FALSIFIED_score_1_43_dispatcher_VALIDATED_20260505.md``;
     the validator emits a WARNING-level finding (``passed=True``) so
     operators can still substrate-transform-compose at int4 for
     forensics / Pareto-mapping but are reminded the score is bad.

Empirical (smoke-test): on a synthetic-state-dict with FIXED_STATE_SCHEMA,
``[Op3(bits=6), Op1]`` makes Op1's own blob smaller than Op1 alone. The final
CPL1 wrapper still carries both op blobs and may be larger; the measured effect
is downstream entropy reduction, not a standalone score claim.
``[empirical:experiments/results/lane_codec_pipeline_apogee_int_<UTC>/manifest.json]``

Cross-references:
  - Contract memo:
    ``.omx/research/four_way_stack_composition_contract_20260507_claude.md``
  - Op 1: :class:`tac.codec_pipeline.Op1_PR101SplitBrotli`
  - Op 2: :class:`tac.codec_pipeline.Op2_PR103ArithmeticCodec`
  - Encoder/decoder: :func:`experiments.block_fp_intN_codec_sketch.encode_intN_block_fp` /
    :func:`submissions.apogee_intN.src.intn_codec.decode_intN_block_fp`
  - Basin-parity boundary:
    ``project_apogee_intN_basin_parity_safety_boundary_20260507.md``
"""
from __future__ import annotations

import io
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from tac.codec_pipeline import EncodeResult, ValidationReport

# We need the encoder side `encode_intN_block_fp` from the experiments
# sketch (the canonical intN producer; the apogee_intN inflate runtime
# only ships the decoder side). Importing from `experiments` requires a
# repo-root path entry; do that defensively.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.block_fp_intN_codec_sketch import (  # noqa: E402
    encode_intN_block_fp,
)
from submissions.apogee_intN.src.intn_codec import (  # noqa: E402
    decode_intN_block_fp,
)

#: bit-widths for which apogee_intN is GREEN to dispatch (basin-parity
#: PASSED or pre-PR106 baseline).
SAFE_BITS: frozenset[int] = frozenset({6, 7, 8})

#: bit-widths the validate gate REFUSES (DEFERRED-pending-research).
REFUSED_BITS: frozenset[int] = frozenset({5})

#: bit-widths the validate gate WARNS on (FALSIFIED at one config but
#: not killed; substrate-transform composition still permitted for
#: forensics).
WARN_BITS: frozenset[int] = frozenset({3, 4})


@dataclass
class Op3_ApogeeIntN_Substrate:
    """Substrate-transform op: quantize fp32 state_dict to intN block-FP.

    The op_state carries:
        - ``bits`` (int): the intN bit-width used for the encode.
        - ``tensor_names`` (list[str]): the fixed encode order; the
          decoder reads the blob in the same order. Pinned to
          ``FIXED_STATE_SCHEMA`` order for determinism.
        - ``block_size`` (int): block-FP block size (default 128 from
          the ``encode_intN_block_fp`` sketch).

    The blob is a deterministic length-prefixed concatenation of
    per-tensor :func:`encode_intN_block_fp` outputs.

    Wire format (Op 3 blob, NOT the CPL1 wrapper - that's the
    pipeline's job):
        magic   : 4 bytes  = b"AP3I"  (apogee int op-3 v1)
        n_tens  : u32_LE
        for each tensor:
            blob_len : u32_LE
            blob     : raw bytes from encode_intN_block_fp
    """
    name: str = "apogee_intN_substrate"
    transforms_state_dict: bool = True
    bits: int = 6  # default int6 - basin-parity PASSED 2026-05-07
    block_size: int = 128

    # Op 3 wire-format magic (distinct from CPL1 since Op 3 is one
    # op's blob; the pipeline wraps multiple ops' blobs in CPL1).
    _BLOB_MAGIC: bytes = field(default=b"AP3I", init=False, repr=False)

    def encode(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> EncodeResult:
        from tac.pr101_split_brotli_codec import FIXED_STATE_SCHEMA

        bytes_in = sum(t.numel() * t.element_size() for t in state_dict.values())

        # Encode each tensor in FIXED_STATE_SCHEMA order (deterministic).
        tensor_names: list[str] = []
        out = io.BytesIO()
        out.write(self._BLOB_MAGIC)
        # Write n_tens = number of schema-resident tensors actually present.
        present = [name for name, _shape in FIXED_STATE_SCHEMA if name in state_dict]
        out.write(struct.pack("<I", len(present)))
        for name, _shape in FIXED_STATE_SCHEMA:
            if name not in state_dict:
                continue
            t = state_dict[name].detach().to(torch.float32).cpu()
            blob = encode_intN_block_fp(t, block_size=self.block_size, bits=self.bits)
            out.write(struct.pack("<I", len(blob)))
            out.write(blob)
            tensor_names.append(name)

        op_blob = out.getvalue()
        op_state: dict[str, Any] = {
            "bits": int(self.bits),
            "tensor_names": tensor_names,
            "block_size": int(self.block_size),
        }
        return EncodeResult(
            blob=op_blob,
            bytes_in=bytes_in,
            bytes_out=len(op_blob),
            op_name=self.name,
            op_state=op_state,
        )

    def decode(
        self,
        blob: bytes,
        *,
        op_state: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        # Bug-hunter v2 fix 2026-05-07 (MEDIUM, re-opened from prior round):
        # `Op3_ApogeeIntN_Substrate.decode` previously trusted the encoder-side
        # `tensor_names` list and silently produced a partial state_dict when a
        # caller bypassed `validate()` via `pipeline.encode(..., skip_validate=
        # True)` and supplied a state_dict missing some FIXED_STATE_SCHEMA
        # tensors. With ``transforms_state_dict=True`` set, the partial dict
        # then fed downstream Op1/Op2 encoders that crashed inside
        # `encode_decoder_compact` / `encode_decoder_ac` with an opaque
        # KeyError — i.e. silent substrate corruption that punted the failure
        # to the next op. Per Selfcomp's wire-format contract, the substrate
        # transform must round-trip ALL FIXED_STATE_SCHEMA tensors or refuse
        # cleanly at THIS boundary, not punt to the next op.
        from tac.pr101_split_brotli_codec import FIXED_STATE_SCHEMA

        if blob[:4] != self._BLOB_MAGIC:
            raise ValueError(
                f"Op3 decode: bad blob magic {blob[:4]!r}, expected "
                f"{self._BLOB_MAGIC!r}"
            )
        cursor = 4
        n_tens = struct.unpack_from("<I", blob, cursor)[0]
        cursor += 4
        tensor_names = list(op_state.get("tensor_names", []))
        if len(tensor_names) != n_tens:
            raise ValueError(
                f"Op3 decode: blob says {n_tens} tensors but op_state has "
                f"{len(tensor_names)} names - wrapper/op_state mismatch"
            )
        # Substrate-transform roundtrip integrity: refuse to decode a partial
        # state_dict. Op3 declares ``transforms_state_dict=True``, so any
        # downstream op consumes our reconstruction; missing schema names will
        # corrupt that op's encode silently.
        schema_names = [name for name, _shape in FIXED_STATE_SCHEMA]
        if n_tens != len(schema_names):
            missing = [n for n in schema_names if n not in tensor_names]
            raise ValueError(
                f"Op3 decode: partial substrate refused — blob carries "
                f"{n_tens} tensors but FIXED_STATE_SCHEMA has "
                f"{len(schema_names)}. Missing: {sorted(missing)}. Re-encode "
                f"with the full state_dict (or run validate() to catch this "
                f"before encode)."
            )
        sd: dict[str, torch.Tensor] = {}
        for name in tensor_names:
            tlen = struct.unpack_from("<I", blob, cursor)[0]
            cursor += 4
            tblob = blob[cursor:cursor + tlen]
            cursor += tlen
            sd[name] = decode_intN_block_fp(tblob)
        if cursor != len(blob):
            raise ValueError(
                f"Op3 decode: trailing {len(blob) - cursor} bytes after "
                f"final tensor"
            )
        return sd

    def validate(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> ValidationReport:
        from tac.pr101_split_brotli_codec import FIXED_STATE_SCHEMA

        findings: list[str] = []

        # Gate 1: bits must be in supported range.
        if self.bits in REFUSED_BITS:
            # Specifically int5 (DEFERRED-pending-research; basin-parity
            # FAIL per project_apogee_intN_basin_parity_safety_boundary_20260507).
            findings.append(
                f"apogee_int{self.bits} DEFERRED-pending-research; not safe "
                f"for substrate-transform composition until basin-parity "
                f"passes (pose_dist_delta 2.26x threshold; see "
                f"project_apogee_intN_basin_parity_safety_boundary_20260507.md)"
            )
            return ValidationReport(
                passed=False, op_name=self.name, findings=findings,
            )
        if self.bits not in (SAFE_BITS | WARN_BITS):
            findings.append(
                f"bits={self.bits} not supported (must be in "
                f"{sorted(SAFE_BITS | WARN_BITS)}; bits=5 refused as "
                f"DEFERRED-pending-research)"
            )
            return ValidationReport(
                passed=False, op_name=self.name, findings=findings,
            )

        # Gate 2: schema match - every FIXED_STATE_SCHEMA name present
        # with correct shape.
        schema_lookup = dict(FIXED_STATE_SCHEMA)
        missing = [name for name, _ in FIXED_STATE_SCHEMA if name not in state_dict]
        if missing:
            findings.append(f"missing tensors: {sorted(missing)}")
            return ValidationReport(
                passed=False, op_name=self.name, findings=findings,
            )
        shape_mismatches = []
        for name, expected_shape in FIXED_STATE_SCHEMA:
            t = state_dict[name]
            if tuple(t.shape) != expected_shape:
                shape_mismatches.append(
                    f"{name}: expected {expected_shape}, got {tuple(t.shape)}"
                )
        if shape_mismatches:
            findings.append(f"shape mismatches: {shape_mismatches}")
            return ValidationReport(
                passed=False, op_name=self.name, findings=findings,
            )

        # Gate 3 (advisory only - passes): WARN_BITS reminder.
        passed = True
        if self.bits in WARN_BITS:
            findings.append(
                f"bits={self.bits} is in WARN_BITS - apogee_int{self.bits} is "
                f"FALSIFIED at one config (apogee_int4 score 1.4287 "
                f"[contest-CUDA T4]; see "
                f"project_apogee_int4_FALSIFIED_score_1_43_dispatcher_"
                f"VALIDATED_20260505.md). Substrate-transform composition "
                f"still permitted for forensics / Pareto-mapping; passed=True."
            )
            # Use schema_lookup to silence the "unused" lint.
            assert "stem.weight" in schema_lookup
        return ValidationReport(
            passed=passed, op_name=self.name, findings=findings,
        )


__all__ = [
    "REFUSED_BITS",
    "SAFE_BITS",
    "WARN_BITS",
    "Op3_ApogeeIntN_Substrate",
]
