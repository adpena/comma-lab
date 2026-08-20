#!/usr/bin/env python3
"""ddm_cd1 - stage a candidate runtime tree that DECOMPOSES its own token stage.

WHY.  ``ddm_wc2`` MEASURED the shipped jg5 token stage at **1,341.540 s = 94.5%** of a
1,419.904 s inflate ``[contest-CUDA T4]``.  ``ddm_rr7`` then MEASURED the full native
token-decode port at 1,546.617 s on the same axis -- byte-perfect and **15.3% SLOWER** --
which retired "lower the whole token stage to C" and left the float64 corrector as the only
remaining decode-wall candidate.

The corrector's price is known only LOCALLY: ``ddm_rr6`` 6 puts it at 59.2% of a 326.2 s
split run on an M5 Max.  Transferring that fraction to T4 is exactly the cross-regime
constant-transfer error rr7 just measured at 2.02x, so it may not be quoted as a shipping
number.  The shipped report emits no sub-stage breakdown at all, so the T4 split cannot be
recovered from any existing row.  **This stager builds the instrument that produces it.**

WHAT IT CHANGES, in exactly one file (``runtime/residual_archive.py``), four recorded
rewrites, all of them TIMING READS:

1. A module-level helper ``_cd1_breakdown`` plus the family map, inserted above
   ``decode_production_tokens``.
2. Accumulator setup beside the existing ``started = time.time()``.
3. The decode loop, with ``time.perf_counter()`` around each unchanged call.
4. The returned token report, gaining ``token_stage_breakdown``.

WHY IT CANNOT MOVE A BYTE.  Every rewrite wraps a call it does not modify.  No argument,
no value, no evaluation order and no control flow changes.  The single expression that is
split -- ``decoder.decode(corrector.coding_row(state))`` -- becomes two statements in the
same order with the same operands.  The claim is not an argument, it is falsifiable: the
staged tree must reproduce ``decoded_token_sha256``, both digest anchors, the decoder bit
position and the whole ``0.raw`` byte-for-byte.  A single differing byte means STOP.

WHAT IT DOES NOT DO.  It does not touch the sealed jg5 tree in place, does not fire
anything, and does not claim a score.  It emits a NEW tree plus a manifest of every source
SHA consumed and produced.  The output owes a full-field identity run before any exact
eval, and its runtime-tree SHA differs from jg5's -- which the candidate seal, not a
hand-typed digest, is responsible for pinning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BASE = REPO / "submissions" / "robust_current" / "jg5_sub015_runtime" / "runtime"
TARGET = "runtime/residual_archive.py"

# --- rewrite 1: the breakdown helper -------------------------------------------------------
HELPER_ANCHOR = '''def decode_production_tokens(
    parts: ResidualArchiveParts,
'''

HELPER_NEW = '''# --- ddm_cd1 SHIPPING-AXIS STAGE DECOMPOSITION --------------------------------------------
# Timing reads ONLY.  Every accumulator below wraps an UNCHANGED call with
# ``time.perf_counter()``; no argument, no value and no evaluation order is touched, so the
# decoded field is bit-identical to the shipped decoder by construction.  The three families
# are the ones the corrector-port decision actually needs:
#   model          the integer HPAC forward and its device round trips (STAYS on CUDA)
#   corrector      the float64 ddm_rr2 free corrector (the port candidate)
#   orchestration  everything a corrector port would NOT touch -- the correction-table
#                  lookup, the probability table, the two digests, the RC64 ctypes decode,
#                  the H2D scatter and the per-frame writeback
CD1_FAMILIES = {
    "model": (
        "model_prepare_context",
        "model_selected_logits",
        "model_d2h_sync",
    ),
    "corrector": (
        "corrector_begin_frame",
        "corrector_group_state",
        "corrector_coding_row",
        "corrector_observe",
        "corrector_end_frame",
    ),
    "orchestration": (
        "orch_boundary_buckets",
        "orch_table_lookup",
        "orch_probability_table",
        "orch_digests",
        "orch_rc64_decode",
        "orch_h2d_scatter",
        "orch_token_writeback",
    ),
}

#: The three corrector calls a port would actually replace (``ddm_rr6`` 6: group_state +
#: coding_row + observe).  ``begin_frame``/``end_frame`` are per-FRAME bookkeeping and are
#: reported separately so nobody prices the port by quietly folding them in.
CD1_PORT_SCOPE = ("corrector_group_state", "corrector_coding_row", "corrector_observe")

CD1_NOTES = (
    "MEASURED wall clock, never a score. The instrumentation is timing-only: the decoded "
    "token field, both digest anchors and the decoder bit position are unchanged, and they "
    "are the falsifier for that claim rather than its justification.",
    "On CUDA the model kernels launch ASYNCHRONOUSLY, so GPU compute is absorbed by "
    "model_d2h_sync (the first blocking read) and NOT by model_selected_logits. Read the "
    "model cost as the FAMILY sum, never one member. The loop is strictly serial -- group "
    "g+1's forward needs group g's symbols -- so no CPU work overlaps pending GPU work and "
    "the families neither double-count nor hide overlap.",
    "orch_rc64_decode carries the symbols' .astype(np.int64) cast, which shared an "
    "expression with coding_row upstream; splitting that expression into two statements "
    "preserves both operand order and values.",
    "port_scope_seconds is the ONLY figure a corrector port may be priced against. Folding "
    "in the probability table, the digests, the RC64 calls or the transfers overstates the "
    "ceiling, which is the error ddm_rr6 6 named when it refused to quote 62.8%.",
)


def _cd1_breakdown(
    accumulators,
    *,
    prelude_seconds,
    loop_seconds,
    total_seconds,
    frame_count,
    group_count,
):
    """Assemble the measured decomposition.  Pure arithmetic over measured seconds."""
    measured = {name: float(value) for name, value in accumulators.items()}
    families = {
        family: sum(measured[name] for name in names)
        for family, names in CD1_FAMILIES.items()
    }
    attributed = sum(families.values())
    return {
        "schema": "ddm_cd1_token_stage_breakdown.v1",
        "measured_seconds": measured,
        "family_seconds": families,
        "port_scope_seconds": sum(measured[name] for name in CD1_PORT_SCOPE),
        "prelude_seconds": float(prelude_seconds),
        "loop_seconds": float(loop_seconds),
        "attributed_in_loop_seconds": attributed,
        "unattributed_in_loop_seconds": float(loop_seconds) - attributed,
        "decode_total_seconds": float(total_seconds),
        "frame_count": int(frame_count),
        "group_count": int(group_count),
        "group_iterations": int(frame_count) * int(group_count),
        "timer_pairs_per_group_iteration": 11,
        "notes": list(CD1_NOTES),
        "is_score_claim": False,
    }


def decode_production_tokens(
    parts: ResidualArchiveParts,
'''

# --- rewrite 2: accumulator setup ----------------------------------------------------------
SETUP_OLD = '''    started = time.time()
    if device.type == "cuda":
        configure_cuda_reproducibility()
'''

SETUP_NEW = '''    started = time.time()
    # ddm_cd1 -- shipping-axis stage decomposition; see _cd1_breakdown above.
    _cd1_clock = time.perf_counter
    _cd1_t0 = _cd1_clock()
    _cd1 = {name: 0.0 for names in CD1_FAMILIES.values() for name in names}
    if device.type == "cuda":
        configure_cuda_reproducibility()
'''

# --- rewrite 3: the timed loop -------------------------------------------------------------
LOOP_OLD = '''    with torch.inference_mode():
        optimize_sparse_evaluator(sparse)
        previous = torch.zeros(
            (1, runtime.EVAL_H, runtime.EVAL_W),
            dtype=torch.long,
            device=device,
        )
        tokens = torch.empty(
            (runtime.N, runtime.EVAL_H, runtime.EVAL_W),
            dtype=torch.uint8,
        )
        for frame in range(runtime.N):
            index = torch.tensor([frame], dtype=torch.long, device=device)
            current = torch.zeros_like(previous)
            context = model.prepare_frame_context(index, previous)
            if frame:
                previous_cpu = previous[0].to(
                    device="cpu", dtype=torch.uint8
                ).numpy()
                boundary = _boundary_buckets(previous_cpu).reshape(-1)
            else:
                boundary = np.full(
                    runtime.EVAL_H * runtime.EVAL_W,
                    4,
                    dtype=np.uint8,
                )
            corrector.begin_frame(boundary)
            for group, (device_positions, flat_positions) in enumerate(group_plans):
                selected = sparse.selected_logits(current, context, group)
                base_logits = selected.cpu().numpy()
                predicted = base_logits.argmax(axis=1).astype(np.int64)
                feature = (
                    boundary[flat_positions].astype(np.int64) * NUM_CLASSES
                    + predicted
                )
                corrected = base_logits + parts.table.values[feature]
                corrected_digest.update(
                    np.ascontiguousarray(corrected, dtype="<f4").tobytes()
                )
                probability = _probability_table(
                    corrected,
                    runtime.HPAC_LOGIT_PRECISION,
                )
                cdf_digest.update(
                    np.ascontiguousarray(probability, dtype="<f4").tobytes()
                )
                state = corrector.group_state(
                    probability,
                    predicted,
                    flat_positions,
                )
                symbols = decoder.decode(
                    corrector.coding_row(state)
                ).astype(np.int64)
                corrector.observe(state, symbols)
                current.reshape(-1)[device_positions] = torch.from_numpy(symbols).to(
                    device
                )
            tokens[frame] = current[0].to(device="cpu", dtype=torch.uint8)
            corrector.end_frame(tokens[frame].numpy().reshape(-1))
            previous = current
'''

LOOP_NEW = '''    _cd1_loop_started = _cd1_clock()
    with torch.inference_mode():
        optimize_sparse_evaluator(sparse)
        previous = torch.zeros(
            (1, runtime.EVAL_H, runtime.EVAL_W),
            dtype=torch.long,
            device=device,
        )
        tokens = torch.empty(
            (runtime.N, runtime.EVAL_H, runtime.EVAL_W),
            dtype=torch.uint8,
        )
        for frame in range(runtime.N):
            index = torch.tensor([frame], dtype=torch.long, device=device)
            current = torch.zeros_like(previous)
            _cd1_t = _cd1_clock()
            context = model.prepare_frame_context(index, previous)
            _cd1["model_prepare_context"] += _cd1_clock() - _cd1_t
            _cd1_t = _cd1_clock()
            if frame:
                previous_cpu = previous[0].to(
                    device="cpu", dtype=torch.uint8
                ).numpy()
                boundary = _boundary_buckets(previous_cpu).reshape(-1)
            else:
                boundary = np.full(
                    runtime.EVAL_H * runtime.EVAL_W,
                    4,
                    dtype=np.uint8,
                )
            _cd1["orch_boundary_buckets"] += _cd1_clock() - _cd1_t
            _cd1_t = _cd1_clock()
            corrector.begin_frame(boundary)
            _cd1["corrector_begin_frame"] += _cd1_clock() - _cd1_t
            for group, (device_positions, flat_positions) in enumerate(group_plans):
                _cd1_t = _cd1_clock()
                selected = sparse.selected_logits(current, context, group)
                _cd1["model_selected_logits"] += _cd1_clock() - _cd1_t
                _cd1_t = _cd1_clock()
                base_logits = selected.cpu().numpy()
                _cd1["model_d2h_sync"] += _cd1_clock() - _cd1_t
                _cd1_t = _cd1_clock()
                predicted = base_logits.argmax(axis=1).astype(np.int64)
                feature = (
                    boundary[flat_positions].astype(np.int64) * NUM_CLASSES
                    + predicted
                )
                corrected = base_logits + parts.table.values[feature]
                _cd1["orch_table_lookup"] += _cd1_clock() - _cd1_t
                _cd1_t = _cd1_clock()
                corrected_digest.update(
                    np.ascontiguousarray(corrected, dtype="<f4").tobytes()
                )
                _cd1["orch_digests"] += _cd1_clock() - _cd1_t
                _cd1_t = _cd1_clock()
                probability = _probability_table(
                    corrected,
                    runtime.HPAC_LOGIT_PRECISION,
                )
                _cd1["orch_probability_table"] += _cd1_clock() - _cd1_t
                _cd1_t = _cd1_clock()
                cdf_digest.update(
                    np.ascontiguousarray(probability, dtype="<f4").tobytes()
                )
                _cd1["orch_digests"] += _cd1_clock() - _cd1_t
                _cd1_t = _cd1_clock()
                state = corrector.group_state(
                    probability,
                    predicted,
                    flat_positions,
                )
                _cd1["corrector_group_state"] += _cd1_clock() - _cd1_t
                _cd1_t = _cd1_clock()
                _cd1_coding_row = corrector.coding_row(state)
                _cd1["corrector_coding_row"] += _cd1_clock() - _cd1_t
                _cd1_t = _cd1_clock()
                symbols = decoder.decode(_cd1_coding_row).astype(np.int64)
                _cd1["orch_rc64_decode"] += _cd1_clock() - _cd1_t
                _cd1_t = _cd1_clock()
                corrector.observe(state, symbols)
                _cd1["corrector_observe"] += _cd1_clock() - _cd1_t
                _cd1_t = _cd1_clock()
                current.reshape(-1)[device_positions] = torch.from_numpy(symbols).to(
                    device
                )
                _cd1["orch_h2d_scatter"] += _cd1_clock() - _cd1_t
            _cd1_t = _cd1_clock()
            tokens[frame] = current[0].to(device="cpu", dtype=torch.uint8)
            _cd1["orch_token_writeback"] += _cd1_clock() - _cd1_t
            _cd1_t = _cd1_clock()
            corrector.end_frame(tokens[frame].numpy().reshape(-1))
            _cd1["corrector_end_frame"] += _cd1_clock() - _cd1_t
            previous = current
    _cd1_loop_seconds = _cd1_clock() - _cd1_loop_started
'''

# --- rewrite 4: the token report -----------------------------------------------------------
REPORT_OLD = '''    elapsed = time.time() - started
    del model
    return tokens, {
        "corrected_quantized_logit_sha256": corrected_digest.hexdigest(),
        "corrected_cdf_input_sha256": cdf_digest.hexdigest(),
        "decoded_token_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
        "decode_runtime_seconds": elapsed,
        "adapter_sha256": "",
        "token_codec": "rc64",
        "decoder_bit_position": decoder.bit_position,
    }
'''

REPORT_NEW = '''    elapsed = time.time() - started
    _cd1_report = _cd1_breakdown(
        _cd1,
        prelude_seconds=_cd1_loop_started - _cd1_t0,
        loop_seconds=_cd1_loop_seconds,
        total_seconds=_cd1_clock() - _cd1_t0,
        frame_count=runtime.N,
        group_count=len(group_plans),
    )
    del model
    return tokens, {
        "corrected_quantized_logit_sha256": corrected_digest.hexdigest(),
        "corrected_cdf_input_sha256": cdf_digest.hexdigest(),
        "decoded_token_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
        "decode_runtime_seconds": elapsed,
        "adapter_sha256": "",
        "token_codec": "rc64",
        "decoder_bit_position": decoder.bit_position,
        "token_stage_breakdown": _cd1_report,
    }
'''

REWRITES = (
    ("breakdown helper", HELPER_ANCHOR, HELPER_NEW),
    ("accumulator setup", SETUP_OLD, SETUP_NEW),
    ("timed decode loop", LOOP_OLD, LOOP_NEW),
    ("token report breakdown", REPORT_OLD, REPORT_NEW),
)


class StagingError(RuntimeError):
    """A base-tree assumption or a produced-tree invariant failed."""


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _apply(text: str, old: str, new: str, label: str) -> str:
    """Apply one recorded rewrite, refusing on any count but exactly one."""
    found = text.count(old)
    if found != 1:
        raise StagingError(
            f"{label}: expected exactly 1 occurrence in the base tree, found {found}; "
            "the base tree has drifted and this transformation is stale"
        )
    return text.replace(old, new)


def _tree_sha256(root: Path) -> str:
    """Order-stable hash over every staged file's relative path and content.

    Byte-compatible with ``ddm_wc2c_stage_native_split_runtime._tree_sha256`` on purpose:
    the two staged trees are compared against the same jg5 base, so a divergent definition
    would make their manifests incomparable.
    """
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.name.startswith("._"):
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(_sha256_file(path).encode())
    return digest.hexdigest()


def _assert_only_target_changed(base: Path, output: Path) -> list[str]:
    """Refuse unless the staged tree differs from its base in exactly ``TARGET``."""
    def census(root: Path) -> dict[str, str]:
        return {
            str(path.relative_to(root)): _sha256_file(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and "__pycache__" not in path.parts
            and not path.name.startswith("._")
        }

    before, after = census(base), census(output)
    if set(before) != set(after):
        raise StagingError(
            "staging changed the FILE SET, which this transformation never does: "
            f"added={sorted(set(after) - set(before))} removed={sorted(set(before) - set(after))}"
        )
    changed = sorted(name for name in before if before[name] != after[name])
    if changed != [TARGET]:
        raise StagingError(
            f"expected exactly one changed file ({TARGET}), got {changed}"
        )
    return changed


def stage(base: Path, output: Path, *, force: bool = False) -> dict[str, Any]:
    base = base.resolve()
    output = output.resolve()
    if not (base / TARGET).is_file():
        raise StagingError(f"no candidate runtime tree at {base}")
    if output.exists():
        if not force:
            raise StagingError(f"refusing to overwrite existing tree: {output}")
        shutil.rmtree(output)

    base_sha = _tree_sha256(base)
    shutil.copytree(base, output, ignore=shutil.ignore_patterns("__pycache__", "._*"))

    target_path = output / TARGET
    text = target_path.read_text()
    for label, old, new in REWRITES:
        text = _apply(text, old, new, label)
    target_path.write_text(text)

    # The staged module must at least PARSE before anything downstream trusts it; a
    # SyntaxError discovered on a paid runner costs the dispatch, not the edit.
    compile(text, str(target_path), "exec")
    changed = _assert_only_target_changed(base, output)

    manifest = {
        "schema": "ddm_cd1_stage_instrumented.v1",
        "base_tree": str(base),
        "base_tree_sha256": base_sha,
        "output_tree": str(output),
        "output_tree_sha256": _tree_sha256(output),
        "changed_files": changed,
        "staged_files": {TARGET: _sha256_file(target_path)},
        "rewrites_applied": [label for label, _, _ in REWRITES],
        "instrumentation": {
            "kind": "wall-clock accumulation only",
            "families": ["model", "corrector", "orchestration"],
            "emitted_at": "report['token_decoder']['token_stage_breakdown']",
            "cannot_move_a_byte_because": (
                "every rewrite wraps an unchanged call; the one split expression keeps "
                "operand order and values"
            ),
        },
        "owed_before_any_exact_eval": [
            "full-field identity run against this staged tree (0.raw + the four token anchors)",
            "a fresh candidate seal measured from this tree -- never a hand-typed digest",
        ],
    }
    _atomic_json(output.parent / f"{output.name}_stage_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    try:
        manifest = stage(args.base, args.output, force=args.force)
    except StagingError as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
