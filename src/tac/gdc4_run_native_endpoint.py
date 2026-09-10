"""Run-native endpoint generator primitives for the ddm_gdc4 generator door.

The move-44 token field is a ``(600, 384, 512)`` uint8 array of 5 class labels.
This module holds the shared, video-independent primitives the arm needs:

* horizontal run extraction (the native symbol space: ``(x_stop, class)``);
* the optimal bounded-endpoint approximation of a row (an ORACLE floor: no
  generator restricted to ``E`` endpoints per row can beat it);
* the row-level rasterizer the receiver uses.

Nothing here contains a video-derived constant. Every routine is generic over
an arbitrary label field and is exercised by the arm's tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_CLASSES = 5


def row_runs(row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(values, stops)`` for the maximal horizontal runs of ``row``.

    ``stops[k]`` is the exclusive end coordinate of run ``k``; the final stop is
    always ``len(row)``. ``values[k]`` is the class of run ``k``.
    """
    if row.ndim != 1:
        raise ValueError(f"row must be 1-D, got shape {row.shape}")
    n = int(row.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    change = np.flatnonzero(row[1:] != row[:-1]) + 1
    stops = np.empty(change.shape[0] + 1, dtype=np.int64)
    stops[:-1] = change
    stops[-1] = n
    starts = np.empty_like(stops)
    starts[0] = 0
    starts[1:] = stops[:-1]
    return row[starts].astype(np.int64), stops


def raster_row(values: np.ndarray, stops: np.ndarray, width: int) -> np.ndarray:
    """Rasterize ``(values, stops)`` left-to-right into a width-``width`` row.

    This is the receiver's free generic expansion step. ``stops`` must be
    strictly increasing and end at ``width``.
    """
    values = np.asarray(values, dtype=np.int64)
    stops = np.asarray(stops, dtype=np.int64)
    if values.shape[0] != stops.shape[0]:
        raise ValueError("values and stops must have equal length")
    if values.shape[0] == 0:
        raise ValueError("at least one run is required")
    if int(stops[-1]) != int(width):
        raise ValueError(f"final stop {int(stops[-1])} must equal width {width}")
    if np.any(np.diff(stops) <= 0) or int(stops[0]) <= 0:
        raise ValueError("stops must be strictly increasing and positive")
    starts = np.empty_like(stops)
    starts[0] = 0
    starts[1:] = stops[:-1]
    out = np.empty(int(width), dtype=np.uint8)
    for value, start, stop in zip(values, starts, stops, strict=True):
        out[int(start) : int(stop)] = np.uint8(value)
    return out


def row_class_prefix(values: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Per-class cumulative run length: shape ``(n_runs + 1, N_CLASSES)``."""
    r = int(values.shape[0])
    table = np.zeros((r + 1, N_CLASSES), dtype=np.int64)
    if r:
        onehot = np.zeros((r, N_CLASSES), dtype=np.int64)
        onehot[np.arange(r), values] = lengths
        table[1:] = np.cumsum(onehot, axis=0)
    return table


def optimal_bounded_endpoint_cost(row: np.ndarray, budgets: np.ndarray) -> np.ndarray:
    """Exact minimum mismatch count for each endpoint budget in ``budgets``.

    A generator that emits at most ``E`` ``(x_stop, class)`` symbols for this row
    produces a piecewise-constant row with at most ``E`` pieces. This returns,
    for each ``E``, the minimum possible number of mismatched cells against
    ``row`` -- an ORACLE floor that holds for any such generator regardless of
    how it is trained or how many bytes its state costs.

    Breakpoints may be restricted to original run boundaries without loss: the
    target is constant inside a run, so sliding an interior breakpoint to the
    nearer run edge never increases cost.
    """
    budgets = np.asarray(budgets, dtype=np.int64)
    values, stops = row_runs(row)
    r = int(values.shape[0])
    starts = np.empty_like(stops)
    starts[0] = 0
    starts[1:] = stops[:-1]
    lengths = stops - starts
    positions = np.concatenate(([0], stops))
    prefix = row_class_prefix(values, lengths)

    max_budget = int(budgets.max())
    out = np.zeros(budgets.shape[0], dtype=np.int64)
    if r <= 1:
        return out

    # cost[i, j] for 0 <= i < j <= r
    span = positions[None, :] - positions[:, None]
    per_class = prefix[None, :, :] - prefix[:, None, :]
    cost = span - per_class.max(axis=2)

    inf = np.int64(1) << 40
    f = np.full(r + 1, inf, dtype=np.int64)
    f[0] = 0
    best = {}
    for e in range(1, min(max_budget, r) + 1):
        nxt = np.full(r + 1, inf, dtype=np.int64)
        for j in range(1, r + 1):
            nxt[j] = int((f[:j] + cost[:j, j]).min())
        f = nxt
        best[e] = int(f[r])
    for idx, budget in enumerate(budgets):
        e = int(min(budget, r))
        out[idx] = best[e] if e >= 1 else int(row.shape[0])
    return out


@dataclass(frozen=True)
class FrameRunStats:
    """Per-frame aggregate run statistics."""

    frame: int
    n_rows: int
    n_runs: int
    max_runs_in_row: int
    run_length_sum: int


def frame_run_counts(frame: np.ndarray) -> np.ndarray:
    """Runs per row for a ``(H, W)`` frame."""
    if frame.ndim != 2:
        raise ValueError(f"frame must be 2-D, got shape {frame.shape}")
    if frame.shape[1] == 0:
        return np.zeros(frame.shape[0], dtype=np.int64)
    changes = (frame[:, 1:] != frame[:, :-1]).sum(axis=1, dtype=np.int64)
    return changes + 1


# --------------------------------------------------------------------------
# Run-native packet codec.
#
# The counted packet is a set of independently coded byte streams. All of the
# machinery below -- reference selection, varint framing, rasterization -- is
# generic receiver code with no video-derived constant: the only counted object
# is the coded stream content.
# --------------------------------------------------------------------------

MODE_COPY_A = 0
MODE_COPY_B = 1
MODE_DELTA_A = 2
MODE_DELTA_B = 3
MODE_EXPLICIT = 4
STREAM_NAMES = ("mode", "nrun", "cls", "dstop", "stop")


def _zigzag(value: int) -> int:
    return (value << 1) ^ (value >> 63) if value >= 0 else ((-value) << 1) - 1


def _unzigzag(value: int) -> int:
    return (value >> 1) if (value & 1) == 0 else -((value + 1) >> 1)


def _put_uvarint(out: bytearray, value: int) -> None:
    if value < 0:
        raise ValueError("uvarint requires a non-negative value")
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return


class _VarintReader:
    __slots__ = ("buf", "pos")

    def __init__(self, buf: bytes) -> None:
        self.buf = buf
        self.pos = 0

    def uvarint(self) -> int:
        value = 0
        shift = 0
        buf = self.buf
        while True:
            byte = buf[self.pos]
            self.pos += 1
            value |= (byte & 0x7F) << shift
            if not (byte & 0x80):
                return value
            shift += 7

    def svarint(self) -> int:
        return _unzigzag(self.uvarint())

    def exhausted(self) -> bool:
        return self.pos == len(self.buf)


def _default_row_runs(width: int) -> tuple[list[int], list[int]]:
    """The generic seed reference: one run of class 0 spanning the row."""
    return [0], [int(width)]


def encode_field_runs(
    field: np.ndarray,
    *,
    prefer_previous_frame: bool = False,
) -> dict[str, bytes]:
    """Encode a ``(F, H, W)`` label field into run-native byte streams.

    ``prefer_previous_frame`` only breaks ties between the two free causal
    references; it never changes what is representable.
    """
    if field.ndim != 3:
        raise ValueError(f"field must be 3-D, got shape {field.shape}")
    n_frames, height, width = (int(v) for v in field.shape)

    mode_s = bytearray()
    nrun_s = bytearray()
    cls_s = bytearray()
    dstop_s = bytearray()
    stop_s = bytearray()

    seed_values, seed_stops = _default_row_runs(width)
    prev_frame_rows: list[tuple[list[int], list[int]]] | None = None

    for f in range(n_frames):
        frame = np.asarray(field[f])
        this_frame_rows: list[tuple[list[int], list[int]]] = []
        prev_row: tuple[list[int], list[int]] = (seed_values, seed_stops)
        for y in range(height):
            values_np, stops_np = row_runs(frame[y])
            values = values_np.tolist()
            stops = stops_np.tolist()

            ref_a = prev_row
            ref_b = prev_frame_rows[y] if prev_frame_rows is not None else ref_a

            mode = MODE_EXPLICIT
            deltas: list[int] | None = None
            cand: list[tuple[int, int, list[int]]] = []
            for tag, ref in ((MODE_DELTA_A, ref_a), (MODE_DELTA_B, ref_b)):
                rv, rs = ref
                if rv == values:
                    if rs == stops:
                        mode = MODE_COPY_A if tag == MODE_DELTA_A else MODE_COPY_B
                        deltas = None
                        cand = []
                        break
                    d = [stops[i] - rs[i] for i in range(len(stops) - 1)]
                    cand.append((sum(abs(v) for v in d), tag, d))
            if mode == MODE_EXPLICIT and cand:
                cand.sort(key=lambda item: (item[0], item[1] if not prefer_previous_frame else -item[1]))
                _, mode, deltas = cand[0]

            mode_s.append(mode)
            if mode in (MODE_DELTA_A, MODE_DELTA_B):
                assert deltas is not None
                for value in deltas:
                    _put_uvarint(dstop_s, _zigzag(value))
            elif mode == MODE_EXPLICIT:
                n = len(values)
                _put_uvarint(nrun_s, n)
                cls_s.extend(bytes(values))
                previous = 0
                for i in range(n - 1):
                    _put_uvarint(stop_s, stops[i] - previous)
                    previous = stops[i]

            prev_row = (values, stops)
            this_frame_rows.append(prev_row)
        prev_frame_rows = this_frame_rows

    return {
        "mode": bytes(mode_s),
        "nrun": bytes(nrun_s),
        "cls": bytes(cls_s),
        "dstop": bytes(dstop_s),
        "stop": bytes(stop_s),
    }


def decode_field_runs(
    streams: dict[str, bytes],
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Reference receiver: expand run-native streams back to a label field."""
    n_frames, height, width = (int(v) for v in shape)
    mode_s = streams["mode"]
    nrun_r = _VarintReader(streams["nrun"])
    cls_s = streams["cls"]
    dstop_r = _VarintReader(streams["dstop"])
    stop_r = _VarintReader(streams["stop"])

    if len(mode_s) != n_frames * height:
        raise ValueError("mode stream length does not match the declared shape")

    out = np.empty((n_frames, height, width), dtype=np.uint8)
    seed_values, seed_stops = _default_row_runs(width)
    prev_frame_rows: list[tuple[list[int], list[int]]] | None = None
    cls_pos = 0
    mode_pos = 0

    for f in range(n_frames):
        this_frame_rows: list[tuple[list[int], list[int]]] = []
        prev_row: tuple[list[int], list[int]] = (seed_values, seed_stops)
        for y in range(height):
            mode = mode_s[mode_pos]
            mode_pos += 1
            ref_a = prev_row
            ref_b = prev_frame_rows[y] if prev_frame_rows is not None else ref_a

            if mode == MODE_COPY_A:
                values, stops = ref_a
                values = list(values)
                stops = list(stops)
            elif mode == MODE_COPY_B:
                values, stops = ref_b
                values = list(values)
                stops = list(stops)
            elif mode in (MODE_DELTA_A, MODE_DELTA_B):
                rv, rs = ref_a if mode == MODE_DELTA_A else ref_b
                values = list(rv)
                stops = list(rs)
                for i in range(len(stops) - 1):
                    stops[i] = rs[i] + dstop_r.svarint()
            elif mode == MODE_EXPLICIT:
                n = nrun_r.uvarint()
                values = list(cls_s[cls_pos : cls_pos + n])
                cls_pos += n
                stops = []
                previous = 0
                for _ in range(n - 1):
                    previous += stop_r.uvarint()
                    stops.append(previous)
                stops.append(width)
            else:
                raise ValueError(f"unknown mode byte {mode}")

            out[f, y] = raster_row(
                np.asarray(values, dtype=np.int64),
                np.asarray(stops, dtype=np.int64),
                width,
            )
            prev_row = (values, stops)
            this_frame_rows.append(prev_row)
        prev_frame_rows = this_frame_rows

    if cls_pos != len(cls_s) or not dstop_r.exhausted() or not stop_r.exhausted():
        raise ValueError("stream residue after decode: packet is not self-consistent")
    return out


# --------------------------------------------------------------------------
# Transition-aligned packet codec (v2).
#
# A row is described as an EDIT of the causally available reference row's
# transition list. Region boundaries are curves in (y, x): consecutive rows
# usually share transitions at nearly the same x, and topology changes are one
# insert or delete. The alignment is encoder-side only; the emitted op script
# determines the decode exactly.
# --------------------------------------------------------------------------

OP_MATCH = 0
OP_SUBST = 1
OP_INSERT = 2
OP_DELETE = 3
OP_END = 4
V2_STREAM_NAMES = ("ref", "lead", "op", "dx", "ins", "icls", "scls")

_COST_MATCH_BASE = 2
_COST_SUBST_BASE = 6
_COST_INSERT = 14
_COST_DELETE = 3


def _row_transitions(row: np.ndarray) -> tuple[int, list[tuple[int, int]]]:
    """Return ``(leading_class, [(x, class_after), ...])`` for one row."""
    values, stops = row_runs(row)
    lead = int(values[0])
    trans = [(int(stops[i]), int(values[i + 1])) for i in range(values.shape[0] - 1)]
    return lead, trans


def _align_transitions(
    cur: list[tuple[int, int]], ref: list[tuple[int, int]]
) -> list[tuple[int, int, int]]:
    """Least-cost edit script turning ``ref`` into ``cur``.

    Returns a list of ``(op, arg0, arg1)`` where the meaning of the args depends
    on the op: MATCH/SUBST carry ``(dx, class)``, INSERT carries ``(x, class)``,
    DELETE carries ``(0, 0)``.
    """
    m, k = len(cur), len(ref)
    inf = 1 << 30
    # dp[i][j] = cost of producing cur[i:] from ref[j:]
    dp = [[inf] * (k + 1) for _ in range(m + 1)]
    back = [[(0, 0)] * (k + 1) for _ in range(m + 1)]
    dp[m][k] = 0
    for j in range(k - 1, -1, -1):
        dp[m][j] = dp[m][j + 1] + _COST_DELETE
        back[m][j] = (OP_DELETE, j + 1)
    for i in range(m - 1, -1, -1):
        dp[i][k] = dp[i + 1][k] + _COST_INSERT
        back[i][k] = (OP_INSERT, k)
        for j in range(k - 1, -1, -1):
            best = dp[i + 1][j] + _COST_INSERT
            choice = (OP_INSERT, j)
            cand = dp[i][j + 1] + _COST_DELETE
            if cand < best:
                best, choice = cand, (OP_DELETE, j + 1)
            dx = cur[i][0] - ref[j][0]
            adx = dx if dx >= 0 else -dx
            step = adx.bit_length()
            if cur[i][1] == ref[j][1]:
                cand = dp[i + 1][j + 1] + _COST_MATCH_BASE + step
                if cand < best:
                    best, choice = cand, (OP_MATCH, j + 1)
            else:
                cand = dp[i + 1][j + 1] + _COST_SUBST_BASE + step
                if cand < best:
                    best, choice = cand, (OP_SUBST, j + 1)
            dp[i][j] = best
            back[i][j] = choice

    script: list[tuple[int, int, int]] = []
    i = j = 0
    while i < m or j < k:
        op, nj = back[i][j]
        if op == OP_DELETE:
            script.append((OP_DELETE, 0, 0))
            j = nj
        elif op == OP_INSERT:
            script.append((OP_INSERT, cur[i][0], cur[i][1]))
            i += 1
            j = nj
        else:
            script.append((op, cur[i][0] - ref[j][0], cur[i][1]))
            i += 1
            j = nj
    return script


def _script_cost(script: list[tuple[int, int, int]]) -> int:
    total = 1
    for op, arg0, _ in script:
        if op == OP_DELETE:
            total += _COST_DELETE
        elif op == OP_INSERT:
            total += _COST_INSERT
        else:
            adx = arg0 if arg0 >= 0 else -arg0
            base = _COST_MATCH_BASE if op == OP_MATCH else _COST_SUBST_BASE
            total += base + adx.bit_length()
    return total


def plan_field_transitions(field: np.ndarray):
    """Yield the per-row coding plan for a ``(F, H, W)`` label field.

    Yields ``(f, y, ref_tag, lead_code, script, trans, ref_trans, prev_script)``.
    Both the byte serializer and the adaptive-cost tracer consume this single
    planner so the two can never drift apart.
    """
    if field.ndim != 3:
        raise ValueError(f"field must be 3-D, got shape {field.shape}")
    n_frames, height, _width = (int(v) for v in field.shape)
    seed: tuple[int, list[tuple[int, int]]] = (0, [])
    prev_frame_rows: list[tuple[int, list[tuple[int, int]]]] | None = None

    for f in range(n_frames):
        frame = np.asarray(field[f])
        this_frame: list[tuple[int, list[tuple[int, int]]]] = []
        prev_row = seed
        prev_script: list[tuple[int, int, int]] = []
        for y in range(height):
            lead, trans = _row_transitions(frame[y])
            ref_a = prev_row
            ref_b = prev_frame_rows[y] if prev_frame_rows is not None else ref_a

            script_a = _align_transitions(trans, ref_a[1])
            cost_a = _script_cost(script_a) + (0 if lead == ref_a[0] else 8)
            if ref_b is ref_a:
                script, ref_tag, ref_lead, ref_trans = script_a, 0, ref_a[0], ref_a[1]
            else:
                script_b = _align_transitions(trans, ref_b[1])
                cost_b = _script_cost(script_b) + (0 if lead == ref_b[0] else 8)
                if cost_b < cost_a:
                    script, ref_tag, ref_lead, ref_trans = script_b, 1, ref_b[0], ref_b[1]
                else:
                    script, ref_tag, ref_lead, ref_trans = script_a, 0, ref_a[0], ref_a[1]

            lead_code = 0 if lead == ref_lead else 1 + lead
            yield (f, y, ref_tag, lead_code, script, trans, ref_trans, prev_script)

            prev_row = (lead, trans)
            prev_script = script
            this_frame.append(prev_row)
        prev_frame_rows = this_frame


def encode_field_transitions(field: np.ndarray) -> dict[str, bytes]:
    """Encode a ``(F, H, W)`` label field into transition-edit byte streams."""
    ref_s = bytearray()
    lead_s = bytearray()
    op_s = bytearray()
    dx_s = bytearray()
    ins_s = bytearray()
    icls_s = bytearray()
    scls_s = bytearray()

    for _f, _y, ref_tag, lead_code, script, trans, _ref_trans, _prev in (
        plan_field_transitions(field)
    ):
        ref_s.append(ref_tag)
        lead_s.append(lead_code)
        prev_x = 0
        i_cur = 0
        for op, arg0, arg1 in script:
            op_s.append(op)
            if op == OP_MATCH:
                _put_uvarint(dx_s, _zigzag(arg0))
                prev_x = trans[i_cur][0]
                i_cur += 1
            elif op == OP_SUBST:
                _put_uvarint(dx_s, _zigzag(arg0))
                scls_s.append(arg1)
                prev_x = trans[i_cur][0]
                i_cur += 1
            elif op == OP_INSERT:
                _put_uvarint(ins_s, _zigzag(arg0 - prev_x))
                icls_s.append(arg1)
                prev_x = arg0
                i_cur += 1
        op_s.append(OP_END)

    return {
        "ref": bytes(ref_s),
        "lead": bytes(lead_s),
        "op": bytes(op_s),
        "dx": bytes(dx_s),
        "ins": bytes(ins_s),
        "icls": bytes(icls_s),
        "scls": bytes(scls_s),
    }


def decode_field_transitions(
    streams: dict[str, bytes], shape: tuple[int, int, int]
) -> np.ndarray:
    """Reference receiver for the transition-edit packet."""
    n_frames, height, width = (int(v) for v in shape)
    ref_s = streams["ref"]
    lead_s = streams["lead"]
    op_s = streams["op"]
    dx_r = _VarintReader(streams["dx"])
    ins_r = _VarintReader(streams["ins"])
    icls_s = streams["icls"]
    scls_s = streams["scls"]

    out = np.empty((n_frames, height, width), dtype=np.uint8)
    seed: tuple[int, list[tuple[int, int]]] = (0, [])
    prev_frame_rows: list[tuple[int, list[tuple[int, int]]]] | None = None
    op_pos = row_pos = icls_pos = scls_pos = 0

    for f in range(n_frames):
        this_frame: list[tuple[int, list[tuple[int, int]]]] = []
        prev_row = seed
        for y in range(height):
            ref_tag = ref_s[row_pos]
            lead_code = lead_s[row_pos]
            row_pos += 1
            ref_a = prev_row
            ref_b = prev_frame_rows[y] if prev_frame_rows is not None else ref_a
            ref_lead, ref_trans = ref_a if ref_tag == 0 else ref_b
            lead = ref_lead if lead_code == 0 else lead_code - 1

            trans: list[tuple[int, int]] = []
            j = 0
            prev_x = 0
            while True:
                op = op_s[op_pos]
                op_pos += 1
                if op == OP_END:
                    break
                if op == OP_MATCH:
                    x = ref_trans[j][0] + dx_r.svarint()
                    trans.append((x, ref_trans[j][1]))
                    j += 1
                elif op == OP_SUBST:
                    x = ref_trans[j][0] + dx_r.svarint()
                    trans.append((x, scls_s[scls_pos]))
                    scls_pos += 1
                    j += 1
                elif op == OP_INSERT:
                    prev_x += ins_r.svarint()
                    trans.append((prev_x, icls_s[icls_pos]))
                    icls_pos += 1
                elif op == OP_DELETE:
                    j += 1
                else:
                    raise ValueError(f"unknown op byte {op}")
                if op in (OP_MATCH, OP_SUBST):
                    prev_x = trans[-1][0]

            values = [lead] + [c for _, c in trans]
            stops = [x for x, _ in trans] + [width]
            out[f, y] = raster_row(
                np.asarray(values, dtype=np.int64),
                np.asarray(stops, dtype=np.int64),
                width,
            )
            prev_row = (lead, trans)
            this_frame.append(prev_row)
        prev_frame_rows = this_frame

    if (
        op_pos != len(op_s)
        or icls_pos != len(icls_s)
        or scls_pos != len(scls_s)
        or not dx_r.exhausted()
        or not ins_r.exhausted()
    ):
        raise ValueError("stream residue after decode: packet is not self-consistent")
    return out
