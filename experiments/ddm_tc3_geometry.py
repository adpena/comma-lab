"""Receiver-only TC2 reference and TC3 coherent run geometry.

Only NumPy and decoded symbols enter this module. Constants specify a generic
geometry algorithm; there are no fitted/video-derived tables. Call ``contexts``
once before each complete HPAC group, then ``observe`` after decoding that group.
``previous`` is the complete decoded previous frame, or None for frame zero.

TC2 uses its original float64 moment arithmetic. TC3 uses integer endpoint
tracking and therefore has no host-dependent floating point bin boundary.
These maps alone do not prove compression gain or a public receiver round trip.
"""

from __future__ import annotations

import numpy as np

H, W, K = 384, 512, 5
Y, X = np.indices((H, W))
GROUP = X % 64 + 2 * (Y % 64)
POSITIONS = tuple(np.flatnonzero(GROUP.reshape(-1) == g) for g in range(190))
ROW_SLOT = Y * 8 + X // 64
VALUES = (np.ones((H, W)), Y, Y * Y, X, X * Y, X * X)


def _above_window(value):
    prefix = np.concatenate([np.zeros_like(value[:, :1]), np.cumsum(value, axis=1)], axis=1)
    return prefix[:, np.arange(H)] - prefix[:, np.maximum(0, np.arange(H) - 16)]


class _GroupPrefix:
    """Enforce the input boundary; a group cannot be partially fed or skipped."""

    def __init__(self, previous):
        if previous is not None:
            previous = np.asarray(previous)
            if previous.shape != (H, W) or previous.dtype.kind not in "iu" or np.any(previous >= K):
                raise ValueError("previous must be a complete decoded HxW class plane")
            if np.any(previous < 0):
                raise ValueError("negative previous symbol")
        self.previous = None if previous is None else previous.copy()
        self.plane = np.full((H, W), K, dtype=np.uint8)
        self.group = 0
        self.pending = False

    def _positions(self, positions):
        positions = np.asarray(positions)
        if (
            self.group >= len(POSITIONS)
            or positions.dtype.kind not in "iu"
            or not np.array_equal(positions, POSITIONS[self.group])
        ):
            raise ValueError("expected the complete sorted positions for the next HPAC group")
        return positions

    def _begin(self, positions):
        positions = self._positions(positions)
        if self.pending:
            raise ValueError("observe must follow contexts before the next contexts")
        self.pending = True
        return positions

    def _observe(self, positions, symbols):
        positions = self._positions(positions)
        symbols = np.asarray(symbols)
        if not self.pending:
            raise ValueError("contexts must precede observe")
        if symbols.shape != positions.shape or symbols.dtype.kind not in "iu":
            raise ValueError("symbols must be the integer decoded group")
        if np.any(symbols < 0) or np.any(symbols >= K):
            raise ValueError("decoded symbol outside class alphabet")
        self.plane.reshape(-1)[positions] = symbols
        self.group += 1
        self.pending = False
        return positions, symbols


class LaneGeometry(_GroupPrefix):
    """TC2's exact slotwise floating point reference, without experiment imports."""

    def __init__(self, previous):
        super().__init__(previous)
        self.current = np.zeros((6, 1, H, 8))
        self.lo, self.hi = np.full((H, 8), W), np.full((H, 8), -1)
        prior = np.zeros((H, W), dtype=bool) if self.previous is None else self.previous == 1
        self.old = np.stack(
            [np.bincount(ROW_SLOT[prior], weights=v[prior], minlength=H * 8).reshape(1, H, 8) for v in VALUES]
        )
        self.old_lo, self.old_hi = np.full((H, 8), W), np.full((H, 8), -1)
        np.minimum.at(self.old_lo.reshape(-1), ROW_SLOT[prior], X[prior])
        np.maximum.at(self.old_hi.reshape(-1), ROW_SLOT[prior], X[prior])
        runs = prior.reshape(H, 8, 64)
        starts = runs & ~np.concatenate([np.zeros((H, 8, 1), dtype=bool), runs[:, :, :-1]], axis=2)
        self.ambiguous_prior = starts.sum(axis=2) > 1

    def contexts(self, positions):
        positions = self._begin(positions)
        count, sy, syy, sx, sxy, sxx = [_above_window(v)[0] for v in self.current + self.old * 0.25]
        den = count * syy - sy * sy
        slope = np.divide(count * sxy - sy * sx, den, out=np.zeros_like(den), where=den > 0)
        my = np.divide(sy, count, out=np.zeros_like(sy), where=count > 0)
        mx = np.divide(sx, count, out=np.zeros_like(sx), where=count > 0)
        centers = mx + slope * (np.arange(H)[:, None] - my)
        variance = np.divide(sxx, count, out=np.zeros_like(sxx), where=count > 0) - mx**2
        residual = variance - slope * np.divide(sxy - my * sx, count, out=np.zeros_like(sxy), where=count > 0)
        width = np.maximum(0.5, np.sqrt(np.maximum(0, 3 * residual)))
        valid = (count >= 4) & (den > 0) & (residual <= 36) & (np.abs(slope) <= 2)
        known = self.plane.reshape(H, 8, 64)
        xx = X.reshape(H, 8, 64)
        gap = ((known < K) & (known != 1) & (xx > self.lo[:, :, None]) & (xx < self.hi[:, :, None])).any(axis=2)
        wide = (np.maximum(self.hi, self.old_hi) - np.minimum(self.lo, self.old_lo)) > 24
        valid &= _above_window((wide | gap | self.ambiguous_prior)[None].astype(np.int64))[0] == 0
        yy, xx = positions // W, positions % W
        distance = np.full(len(positions), np.inf)
        for slot in range(8):
            d = np.minimum(
                np.abs(xx - (centers[yy, slot] - width[yy, slot])),
                np.abs(xx - (centers[yy, slot] + width[yy, slot])),
            )
            distance = np.minimum(distance, np.where(valid[yy, slot], np.rint(d), np.inf))
        bins = np.searchsorted([0, 1, 2, 4, 8, 16], distance, side="left").astype(np.uint8)
        bins[~np.isfinite(distance)] = 7
        bins[(yy < 128) | (yy >= 320)] = 8
        return bins

    def observe(self, positions, symbols):
        positions, symbols = self._observe(positions, symbols)
        lane = positions[symbols == 1]
        code = ROW_SLOT.reshape(-1)[lane]
        for j, value in enumerate(VALUES):
            np.add.at(self.current[j].reshape(-1), code, value.reshape(-1)[lane])
        np.minimum.at(self.lo.reshape(-1), code, X.reshape(-1)[lane])
        np.maximum.at(self.hi.reshape(-1), code, X.reshape(-1)[lane])


def _round_div4(value):
    """Nearest integer, ties away from zero, with unbounded Python integers."""
    return (value + 2) // 4 if value >= 0 else -((-value + 2) // 4)


def _run_forecasts(mask, query_rows):
    """Track separate runs down rows; forecast each row before reading that row.

    State per track is (left, right, left_velocity, right_velocity, gap, hits)
    in quarter-pixels. Alpha=3/4, beta=1/4 is a fixed local linear filter.
    One-to-one nearest-endpoint association preserves run identities, including
    across HPAC slots. A four-row absence is an explicit dash/occlusion carry;
    it does not become a long polynomial bridge through an absent lane.
    """
    starts = mask & ~np.pad(mask[:, :-1], ((0, 0), (1, 0)))
    ends = mask & ~np.pad(mask[:, 1:], ((0, 0), (0, 1)))
    sy, sx = np.nonzero(starts)
    ey, ex = np.nonzero(ends)
    if not np.array_equal(sy, ey):
        raise ValueError("unpaired horizontal runs")
    offsets = np.searchsorted(sy, np.arange(H + 1))
    query_row_set = {int(y) for y in query_rows}
    tracks = []
    forecasts = {}
    for y in range(int(query_rows[-1]) + 1):
        # Propagation uses only the track state from strictly above this row.
        predicted = [(left + vl, right + vr, vl, vr, gap, hits) for left, right, vl, vr, gap, hits in tracks]
        if y in query_row_set:
            forecasts[y] = [
                (left, right) for left, right, _, _, gap, hits in predicted if hits >= 2 and gap <= 4 and left <= right
            ]
        observations = [
            (4 * int(left), 4 * int(right))
            for left, right in zip(sx[offsets[y] : offsets[y + 1]], ex[offsets[y] : offsets[y + 1]], strict=True)
        ]
        candidates = []
        for j, (left, right) in enumerate(observations):
            for i, (pl, pr, _, _, gap, _) in enumerate(predicted):
                # Endpoint gate: 4px + at most 4px extra for a carried dash gap.
                gate = 16 + 4 * gap
                if abs(left - pl) <= gate and abs(right - pr) <= gate:
                    candidates.append((abs(left - pl) + abs(right - pr), i, j))
        matched_tracks, matched_runs = set(), set()
        updated = []
        for _, i, j in sorted(candidates):
            if i in matched_tracks or j in matched_runs:
                continue
            matched_tracks.add(i)
            matched_runs.add(j)
            pl, pr, vl, vr, _, hits = predicted[i]
            left, right = observations[j]
            el, er = left - pl, right - pr
            updated.append(
                (
                    pl + _round_div4(3 * el),
                    pr + _round_div4(3 * er),
                    max(-8, min(8, vl + _round_div4(el))),
                    max(-8, min(8, vr + _round_div4(er))),
                    0,
                    min(hits + 1, 16),
                )
            )
        for i, (left, right, vl, vr, gap, hits) in enumerate(predicted):
            if i not in matched_tracks and gap < 4 and hits >= 2 and left <= right:
                updated.append((left, right, vl, vr, gap + 1, hits))
        for j, (left, right) in enumerate(observations):
            if j not in matched_runs:
                updated.append((left, right, 0, 0, 0, 1))
        tracks = updated
    return forecasts


class RunTrackingGeometry(_GroupPrefix):
    """TC3 coherent endpoint tracks using only the strict HPAC group prefix.

    Unknown current pixels fall back to the previously decoded plane, or to
    non-Lane for the first frame. Above-row current symbols overwrite that prior.
    The frame-local tracking pass is deterministic and recomputed as the legal
    prefix grows, so no current or future group labels can enter a forecast.
    """

    def contexts(self, positions):
        positions = self._begin(positions)
        yy, xx = positions // W, positions % W
        band = (yy >= 128) & (yy < 320)
        result = np.full(len(positions), 8, dtype=np.uint8)
        if not band.any():
            return result
        prior = False if self.previous is None else self.previous == 1
        mask = np.where(self.plane < K, self.plane == 1, prior)
        rows = np.unique(yy[band])
        forecasts = _run_forecasts(mask, rows)
        count = max((len(value) for value in forecasts.values()), default=0)
        if count == 0:
            result[band] = 7
            return result
        left = np.full((H, count), 1 << 20, dtype=np.int64)
        right = np.full((H, count), 1 << 20, dtype=np.int64)
        valid = np.zeros((H, count), dtype=bool)
        for y, intervals in forecasts.items():
            if intervals:
                left[y, : len(intervals)], right[y, : len(intervals)] = np.asarray(intervals).T
                valid[y, : len(intervals)] = True
        # Interface endpoints are both the Lane boundary pixel and its adjacent
        # non-Lane pixel, matching the two-sided TC2 oracle's edge convention.
        bx, by = xx[band, None] * 4, yy[band]
        distance = np.minimum.reduce(
            [np.abs(bx - left[by]), np.abs(bx - left[by] + 4), np.abs(bx - right[by]), np.abs(bx - right[by] - 4)]
        )
        distance = np.min(np.where(valid[by], distance, 1 << 20), axis=1)
        # Quarter-pixel unsigned distances: nearest integer, exact half rounds up.
        bins = np.searchsorted([0, 1, 2, 4, 8, 16], (distance + 2) // 4, side="left").astype(np.uint8)
        bins[~valid[by].any(axis=1)] = 7
        result[band] = bins
        return result

    def observe(self, positions, symbols):
        self._observe(positions, symbols)
