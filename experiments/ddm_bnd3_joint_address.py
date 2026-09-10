"""Research-only exact path-cover assignment for real boundary crack candidates.

The MILP minimizes the byte length of a SPECIFIED uncompressed address wire:
4 bytes/frame count, 8 bytes/segment start+length, one byte/visited edge.
It does not optimize Brotli length or establish an entropy lower bound.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array

from experiments import ddm_bnd2_segment_encode as old


def candidates(plane, misses, orientation):
    """Every allowed (class pair, normal, directed unit crack) per miss."""
    nodes = []
    for pos in misses.tolist():
        y, x = divmod(pos, old.W)
        symbol = int(plane[y, x])
        for normal in (0, -1, 1):
            for direction, (ny, nx) in enumerate(old.RIGHT):
                ry, rx = y - normal * ny, x - normal * nx
                ly, lx = ry - ny, rx - nx
                if not (0 <= ry < old.H and 0 <= rx < old.W and 0 <= ly < old.H and 0 <= lx < old.W):
                    continue
                a, b = int(plane[ry, rx]), int(plane[ly, lx])
                if a == b or (a < b) != (orientation == 0) or (b if normal == -1 else a) != symbol:
                    continue
                sy, sx = old.START_FROM_RIGHT_CELL[direction]
                vertex = (ry + sy) * (old.W + 1) + rx + sx
                nodes.append(((a, b, normal), vertex, direction, pos))
    return nodes


def end_vertex(vertex, direction):
    dy, dx = old.DIRECTIONS[direction]
    return vertex + dy * (old.W + 1) + dx


def valid_bridge(plane, best, key, vertex, direction):
    """A skipped edge must address a correctly predicted cell on a true crack."""
    y, x = divmod(vertex, old.W + 1)
    dy, dx = old.DIRECTIONS[direction]
    ey, ex = y + dy, x + dx
    if not (0 <= y <= old.H and 0 <= x <= old.W and 0 <= ey <= old.H and 0 <= ex <= old.W):
        return False
    a, b, normal = key
    ry, rx = old.sample_edge(y, x, direction, 0)
    ly, lx = old.sample_edge(y, x, direction, -1)
    py, px = old.sample_edge(y, x, direction, normal)
    if not all(0 <= yy < old.H and 0 <= xx < old.W for yy, xx in ((ry, rx), (ly, lx), (py, px))):
        return False
    return (int(plane[ry, rx]), int(plane[ly, lx])) == (a, b) and int(plane[py, px]) == int(best[py * old.W + px])


def links_for(nodes, plane, best, gap):
    """All shortest legal links through zero to gap correctly predicted cells."""
    starts = defaultdict(list)
    for j, (key, vertex, _direction, _pos) in enumerate(nodes):
        starts[key, vertex].append(j)
    links = []
    for i, (key, vertex, direction, pos) in enumerate(nodes):
        queue = [(end_vertex(vertex, direction), ())]
        seen = {queue[0][0]}
        targets = {}
        for at, path in queue:
            for j in starts[key, at]:
                if nodes[j][3] != pos and j not in targets:
                    targets[j] = path
            if len(path) == gap:
                continue
            for d in range(4):
                to = end_vertex(at, d)
                if to not in seen and valid_bridge(plane, best, key, at, d):
                    seen.add(to)
                    queue.append((to, (*path, d)))
        links.extend((i, j, path) for j, path in sorted(targets.items()))
    return links


def joint_segments(plane, best, misses, orientation, gap):
    """Exact global minimum of the raw address wire, with explicit cycle cuts.

    Frames are independent in this wire, so their certified minima sum to the
    global n600 raw-wire minimum. No greedy assignment is substituted. HiGHS
    must prove zero integer gap; a failed solve is a blocker, never a result.
    """
    nodes = candidates(plane, misses, orientation)
    links = links_for(nodes, plane, best, gap)
    by_pos, incoming, outgoing = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, node in enumerate(nodes):
        by_pos[node[3]].append(i)
    for j, (a, b, _path) in enumerate(links):
        outgoing[a].append(len(nodes) + j)
        incoming[b].append(len(nodes) + j)
    entries, lower, upper = [], [], []

    def constraint(values, lo, hi):
        row = len(lower)
        entries.extend((row, col, value) for col, value in values)
        lower.append(lo)
        upper.append(hi)

    for indices in by_pos.values():
        constraint([(i, 1) for i in indices], 1, 1)
    for i in range(len(nodes)):
        for attached in (incoming[i], outgoing[i]):
            constraint([(i, -1)] + [(j, 1) for j in attached], -np.inf, 0)
    cost = np.array([0.0] * len(nodes) + [len(p) - 8.0 for _, _, p in links])
    cuts = 0
    if not nodes:
        return [], {"eligible": 0, "raw_address_bytes": 4, "segments": 0, "links": 0, "cycle_cuts": 0}
    while True:
        rows, cols, values = zip(*entries, strict=True)
        matrix = coo_array(
            (values, (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))), shape=(len(lower), len(cost))
        ).tocsc()
        result = milp(
            cost,
            integrality=np.ones(len(cost)),
            bounds=Bounds(0, 1),
            constraints=LinearConstraint(matrix, lower, upper),
            options={"mip_rel_gap": 0.0, "presolve": True, "threads": 1, "random_seed": 20260910},
        )
        if not result.success or result.mip_gap != 0 or np.max(np.abs(result.x - np.round(result.x))) > 1e-6:
            raise RuntimeError(f"joint address minimum unproved: {result.message}")
        selected = set(np.flatnonzero(result.x[: len(nodes)] > 0.5).tolist())
        chosen = [j for j in range(len(links)) if result.x[len(nodes) + j] > 0.5]
        successor = {links[j][0]: (links[j][1], j) for j in chosen}
        visited, cycles = set(), []
        for start in sorted(selected):
            at, walk, index = start, [], {}
            while at not in visited and at not in index:
                index[at] = len(walk)
                walk.append(at)
                if at not in successor:
                    break
                at = successor[at][0]
            if at in index and at in successor:
                cycles.append([successor[v][1] for v in walk[index[at] :]])
            visited.update(walk)
        if not cycles:
            break
        for cycle in cycles:
            constraint([(len(nodes) + j, 1) for j in cycle], -np.inf, len(cycle) - 1)
            cuts += 1
    destinations = {links[j][1] for j in chosen}
    segments = []
    for start in sorted(selected - destinations):
        key, vertex, _direction, _pos = nodes[start]
        at, directions, positions, skip = start, [], [], []
        while True:
            directions.append(nodes[at][2])
            positions.append(nodes[at][3])
            skip.append(False)
            if at not in successor:
                break
            to, j = successor[at]
            for d in links[j][2]:
                directions.append(d)
                skip.append(True)
            at = to
        segments.append((key, vertex, directions, positions, skip))
    recovered = [p for s in segments for p in s[3]]
    if sorted(recovered) != sorted(by_pos) or len(recovered) != len(set(recovered)):
        raise ValueError("joint assignment lost or duplicated a miss")
    raw = 4 + sum(8 + len(s[2]) for s in segments)
    expected = 4 + 9 * len(by_pos) + round(result.fun)
    if raw != expected:
        raise ValueError("MILP objective does not equal actual raw address bytes")
    return segments, {
        "eligible": len(by_pos),
        "raw_address_bytes": raw,
        "segments": len(segments),
        "links": len(links),
        "cycle_cuts": cuts,
        "mip_gap": float(result.mip_gap),
        "objective": float(result.fun),
        "gap": gap,
        "orientation": orientation,
    }
