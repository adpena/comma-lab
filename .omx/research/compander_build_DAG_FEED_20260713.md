# Standalone DAG FEED — inverse-depth Riemannian compander build

- Date: 2026-07-13
- Lane/checkpoint: `compander_build`
- Node: `FEED-COMPANDER-BUILD-20260713`
- Status: `BUILT_NEVER_FIRED__A_B_OWED`
- Authority: `$0` local build plus cached-artifact analysis; no launch
- Pointer delta: `NONE`

## Parent edges consumed

```text
FEED-manifold-S1
  -> MEASURED all-class n600 row density on v>174
  -> fixed softened-inverse-depth delta = 32.5257801441824 rows
  -> JS 0.06994264689610602 vs log-depth 0.16044044809089875 vs uniform 0.24818817978749463
  -> PROCEED-with-gate only after ground-class-pair ledger + counted receiver-close A/B

GroundFrameChart #194 / perspective-aware chart #185
  -> existing projective xi-homography chart
  -> compander must compose after it, not replace it

flip_density_chart_metric_v1
  -> sqrt(g_vv) proportional to measured flip density w_flip(v)
  -> canonical equation reused; no duplicate law registered
```

## Build edge

```text
GroundFrameChart.coords_for_pair(frame coords)
  -> projective ground coordinates
  -> C(v) = normalized integral of (v-v_h+delta)^-2
  -> margin-companded ground coordinates
  -> unchanged curvelet feature width/cache shape

DSL MarginCompandedGroundChart (default OFF)
  -> --ground-frame-chart
  -> --margin-companded-ground-chart
  -> exact measured v_h/delta + deterministic seed identity
  -> trainer wraps the already-built projective chart
  -> resume registry name margin_compander / prefix __mcc_
  -> additive config/divergence custody
```

The coordinate transform has deterministic NumPy-fp32 forward/inverse references and an MLX twin
whose CPU bit-parity assertion is committed in the focused test. The managed build session could not
initialize any MLX device (`No Metal device available`), so the MLX assertion is environment-blocked,
not reported as measured parity. No linalg was added; the existing projective chart keeps its pinned
CPU construction.

## Ground-class-pair ledger edge

```text
GT cache lstars + six cached witness-argmax chunks
  -> SHA-256 custody for all seven files
  -> exactly 600 unique pair indices, gap-free 0..599
  -> 20 directed + 10 undirected class-pair row-density records
  -> strict planar-ground definition = Road<->Lane only
```

`MEASURED [cached-argmax local-CPU analysis], n600`: total cached source-to-witness flips are
`785058` (`0.006655019124348958` of pixels). Road<->Lane accounts for `447620` flips, or
`0.5701744329718313` of all flips. On that undirected pair and `v>174`, the fixed S1 profile has JS
`0.13905534495630045`, versus `0.25990849245359443` for unshifted log-depth and
`0.23730967324348934` for uniform. The fixed S1 constant was not refit. This is mechanism evidence
only: cached flips do not measure the trained chart arm, receiver parse-back, PoseNet, or archive
rate.

## Remaining promotion edge

```text
governed future n600 control/treatment launch
  -> same seed / exact optimizer steps / total archive bytes / authority axis
  -> treatment factory = MarginCompandedGroundChart
  -> fitted chart payload has bytes + SHA and is counted inside treatment archive
  -> archive parse-back passes and decoded SHA == pre-archive reference SHA
  -> per-class d_seg for Road/Lane/Undrivable/Movable/MyCar
  -> PRIMARY effect = Lane d_seg(treatment-control)
  -> d_pose must not worsen; rate is exactly matched
  -> only then may promotion be decided
```

`tools/probe_compander_receiver_close_ab.py` is the pure receipt comparator for this edge. It was
built and synthetic-tested but not run on real arms. It issues no score or promotion claim.

## Triality and consumers

- **DAG:** this standalone node; no hot shared DAG was edited.
- **Equation:** reuses `flip_density_chart_metric_v1` in
  `src/tac/canonical_equations/manifold_geometry_slots_20260713.py`.
- **DSL:** `MarginCompandedGroundChart`, visible through the canonical lever registry and never-fired
  activation surface; pool state is `built-never-fired`.
- **Sensitivity:** the class-pair row ledger supplies a spatial prior only. It is not an empirical
  chart-arm gradient or score sensitivity; the matched A/B remains the required effect update.
- **Pareto:** admission is Lane/all-class d_seg improvement with non-worsening d_pose at exactly
  matched bytes and steps.
- **Bit allocator:** the build changes where fixed capacity lands, not how much exists. Any fitted
  chart payload is counted; no byte benefit is claimed.
- **Cathedral/autopilot:** the pool duty queue holds the lever for the next governed treatment arm;
  no dispatch hook or ticket was mutated.
- **Continual learning:** the custody-complete ledger and pool transition prevent re-deriving the S1
  constant or forgetting the owed receiver A/B.
- **Probe disambiguator:** the future receipt comparator arbitrates chart-arm effect versus the
  cached-density mechanism proxy.

No trainer, scorer, evaluator, archive, provider, GPU, live run, launch ticket, memory-envelope
config, or frontier pointer was mutated. Feature/cache tensor shapes are unchanged, so the sibling
launch-ticket recompile needs no projected-memory adjustment for this lever. All files remain
uncommitted for main review.
