# Frame 5 — Causal / constructor-theoretic (alien-tech ledger 2026-05-13)

**Parent memo**: `.omx/research/alien_technology_unknown_unknowns_research_20260513.md` §5.
**Lane**: `lane_alien_technology_unknown_unknowns_research_20260513` (L0).

## Worldview

Pearl + Deutsch as foundational. Compression encodes **causes**, not effects.

## Core inductive bias

**Same SCM → same compression.** Encode the generative process; the artifact reproduces deterministically.

## Concrete technique 5A — Pearl do-calculus pose-driven encoding

SCM for driving video:
```
ego_pose(t), scene_static, lighting(t), camera_intrinsics → frame(t)

Generative function: render(ego, scene, light, cam) → YUV6
```

Archive stores:
- Pose trajectory (1200 × 6 floats, codec-compressed): ~5 KB
- Scene mesh (2D ground plane + sparse 3D obstacles): ~5 KB
- Lighting model (4-6 spherical harmonics + per-frame intensity): ~1 KB
- Camera intrinsics: ~50 bytes (one constant)

**Total: ~11 KB.** Inflate-time runs deterministic shader → YUV6 frames.

**Catch**: shader fidelity is the limiting factor. A 5% modeling error in scene geometry → ~0.05 score regression. Needs **NEURAL RESIDUAL** to close the gap (~30 KB).

**Net byte budget**: ~50 KB for causal scaffold + neural residual. Competitive with HNeRV.

## Concrete technique 5B — Constructor-theoretic archive (Deutsch-Marletto)

[Constructor Theory of Information](https://royalsocietypublishing.org/rspa/article/471/2174/20140540/100308/Constructor-theory-of-informationConstructor) (Deutsch-Marletto 2014).

Archive = DAG of (sub-task, executor) pairs.

**Example primitive library:**
- `generate_road_segment(type_K, length, curvature)` → road segment
- `apply_ego_motion(pose_p, scene)` → updated scene
- `render_vehicle(class_C, position_q, rotation_r)` → vehicle pixels
- `render_lane_marker(...)`, `render_horizon_band(...)`, `render_sky_gradient(...)`
- `composite(layer1, layer2, mask)`

Archive structure:
```
HEADER: magic "CTAA" + primitive_count + total_dag_size
PRIMITIVE_TABLE: [(primitive_id, executor_name, parameter_count)] × N
DAG_NODES: [(node_id, primitive_id, parameter_bytes, dependencies)] × M
PARAMS: concatenated parameter bytes
```

Estimated size: ~5 KB. **Reviewable in 30 seconds** — every primitive is named. Audit-friendly.

**MAJOR ADVANTAGE**: each construction task carries a **construction certificate**. The contest scorer is just one consumer of this DAG; the SAME DAG is reusable for other tasks (drive policy training, simulation, etc.). Substrate-independent.

## Concrete technique 5C — Causal-discovery across many videos

[PC algorithm](https://en.wikipedia.org/wiki/Spirtes%E2%80%93Glymour%E2%80%93Scheines_algorithm), [LiNGAM](https://www.cs.helsinki.fi/group/neuroinf/lingam/).

If we had 100 driving videos, discover the **shared SCM** structure. Encode shared SCM ONCE; per-video, encode only deviations.

**Single-video relevance**: for the contest, we have ONE video. **BUT** there's an implicit shared SCM with all driving videos in our training set. Tap that shared prior.

## Concrete technique 5D — Mereotopological description

[Casati-Varzi 1999 — Parts and Places](https://mitpress.mit.edu/9780262513784/parts-and-places/).

Mereotopological formal logic of part-of / connected-to / surrounded-by.

Archive structure:
```
SCENE_DESCRIPTION:
  - sky IS-CONNECTED-TO horizon_band
  - road IS-A-PART-OF ground_plane
  - lane_marker_i IS-CONTAINED-IN road
  - vehicle_j IS-SURROUNDED-BY road on (left, right)
  - ...
```

Encoded as a few hundred Horn clauses → ~1-2 KB.

**Catch**: pixel synthesis from logical predicates is a hard inverse problem. Needs a **mereotopological renderer** (which itself is a neural network → bytes spent there).

**Net**: ~30-50 KB archive, similar to other paths. Novelty: provable spatial-consistency guarantees.

## Closest extant work

- [DoWhy (Microsoft causal inference)](https://github.com/py-why/dowhy)
- [Pearl 2009 — Causality 2nd ed](http://bayes.cs.ucla.edu/BOOK-2K/)
- [Deutsch-Marletto — Constructor theory of information](https://arxiv.org/abs/1405.5563)
- [arXiv:2502.04210 — Algorithmic causal structure through compression](https://arxiv.org/pdf/2502.04210)

## SHOCK-AND-AWE recommendation

**Constructor-DAG archive (Frame 5B)** is the most **AUDIT-FRIENDLY** alien-tech idea. Recommended dispatch: $1 design memo + $5 substrate engineering.

## Wire-in declaration

All 6 hooks: N/A. If operator approves Decision C (constructor-DAG archive grammar design memo), the implementation subagent will register hooks 1-4.

## Research-only tag

`research_only=true`.
