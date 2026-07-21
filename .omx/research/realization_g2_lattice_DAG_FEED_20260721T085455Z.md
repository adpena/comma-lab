---
schema: realization_g2_lattice_dag_feed.v1
task_id: "578"
lane_id: lane_realization_g2_lattice_578_20260721
research_only: true
status: BLOCKED_INPUT_DOMAIN_LABEL_FIELD_IS_NOT_RGB_PLANE
---

# DAG FEED: realization G2 lattice

## Measured current path

```text
seed_compose_b2 bytes
  -> scorer-free seed parser
  -> predict_cell_field: HxW uint8 class IDs
  -> plane/cache cell and pose-tube replay
  -> BLOCKER: no decoder-derived or counted HxWx3 uint8 projected RGB plane
```

At n16/n64/n600 this path yielded 16/64/600 label fields, zero RGB projections, and zero valid lattice attempts. The blocker is therefore before the factor-2 lattice, not a failed lattice solve.

## Landed strict continuation

```text
projected cells: HxW uint8
        +
projected RGB plane: HxWx3 uint8
        +
hash-bound custody
  [source kind, generator, seed hash, RGB hash, cell hash,
   added seed bytes, decoder scorer invocations == 0]
        |
        v
realize_projected_rgb_plane_camera_uint8
        |
        +--> #547 exact uint8 factor-2 lattice solver
        +--> #580 full resize-kernel operator
        +--> exact integer parse-back verification
        |
        v
camera Hc x Wc x 3 uint8
        |
        +--> hard CPU-torch argmax oracle          [OWED]
        +--> realized-frame PoseNet tube replay    [OWED]
        +--> time and additional-byte accounting   [OWED]
        |
        v
real n16 -> n64 -> n600 anchor
        |
        +--> canonical equation registration       [ONLY AFTER D1]
```

## Triality delta

- DSL/code: the receiver now has a typed RGB-plane custody boundary and exact realization callable; label fields cannot impersonate RGB.
- DAG: the missing label-to-RGB generator/payload edge is explicit and fail-closed.
- Equations: no equation row is added because no real n600 composed RGB anchor exists. M2 remains a separate counted-RGB existence proof.

The pointer remains unchanged and this feed carries no score or promotion authority.
