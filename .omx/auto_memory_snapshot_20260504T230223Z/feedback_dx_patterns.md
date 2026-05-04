---
name: DX patterns — canonicalization lessons from 54-commit session
description: Recurring bug patterns and their fixes. ArchConfig single source of truth. Model self-description. No mutable singletons. Separate data paths for different purposes. Precompute GT once.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## Bug Patterns (each caused at least one wasted run)

1. **Default override**: function default ≠ CLI default ≠ profile value.
   Fix: ArchConfig as single source of truth. All configs reference it.

2. **Stale documentation**: comment says (B,2,3,H,W), code produces (B,2,H,W,3).
   Fix: assert shapes at format boundaries. Trust code, not comments.

3. **Overloaded data path**: one mask file for both training (1200) and archive (600).
   Fix: separate cfg.masks (full) and cfg.masks_archive (half). Different purposes = different paths.

4. **Mutable singleton**: _make_zoom_flow cached on function attribute.
   Fix: create once in main(), pass explicitly. No global state.

5. **Dead config**: field stored but never consumed (error_boost_phase3, use_boundary_hinge, focal_gamma with wrong loss_mode).
   Fix: validate at construction (__post_init__), raise on incompatible combinations.

6. **Implicit dependencies**: script assumes ffmpeg, av, pydantic exist.
   Fix: explicit apt-get + pip install in deployment script.

7. **Redundant computation**: optimize_poses.py recomputes GT masks + pose targets from video every run (~30 min).
   Fix: accept --precomputed-gt-masks and --precomputed-pose-targets flags. Cache once, reuse.

8. **Proxy mismatch**: curriculum used PoseNet-only difficulty instead of full score formula.
   Fix: always use the contest formula (100*seg + sqrt(10*pose)) for any difficulty/priority metric.

## Canonical Abstractions (implemented 2026-04-25)

- **ArchConfig** (renderer.py): single source of truth for architecture params
- **arch_dict()** (AsymmetricPairGenerator): model self-describes for serialization
- **_compute_ego_flow**: pure function, no side effects, explicit lifecycle
- **step_extract_masks → (full, half)**: separate paths for training vs archive
- **DistillConfig.__post_init__**: validates range and compatibility at construction

## How to Apply

Before writing any new training/experiment script:
1. Import ArchConfig for default architecture params
2. Use model.arch_dict() for checkpoint metadata, not external config reconstruction
3. Never cache on function attributes — create in main(), pass through
4. If a data file serves two purposes, make two paths
5. Every CLI arg must map to a config field, every config field must have a CLI arg
6. Every deployment script must explicitly install ALL dependencies
7. Precompute expensive GT operations once, save to disk, reuse
