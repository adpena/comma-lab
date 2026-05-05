# Lane PSD-LumaSkip — Phase A council APPROVE-FOR-SCAFFOLD; scaffold landed (2026-04-30)

**Status**: SCAFFOLD LANDED. GPU dispatch DEFERRED pending separate council per Council #271 reactivation criterion #1.

**Council Phase A vote**: 10 APPROVE-FOR-SCAFFOLD / 0 REJECT / 0 ABSTAIN (unanimous; conservative-bias check PASSED, dispatch-scope check BOUNDED).

**Authority**: `.omx/research/council_lane_psd_lumaskip_design_20260430.md`

---

## Why this lane exists

Council #271 unanimously KILLED vanilla PSD dispatch on 2026-04-30 (`council_lane_7_psd_dispatch_review_20260430.md`). Reactivation criterion #1 was: "PoseNet-aware luma-skip variant designed AND separate council approves." This Phase A deliverable satisfies #1 partially: the design is approved for SCAFFOLD landing only; dispatch requires a separate council convened with empirical predicted-band evidence.

## Mechanism

PSD's PixelUnshuffle/PixelShuffle bottleneck destroys high-frequency luma content that FastViT-PoseNet's attention layers attend to via the YUV6 polyphase decomposition (Y → 4 polyphase planes y00/y10/y01/y11 at half res, exact reconstruction). The luma-skip path keeps a full-resolution luma residual that bypasses the bottleneck entirely. Chroma (which is already half-res inside YUV6 anyway) still flows through PSD's bottleneck, preserving PSD's 12.8% historical SegNet advantage.

**Score arithmetic floor estimate** (Phase A memo §F5):
- Optimistic (luma-skip recovers >80% FastViT signal, 25% prob): floor 1.024 (Pareto-dominates Lane G v3)
- Central (~50% recovery, 45% prob): floor 1.05-1.20
- Pessimistic (dual-path destabilizes, 30% prob): floor 1.20-1.40

**Predicted band [contest-CUDA] [prediction]: [0.95, 1.40]**, central ~1.10.

## Scaffold landed (Phase B output)

| File | Purpose | Status |
|---|---|---|
| `src/tac/psd_lumaskip_renderer.py` | `PSDLumaSkipPostFilter` class (~90K params at h=64, luma_hidden=16) | NEW |
| `src/tac/profiles.py` | `PSD_LUMASKIP_LANE_G_V3` profile + PROFILES registry entry `psd_lumaskip_lane_g_v3` | EDITED |
| `src/tac/architectures.py` | `VARIANTS["psd_lumaskip"]` registration via lazy import | EDITED |
| `src/tac/tests/test_psd_lumaskip_renderer.py` | 15 tests covering shapes, value range, identity-at-init, dual-path gradients, EMA, registry wiring, robustness | NEW (15/15 PASS) |

**Architecture details**:
- Chroma path: identical to PSDPostFilter (PixelUnshuffle(2) → conv1(12→64) → conv2(64→64 dilated=2) → conv3(64→64) → conv4(64→12 zero-init) → PixelShuffle(2))
- Luma path: rec.601 luma extraction → conv(1→16) → conv(16→16 dilated=2) → conv(16→1 zero-init) → learned 1×1 projection (1→3, broadcast-prior init)
- Both output convs zero-init → starts as identity
- Param breakdown at h=64, luma_hidden=16: ~90K total (chroma path ~85K, luma path ~3K, projection ~6 params)

**Council recommendations baked into scaffold**:
- Yousfi: learned 1×1 luma projection (not naive `.expand`) → broadcast-prior init for symmetric R/G/B gradient flow at start
- Fridrich: `luma_hidden=16` default (NOT 8 — 8 under-allocates capacity to luma)
- Contrarian: predicted band widened to [0.95, 1.40]; dispatch council must monitor dual-path gradient stability

## Tests verified (15/15 PASS)

`test_psd_lumaskip_renderer.py`:
1-2. Forward shape preserved at SegNet input (384×512) and camera_size (874×1164)
3. Output clamped to [0, 255] after training perturbation
4. Identity at init (zero residuals from zero-init conv4 + luma_out)
5. Luma path is full-resolution (architectural invariant)
6. Chroma path operates at half-res (PSD bottleneck preserved)
7. Learned 1×1 projection correctly wired with broadcast-prior init; `use_learned_luma_projection=False` path uses `.expand`
8. Backward gives nonzero gradients on BOTH paths (luma_in/luma_mid/luma_out + conv1-4) — dual-path stability sanity
9. Param count at default config falls in [90K, 105K] target band; luma path ≤5K, chroma path ≥80K
10. EMA snapshot+update+apply+restore round-trips cleanly
11. VARIANTS registry wires `"psd_lumaskip"` correctly
12. PROFILES registry wires `"psd_lumaskip_lane_g_v3"` with all expected keys
13-15. Robustness: rejects odd dimensions (H=63), wrong channel count (4 instead of 3), invalid ctor args (hidden=0, luma_hidden=0, kernel=2)

Pre-existing hardening test failures (lane_19_logit_margin, j_jbl, q_faithful) are UNRELATED — our profile passes both `test_profile_creates_valid_config` AND `test_profile_architecture_buildable`.

## NOT done in this scaffold (Phase C+ deferred)

- Optional STRICT preflight Check 93+ `check_psd_lumaskip_preserves_luma_resolution` (AST scan that asserts no downscaling primitive on luma path) — design-level recommendation; not landed in this Phase A/B because it requires preflight infrastructure changes (warn-only first, then strict-flip after audit). This is a follow-up task.
- `scripts/remote_lane_psd_lumaskip.sh` dispatch script — INTENTIONALLY NOT CREATED. Per Council #271 reactivation criterion #1, dispatch infrastructure should NOT land until separate council approves GPU run.
- Lane registry entry via `python tools/lane_maturity.py add-lane lane_psd_lumaskip --name "PSD-LumaSkip variant" --phase 1` — pending; will be done in commit follow-up if registry tool tolerates the call without git-state mutation.

## Reactivation status

- Council #271 reactivation criterion #1: **PARTIALLY SATISFIED** (design done + council approves SCAFFOLD; dispatch council still required)
- Council #271 reactivation criterion #2 (current floor < 0.50): NOT MET (Lane G v3 = 1.05)
- Council #271 reactivation criterion #3 (Lane 19 logit-margin demonstrates SegNet improvements transfer architecture-agnostically): IN PROGRESS (Lane 19 design landed 2026-04-30; empirical run pending)

## Cross-references

- `.omx/research/council_lane_psd_lumaskip_design_20260430.md` — Phase A design memo (load-bearing source)
- `.omx/research/council_lane_7_psd_dispatch_review_20260430.md` — Council #271 KILL of vanilla PSD
- `.omx/research/lane_7_psd_kill_memo_20260430.md` — Formal kill memo with reactivation criteria
- `src/tac/psd_lumaskip_renderer.py` — module
- `src/tac/profiles.py:PSD_LUMASKIP_LANE_G_V3` — profile
- `src/tac/architectures.py:VARIANTS["psd_lumaskip"]` — variant registration
- `src/tac/tests/test_psd_lumaskip_renderer.py` — 15 tests

## Next steps (for separate dispatch council to consider)

1. Local CPU smoke test on synthetic 8-frame batch (~30s) for forward/backward shape correctness — already covered by Test 1, 2, 8; confirms via Phase B
2. Add the recommended STRICT preflight check (`check_psd_lumaskip_preserves_luma_resolution`)
3. Create `scripts/remote_lane_psd_lumaskip.sh` ONLY after dispatch council approval
4. Dispatch council MUST consider both standalone (predicted EV negative per Hotz) AND stack-composition (with Lane 19 logit-margin as primary partner per Quantizr)
5. Kill criterion if dispatched: proxy auth at epoch 200 with pose > 0.008 → kill; or proxy total > 1.20 at epoch 1000 → kill
