# Capstone HiNeRV decoder upgrade: grid-PE (opt-in, default-off) — landed 2026-06-11

**Subagent:** `capstone_hinerv_skip_gridpe_20260611` (resumed a predecessor checkpoint of the same id;
crash-resume protocol). **Task:** add the HiNeRV-class decoder improvements (bilinear-skip + grid
positional-encoding — the ~72.3% BD-rate-over-HNeRV lever) to the capstone VQ-NeRV decoder as an OPT-IN
config, default-OFF (byte-identical when off), numpy-portable, parity-tested. This is the established
"missing lever" from the C1′ carrier verdict + the lever-B → lever-C pivot.

**Authority:** all results are `[macOS-MLX research-signal]` / `[macOS-CPU advisory]` — frozen proto +
real-EfficientNet DistortionNet on CPU, MLX-GPU render. NON-PROMOTABLE per the GOAL authority ladder.
`$0` spend, NO GPU training, NO Modal/paid dispatch, NO MPS. `promotable=false`, `score_claim=false`,
`ready_for_exact_eval_dispatch=false`. Frontier read from pointer: **0.19109982 [contest-CPU], 177,169 B,
sha `b46897267…`** (UNMOVED — this is a decoder-capability upgrade, not a score row). The running daemon
(`experiments/results/capstone_c1prime_honest_b20_n48`) was NOT touched (default-off keeps it unchanged).

---

## 0. HEADLINE

The HiNeRV upgrade is LANDED as a default-off opt-in on the EXISTING capstone decoder (no parallel
decoder built — the predecessor + this session extended the live `CapstoneVqNervBundle`). The audit found
the **bilinear-skip + PixelShuffle + sin upsample blocks are ALREADY structurally present** in every
`HNeRVDecoderMLX` block, so the genuinely-missing HiNeRV delta is the **multi-resolution grid
positional-encoding (grid-PE)** fed to the stem. That is what is now wired — in BOTH the MLX forward AND
the pure-numpy inflate, op-for-op, with a **real NO-FAKE bug fixed** (a weight-key mismatch that silently
dropped grid-PE in the numpy path → 34-unit render divergence under a trained projection). 7 new grid-PE
tests + the full 14-test parity suite + the 76-test capstone suite all pass.

---

## 1. AUDIT — what the capstone decoder ALREADY had vs. what HiNeRV adds (Step 1)

The HiNeRV BD-rate-over-HNeRV lever has two structural halves: (a) per-stage **bilinear-skip**
connections in each upsample block, and (b) a multi-resolution **grid/coordinate positional encoding** at
the decoder input. Audited against `src/tac/local_acceleration/pr95_hnerv_mlx.py` (the shared PR95 decoder
backbone the capstone composes over):

| HiNeRV/HNeRV mechanism | already present? | evidence (file:line) |
|---|---|---|
| **Per-stage bilinear-skip identity** | ✅ ALREADY PRESENT | `bilinear_resize2x_align_corners_false_nhwc` (`pr95_hnerv_mlx.py:1008`); each block: `identity = bilinear_2x(x)` + 1×1 `skip_conv` when channels change (`blocks.{i}.skip_conv`, lines 91/108/116/896) |
| **PixelShuffle(2) upsample** | ✅ ALREADY PRESENT | `pixel_shuffle_2x_nhwc` (`pr95_hnerv_mlx.py:937`); `decoded = pixel_shuffle_2x(conv3x3(x))` per block |
| **sin (NeRF-style) activation** | ✅ ALREADY PRESENT | `x = sin(decoded + identity)` per block; `feat = x + 0.1·sin(refine1(refine0(x)))` (`features_nhwc`, `pr95_hnerv_mlx.py:1420`) |
| **Grid / coordinate positional encoding** | ❌ MISSING (the HiNeRV delta) | the stem is `latent → Linear → reshape → sin`; NO coordinate grid fed to the decoder. This is the spatial inductive bias HiNeRV adds over HNeRV. |
| **Hierarchical-skip structure** | ✅ (the bilinear-skip + refine residual IS the hierarchical skip) | per-block `sin(decoded + identity)` + the `+0.1·sin(refined)` residual = the coarse→fine skip chain |

**Conclusion of the audit:** the only genuinely-missing HiNeRV mechanism is the **grid positional
encoding**. The bilinear-skip half is structural in every block already (so we do NOT rebuild it — doing so
would duplicate working code, an explicit anti-pattern). Therefore the build is exactly the grid-PE.

---

## 2. WHAT WAS ADDED (Step 2) — grid-PE, opt-in, default-off, numpy-portable

Implemented in BOTH the MLX forward and the numpy reference + contest inflate so the portability contract
holds (`MLX fast path → numpy reference → torch`). Config flags on `CapstoneVqNervConfig`
(`src/tac/capstone_vq_nerv/vq_nerv_bundle.py`):

```python
hinerv_grid_pe: bool = False     # default-off => byte-identical to the pre-switch decoder
grid_pe_num_freqs: int = 4       # pe_dim = 4 * num_freqs (sin/cos × {x,y} × num_freqs)
```

Mechanism (when ON):
- A **DETERMINISTIC** multi-frequency sinusoidal coordinate grid `(base_h·base_w, pe_dim)` is built from
  coordinates at decode (`grid_positional_encoding(base_h, base_w, num_freqs)`; NeRF-style frequencies
  `2^k·π`). It stores **0 archive bytes** — the inflate regenerates it from `(base_h, base_w, num_freqs)`.
- A tiny learned `nn.Linear(pe_dim, channels[0])` projection (`grid_pe_proj`) maps the grid to the stem
  channels and is **ADDED to the stem feature BEFORE the `sin`** activation. The projection is
  **zero-init**, so at init the grid-PE contributes 0 — the untrained ON-render == the OFF-render. Only
  training writes the grid grammar into the projection.
- The ONLY new stored params are `grid_pe_proj.proj.{weight,bias}` (`channels[0] × pe_dim`).

Files touched (extending existing files — NO new decoder):
- `src/tac/capstone_vq_nerv/vq_nerv_bundle.py` — `CapstoneVqNervConfig.{hinerv_grid_pe,grid_pe_num_freqs}`;
  `_GridPE` module; gated injection in `_decode_with_film` (predecessor; verified this session).
- `src/tac/capstone_vq_nerv/numpy_reference.py` — `grid_positional_encoding`; gated grid-PE branch in
  `_features_nhwc`; `CapstoneDecodeConfig.{hinerv_grid_pe,grid_pe_num_freqs}`; export of
  `grid_pe_proj.*` in `full_render_weights_from_bundle`; carry of flags in `decode_config_from_bundle`.
  **(this session)** added `_grid_pe_weight_keys()` — the NO-FAKE key-mismatch fix (see §4).
- `src/tac/capstone_vq_nerv/inflate.py` — carry `hinerv_grid_pe`/`grid_pe_num_freqs` into BOTH the
  `stored_latent` AND the `vq_index` `CapstoneDecodeConfig`. **(this session)** added the `vq_index`
  branch (the predecessor had only wired the `stored_latent` branch — a GAP that would have silently
  dropped grid-PE for a vq_index archive).
- `src/tac/capstone_vq_nerv/tests/test_numpy_reference_parity.py` — **(this session)** 7 grid-PE tests.

### Canonical-vs-unique decision per layer (Catalog #290)

| layer | decision | rationale |
|---|---|---|
| upsample blocks (bilinear-skip + PixelShuffle + sin) | **ADOPT_CANONICAL (reuse `HNeRVDecoderMLX`)** | already the PR95 canonical decoder; the bilinear-skip HiNeRV half is structurally present; rebuilding = duplication anti-pattern. No fork. |
| grid positional-encoding | **FORK_PRINCIPLED (new `_GridPE` + `grid_positional_encoding`)** | no canonical grid-PE helper exists for this decoder; HiNeRV's delta is exactly this; the deterministic-grid + zero-init-proj design is the substrate-optimal, byte-free, default-off form. |
| grid-PE injection point | **FORK_PRINCIPLED (add-before-sin at the stem)** | matches HiNeRV's coordinate-at-input inductive bias; injecting after sin or at a block would not be the HiNeRV mechanism. |
| numpy/inflate port | **ADOPT_CANONICAL (extend `numpy_reference` op-for-op)** | the portability contract requires the numpy decode to reproduce the MLX forward exactly; reuse the existing linear/conv primitives. |
| archive grammar | **ADOPT_CANONICAL (unchanged 4-section / 3-section monolith)** | grid is storage-free; only the tiny `grid_pe_proj.*` params ride the existing decoder weight section (sorted-name int8/fp16 brotli). No grammar change. |

---

## 3. BYTE-DELTA of the grid-PE (the rate accounting)

- **Grid itself:** 0 archive bytes (regenerated from `(base_h, base_w, num_freqs)` at inflate).
- **New stored params:** `grid_pe_proj.proj.weight` `(channels[0] × pe_dim)` + `grid_pe_proj.proj.bias`
  `(channels[0],)`. With the frontier-class `base_channels=36` (`channels[0]=36`) and `num_freqs=4`
  (`pe_dim=16`): `36×16 + 36 = 612` params. At the int8 codec (~1 byte/param + per-tensor fp16 scale +
  brotli q11), that is **≈ 0.6 KB** added to the decoder section (rate Δ ≈ `25·600/37_545_489 ≈ +0.0004`).
  At `base_channels=20` (`channels[0]=20`): `20×16+20 = 340` params ≈ **0.34 KB** (rate Δ ≈ +0.0002).
- **When OFF (default):** **0 bytes added** — no `_GridPE` module is constructed, no `grid_pe_*` keys exist
  in the render basis (proven by `test_grid_pe_off_is_byte_identical_no_extra_params`). The archive is
  byte-identical to the pre-switch decoder.

So the upgrade is essentially byte-free (≤0.6 KB when ON) and exactly byte-identical when OFF — the cost is
trivially dominated by any BD-rate gain the spatial inductive bias buys at training time.

---

## 4. PARITY + NO-FAKE TEST RESULTS (Step 3 — the gate)

New tests in `src/tac/capstone_vq_nerv/tests/test_numpy_reference_parity.py` (7), all PASS:

| test | what it gates | result |
|---|---|---|
| `test_grid_pe_off_is_byte_identical_no_extra_params` | DEFAULT-OFF: no `_GridPE` module, no `grid_pe_*` keys, flag off | PASS |
| `test_grid_pe_on_at_init_equals_off_render` | zero-init proj ⇒ ON-render == OFF-render (safe opt-in) | PASS (max|Δ| < 1e-3) |
| `test_grid_pe_numpy_matches_mlx_render[False]` | PORTABILITY: numpy == MLX at init | PASS (drift 6.7e-4) |
| `test_grid_pe_numpy_matches_mlx_render[True]` | PORTABILITY: numpy == MLX with **trained** proj (the bug case) | PASS (drift 0.017, fp32 conv-accum order, NOT score-affecting) |
| `test_grid_pe_trained_actually_changes_render_not_a_noop` | **NO-FAKE**: trained grid-PE measurably changes the render vs OFF | PASS (max|Δ| 33.5 ≫ 1.0) |
| `test_grid_pe_is_deterministic_and_storage_free` | grid is a pure fn of coords, band-limited, deterministic, `num_freqs=0`→ValueError | PASS |
| `test_grid_pe_inflate_end_to_end_with_trained_proj` | END-TO-END: trained grid-PE archive byte-closes + real numpy inflate score-matches MLX render on frozen DistortionNet | PASS (d_seg parity <1e-3) |

**THE REAL NO-FAKE BUG I FOUND + FIXED (this is the load-bearing finding):** the predecessor's
`numpy_reference._features_nhwc` looked for the weight key `grid_pe_proj.weight`, but the MLX export
(`full_render_weights_from_bundle` → `tree_flatten`) emits `grid_pe_proj.proj.weight` (the `_GridPE`
module nests the projection in `self.proj = nn.Linear`, mirroring how FiLM emits `pose_film0.fc1.weight`).
With a **zero-init** proj this coincidentally passed (the missing-key path and a zero projection both add
nothing). With a **trained** proj the numpy path silently SKIPPED the grid-PE while MLX applied it → a
**34-unit render divergence** = a portability-contract violation = exactly the forbidden
"returns-canonical-markers-without-doing-work" / silent-no-op NO-FAKE class. Fix: `_grid_pe_weight_keys()`
resolves the canonical `grid_pe_proj.proj.{weight,bias}` (with the flat form accepted for forward-compat).
Post-fix trained parity = **0.017** (down from 34.1). A second smaller GAP was the `vq_index` inflate
branch missing the grid-PE config (only `stored_latent` had it) — also fixed.

**Broader regression:** the full parity file (14 tests) PASS; the full capstone suite **76 passed, 1
deselected** (the slow real-scorer test). The slow test
(`test_real_scorer_joint_loop_moves_seg_logits_and_holds_pose`, `@pytest.mark.slow`,
`@skip_no_real_scorer`, ~80–150s) is **PRE-EXISTING FLAKY, independent of this change — PROVEN, not
asserted**:
- It builds a default-config bundle (grid-PE structurally OFF — no `_GridPE`, all my edits gated behind
  `if cfg.hinerv_grid_pe` / never imported unless ON). `numpy_reference` is imported ONLY inside
  `_GridPE.__init__`, which is never constructed here, so my numpy/inflate/test edits are UNREACHABLE
  from this MLX training path.
- **Root cause of the flake (isolated):** the bundle's FiLM + decoder `nn.Linear` inits draw from MLX's
  UNSEEDED global RNG stream (only `self.latents` uses the explicit `key=mx.random.key(seed)`). A
  per-process construction probe showed `pose_film0.fc1.weight[0]` = −0.3826 / −0.0129 / +0.1234 across
  three separate processes of the SAME tree — i.e. the init (hence the 8-epoch aggressive-LR `muon_lr=3e-2`
  trajectory) is non-reproducible run-to-run. The test's `d_pose1 <= d_pose0 + 1.0` tolerance sits at the
  edge of that variance.
- **The decisive evidence:** with my changes applied the test both FAILS and PASSES across runs (observed:
  fail `d_pose 131→167`, then PASS, …) while the committed baseline passed 3/3 — i.e. it is a coin-flip at
  the tolerance edge, NOT a deterministic effect of this default-off upgrade. The construction-SHA differing
  between HEAD and the current tree was likewise just two unseeded random draws (param COUNT identical,
  `__latents` identical since it IS seeded), not a code-numeric difference.
- **Not fixed here** (out of scope for this default-off decoder-capability task): the robust fix is to seed
  the FiLM/decoder init in the test (or widen the pose-hold tolerance for the 8-epoch probe). Filed as the
  reactivation note below.

---

## 5. HOW TO ENABLE FOR THE NEXT FRONTIER-CLASS DAEMON

Default-off-safe: the current daemon + all existing behavior are unchanged. To turn the HiNeRV grid-PE ON
for a new training run, set on `CapstoneVqNervConfig`:

```python
CapstoneVqNervConfig(
    num_pairs=600, base_channels=36, latent_dim=28,
    carrier="stored_latent",      # the C1′ pose-capable carrier (frontier's own)
    hinerv_grid_pe=True,          # <-- the HiNeRV grid-PE upgrade (this landing)
    grid_pe_num_freqs=4,          # pe_dim = 16; ~0.6 KB stored proj at base_ch=36
)
```

The trainer/export must thread `hinerv_grid_pe` + `grid_pe_num_freqs` into the `capstone_config_v1`
inflate sidecar (the same dict `decode_archive` consumes — `decode_config_from_bundle` already produces
both flags; `_build_grid_pe_config` in the test is the reference sidecar builder). Then the byte-closed
advisory (`tac.capstone_vq_nerv.advisory.score_reloaded_int8_archive`) and the contest `inflate.py`
reproduce the grid-PE render exactly (the parity gate above proves it).

**Recommended next step (not done here — needs GPU/long-run, out of this CPU-only task's scope):** a
paired A/B `hinerv_grid_pe=False` vs `True` at base_ch=20/36, ≥48 pairs, on the trustworthy cosine-LR +
EMA stack, measuring whether the grid-PE's spatial inductive bias lowers d_seg at equal bytes (the
falsifiable BD-rate-over-HNeRV claim). The upgrade is ready to train; this landing is the byte-free,
parity-proven, default-off mechanism + its NO-FAKE gate.

---

## 6. WIRE-IN (Catalog #125) + scoreboard

1. **sensitivity-map — N/A** (decoder-capability mechanism; no per-axis byte-savings row until the A/B
   measures the d_seg-per-byte of grid-PE).
2. **Pareto — ACTIVE** (adds a near-zero-byte capacity axis: grid-PE buys spatial inductive bias for
   ≤0.6 KB; the feasible move is to test it at equal bytes on the C1′ carrier).
3. **bit-allocator — N/A** (the grid is storage-free; the proj rides the existing decoder section codec).
4. **cathedral-autopilot — gate-conditional** (no dispatch; CPU-only build, advisory non-promotable).
5. **continual-learning — ACTIVE** (reseeds the judge: HNeRV bilinear-skip is ALREADY present in our
   decoder; the HiNeRV delta is grid-PE, now landed default-off; a nested-module export emits
   `grid_pe_proj.proj.*` not `grid_pe_proj.*` — the key-mismatch NO-FAKE class to watch in future ports).
6. **probe-disambiguator — RESOLVED** ("what HiNeRV mechanism is missing?" → grid-PE, not bilinear-skip;
   "is the numpy port faithful with a trained proj?" → NOW yes, after the key fix; "does OFF stay
   byte-identical?" → yes, no `_GridPE` constructed).

**UPPER:** frontier 0.19110 [contest-CPU] UNMOVED (decoder-capability upgrade, not a score row).
**LOWER:** S_floor 0.11797 [advisory] unchanged. The grid-PE is a candidate BD-rate lever toward the C1′
decoder-shrink path; its score effect is a TRAINING outcome a future A/B falsifies.

## 7. CROSS-REFERENCES
`capstone_optimal_carrier_design_20260611T041937Z.md` (C1′ = smaller score-aware HNeRV decoder; the lever
this grid-PE feeds) · `capstone_carrier_pivot_vq_index_impoverishment_20260611T034500Z.md` (stored-latent
carrier the grid-PE composes with) · `lever_b_byte_close_exact_eval_readiness_20260611.md` (lever B → lever
C pivot: "can a per-pair-latent CONVOLUTIONAL HNeRV-class carrier reach the frontier cell?" — grid-PE is
part of making that carrier maximally expressive) · `src/tac/local_acceleration/pr95_hnerv_mlx.py` (the
shared decoder backbone; bilinear-skip + PixelShuffle + sin audited present) ·
`src/tac/capstone_vq_nerv/{vq_nerv_bundle,numpy_reference,inflate}.py` (the landed grid-PE) ·
`src/tac/capstone_vq_nerv/tests/test_numpy_reference_parity.py` (the 7-test grid-PE gate).
