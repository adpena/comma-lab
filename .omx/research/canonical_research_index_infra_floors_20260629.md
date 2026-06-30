# Canonical Research Index — INFRA / MEASURED-FLOORS / EXACT-ROWS / THEORY / MOLT / DSL

**Date:** 2026-06-29/30 · **Author:** consolidation subagent `idx-infra` · **$0 (no GPU, read-mostly)**
**Pointer UNMOVED: contest-CPU 0.19109982 · contest-CUDA 0.20533003** (`.omx/state/canonical_frontier_pointer.json`)

**Purpose (operator 2026-06-29 "signal loss + rediscovery + starting LESS optimal than perfect recollection"):**
a deduplicated, calibrated slice of the INFRA/FLOORS/EXACT-ROWS/THEORY/MOLT/DSL axis so we never re-derive
the measurement substrate or mis-state a floor/row. Every score here is pointer-backed and axis-tagged.
NO-FAKE: only `upstream/evaluate.py` 600-sample rows on exact archive bytes are scores; advisory/MLX/MPS are
explicitly marked non-authority. This index is a MEANS — the END is a lower exact score; pointer UNMOVED.

---

## 0. CALIBRATION LEGEND (axis discipline — binding)

- **[contest-CPU]** = `upstream/evaluate.py --device cpu` on Linux x86_64, 600 samples, exact archive bytes. **The contest leaderboard ranks by THIS axis.** Authority.
- **[contest-CUDA]** = `--device cuda` on Linux x86_64 T4, 600 samples, exact bytes. Authority (separate axis; never inferred from CPU).
- **[advisory]** = macOS-CPU / macOS-MLX research-signal. NOT a score; `score_claim=false`. (For g3 bc20, advisory==contest-CPU to ~0.001% — a calibration fact, NOT an authority promotion.)
- **[MPS-NEVER]** = MPS is NEVER a score authority (PoseNet drifts 23×, SegNet 2×, score 2.5×). Valid only as an fp32 *training-gradient* device.
- **[theory-bound]** = a derived floor / lower bound, not a measured row.

**Measured CPU→CUDA drift (per-archive, empirical):** CUDA d_pose is consistently WORSE than CPU → CUDA score ≈ CPU + ~0.034 on the entropy-recode family. The CPU-best archive ≠ the CUDA-best archive (see §1). Never infer one axis from the other.

---

## 1. MEASURED EXACT ROWS (the byte-closed contest-CPU/CUDA truth table) — DEDUPLICATED

Source: `.omx/state/active_lane_dispatch_claims.md` (2582 rows) + `.omx/state/canonical_frontier_pointer.json`. `score_recomputed` = recomputed-from-components (NOT the rounded `final_score` field — the rounding trap).

| Row (lane / archive) | Axis | Score | d_seg | d_pose | bytes | archive sha256 (prefix) | Status / note |
|---|---|---|---|---|---|---|---|
| **recoded-R3 / pr110 payload-entropy-recode** — **THE CPU FRONTIER** | **[contest-CPU]** | **0.19109982** | (==R3) | (==R3) | **177169** | `b46897267ded…` | Frontier pointer. Lossless recode of R3 → d_seg/d_pose byte-identical. Beats PR112 0.19112577 + prior 0.19198275. **Borrowed PR101/PR110 recode (NO-FAKE #7 — defensive bank, not original).** |
| recoded-R3 (same archive) | [contest-CUDA] | 0.22528084 | (==R3) | (==R3) | 177169 | `b46897267ded…` | Same bytes, CUDA axis (+0.034 vs CPU). |
| **PR106 format0d latent-score-table** — **THE CUDA FRONTIER** | **[contest-CUDA]** | **0.20533003** | — | — | **186876** | `9cb989cef519…` | CUDA frontier pointer. CUDA-best ≠ CPU-best archive. |
| **g3 torch_vehicle bc20** — **OUR-ORIGINAL witness, dual exact** | **[contest-CPU]** | **0.37797132** | 0.00260094 | 0.00034168 | **89244** | `856e3bf076a5…` | Real calibration row; pointer unmoved. Advisory==contest-CPU to 0.001%. The only OUR-original byte-closed dual row. |
| g3 torch_vehicle bc20 (same archive) | [contest-CUDA] | 0.39153009 | 0.00262703 | 0.00048168 | 89244 | `856e3bf076a5…` | CPU→CUDA d_pose +41%. |
| frontier-decoder-waterfill baseline (b7106c9b) | [contest-CPU] | 0.19198534 | — | — | 178493 | `b7106c9bdbb8…` | Prior CPU frontier (pre-recode); == archived 0.19199. |
| same (b7106c9b) | [contest-CUDA] | 0.22614782 | — | — | 178493 | `b7106c9bdbb8…` | fp11 source-brotli-recode reproduced bit-exact. |
| DQS1 selective-decoderq (best pair-drop rank021) | [contest-CPU] | 0.19202828 | — | — | 178559 | `7a0da5d0fc32…` | DQS1 family floor; ~30 pairset-drop variants all ≈0.1920293 (no movement). |
| DQS1 (rank021 pair0371) | [contest-CUDA] | 0.22619177 | — | — | 178559 | `7a0da5d0fc32…` | |
| pr110pp R3 candidate (per-pair pose table) | [contest-CUDA] | 0.22616377 | — | — | 178495 | `1ccae18d8632…` | **NO TRANSFER** — ΔS+0.02083 vs CUDA control; macOS-CPU per-mode pose ordering did NOT transfer to Linux CPU → KILL criterion fired (paradigm intact, macOS-CPU selector FALSIFIED). |
| ias1 inverse-scorer runtime-parity top4 | [contest-CPU] | 0.19380912 | — | — | 181232 | `2d08507894 83…` | Above frontier. |
| same | [contest-CUDA] | 0.22796961 | — | — | 181232 | `2d0850789483…` | |
| v14 fec10-hybrid stacked | [contest-CPU] | 0.19204266 | — | — | 178504 | `fed97266e88d…` | NOT PR111 candidate (+1.4e-5 vs CPU frontier). |
| same | [contest-CUDA] | 0.22620137 | — | — | 178546 | `0a3abfe645c4…` | |
| feca selector-reparam scale64/256 | [contest-CPU] | 0.19200897–0.19201630 | — | — | 178530–178541 | `18e3155fbbbe…` | Near-frontier, no cross. |
| L1 sidecar-drop (frontier latent waterfill) | [contest-CPU] | 0.19486984 | — | — | — | — | Above frontier; sidecar **pays rent ~7×**; L2/L3/L4 killed by kill-gate. DEFER. |
| frontier-decoder-waterfill c1/c2/c3 | [contest-CPU] | 0.26287 / 0.28219 / 0.35675 | — | — | 169185/167498/159936 | — | Decoder-quant ladder; all above frontier (rate-saved, distortion-swamped). |

**NOT-a-row (flag):** triple-wave-N6 composite predicted [contest-CPU] **0.156006** (sub-0.16) appears in the ledger but only at `active_modal_*_spawning` status — it **never completed an exact eval**. It is a PREDICTION, not a measured row. Do not cite as achieved.

### Frontier-family takeaways (deduplicated)
- **The whole 0.191–0.192 cluster is the borrowed PR101/PR110/DQS1/FEC entropy-recode plateau.** Dozens of pairset-drop / selector / FEC-hybrid variants land within ~1e-4 of 0.19203 (CPU). The −2.58e-6 recoded-R3 win (0.19110) is the deepest lossless squeeze; **lossless rate on this frontier is EXHAUSTED** (FEED-lb; finishing-kit byte_delta=0; a NO-FAKE catch killed a double-counted "−0.005 finishing kit" estimate).
- **OUR-original (g3 bc20) is the honest non-borrowed row but at S~0.38** (d_seg ~0.0026 adequate, but rate 89244B → ~0.059 rate-term yet seg/pose distortion dominate). The witness RD-CURVE projects ~89KB→S0.216, B*~122KB optimal-form+directional→S0.134 (sub-0.15) — *advisory/through-R projection, the optimum is not the current point; map the curve.*

---

## 2. ★ OPTIMAL-CONFIG CONTRIBUTION — the measurement + deploy + repro substrate (the marshaled best)

The optimal substrate for turning a candidate into a trustworthy exact row, with zero re-derivation:

### 2A. Byte-close → DUAL-EXACT pipeline (the only thing that produces a "score")
1. **Export contract** → byte-closed `archive.zip` (sha256 + size recorded). One-command witness path: **`tools/witness_byte_close_and_eval.py`** (trained MLX ckpt → int8+brotli archive whose `st_size` IS the rate term → MLX-free numpy+torch `inflate.py` → realized d_seg/d_pose → staged contest-CPU command). Full-scale advisory characterization: `tools/z8_600pair_byte_closed_contest_score_advisory.py`.
2. **Local parity smoke** → run contest `inflate.sh` in a clean env; verify raw-output byte count; **byte-mutation smoke** (Catalog #105/#139) proves bytes are consumed.
3. **Canonical eval** = **`experiments/contest_auth_eval.py`** (`archive.zip → submission inflate.sh → upstream/evaluate.py → score`). `--device {cuda,cpu}`; stamps `score_axis="contest_cuda"` only when device==cuda AND `tac.device_axis_eval.is_contest_cuda_equivalent_gpu` passes (T4 match). Result JSON → `work_dir/contest_auth_eval.json` (+ `RESULT_JSON:` stdout).
4. **DUAL exact eval** (mandatory for any shippable/frontier claim): paired CPU (Linux x86_64) **AND** CUDA (T4) on the **EXACT same archive bytes**. Planner: `tools/plan_dual_device_auth_eval.py`. Canonical Modal paired dispatch: **`tools/dispatch_modal_paired_auth_eval.py`** (plan-only by default; `--execute` spawns both detached; `--skip-axis-if-promotable-anchor-exists` via `tac.deploy.modal.anchor_lookup`). Recover detached: `tools/recover_modal_auth_eval.py`. Results land in `experiments/results/modal_auth_eval{,_cpu}/…/modal_{cpu,cuda}_auth_eval_result.json`.
5. **Recompute score FROM COMPONENTS** (never the rounded `final_score`): `contest_auth_eval._parse_report` sets `canonical_score = score_recomputed_from_components` = `100·d_seg + sqrt(10·d_pose) + 25·archive_bytes/uncompressed`. **Refuses on:** rounded-rate custody drift, NaN/inf, negatives, `n_samples != 600`, >0.01 formula divergence. Shared device-axis math: `src/tac/device_axis_eval.py` (`score_terms`, `cuda_minus_cpu_gaps`, `raw_output_pairing`).
6. **Harvest + reseed**: ledger row via `tools/claim_lane_dispatch.py`; posterior via `src/tac/continual_learning.py` (`contest_result_from_auth_eval_payload` reads `score_recomputed_from_components` FIRST); pointer auto-refreshes on dispatch completion. Pre-submission gate: `scripts/pre_submission_compliance_check.py` (verifies artifact+archive+manifest+report+ledger agree; does NOT itself score).

### 2B. FREE small-n exact-eval loop ($0 local) — INFRA WIN (mechanism corrected)
**Accurate mechanism (read from `upstream/evaluate.py:10-18`):** there is NO `--num-samples`; `--batch-size` is the dataloader batch (default 16), NOT a sample count. The free small-n loop = pass a **`--video-names-file`** containing only the first n names → the distortion accumulators iterate exactly those pairs and divide by the real count → **d_seg/d_pose on the n-subset are REAL through the actual scorer.** **BUT** `rate = compressed_size / uncompressed_size` uses the FULL 37,545,489-byte denominator regardless of n (`:63-65`) → S is **NOT 600-comparable**, and `contest_auth_eval.py` refuses `n_samples != 600` for a promotable score. Use for distortion go/no-go ONLY. *(The CURRENT-memory shorthand "`--batch-size n`" is imprecise — the real subset knob is `--video-names-file`.)*

### 2C. Determinism / reproducibility spine (one of two hard limits)
- **numpy-fp32 = bit-identical verdict authority**; torch/MLX must match (parity ≥0.9997). Realized as pure-numpy float64 mirrors of the generators, e.g. `src/tac/boundary_math/lever_b_generator.py:251` (mirror of `ScoreNativeSegGenerator.__call__`). Parity audited under `src/tac/analysis/*_official_source_parity_audit.py` + `hinerv_training_parity_guard.py`.
- **MPS is NEVER authority** [MPS-NEVER]; macOS-CPU/MLX are advisory; only contest-CPU/CUDA score. Canonical device gate: **`src/tac/substrates/_shared/trainer_skeleton.py:668` `device_or_die(...)`** (cuda default; cpu only with `--smoke`/waiver; **mps FORBIDDEN**, SystemExit). MPS training-gradient patch (no upstream edit): `src/tac/torch_mps_compat.py` (~104× faster than torch-CPU at fp32, grad cosine ~1.0; fp32 sweet spot — fp16/bf16 worse).
- **Seeded everywhere** (single recorded `seed` across torch/numpy/random/MLX); per-substrate seed helpers (`src/tac/openpilot_seeding.py`, MLX ports under `src/tac/mlx_pr95_port/*`).
- **Resumable-from-disk + per-stage checkpoint + EMA-shadow** (NON-NEGOTIABLE): substrate = `src/tac/torch_vehicle/checkpoint.py` (per-stage/EMA-shadow save+resume) + `curriculum.py` / `src/tac/witness_dsl/curriculum_dsl.py` (staging). Never launch anything not `--resume-from`-able that doesn't save a complete byte-close-loadable checkpoint at EVERY stage boundary (CE/tau/l7/Muon), atomically (tmp+rename), distinct filename per stage, EMA shadow (not live weights). Loop-end-only saving FORBIDDEN.
- **Provenance with every result**: git hash, seed, config, upstream snapshot sha, hardware/axis, archive sha256+size, realized-through-R deltas.

### 2D. Dispatch / harvest infra
- `tools/claim_lane_dispatch.py` — fcntl-locked cross-agent dispatch claim (24h TTL; `--allow-parallel`/`--force`/`--dry-run`).
- `experiments/modal_train_lane.py` — runs `scripts/remote_lane_*.sh` on Modal T4/A10G (`.spawn()`/`--detach`); recover via `experiments/modal_recover_lane.py`.
- Modal `.spawn()` → result-cache (~24h TTL, NOT a Volume) → **HARVEST OR LOSE** (`tools/harvest_modal_calls.py`, `--execute`); call_id ledger `.omx/state/modal_call_id_ledger.jsonl` (code `src/tac/deploy/modal/`).
- `tools/parallel_dispatch_top_k.py` (strict: refuses prediction-only/forensic/local-proxy; requires `--max-dph`+`--estimated-cost`) + `tools/harvest_and_reseed.py` — the race-mode fan-out actuator.
- Canonical NVML/CUDA env block (Catalog #244): emitted by `src/tac/substrate_registry/driver_generator.py:123` (`tac.deploy.modal.runtime.DALI_DISABLE_NVML_VALUE`); gate `check_remote_lane_scripts_carry_canonical_nvml_block` at `src/tac/preflight.py:59456`.

### 2E. Scale / containment safeguards
- `tools/memory_guard.py` (3-layer: launch-preflight `--check` exit3=REFUSE, whole-machine watchdog `--watch` sheds LARGEST arm, per-arm RSS cap) + `tools/safe_run.py` (per-arm wall-time+RSS cap, `start_new_session=True` so it can ONLY kill its own group — control-plane safe by construction). Both **vendored from molt**.
- **⚠️ FLOOR CONFLICT (telemetry-accuracy flag):** the CODE default is **`DEFAULT_MIN_FREE_GB = 30.0`** (`tools/memory_guard.py:101`), but the operator binding 2026-06-26 RELAXED the floor to **≥10GB** (128GB all-ours). The vendored code is STALE vs the binding → pass an explicit `--min-free-gb 10` until the molt-vendored default is refreshed, OR re-vendor. Do not assume 10GB is enforced by default.
- Guard NEVER kills the control-plane (claude/codex): kill-selector is custody-gated (durable-daemon registry membership = PRIMARY gate) + identity-gated (live pgid + live command); control-plane structurally excluded. Selector fix vendored from molt HEAD 3b1e49b18.
- Containment when optimizing: PRESERVE earned ckpts before EMA-overwrite, CONTAIN blast radius (worktree/default-OFF/one-GPU), PROTECT live run + control-plane.

### 2F. The molt-compile path (the FREE deterministic decoder runtime)
molt = our owned Nuitka-style/Codon-fast Python→WASM+WebGPU compiler. Per rule-118 the GENERIC generator algorithm is FREE in inflate.py (not sized) — a molt-compiled deterministic witness generator is a contest-LEGAL fast decode runtime (sister of the runtime-rs Rust path). See §5.

### 2G. Key code paths (vehicle + math substrate — verified to exist)
- **`src/tac/torch_vehicle/`** — the torch vehicle. `driver.py` (`TorchVehicleConfig` L531, `TorchVehicleDriver` L1374 = the bc20/g3 driver w/ `PoseFiLMHNeRVWrapper`); `checkpoint.py`, `curriculum.py`, `boundary_head.py`/`boundary_routing.py`, `pose_film.py`/`pose_film_v2.py`, `score_aware_qat.py`, `scorer_context.py`, `run.py`.
- **`src/tac/boundary_math/`** — `lever_b_generator.py` (`ScoreNativeSegGenerator` MLX coord-INR + FiLM + 5-class head L12; numpy mirror L251), `lever_b_levelset_generator.py`, `lane_sdf_component.py`, `hood_static_component.py`, `amortized_luma_carrier.py`.
- **`src/tac/se3.py`** (SE(3) exp/log/transforms) · **`src/tac/camera.py`** (intrinsics/projection, 1164×874) · **`src/tac/scorer_targets.py`** (pose sidecar: precomputed `PoseNet(orig)[:6]` per 600 pairs).
- **Range/entropy coder:** **`src/tac/lossless/range_coder.py`** (+ `argmax_codec.py`) — NOT `src/tac/range_coder*` (corrects a common mis-path).

---

## 3. INFORMATION-THEORETIC FLOORS [theory-bound]

| Floor | Value | Kind | Caveat | Source |
|---|---|---|---|---|
| S_floor (rate-dominated) | **0.11797** (≈0.118) | rate-only lower bound | **LOOSE** — assumes d_seg→0 byte-cheaply, which the pincer FALSIFIED. The achievable task-RD floor is strictly higher. (value confirmed across 47 memos.) | `adjusted_theoretical_floor_v3_post_pr106_falsification_20260513.md`; `active_pr103_pr106_floor_anatomy_20260507_worker_p.md` |
| Task-space byte floor (witness) | (formulation) | task-space roundtrip byte floor | The witness-specific byte-floor formulation (survives eval roundtrip + frozen-SegNet reproduction). | `CAPSTONE_witness_taskspace_roundtrip_byte_floor_formulation_20260621.md` |
| Task-RD floor S* | strictly inside **(0.118, 0.191)** | indirect-RD / CEO / coding-for-machines task floor | Reachable ONLY by a task-space (feature-space) representation we have never built; every vehicle (PR95/HNeRV/bc20/0.191) sits on the DOMINATED reconstruct-RGB rung. | `project_contest_is_indirect_rate_distortion_task_space_coding_20260619.md`; `vcm_theory_primitive_layer_20260619T033429Z.md`; `frozen_instance_exploit_catch_up_then_surpass_vcm_20260619.md` |
| T_floor | (info-theoretic LB) | headroom proof, not the target | The hard limit; the proof that sub-0.15 is reachable. | CLAUDE.md GOAL ladder |

**Frozen-instance exploit:** the field optimizes task-RD *in expectation*; ours is ONE frozen instance (known video, frozen scorer weights, frozen eval roundtrip) → we can compute the EXACT per-instance optimum (exact histograms, sufficient statistic, polytopes/null-spaces), provably ≤ any learned-general codec → catch up then surpass. The decisive $0 go/no-go = **P-SUFF** (how many reconstructed-RGB bits the frozen scorer is invariant to).

---

## 4. CONTEST MECHANICS (the two hard limits — confirmed from upstream)

- **Score (`upstream/evaluate.py:92`):** `score = 100*segnet_dist + math.sqrt(posenet_dist*10) + 25*rate`.
- **Rate (`:63-65`):** `rate = compressed_size / uncompressed_size`; `compressed_size = (submission_dir/'archive.zip').stat().st_size` — **ONLY archive.zip is sized**; `inflate.py`/`inflate.sh` are NOT counted. `uncompressed_size` ≈ 37,545,489 (the 25·B/37_545_489 term).
- **No time term** — the only constraint is the **30-min full-eval budget** on **T4 (16GB VRAM) OR CPU (4-core/16GB)** (README).
- **Rule-118:** external CODE/tools are FREE (don't count); **video-derived LEARNED artifacts (NN weights, meshes, point clouds) MUST be in archive.zip and ARE counted.** The FREE/COUNTED/FORBIDDEN boundary: generic algorithm + deterministically-generated tables = free; learned/video-derived payload = counted; smuggling video-derived data into "code" = the hide-data-in-code FAKE (NO-FAKE #6/#7).
- **Source provenance:** the contest video = comma2k19 RAV4 segment; comma10k MEMBERSHIP=NO exact contest frames (0) but SAME RAV4/device IS present (30 frames/26 drives) → SegNet did not memorize the clip; the 4.67% lane-edge residual is genuine (`feedback_comma10k_membership_no_exact_frames_same_rig_deterministic_gen_exploit_20260627.md`).

---

## 5. MOLT COLLAB (the FREE-generator / WASM decode path) — #187

- **molt** = operator-owned Python compiler (Nuitka-style AOT, Codon-class speed, "pyodide on steroids"); targets WASM **and WebGPU**; CPython-ecosystem+extensions compatible; smaller/faster binaries than pyodide. A separate molt team actively develops it. Repo `github.com/adpena/molt`. Local in-tree copy (`tools/memory_guard.py`, `tools/safe_run.py`) is STALE — always pull latest to a scratch path OUTSIDE pact.
- **LIVE two-way channel** in `collab/pact/` on branch `pact-collab` (`github.com/adpena/molt/tree/pact-collab`, head 1976e2e): reports 001–008 + a runnable `pact_witness_kernel/` bundle (molt team already building our witness decoder kernels).
- **report 007 = molt team REPLY:** greened the NumPy/SciPy C-API scan (0 missing symbols); milestone = **Kernel-A WASM parity**.
- **our 008 addendum** (additive, no clobber): full decode-chain compile targets (se3/camera/lane_sdf/levelset/range/coord-INR); rule-118 free-generator rate-half framing; **bit-exact-vs-numpy-fp32 + 30min-T4-or-CPU contracts**; **OPEN-Q: WebGPU-in-contest-runner** (does the contest eval env expose WebGPU? gates the GPU-decode path).
- **Local fallback:** `.omx/research/molt_collab_addendum_20260629/`. **FOLLOW-UP:** read report 007 + their live blocker → respond/help (operator directs molt team).

---

## 6. DSL / TRIALITY + canonical-equations meta-layer — #189

- **`src/tac/witness_dsl/`** (`curriculum_dsl.py` + `campaign.py`): a declarative recursion+math front-end compiling to the trainer CLI. **VALIDATES every emitted flag against the real argparse** (never-invent-flags by construction); **enforces preserve/contain/authority** invariants; **emit-only** (containment — never auto-fires a GPU run); deterministic (pure fn of seed + on-disk results + policy). Campaign engine = cyclic-recursion (warm-start chain) + harvest (d_seg Δ) + compose (θ*=bind winning levers). Search over (stage × config × pass × scale); curvelet coarse→fine scaling; UniWARD + margin-saliency levers; RERUN/EXTEND/ADVANCE/ROLLBACK policy. **294 tests pass.**
- **TRIALITY (one object, three views):** the **DAG** (work-graph / FEED chain = trajectory) ↔ the **DSL** (witness-program = executable, registered as equation E10) ↔ the **system-of-equations** (math, E0–E12 in `tac.canonical_equations`). `project_witness_dsl_and_dag_dsl_duality_20260629.md`.
- **Canonical equations registry:** `.omx/state/canonical_equations_registry.jsonl` — **424 equations**; the GR-unified action `S_τ = 100·d_seg + √(10·d_pose) + 25·rate` (stationarity in the frozen-scorer Fisher metric) is E0; E0–E12 formalize conditioning/distortion/rate. CLI: `tools/list_canonical_equations.py`.

---

## 7. OPEN / HEADROOM (ranked by sub-0.15 leverage)

1. **RATE is the binding sub-0.15 lever** (seg axis capped ~0.012 ΔS by label-noise → best-case ≈0.184). The only way to cut rate now = a **SMALLER REPRESENTATION**. THE OPEN FORK: **v2-done-right** (warp-fix + build the real 6-section codec + fix the confounded θ* campaign — the principled task-space rate lever) **OR distortion-quant** of the frontier (gentle/sensitivity-aware mixed int8/int6; prior int5 capped ~0.49).
2. **Task-space (feature-space) representation = the unbuilt prize** — the indirect-RD floor S* ∈ (0.118, 0.191) is reachable ONLY off the dominated reconstruct-RGB rung. Decisive $0 go/no-go = **P-SUFF** measurement.
3. **The HYBRID Pareto solution** (FEED-ll): rate de-risked GREEN (partition keyframes + screw-warp + pose sidecar → rate ~0.006–0.060 ≪ 0.277 store-everything) BUT the deterministic-render d_seg FLOOR (R1 ≈ 0.0185 bulk) is 15–40× the sub-0.15 d_seg budget → needs the TRAINED amortized-residual generator (paused level-set witness) for the d_seg floor, composing with the cheap deterministic substrate.
4. **WebGPU-in-contest-runner** (molt OPEN-Q) — if the contest env exposes WebGPU, a molt-compiled GPU decoder widens the legal free-generator class.
5. **CUDA-axis pose drift** — every recode is ~0.034 worse on CUDA than CPU (pose). The CUDA frontier (PR106 0.20533) is a different archive than the CPU frontier; a CUDA-targeted recode is unexplored headroom on that axis.

---

## 8. CONFLICTS / SUPERSEDED (pointer-only enforcement)

- **Frontier score literals:** the ONLY canonical frontier numbers are contest-CPU **0.19109982** and contest-CUDA **0.20533003** (`canonical_frontier_pointer.json`). Any memo citing "0.19199" / "0.19198" as *current* frontier is SUPERSEDED by the recoded-R3 0.19110 (2026-06-10). "0.193"/"0.196-0.199 cluster"/"0.229" = HISTORICAL.
- **triple-wave-N6 0.156006** = PREDICTION only (never completed exact eval) — do NOT cite as a measured row or a sub-0.16 achievement.
- **macOS-CPU / MLX / MPS** rows are NEVER frontier or promotion authority. The pr110pp R3 per-pair pose table was KILLED precisely because macOS-CPU selector ordering did not transfer to contest-CPU.
- **g3 bc20 0.37797/0.39153** is the ONLY OUR-original byte-closed dual row; the 0.191 frontier is a BORROWED PR101/PR110 recode (NO-FAKE #7 — a defensive bank, not the innovative submission). Keep the distinction explicit in any "our score" statement.
- **"finishing-kit −0.005..−0.008 sub-0.19"** = a DOUBLE-COUNTED estimate (cited already-spent bytes); RETRACTED by a NO-FAKE catch (FEED-lb). Lossless rate on the 0.191 frontier is EXHAUSTED.

### Telemetry-accuracy flags surfaced this sweep (code-vs-memory drift)
1. **memory-guard floor:** operator binding = ≥10GB (2026-06-26) but `tools/memory_guard.py:101 DEFAULT_MIN_FREE_GB = 30.0` (vendored-from-molt, stale). Pass `--min-free-gb 10` explicitly; the relaxation is NOT enforced by default. (§2E)
2. **free small-n eval mechanism:** the CURRENT-memory shorthand "`--batch-size n` scores the first n pairs" is imprecise — `upstream/evaluate.py` has no `--num-samples`, `--batch-size` is the dataloader batch (default 16); the real subset knob is **`--video-names-file`** with the first n names. Distortion subset-real, rate full-denominator, NOT 600-comparable. (§2B)
3. **range coder path:** it is `src/tac/lossless/range_coder.py`, not a top-level `src/tac/range_coder*`. (§2G)

---

*MEANS ≠ ENDS — pointer UNMOVED 0.19110. The verdict for any path here is a byte-closed n600 exact row that crosses the threshold.*
