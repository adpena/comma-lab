# SNeRV FULL-STACK EXTREME-SCRUTINY audit vs the frozen `evaluate.py` — per-component A/B/C provenance

Date: 2026-06-09
Author: Claude MAX-REASONING full-stack extreme-scrutiny subagent (READ-ONLY; no code edits, no training, no GPU, no dispatch).
Operator directive 2026-06-09 verbatim: *"what about our snerv implementation? all vehicles need extreme scrutiny all lines full stack"*.

Sister memos this DEEPENS (not duplicates):
- `.omx/research/snerv_all_vehicles_fidelity_review_vs_evaluate_py_20260609.md` (the fidelity MANIFEST: arch real, fails from recon-MSE-only + mean-collapsed finest skip).
- `.omx/research/deep_hinerv_snerv_fidelity_review_vs_evaluate_py_20260609.md` (the HiNeRV template + the "L0 SKETCH dispatched before maturation" precedent).

**Authority discipline (CLAUDE.md non-negotiables):** everything here is `[macOS-CPU advisory]` / mechanism-only. NO score claims, NO promotions, NO kills (Forbidden premature KILL). The ep22399 numbers cited are `[macOS-MLX research-signal]` telemetry, NOT scores. Default verdict on any weak result is DEFERRED-pending-research. The architecture verdicts are *mechanism + provenance* verdicts, NOT score verdicts.

**The contest geometry (never edit `upstream/`):** `S = 100·d_seg + sqrt(10·d_pose) + 25·bytes/37,545,489`. SegNet reads ONLY frame1 argmax (5-class, keys on HF boundary structure); PoseNet reads BOTH frames via RGB→YUV6, MSE on first 6 of 12 pose dims. The universal failure mode: blur → SegNet argmax collapse → d_seg≈0.5.

---

## 0. THE HEADLINE (the operator's exact provenance question, answered)

**SNeRV is NOT fake and NOT a "SKETCH dispatched before maturation" in the HiNeRV sense. Its full architecture stack — DWT, official conv MFU/HFR/TUB, archive grammar, inflate, parse-back, receiver proof — is REAL-AND-COMPLETE and byte-closed, with honest fail-closed blockers everywhere a claim outruns evidence. SNeRV's failure at ep22399 (d_seg=0.711 `[macOS-MLX research-signal]`) is a class-B INCOMPLETENESS/ORCHESTRATION ARTIFACT, but a DIFFERENT and narrower one than HiNeRV's:**

> **The wrong loss path trained the right architecture.** SNeRV has TWO loss surfaces. The real-frozen-scorer surface (`scorer_loop_decoder_qat.py`) is wired ONLY to the *tiny path-A linear* decoder smoke. The *at-scale official-conv* decoder (path B) was trained through the SHARED MLX harness whose base term is reconstruction-MSE with every SegNet/PoseNet weight defaulting to **0.0** (`bundle.py:519,527,540,541`; `score_aware_loss(recon_weight=1.0)` `loss.py:3099,3174-3201`). The ep22399 run artifact literally records `observed_segnet_distillation_weight = None` AND `recommended_segnet_distillation_weight = None`. So the most-faithful conv decoder in the entire fleet was optimized scorer-blind against MSE, whose minimizer is the conditional-mean blur that collapses SegNet's argmax.

**Provenance proportions for SNeRV (per-component, evidenced in §2):**
- **REAL-AND-COMPLETE: ~70%** of the stack by component count (DWT + adjoint; official conv MFU/HFR/TUB numeric primitives; archive grammar SNAR1/SNAR2; inflate runtime; parse-back/no-op falsification; receiver proof; real-frozen-scorer QAT loop). These are genuine, byte-closed, tested, and honestly labeled.
- **JUSTIFIED ADAPTATION (class A): ~15%** (LF-store/generate-HF paradigm; pywt ADOPT; L∞ allocator ADOPT; SNAR2 binary header; `x` member-name byte-save) — each carries a `## Canonical-vs-unique decision per layer` row in the landing memo (`snerv_inverse_steganalysis_carrier_landed_20260601.md`).
- **INCOMPLETENESS / ORCHESTRATION ARTIFACT (class B): ~15%** and it is the SCORE-BINDING fraction:
  - **B1 (the binding defect): the at-scale trainer used recon-MSE-base/scorer-off, not the real-scorer objective.** This is an orchestration artifact (the long run was launched against the shared-harness default config, not the substrate's own real-scorer loop), NOT a justified adaptation and NOT fake. There is NO design memo that says "train SNeRV recon-MSE-only on purpose."
  - **B2: path-A HF generator is a degenerate LINEAR `einsum` map** (`carrier.py:837-843`). The *fork decision* is justified (memo: `FORK_PRINCIPLED`), but the memo justifies "generate HF from LF," NOT "do it with a fixed linear filter." A linear map of the LF approximation cannot synthesize the boundary HF SegNet keys on. Unexamined-degeneracy = class B.
  - **B3: official-conv path B is export-blocked + source-forward-unproven** (`carrier.py:312-316`; `official_tub.py:40-44`). These are HONEST, fail-closed blockers (so NOT fake), but they are *unfinished* (class B) — the conv path B that should carry SNeRV cannot yet emit a byte-closed contest archive from its own trained MLX weights.
- **FAKE (class C): 0%.** I found NO instance of any of the 5 forbidden FAKE classes. Every "official" primitive computes real conv/DWT math on real inputs; every authority field is fail-closed `False`; every blocker is a real string; the MFU/HFR/TUB enum is 3 structurally-distinct modules (not enum-padding); the receiver/parse-back/TUB-prune reports run real byte mutations and real replays.

**Full-stack integrity verdict (does SNeRV emit a scorable contest archive from real-teacher training?):** **NO — blocked, but the block is in the orchestration/export wiring, not the architecture.** The official-payload inflate chain IS byte-closed and IS consumed (proven by §3), so an *official-payload archive built by the receiver-proof path* scores. But the **at-scale MLX-trained official conv weights are not bound to that official payload** (B3 export blocker), and the **trainer that ran at scale used the scorer-blind recon-MSE loss** (B1). So end-to-end "real-scorer training → byte-closed official archive → scorable" does NOT currently close. The path-A linear archive DOES close end-to-end (it is numpy-portable and the receiver proof builds an `archive.zip`), but path A is the degenerate linear decoder (B2) that cannot synthesize HF.

**The single cheapest, highest-value, binding fix:** the sister memo's **G1 is directionally right but mis-targets the binding line.** The binding fix is TWO config-only moves that must land together, verified line-by-line below in §5:
1. **Flip the trainer's loss config** so the long official-MFU MLX run sets `segnet_direct_live_distillation_weight > 0` (+ a class-balanced subcontrol, since d_seg is the contest's per-pixel argmax-flip rate) and `pose_direct_live_distillation_weight > 0`, and anneals `recon_weight` toward a small anchor. These knobs already exist as `score_aware_loss` extra terms (`loss.py:3204-3269+`) and SNeRV policy params (`mlx_native_train_export.py:499-515`) — they are simply defaulted to 0.0.
2. **Keep `official_skip_high_mode='full'`** (it is already the carrier DEFAULT, `carrier.py:170`) — do NOT use the `channel_mean`/`scalar_mean` byte-saver modes that collapse the finest skip to a frame-invariant blob (`mlx_renderer.py:745-758`).

Necessary-but-deferred (does NOT block the G1 smoke, blocks PROMOTION): **bind the trained official-conv MLX weights into the official MFU/HFR/TUB decoder payload** so the official conv path (not just path-A linear) emits the byte-closed archive (B3). Until then the G1 result is a `[macOS-MLX research-signal]` mechanism check, not a scorable archive.

---

## 1. The two-loss-path structural map (the crux nobody else stated)

SNeRV is unusual: it has TWO distinct loss surfaces and TWO distinct decoders, and the score-binding bug is the CROSS-WIRING between them.

| | Decoder | Loss surface | Trained at scale? | Real frozen scorer? | Emits byte-closed archive? |
|---|---|---|---|---|---|
| **Path A** | linear ridge `einsum` (`carrier.py:837-843`) | `scorer_loop_decoder_qat.py` (REAL exact SegNet/PoseNet, perturbative QAT smoke) | **NO** — "intentionally tiny … false-authority" (`scorer_loop_decoder_qat.py:2-9`) | **YES** (`load_score_exact_scorers`, `compute_s_seg_flip_risk`, line 24-30, 449, 1682) | **YES** (numpy-portable SNAR1, receiver proof builds `archive.zip`) |
| **Path B** | official conv MFU/HFR/TUB (`official_mfu.py`/`official_hfr.py`/`official_tub.py`) | SHARED MLX harness `score_aware_loss` (recon-MSE base, scorer weights default 0.0) | **YES** — ep22399 | **NO** (defaults off; ep22399 `observed_segnet_distillation_weight=None`) | **NO** — export blocked (`carrier.py:312-316`) |

**The defect in one sentence:** the real-scorer objective is wired to the decoder that cannot win (path-A linear), and the decoder that could win (path-B conv) is trained against the objective that produces blur. The fix (§5) is to route the real-scorer objective to path B — which is config-only on the loss side, plus the export binding for promotion.

---

## 2. Per-file, per-component A/B/C provenance table (every verdict cited to file:line)

Legend: **RC** = REAL-AND-COMPLETE · **A** = JUSTIFIED ADAPTATION · **B** = INCOMPLETENESS/ORCHESTRATION ARTIFACT · **C** = FAKE.

### 2.1 DWT / wavelet LF·HF split (`dwt.py`)

| Component | Claim | Verdict | Evidence (claim + justification) |
|---|---|---|---|
| Orthonormal Haar DWT | exact normalized Haar | **RC** | `_haar_dwt2_level` `dwt.py:408-422` (`(a±b±c±d)*0.5` = orthonormal); inverse `_haar_idwt2_level` 425-438 |
| db2 pywt periodization | square orthonormal on padded canvas | **RC** | `dwt2_multilevel` 196-225; `DWT_MODE="periodization"` 70 |
| Exact synthesis adjoint (G3) | `idwt2 == analysis adjoint`, rel-residual 0.0 | **RC** | `synthesis_adjoint_residual` 310-355 is a REAL dot-product test `<S c,g>==<c,S^T g>`; crop-aware zero-embed `dwt2_native_synthesis_adjoint` 228-262; test `tests/test_dwt_adjoint.py` |
| pywt ADOPT vs Z8 fork | canonical-helper decision | **A** | docstring rationale `dwt.py:33-40` + landing-memo row "DWT … ADOPT_CANONICAL" |
| Gradient reachability | DWT differentiable for training | **RC (numpy adjoint) / N/A in MLX render** | adjoint exists for pixel-cotangent → coeff; the renderer's HF is generated by conv (path B) or linear (path A), not by inverting a learned DWT, so DWT-grad-reach is not the training bottleneck |

**DWT verdict: REAL-AND-COMPLETE.** The "exact DWT adjoint" claim is true and tested; the pywt ADOPT is a documented class-A decision.

### 2.2 Official MFU / HFR / TUB (`official_mfu.py`, `official_hfr.py`, `official_tub.py`, `official_tub_torch.py`)

| Component | Claim | Verdict | Evidence |
|---|---|---|---|
| `OfficialConv2dNchw.forward` | real conv2d | **RC** | `np.einsum("nchwkl,ockl->nohw", windows, w64)` `official_hfr.py:257` (genuine sliding-window) |
| `OfficialResidualBlockNoBN` | conv→LeakyReLU(0.1)→conv→**skip-add** | **RC** | `official_mfu.py:317-324` `arr + conv2(leaky_relu01(conv1(arr)))` — the residual HF path the rest of the fleet LACKS |
| `OfficialSnervMfu.forward` | ConvT→cat(skip_mid)→RB→ConvT→cat(skip_high)→RB U-Net | **RC** | `official_mfu.py:480-514` (real skip-CONCATs + nonlinearity); NumPy + MLX forwards 516+ |
| HFR heads LH/HL/HH | `1x1 Conv→LeakyReLU(0.1)→3x3 Conv→3ch`, stack dim2 | **RC** | `official_hfr.py:106-131,166-197` (matches official torch layout) |
| MFU/HFR/TUB enum = 3 distinct primitives | not enum-padding | **RC (not C-class enum-padding)** | 3 separate modules, distinct math: MFU=fusion U-Net, HFR=detail conv heads, TUB=temporal output_2 split/concat/pixel-shuffle; `_forward_trace` distinct |
| TUB `output_2` fusion algebra | source split/concat/shuffle | **RC** | `official_output2_fusion_numpy` `official_tub.py`; torch shim `official_tub_torch.py:19-46` (thin backend dispatch, NOT a fake) |
| TUB inputs consumed at frame synthesis | — | **B (honest, surfaced)** | `build_snerv_official_tub_input_prune_report` docstring `archive.py:1838-1842`: "current official receiver path does NOT consume `inputs.tub.*` during frame synthesis"; DROP_OR_REIFY gate 1919-1956 with real byte-mutation no-op proof (Catalog #105/#139) |
| Source-forward parity to real official torch+weights | — | **B (honest, fail-closed)** | `OFFICIAL_SNERV_MFU_NUMERIC_PARITY_BLOCKERS` `official_mfu.py:38-41` (`official_weight_tensor_mapping_not_loaded`); `OFFICIAL_SNERV_T_TUB_SOURCE_FORWARD_BLOCKERS` `official_tub.py:40-44`; proof-status validator `archive.py:5311-5400` REFUSES `complete` without a real numerical action-effect proof; authority-claim guard `_validate_official_source_forward_authority_claims` 5403+ |
| Torch oracle for the official primitives | byte-stable parity | **B (partial)** | no `forward_torch` *method* on MFU/HFR; the torch leg lives in `tests/test_official_mfu_source_parity.py`. That test SKIPS if the official checkout is absent (line 34) and `test_local_receiver_safe_mfu_falsifies_official_mfu_parity` (line 106) EXPLICITLY proves the local kernels are NOT bit-parity with official — an honest falsification, not a fake parity claim |

**Official MFU/HFR/TUB verdict: REAL-AND-COMPLETE numeric primitives + honest class-B source-forward/parity gaps.** Not fake. The conv math is genuine and runs; the gaps (TUB-input non-consumption, no-weights-loaded parity, no source-forward replay) are all surfaced by fail-closed blockers and self-audit reports.

### 2.3 Codebook / LF payload (`lf_payload_codec.py`, `joint_lf_hf_codebook.py`, `allocation.py`)

| Component | Claim | Verdict | Evidence |
|---|---|---|---|
| Store-LF / generate-HF paradigm | the Z8-disease cure | **A** | `__init__.py:2-13`; landing-memo "HF-generation decoder … FORK_PRINCIPLED" |
| L∞ pose-Fisher LF allocator | §7-proven allocator pushed to LF domain | **A (ADOPT)** | `allocation.py` `allocate_lf_linf`/`push_pixel_saliency_to_lf`; memo "L∞ allocator ADOPT_CANONICAL" |
| LF quant + step-map codec | real quantization | **RC** | `quantize_lf`/`dequantize_lf` (carrier); `encode_step_maps` (`tac.analysis.snerv_step_map_coder`) |
| Joint LF/HF codebook | present | **RC (scaffold, byte-pressure not yet stressed)** | `joint_lf_hf_codebook.py` exists; not the at-scale-trained path; honest |

### 2.4 Archive → inflate → parse-back full stack (`archive.py`, `inflate.py`, `section_value.py`, `receiver_proof.py`, `archive_candidate.py`)

| Component | Claim | Verdict | Evidence |
|---|---|---|---|
| SNAR1 grammar | magic + len-prefixed JSON header (offsets+sha256) + concatenated section blobs | **RC** | `pack_snerv_archive` `archive.py:1218-1292` (real monolithic length-prefixed grammar, PR95-L20-class) |
| SNAR2 grammar | compact binary header (byte-minimal) | **A (byte-save)** | `pack_snerv_archive_snar2` 1295+; docstring "removes the human-readable outer JSON" |
| `decode_snerv_archive_frames` (inflate entry) | dual-mode dispatch | **RC** | `archive.py:1639-1648` → `DecodedSnervArchive.decode_frames` 433 → `decode_snerv_archive_frames_from_decoded` 2709; dispatch on `is_official_mfu_hfr_tub_decoder_payload` `archive.py:2545` (official conv path) ELSE path-A linear `decode_frame` 2585-2649 |
| Official inflate frame synth | conv MFU LL + HFR HF → Haar synth → frames | **RC** | `_decode_official_mfu_hfr_tub_selected_frames` 3602-3637 builds real `OfficialMfuHfrTubReceiverPayload` + `.decode_frames()` 726-777; `_official_mfu_hfr_frame_planes` 998-1049 combines LL+HF via real `idwt2_multilevel` |
| Byte-close | archive bytes actually consumed | **RC** | LZMA decompress + sha256 of compressed + raw tensors validated `archive.py:3586-3595`; `pack`→`unpack`→`decode` roundtrip in `receiver_proof.py:141-186` |
| No-op detector (would frames change if a byte changed?) | proven | **RC** | `build_snerv_archive_payload_bitflip_falsification` `archive.py:1665-1728+` flips one section bit, repacks, re-renders, asserts frames change; TUB-prune report uses it to PROVE TUB bytes non-causal `archive.py:1906-1910` |
| inflate.py numpy-portable + scorer-free + torch-free | contest runtime | **RC** | `inflate.py:1-8` (NumPy-only); path-traversal guards 191-209; bilinear→camera HW 152-188; reads `x` or `0.bin` fail-closed 132-149 |
| Receiver proof | reconstructs frames from archive-visible state | **RC, honestly-scoped TOY** | `receiver_proof.py:76-212` real decode-vs-direct check (`max_abs_diff`, `np.allclose atol=0`); blockers `toy_receiver_proof_not_full_600_pair_replay`, `not_packaged_as_contest_archive_zip` (63-67) — honest TOY scope, NOT a full-600 claim |
| Section neutralization | receiver-valid section cuts | **RC** | `section_value.py:42-147` real repack + receiver-decode verify; `FALSE_AUTHORITY` stamped 146 |
| Archive-bound candidate package | builds `archive.zip` + runtime + proof | **RC, false-authority** | `archive_candidate.py:82-183`; member-name `x`; blockers incl `snerv_packet_not_full_600_pairs`, `paired_contest_cpu_cuda_auth_eval_missing` 126-130; `score_claim=False` etc. 170-172 |

**Archive/inflate/parse-back verdict: REAL-AND-COMPLETE and byte-closed.** The official conv decode path IS reachable from contest inflate (data-driven by the decoder_payload schema), the bytes ARE consumed, and there is a real no-op/bit-flip falsification surface. No hidden sidecars, no scorer load at inflate time.

### 2.5 Score-aware training loss (the B1 binding defect)

| Component | Claim | Verdict | Evidence |
|---|---|---|---|
| Shared MLX harness base term | reconstruction MSE | **RC math / B1 default** | `loss.py:3-8` docstring; `score_aware_loss(recon_weight=1.0)` 3099; `recon = mse_0 + mse_1` 3174-3176; `total = recon_weight*recon_stage_weight*recon` 3201 |
| SegNet/PoseNet distill weights default | OFF | **B1** | `bundle.py:519 distillation_weight=0.0`, `527 segnet_direct_live_distillation_weight=0.0`, `540 pose_distillation_weight=0.0`, `541 pose_direct_live_distillation_weight=0.0`; SNeRV policy `mlx_native_train_export.py:501-515` all `=0.0` |
| Scorer terms added only if weight>0 | opt-in | **B1** | `loss.py:3204` (`if bundle.scorer_input_distribution_guard_weight > 0.0`), and the analogous `if … > 0.0` gates for every scorer term |
| ep22399 long run loss config | what actually trained | **B1 (decisive receipt)** | `…/snerv_epoch22399_full_video_mlx_feedback_20260604T004900Z/nerv_full_video_mlx_scorer_feedback.json`: `observed_segnet_distillation_weight=None`, `recommended_segnet_distillation_weight=None`, `avg_segnet_dist=0.7115`, `avg_posenet_dist=163.19`, `nonrate_score_estimate=111.54` `[macOS-MLX research-signal]` |
| eval_roundtrip + differentiable YUV6 | NON-NEGOTIABLE present | **RC (available)** | `loss.py:3168-3169` `_apply_eval_roundtrip_ste_nhwc01` applied to both frames; direct-live SegNet/PoseNet terms exist (`mlx_native_train_export.py:497-616`) and CAN backprop through the frozen scorer |
| Surrogate-vs-authority | seg via learnable student head vs frozen SegNet | **RC, by design** | `loss.py:10-18` "student is a learnable head on the decoded frame … KL→decoded→renderer"; the DIRECT-live terms (`segnet_direct_live_*`) backprop through the FROZEN scorer — both available, both default-off |
| Real-frozen-scorer QAT loop | exists but wired to path A | **RC / B1 cross-wiring** | `scorer_loop_decoder_qat.py:2-9` (real exact scorers, perturbative QAT, "intentionally tiny … false-authority"); imports `load_score_exact_scorers`/`compute_s_seg_flip_risk` (24-30) — but operates on the path-A linear decoder, not the path-B conv trainer |

**Score-aware loss verdict: the loss MATH is REAL-AND-COMPLETE; the at-scale TRAINING CONFIG is class-B1 (scorer-off recon-MSE default).** This is the score-binding defect. It is an orchestration artifact (a default config trained the long run), not a justified adaptation (no memo defends it) and not fake (the knobs exist and work).

### 2.6 HF-generation path A (the B2 degeneracy)

| Component | Claim | Verdict | Evidence |
|---|---|---|---|
| Path-A HF generator | generate HF from LF | **A (fork) / B2 (linearity)** | fork is `FORK_PRINCIPLED` (landing memo); but the readout is LINEAR `np.einsum("...i,i->...", feats, kernel)` `carrier.py:837-843` — a fixed linear map of LF features. The memo justifies "generate HF," NOT "with a linear filter." A linear function of the LF approximation cannot synthesize boundary HF (it is approximately a sharpening kernel). Unexamined degeneracy = class B2 |
| Path-A fit | ridge least-squares | **RC** | `fit_hf_decoder_least_squares`/`fit_hf_decoder_weighted_least_squares` (carrier) — genuine LS fit, but of a linear map |

### 2.7 MLX↔PyTorch parity / export (the B3 blocker)

| Component | Claim | Verdict | Evidence |
|---|---|---|---|
| MLX forwards for official primitives | parity surface | **RC** | `forward_mlx` on conv/RB/MFU/HFR (`official_mfu.py:275-291,326-394,516+`; `official_hfr.py:133-216`) |
| Native MLX → official payload export | binds trained weights to archive | **B3 (honest blocker)** | `official_mfu_hfr_tub_export_blockers` `carrier.py:312-316`: `native_mlx_export_not_bound_to_official_payload`, `weight_mapping_missing`, `source_forward_replay_missing` |
| MLX-trained conv weights → scorable archive | — | **B3 (NOT closed)** | the at-scale MLX official-conv weights are not yet mapped into the official decoder payload; only path-A linear archives byte-close end-to-end today |

**This is the SAME failure-mode-CLASS as HiNeRV's grid-PE (started-but-parked-unproven export), but SNeRV is FURTHER along:** the official conv decode path IS reachable from inflate (HiNeRV's grid-PE forward parity was `official_core_forward_parity_proven=False`); SNeRV's gap is specifically the trained-MLX-weights → official-payload binding, not the decode-path existence.

---

## 3. Full-stack integrity verdict (the chain, end to end)

| Stage | Status | Evidence |
|---|---|---|
| Archive byte-close | **PASS** | `pack_snerv_archive` len-prefixed + sha256 `archive.py:1218-1292`; LZMA + sha256 validation on decode 3586-3595 |
| inflate reads + USES bytes (no-op?) | **PASS (consumed)** | dual-mode decode `archive.py:2545`; bit-flip falsification `archive.py:1665+`; TUB-prune no-op proof 1906-1910 |
| Parse-back survives | **PASS** | `unpack_snerv_archive`→`decode_frames` roundtrip; `receiver_proof.py` `max_abs_diff` check |
| inflate numpy-portable + budget + scorer-free | **PASS** | `inflate.py:1-8` NumPy-only, torch-free, scorer-free; bilinear to camera HW |
| No hidden sidecars / scorer at inflate | **PASS** | inflate imports only `decode_snerv_archive_frames`; receiver "NEVER loads the scorer" (`__init__.py:21`) |
| Gradient-reachable training | **PASS (available)** | eval_roundtrip STE + differentiable-scorer terms exist (`loss.py:3168`, `mlx_native_train_export.py:497-616`) |
| Real-scorer objective drives the AT-SCALE trainer | **FAIL (B1)** | ep22399 `observed_segnet_distillation_weight=None`; harness defaults 0.0 |
| At-scale conv weights → byte-closed official archive | **FAIL (B3)** | `carrier.py:312-316` export blockers |
| **End-to-end: real-teacher training → scorable official archive** | **BLOCKED** | B1 (loss) + B3 (export) both open |
| End-to-end: path-A linear → scorable archive | **PASS but degenerate** | numpy-portable archive closes (`archive_candidate.py`), but path-A is the linear decoder (B2) |

**Net:** SNeRV does NOT currently emit a scorable contest archive from real-teacher training. The blockage is in the orchestration (which loss config trained the long run) and the export binding (MLX conv weights → official payload), NOT in the architecture, archive grammar, inflate runtime, or parse-back — all of which are real and byte-closed.

---

## 4. The HiNeRV-precedent comparison (the operator's framing)

| Dimension | HiNeRV (precedent) | SNeRV (this audit) |
|---|---|---|
| Base architecture honesty | "L0 SKETCH hierarchical 3-scale NeRV" (commit `7a004e5bd`) | honestly-labeled "PARTIAL (advisory)" from first commit `00f3af9a2` |
| Missing residual/skip | bilinear-skip never added (class B) | official conv path B HAS the residual skip-add (`official_mfu.py:317-324`) — SNeRV is BETTER here |
| Defining-feature OFF by default | grid-PE `use_hierarchical_feature_grid=False` | scorer distill weights default 0.0 (B1) — analogous "defining capability off by default," but it's the LOSS not the arch |
| Forward-parity proven | `official_core_forward_parity_proven=False` | official decode path reachable from inflate (further along); source-forward replay proven False but fixture replay bit-exact (`test_official_tub_source_forward_replay.py:31-44`) |
| Trained/dispatched before maturation | yes | yes — but the IMMATURITY is the loss config + export binding, NOT the architecture |
| Provenance class | B (incompleteness) | **B (incompleteness/orchestration), 0% C** |

**SNeRV's analog of HiNeRV's bilinear-skip incompleteness is NOT a missing architecture piece — it is the missing wiring of the existing real-scorer objective to the existing real conv decoder.** SNeRV is the most architecturally-mature NeRV vehicle in the fleet; its failure is the narrowest and the cheapest to fix.

---

## 5. The single binding fix (verified line-by-line, corrects sister G1)

The sister memo's G1 ("turn ON distill weights + anneal recon + skip_high='full'") is **directionally correct and I CONFIRM the binding lines**, with one correction:

**CONFIRMED binding lines:**
1. The scorer-off default is real and is the binding term: `bundle.py:519,527,540,541` + `mlx_native_train_export.py:501-515` (all `=0.0`) + `score_aware_loss` adds scorer terms only `if weight>0` (`loss.py:3204+`). Turning `segnet_direct_live_distillation_weight>0`, a class-balanced subcontrol (`segnet_direct_live_class_balanced_*`, `mlx_native_train_export.py:504-506`), and `pose_direct_live_distillation_weight>0`, and lowering `recon_weight`, is the binding change. **This is config-only — no architecture edit.**
2. `official_skip_high_mode='full'` is ALREADY the carrier default (`carrier.py:170`) — so the fix is "do NOT override it to a `*_mean` byte-saver" (`mlx_renderer.py:745-758`), rather than "change the default."

**CORRECTION to G1:** G1 implies this is fully config-only end-to-end. It is config-only **for the mechanism check**, but NOT for a SCORABLE archive: the at-scale official-conv path is export-blocked (B3, `carrier.py:312-316`). So the honest framing is:
- **G1a (cheapest, $0 local MLX, mechanism check):** flip the loss config on the official-MFU MLX trainer + keep skip='full'. Falsifiable prediction: `avg_segnet_dist` drops from 0.71 toward <0.2 within the same epoch budget, scored live through DistortionNet `[macOS-MLX research-signal]`. If d_seg stays ≈0.5–0.7 with scorer weights on + skip='full', then B1 is falsified and B2 (the linear path-A leaking in) or a deeper binding bug dominates → escalate.
- **G1b (required for PROMOTION, NOT $0):** land the `native_mlx → official decoder payload` weight-mapping export (B3) so the trained official-conv weights — not the path-A linear decoder — emit the byte-closed `archive.zip`. Until G1b, G1a is mechanism-only.

**Why this is the highest EV:** it tests the dominant defect (B1) on the most-faithful, already-built conv decoder with a config-only change, costs $0, and is falsifiable in d_seg. It does NOT require touching the architecture (which is real) or rebuilding anything.

Secondary (lower priority, per sister §4): G2 (add bilinear-skip+refine to the PyTorch-only fleet's `_UpBlock` — but SNeRV path B already has the skip, so this is for the OTHER vehicles); G4 (replace path-A's linear `einsum` with path-B's conv-HFR, retiring B2); the TUB DROP_OR_REIFY decision (`archive.py:1919` — drop the unconsumed TUB input bytes, or reify the temporal encoder/decoder weights).

---

## 6. Honest scope / limits

- **Verified by reading source (every file:line above):** DWT adjoint exactness + tests; the official conv MFU (ConvT+cat-skip+RB U-Net) / HFR (1x1→LeakyReLU→3x3 heads) / TUB (output_2 split/concat/shuffle) numeric primitives compute real math; the dual-mode inflate dispatch (official conv path reachable from contest inflate ELSE path-A linear); byte-close + LZMA/sha256 validation; the no-op bit-flip falsification + TUB-prune DROP_OR_REIFY report; the receiver proof's honest TOY blockers; the source-forward proof-status validator's fail-closed refusal of unproven authority claims; the score_aware_loss recon-MSE base with scorer weights defaulting 0.0; the path-A linear `einsum`; the carrier `official_skip_high_mode='full'` default + the mlx_renderer mean-collapse modes; the export blockers; the landing-memo canonical-vs-unique section; the 31 test files including source-parity + source-forward-replay tests.
- **Verified by telemetry (`[macOS-MLX research-signal]`, NOT a score):** ep22399 `avg_segnet_dist=0.7115`, `avg_posenet_dist=163.19`, `nonrate_score_estimate=111.54`, `observed_segnet_distillation_weight=None`, `recommended_segnet_distillation_weight=None` (run JSON cited in §2.5).
- **Verified by git forensics:** carrier landed honestly-labeled "PARTIAL (advisory)" from `00f3af9a2`/`5ec8489c6`; official primitives added `8aeda349a` with the source-authority guard `ea81e9aea` landed alongside; the layered lane registry (~20 SNeRV lanes, official primitives each separately registered L1) shows deliberate scoping, not a single rushed sketch.
- **INFERRED (mechanism argument, not measured here):** that flipping the loss config (G1a) will move d_seg (falsifiable hypothesis, not a measurement); that the linear path-A map cannot synthesize boundary HF (linear-algebra argument); that the conv path B has enough capacity to beat the mean-field once the objective rewards it.
- **NOT done (out of scope for a read-only audit):** no training, no GPU, no paid dispatch, no edits to any carrier source or `upstream/`. No byte-closed archive built. No `[contest-CPU]`/`[contest-CUDA]` score produced. I did not execute the official primitives (read-only); the conv math is verified by reading, not by running a forward.
- **Authority:** everything here is `[macOS-CPU advisory]` / `mechanism_update_eligible`. It updates next-experiment routing; it does NOT promote, rank, kill, or close any lane (CLAUDE.md "Meta-Lagrangian/Pareto solver" + "Forbidden premature KILL"). Per the Catalog #307 paradigm-vs-implementation distinction, B1/B2/B3 are IMPLEMENTATION-LEVEL findings on the current config/wiring; the SNeRV paradigm (store-LF / generate-HF with conv MFU/HFR/TUB) is intact.

### One-line headline for the operator

**SNeRV's stack is real and byte-closed (0% fake); its failure is a class-B orchestration artifact — the right architecture (official conv MFU/HFR/TUB with a residual skip) was trained by the wrong loss (the shared MLX harness's recon-MSE base with SegNet/PoseNet distill weights defaulting to 0.0, confirmed by ep22399's `observed_segnet_distillation_weight=None`), while the real-frozen-scorer objective sits wired to the tiny path-A linear decoder; the cheapest binding fix is the config-only G1a (turn the direct-live scorer weights on, keep `skip_high_mode='full'`) with the MLX→official-payload export binding (B3) deferred as the promotion gate.**
