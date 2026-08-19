# ddm_up1 — the pose axis was never blocked on CUDA. It was blocked on our own GT cache.

- **arm** `ddm_up1` (ns1 P3 / uncapped exact pose solve on the live pointer body)
- **date** 2026-08-19
- **axis** `[macOS-CPU advisory, frozen CPU-torch PoseNet]` · `score_claim=false` ·
  `promotable=false`. **Pointer UNMOVED** at contest-CUDA `0.15659459685822907`.
  This arm fired no Modal job, compiled no archive, and produced no exact row.
- **cost** $0. ~7 minutes of local compute total.
- **code** `experiments/ddm_up1_decode_axis_photometric_probe.py` (+27 tests)
- **store** `/Volumes/APDataStore/pact/ddm_up1/`

STORES CONSULTED: `.omx/state/canonical_frontier_pointer.json` (re-read at start) ·
`ddm_ps1u_uncapped_pose_solve_20260816.md` §1/§3/§4/§5b · `ddm_ps1u_r2_dual_axis_pose_verdict_20260816.md` ·
`ddm_pv1_pose_floor_and_admission_bar_20260816.md` §1/§2/§3b/§4 · `ddm_pi2_pose_axis_attribution_20260816.md` ·
`ddm_pu3_falsified_premise_propagation_20260816.md` · `ddm_t1h_pose_coeff_resolve_headroom_20260817.md` §11 ·
`ddm_qs5_verdict_and_no_toy_enforcement_20260813.md` · `ddm_to1_tail_override_twelfth_move_20260819.md` ·
`ddm_rr4_cuda_prob_reencode_20260817.md` · `.omx/state/canonical_task_status.jsonl` ·
`.omx/state/active_lane_dispatch_claims.md` ·
`/Volumes/APDataStore/pact/ddm_to1/{t4_row_r1/MODAL_REMOTE_RESULT.json,advisory/attempt_0002/**,generations/to1_tail_override_r1/**}` ·
`/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_dali.pt`.

---

## ANSWER FIRST

1. **My charter's premise was dead on arrival, twice over, and I did not re-run it.**
   #850's relinearization cap was closed by `ddm_pj2` (08-03) and `ddm_ps1u` (08-16);
   uncapping is worth **0.1549%** of the achievable reduction, not "13–23%/iter".
   That figure traces to a **n=1 `STALE_REHEARSAL`** and has now propagated into
   **eight** charters including mine (`ddm_pu3:64-99`). The pose-carrier *target* was
   then tried twice on the paid axis and **REFUSED twice**: `ps1u` r2 at **+1.686e-02 S**
   and `t1h` at **+0.012557 S**.
2. **NEW — the pose axis has an exact $0 local instrument, and has had one all along.**
   On the pointer's cpu-decoded frames, scored against the **DALI-lineage** GT, n600
   d_pose is **7.769511e-06** against the T4 row's **7.77e-06** — **ratio 0.9999**, in
   **140 s**. Every prior "only a CUDA-side solve could measure this" blocker
   (`ddm_pv1:§2`, `ddm_pu3:262-265`) is **dissolved**, not deferred.
3. **NEW — the 19–24× advisory-vs-contest pose gap is 100% GT-lineage, 0% decode.**
   Same frames, same frozen scorer, only the GT cache changed: **23.74×** at n=64.
   This confirms `ddm_pi2` on the live pointer body, which pi2 itself could not do
   (it predated to1 and disclaimed its own frame-identity premise at `:180-183`).
4. **NEW — the device-dependent decode is confirmed on a SECOND lineage (the live
   pointer body) and MECHANISM-LOCALIZED for the first time** to two lines:
   `cpr1/inflate.py:312` and `:335`. It is a real deterministic-decode compliance
   defect. It is **not** a score lever — bounded here at ≤ the 8dp report resolution.
5. **FALSIFIED (my own hypothesis, killed in 88 s)** — the gap is not a global
   photometric shift. d_pose is minimised at offset 0.
6. **REOPENED — `ddm_pv1`'s pose-carrier closure is lineage-confounded.** It compared
   an AV-GT converged floor (`1.285917e-05`) against a DALI-GT shipping value
   (`6.88e-06`) and concluded "1.87× worse, no headroom". Those are two different
   objects. The comparison must be redone on one instrument. §6.

---

## 1. The two decodes of the live pointer body — both re-derived, not inherited

Archive `50e561454b23026d3870f056747e848a49bd5f2b1e23930155d1281aeee91927`, 176,420 B.

| axis | inflate device | raw `0.raw` sha256 | bytes |
|---|---|---|---|
| local advisory | `cpu` | `ccbfa3327d0f2486f8a2d7970fe89c5d56302eb1e04714d05eabff52278f1f9d` | 3,662,409,600 |
| T4 row r1 | `cuda` (`policy=auto`) | `3c810cc4adff01a6783e8727f2cd7161e47d83693acc2aba7941b8ee7b115f6d` | 3,662,409,600 |

I hashed the CPU raw and the advisory's `archive.zip` myself (`shasum -a 256`); the
CUDA digest comes from the T4 run's own
`artifacts.inflated_outputs_manifest.json` inside
`/Volumes/APDataStore/pact/ddm_to1/t4_row_r1/MODAL_REMOTE_RESULT.json`
(`provenance_device = cuda`, `expected_archive_sha256 = 50e56145…`). One archive,
identical geometry, two decodes. **`DEVICE_DEPENDENT_DECODE_CONFIRMED`, second
lineage** (`ddm_ps1u` proved it on hv1/e480b; this is ck2/F26/RX1M — the body that
*is* the frontier). `verdict_scope: formulation`.

### 1b. Mechanism — localized for the first time

`inflate.py:56` picks the device: `device_name = "cuda" if torch.cuda.is_available()
else "cpu"`. That is the forbidden device-selection default from CLAUDE.md, shipped.

Across the whole 33-file runtime, exactly **two** lines let `device.type` change the
computation rather than the placement:

    cpr1/inflate.py:312    semantic_batch = 8  if device.type == "cuda" else 1
    cpr1/inflate.py:335    pose_batch     = 64 if device.type == "cuda" else 1

CPU renders batch-1, CUDA batch-8/64 — different `GroupNorm` reductions, different
`einsum` contraction shapes, different `interpolate` kernel selection, all funnelled
through a `clamp(0,255).round()` at `cpr1/inflate.py:323-324` and `:347-348` that
turns any ULP into a byte. This is memory's
`batch_shape_is_part_of_the_forward_instrument` law firing literally. The entropy
half is *proven* device-exact (`ddm_rr4:91-105`, identical
`corrected_quantized_logit` / `corrected_cdf_input` digests), and to1's own receipts
show the decoded token field is `9ba2e52b…` on both sides — so the divergence is
downstream of the tokens, inside `render_video`, exactly where these two lines sit.

**Cheapest confirming experiment, needing no CUDA host:** re-run the CPU inflate with
`semantic_batch`/`pose_batch` forced to 8/64 and compare the raw digest to
`3c810cc4…`. A match makes the cure a two-line constant in FREE bytes. Not run here
(≈16 min decode); it is the single cheapest open decode-axis measurement and it is
listed as fire-order 1 in §7.

---

## 2. FALSIFIED — the gap is not a global photometric shift

My charter reasoned from the asymmetry (pose 19.08×, seg 1.43×) that a small uniform
photometric offset was the cause, because pose is a global readout and seg is a local
argmax. I built the sweep to test it and it died. n=64 seeded-random (never a prefix),
realized on the uint8 lattice:

| offset (LSB) | mean d_pose | ratio vs base |
|---:|---:|---:|
| −4 | 4.410097e-04 | 3.6153 |
| −2 | 1.856016e-04 | 1.5215 |
| −1 | 1.334004e-04 | 1.0936 |
| **0** | **1.219851e-04** | **1.0000** |
| +1 | 1.440857e-04 | 1.1812 |
| +2 | 1.955167e-04 | 1.6028 |
| +4 | 3.656592e-04 | 2.9976 |

**Offset 0 is the optimum.** A sub-LSB float sweep puts the continuous optimum at
δ*≈−0.2 for a −0.45% gain (ΔS_pose −7.9e-05) — real but small, measured on the CPU
object, and **not transferable**: the shipping object sits 15.7× lower in d_pose and
transporting a constant across that gap is precisely the `qs4`/`ps1u` cross-regime
move that already bought a refused row. I am not proposing it.

Decisive arithmetic against my own hypothesis: ±4 LSB moves d_pose only 3.6× and in
the *wrong* direction, so a rounding-scale difference **cannot** manufacture the
15.7× *improvement* the CUDA object appeared to have. Something structural had to be
wrong with the comparison — §3.

---

## 3. CONFIRMED — the gap is the GT decode lineage, and the instrument is exact

Same cpu-decoded frames, same frozen CPU-torch PoseNet, same pairs. **Only the GT
cache changes.**

| n | GT lineage | mean d_pose | note |
|---:|---|---:|---|
| 64 | AV / PyAV (`gt_n600.npz`) | 1.219851e-04 | |
| 64 | DALI (`gt_cache_dali.pt`) | 5.137472e-06 | **23.74× lower** |
| **600** | **DALI** | **7.769511e-06** | vs T4 `7.77e-06` → **0.9999×** |

The n600 DALI reading rounds to exactly the 8dp value the T4 row reports, and its
pose contribution `sqrt(10·d_pose)` is `0.008814` against the T4's `0.008814760`.

**Three things follow.**

1. **The cpu-decoded frames are pose-equivalent to the cuda-decoded frames.** The raw
   digests differ (§1) but the pose consequence is at or below the 8dp report
   resolution. The decode defect is a *compliance* defect, not a score lever —
   consistent with `pi2`'s ≤0.0040% / ≤3.4e-06 S bound on our frames.
2. **An exact local pose instrument exists**: n600 in ~140 s at $0, reproducing
   contest-CUDA authority at 0.9999×, at *better* precision than the T4's 8dp report.
3. **Both paid pose refusals are explained.** `ps1u` r2 and `t1h` each solved against
   an AV-lineage GT, removing ~23.7× of phantom error that does not exist on the
   shipping axis, then paying real error. `t1h`'s own postmortem reached the same
   place from the other side ("the oracle removed 1.234e-4 of phantom energy and paid
   3.657e-5 of REAL energy", `:600-601`); this is the direct receipt for why.

---

## 4. What I did NOT do, and why

- **No uncapped solve.** Re-running it would have been ~100% duplicative of `ps1u`
  (n=50, 49/50 terminating on a convergence *proof*, cap 0/50) and its answer is
  already banked. Spending the unit on the instrument instead is what produced §3.
- **No candidate, no byte-close, no Modal fire.** MAIN owns the T4 slot, and there is
  no candidate worth a row until a solve runs on the *correct* instrument (§6).
- **No δ*=−0.2 proposal.** §2: cross-regime constant transfer, known-fatal.
- **No MPS decode.** Attempted and correctly refused by the receiver
  (`f26_inflate.py:144`, fail-closed to `cpu`/`cuda`). The launch guard also refused
  my first hand-rolled `nohup` detach; both refusals were right.

---

## 5. My own round-1 adversarial review

1. **Is the 0.9999× a coincidence of aggregation?** It is a mean over all 600 pairs
   (mean-of-d_pose, never mean-of-ratios — `rt1`: the sign flips). n64 DALI read
   0.66× of the n600 value, so the *sample* is noisy while the *population* is exact;
   only the n600 figure is quoted as the fidelity claim.
2. **Am I certain the DALI cache is the contest lineage and not a lucky third
   object?** Not from first principles — it is inferred from `pi2:105-106` plus the
   0.9999× agreement at n600. Two independent routes agreeing to 4 s.f. is strong,
   but the *structural* proof (that `DaliVideoDataset` is what `evaluate.py` builds)
   is not re-derived here. Stated as a limit, and it is fire-order 2.
3. **Does §1 contradict §3?** No, and the tension is the point: the decodes genuinely
   differ in bytes *and* agree in pose. Both are measured; neither is inferred from
   the other.
4. **Is my instrument's GT the same object my seg numbers would need?** Unknown — I
   measured pose only. `pi2:253` warns seg has its own 1.4425× lineage cost. Any seg
   claim on this instrument is unwarranted until separately checked.
5. **Did I verify the frames I scored are the pointer's?** Yes — I hashed
   `advisory/attempt_0002/work/archive.zip` to `50e56145…` and its `0.raw` to
   `ccbfa332…` myself before scoring.

---

## 6. REOPENED — pv1's pose-carrier closure is lineage-confounded

`ddm_pv1:§2` closed the pose-carrier route with: the uncapped solve converges to
`1.285917e-05`, which is **1.87× worse** than the shipping `6.88e-06`, therefore "the
pose carrier has no measured headroom to transport onto the shipping axis."

`1.285917e-05` is an **AV-lineage** number (it aggregates `ps1u`'s shards, which
scored against the AV GT). `6.88e-06` is a **DALI-lineage** number (a contest row).
Per §3 those instruments differ by 23.74× on identical frames, so the ratio is not a
ratio of anything. pv1 flagged the object-mismatch honestly at its own `:§2` ("the
CUDA-decode object is a *different object* … its floor is genuinely unmeasured") but
attributed it to the decode; §3 shows the real split is the GT cache, and that one is
fixable at $0.

**The pose-carrier route is therefore REOPENED** — not on an argument that it will
work, but because the instrument that closed it was measuring a different object. Its
verdict is `UNMEASURED`, not `NEGATIVE`. What is *not* reopened: the relinearization
cap (closed, 0.1549%, three solvers) and the post-hoc/stored-correction family.

---

## 7. Fire-order

1. **Force `semantic_batch=8` / `pose_batch=64` on a CPU inflate; diff the raw digest
   against `3c810cc4…`.** No CUDA host needed. A match reduces the deterministic-decode
   defect to a two-line free-byte constant. ~16 min, $0.
2. **Structurally confirm the DALI cache is `evaluate.py`'s lineage** (re-derive from
   `DaliVideoDataset`, not from the 0.9999× agreement). Cheap, and it is what lets §3
   be cited as authority rather than as a strong coincidence.
3. **Re-run the uncapped per-pair pose solve against the DALI GT** — the first pose
   solve ever pointed at the object that ships. `ps1u`'s actuator
   (`ddm_qs1_frame0_schur_coupled_solve.py:490`, already `while True`) plus this
   module's GT loader is most of the build. Only this answers "is there pose headroom".
4. **Point every live pose instrument at the DALI lineage, fail-closed.** `pi2`'s
   fire-order 2 (a GT-lineage gate) is still unbuilt; §3 is the second time the same
   bug has cost a paid row. `rc4`, `ra3`, `hg1`, `wd3` are the named consumers.
5. **Own the decode axis.** `grep` over `canonical_task_status.jsonl` returns **zero**
   rows for it; the only lane is terminal and its mechanism was later falsified. A
   confirmed compliance defect on the shipping body has no owner.

## 8. Retained payload

`/Volumes/APDataStore/pact/ddm_up1/` — `sweep_realized_n64/`, `sweep_float_n64/`,
`dali_gt_baseline_n64/`, `dali_gt_baseline_n600/`, `smoke_baseline/`, each with
`rows.jsonl` (per-pair d_pose), `SUMMARY.json`, and
`retained/pose_vectors_offset_*.npy` — **every pose vector kept, not just the
scalars**. `mps_decode/` holds the refused diagnostic runtime copy and its launch
receipt.
