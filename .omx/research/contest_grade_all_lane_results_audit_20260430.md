# Contest-Grade All-Lane Results Audit

**Date:** 2026-04-30T12:53:36Z  
**Scope:** all lane results currently reported or collected in `experiments/results/`, `submissions/`, `reports/`, `.omx/research/`, `.ralph/run_log.md`, and the Claude project memories.  
**Purpose:** enforce the user standard that every strategic result must be contest grade. Anything else is advisory, empirical-only, or unreproducible until re-run.

---

## 1. Contest-Grade Standard

A lane score is **Grade A score-grade** only if all of these hold:

1. Exact archive artifact is preserved locally, with byte size matching the eval report.
2. Archive SHA-256 matches `provenance.archive_sha256` when the report provides it.
3. Eval used CUDA (`provenance.device == "cuda"` or equivalent documented CUDA runner).
4. Eval used the contest evaluator path or a report with equivalent component fields.
5. `n_samples == 600` or equivalent full public-test pair count.
6. Score recomputes from:

```text
score = 100 * seg_dist + sqrt(10 * pose_dist) + 25 * archive_bytes / 37,545,489
```

Small deltas up to about `0.005` are accepted when the report stores a rounded display score.

**Grade A++ 1:1 contest-grade** additionally requires:

1. Archive manifest is clean: no unexpected files, no housekeeping files, no external score-relevant artifacts.
2. `inflate.sh` and `config.env` provenance are captured; `PYTHON_INFLATE=renderer` or the lane-specific compliant dispatch is recorded.
3. The eval path is `archive.zip -> inflate.sh -> upstream/evaluate.py`; no scorer patching and no local renderer shortcut.
4. Hardware is T4-matched or officially equivalent to the contest judge for neural forward behavior.
5. Inflate duration is inside the 30-minute contest budget with strict raw-output cardinality and byte-size validation.

Current controlling frontier: Lane G v3 PFP16 now has Grade A++ evidence from Lightning AI Tesla T4 with `gpu_t4_match=true`. Older RTX 4090 PFP16 evidence remains a corroborating Grade A run only.

**Grade B diagnostic CUDA** means CUDA and component math are present, but artifact custody or provenance is incomplete. It can explain a failure mode, but it is not sufficient to promote, ship, or anchor a new floor.

**Everything else is not contest-grade**: CPU, MPS, local proxy, byte proxy, smoke test, memory-only score, crash trace, failed inflate, prediction band, or exact archive missing.

---

## 2. Grade A / A++ Score Ledger

These are the only fully reproducible exact CUDA score-grade lane results found in this pass. PFP16 has since passed the stricter Grade A++ 1:1 gate on T4/equivalent hardware; the remaining rows are Grade A score-grade unless separately upgraded.

| Lane / artifact | Score field | Recomputed | Pose | Seg | Bytes | Archive SHA status | Verdict |
|---|---:|---:|---:|---:|---:|---|---|
| Lane G v3 PFP16 `experiments/results/lane_g_v3_pfp16/pfp16_a_plus_plus_t4_20260430T1620Z_codex/contest_auth_eval.json` | 1.04 | 1.043988 | 0.00346442 | 0.00400656 | 686,635 | matches `0af839...ded7f` | **Current A++ authoritative frontier** |
| Lane G v3 `experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip` | 1.05 | 1.048866 | 0.00345458 | 0.00400846 | 694,074 | matches `9b20...6870b` | Previous frontier |
| Lane A `experiments/results/lane_a_landed/archive_lane_a.zip` | 1.15 | 1.145767 | 0.00496876 | 0.00460724 | 694,045 | matches `a992...8e84` | Valid historical baseline |
| Lane M-V2 `experiments/results/lane_m_v2_landed/archive_lane_m_v2.zip` | 1.84 | 1.839523 | 0.07602696 | 0.00505453 | 694,044 | matches `88cd...73aa` | Valid regression; config/architecture lesson |
| Lane H CRF56 `experiments/results/lane_h_crf56/archive.zip` | 3.20 | 3.203149 | 0.51792711 | 0.00554966 | 559,250 | matches `4829...900f5` | Valid regression |
| Baseline CRF50 `submissions/baseline_dilated_h64_0_90/archive_fullres_crf50.zip` | 2.29 | 2.293579 | 0.24758595 | 0.00258082 | 693,857 | matches `dcdb...ca30b` | Valid baseline, not frontier |

**Mathematical check:** all six rows satisfy the scoring equation within report rounding tolerance.

### Archive manifest check

All six preserved Grade A archives have clean three-member manifests. PFP16 uses `renderer.bin`, `masks.mkv`, and `optimized_poses.bin`; the legacy renderer archives use `renderer.bin`, `masks.mkv`, and `optimized_poses.pt`. No `.DS_Store`, `__MACOSX`, stale debug payloads, sidecar checkpoints, or external postfilter artifacts are present in these archives.

| Lane | Manifest | External neural artifact dependency | Hardware note |
|---|---|---|---|
| Lane G v3 PFP16 | clean 3-file renderer archive | none found; FP16 pose payload is inside archive | Lightning AI Tesla T4, `gpu_t4_match=true`; log proves inflate budget |
| Lane G v3 | clean 3-file renderer archive | none found | CUDA RTX 4090, not T4-matched |
| Lane A | clean 3-file renderer archive | none found | CUDA RTX 4090, not T4-matched |
| Lane M-V2 | clean 3-file renderer archive | none found | CUDA RTX 4090, not T4-matched |
| Lane H CRF56 | clean 3-file renderer archive | none found | CUDA RTX 4090, not T4-matched |
| Baseline CRF50 | clean 3-file renderer archive | none found | CUDA RTX 4090, not T4-matched |

**Compliance implication:** PFP16 may be called the current 1:1 contest-grade baseline after final bundle review. The other preserved archives remain structurally clean Grade A score-grade rows, not A++, until T4/equivalent hardware and inflate-budget evidence are attached to their exact archive SHA.

---

## 3. Grade B Diagnostic CUDA

These results are useful but fail the stricter artifact-custody standard.

| Lane / source | Score | Why not Grade A | Allowed use |
|---|---:|---|---|
| Lane G v3 + Ω-W-V2 stack, `experiments/results/lane_g_v3_omega_w_v2_stack_landed/contest_auth_eval.json` | 1.07 | CUDA provenance and math are present, but the exact archive `eba8e436...` is not present locally. | Diagnostic only: the PoseNet regression finding is credible, but do not promote or stack without recovering/rebuilding the exact archive. |
| Modal re-eval of Lane G v3, `experiments/results/modal_auth_eval_9b20bdfca246.json` | 1.04 | Minimal JSON: no device/provenance/archive SHA fields in the file, though filename ties it to Lane G v3 SHA. | Corroborates Lane G v3 within CUDA noise; not an independent anchor. |
| SHIRAZ v4, `experiments/results/shiraz_v4_archive/auth_eval_with_poses.json` | 2.700102 | Report says `device=cuda` and math recomputes, but it is not the canonical `contest_auth_eval.py` schema and lacks exact archive SHA/provenance. | Historical CUDA diagnostic only. |
| Lane B 2.40, `.ralph/run_log.md` | 2.40 | Run-log narrative only; no exact preserved report/archive pair found in this pass. | Historical lesson about proxy-auth gap only. |
| Lane M+N 2.35, memory `project_lane_mn_radial_zoom_negative_20260427.md` | 2.35 | Memory-only score; exact report/archive not found in this pass. | Historical diagnostic only. |
| Lane F / F-V2 2.73 / 1.79, memories and run log | 2.73 / 1.79 | Memory/run-log only in this pass; also hardware quantization disclosure says FP4 was simulated on 4090, not true hardware FP4. | Do not use as final FP4 theory. Use only as a warning about PoseNet sensitivity and simulated-quant methodology. |

---

## 4. Explicitly Non-Contest-Grade Results

These must be demoted in strategy docs unless rerun under Grade A rules.

### CPU advisory only

| Lane | Score | Source | Reason |
|---|---:|---|---|
| Lane GP v2 | 89.66 | `lane_lane_gp_v2_modal/.../contest_auth_eval.json` | `provenance.device == "cpu"` |
| Lane GP v3 | 89.67 | `lane_lane_gp_v3_modal/.../contest_auth_eval.json` | `provenance.device == "cpu"` |
| Lane MM v2 | 2.63 | `lane_lane_mm_v2_modal/.../contest_auth_eval.json` | `provenance.device == "cpu"` |
| Lane UNIWARD v7 | 53.61 | `lane_uniward_v7_modal/.../contest_auth_eval.json` | `provenance.device == "cpu"` plus known 48x64-mask catastrophe |
| Lane UNIWARD v8 | 1.14 | `lane_uniward_v8_modal/.../contest_auth_eval.json` | `provenance.device == "cpu"` plus NO-OP mask copy finding |

### MPS / local proxy only

| Result | Score | Source | Reason |
|---|---:|---|---|
| `true_auth_eval_round1.json` | 7.49434 | `experiments/results/true_auth_eval_round1.json` | `device == "mps"` |

### Repeated stale or non-provenance auth files

Many harvested `submissions/robust_current/auth_eval_renderer_fp4.json` files report the same `score=6.9431`, `bytes=375,548`, with no CUDA provenance. These are not lane results. They are stale or generic robust-current eval artifacts and must not be attributed to the harvested lane.

### Empirical byte/smoke-only lanes

| Lane | Evidence | Current grade |
|---|---|---|
| Lane 12 NeRV pre-eval report | `reports/lane_12_nerv_real_archive.json`: 94.4% byte saving, 2.003% argmax disagreement, CPU partial training | Superseded by exact negative eval for `jsonfix40`; keep only as a warning that byte/mask proxies were insufficient |
| Lane 12 NeRV `jsonfix40` exact eval | `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/contest_auth_eval.json`: recomputed `26.03719330455429`, PoseNet `49.77849960`, SegNet `0.03528685`, archive `296,478` bytes | Exact negative for this implementation; do not generalize to all alpha/mask compression |
| OWV3/Fisher Modal smoke | `experiments/results/lane_lane_g_v3_owv3_fisher_smoke_20260430_codex_modal/lane_g_v3_owv3_fisher_stack_results/build_provenance.json`: archive `912,971` bytes, `+218,897` vs Lane G v3, no exact eval | Suspicious negative smoke only; diagnose encoder overhead/config before method conclusions |
| Lane 17 IMP | `reports/lane_17_imp_real_archive.json`: 40.2% renderer-byte saving at cycle 9 | Empirical byte-only; no contest score |
| Lane 19 logit-margin | `reports/lane_19_logit_margin_local_smoke.json` | Smoke-only; no contest score |
| Lane 20 Ballé | `reports/lane_20_balle_real_archive.json` and recovered run: `STATIC_WINS_FALLBACK` | Empirical codec finding; no score improvement |
| Lane 8 multipass | `reports/lane_8_multipass_real_archive.json` | Offline byte-proxy only |

---

## 5. Required Retags / Corrections

1. **Lane G v3 PFP16 is now the Grade A++ frontier.** Use score `1.04` with recomputed value `1.043987524793892` from `experiments/results/lane_g_v3_pfp16/pfp16_a_plus_plus_t4_20260430T1620Z_codex/contest_auth_eval.json`; the earlier RTX 4090 run is superseded for contest-identity wording.
2. **Ω-W-V2 1.07 remains a CUDA diagnostic, not a fully reproducible Grade A artifact.** The PoseNet regression math is credible, but the exact archive must be recovered or rebuilt.
3. **The old 0.9001 baseline is retracted as an anchor.** Memory `project_baseline_0_9001_lost_archive_test.md` says the committed `archive_baseline_0_9001.zip` is actually the 53.6-scoring 48x64-mask archive, and the real 0.9001 archive was not preserved.
4. **All CPU/MPS results are advisory.** They cannot kill or promote lanes. This includes GP v2/v3, MM v2, UNIWARD v7/v8, and `true_auth_eval_round1`.
5. **All remaining byte-only reports must stay `[empirical:<path>]`.** IMP, Ballé, multipass, and logit-margin do not have contest scores yet. Lane 12 NeRV `jsonfix40` does have exact CUDA evidence now, but it retires that measured implementation/config only.
6. **PFP16 remote provenance parser fields are superseded.** The harvested `remote_provenance.json` says `contest_cuda_score=100.0`, `hard_kill_triggered=true`, and `lane_status=HARD_KILL_REGRESSION`; those fields are invalid legacy parser/adjudication output and must be ignored in favor of `contest_auth_eval.json`. The current adjudicator uses scoped `regression_triggered` / `REGRESSION_REVIEW_REQUIRED` wording.
7. **Memory-only historical CUDA claims need provenance recovery before ranking.** Lane B, M+N, F/F-V2, and SHIRAZ v4 can inform failure analysis but should not be placed in the same authoritative table as Grade A results unless exact archive/report custody is restored.

---

## 6. Scientific / Mathematical Conclusions

### Current verified score ordering

Using only Grade A artifacts:

1. **Lane G v3 PFP16:** 1.043988 recomputed, best verified and A++ contest-grade.
2. **Lane G v3:** 1.048866 recomputed.
3. **Lane A:** 1.145767 recomputed.
4. **Lane M-V2:** 1.839523 recomputed.
5. **Baseline CRF50:** 2.293579 recomputed.
6. **Lane H CRF56:** 3.203149 recomputed.

### Load-bearing quantitative deltas

Against Lane G v3:

- Ω-W-V2 diagnostic: rate improved by about `-0.034` score, but PoseNet worsened by about `+0.052`, net `+0.021`. This is enough to enforce PoseNet-sensitivity weighting for every renderer-weight codec.
- PFP16 actual rate gain: archive bytes dropped by `7,439`, worth `25 * 7439 / 37,545,489 = 0.004953` before tiny Pose/Seg drift. The measured recomputed score improvement versus Lane G v3 is about `0.004878`, and T4 CUDA validated the lane at `1.043987524793892`.
- Mask byte leverage: every `100,000` bytes saved is `25 * 100000 / 37,545,489 = 0.06659` score. This is why NeRV/mask payload remains the highest-EV rate lane, but it still needs SegNet/PoseNet contest validation.

### Kill/promote discipline

No lane may be marked killed or promoted by:

- proxy loss,
- local MPS,
- Modal CPU,
- byte savings without exact archive eval,
- memory-only scores,
- run-log-only scores,
- score reports lacking exact archive custody.

The narrow exception is pure byte/hash/round-trip claims, which can be local if no neural forward pass is in the dependency chain. Those claims must be tagged empirical, not contest-score.

---

## 7. Immediate Contest-Grade Work Queue

Priority order for converting promising lanes into Grade A score-grade and, when needed, Grade A++ 1:1:

1. **Freeze PFP16 A++ bundle:** attach final manifest, source/staged-tree provenance, log timing, and review notes to the exact archive SHA.
2. **PFP16 provenance/adjudication fix:** keep the parser bug-class fix enforced so stale `contest_cuda_score=100.0` and legacy `hard_kill_triggered=true` fields never override `contest_auth_eval.json`.
3. **Recover/rebuild Ω-W-V2 archive:** find or regenerate SHA `eba8e4360e6366ca10905b58e9ec1d12b3480c78b041548daa21da0c46c31625`; otherwise keep as diagnostic only.
4. **Lane 12 alpha redesign:** current NeRV `jsonfix40` is retired by exact-CUDA regression at `26.03719330455429`; redesign the alpha objective before any new NeRV/INR spend.
5. **Lane 17 IMP:** CUDA full cycle + exact archive + contest eval. Must include PoseNet regression guard.
6. **Lane 19 logit-margin:** A/B against Lane G v3 with exact archive eval.
7. **Lane 20 Ballé:** no Grade A rerun needed unless a non-static mode beats static in bytes before eval; current result is fallback/no-op.
8. **Retag all historical reports:** Grade A table only for preserved exact CUDA artifacts; move everything else to diagnostic/advisory sections.

---

## 8. 1:1 Contest Compliance Gate

Every deployment candidate must pass this exact gate before it can be called production-hardened or contest-ready.

| Gate | Requirement | Failure action |
|---|---|---|
| C1 artifact custody | Exact `archive.zip` preserved locally and remote path recorded | No promotion; rebuild or recover |
| C2 hash identity | SHA-256 matches report provenance and any copied archive | No promotion; identify artifact drift |
| C3 manifest | Known members only; no resource forks, hidden files, debug files, or unscored artifacts | Repack with canonical archive builder |
| C4 payload closure | Every neural/runtime payload used at inflate is inside `archive.zip` or is fixed contest code, never a local sidecar | Non-compliant until repacked |
| C5 inflate dispatch | `inflate.sh` + `config.env` provenance captured; correct lane dispatch set | Non-compliant until rerun |
| C6 strict inflate | All expected `.raw` files produced with exact byte size | Failed eval, no score |
| C7 upstream scorer | `upstream/evaluate.py` invoked directly; no patched scorer and no renderer shortcut | Diagnostic only |
| C8 device | CUDA for score-grade; T4/equivalent for 1:1 contest-grade | Demote label to advisory/score-grade |
| C9 budget | Inflate completes inside 1800 seconds on contest-equivalent hardware | Not shippable |
| C10 math | Score recomputes from components and `n_samples=600` | Invalid report |
| C11 report custody | `contest_auth_eval.json`, `provenance.json`, archive manifest, and logs stored together | Do not rank |
| C12 adversarial review | Three consecutive clean review passes for strategic promotion or kill | Continue review |

**Promotion rule:** a lane may guide next experiments at Grade B, but cannot replace the frontier, kill a competing lane, or anchor Shannon-floor math unless it reaches Grade A score-grade. A final submission candidate must reach Grade A++.

---

## 9. Implementation Pivot Rules

The shortest path to the Shannon floor should run in this order:

1. **Evidence hygiene first:** fix PFP16 remote provenance/adjudication parsing; attach A++ evidence to PFP16 or Lane G v3 if possible; recover or rebuild Ω-W-V2 exact archive.
2. **β foundation:** land sensitivity maps with 480/120 cross-validation and per-channel score gradients. This gates Ω-W-V3, logit-margin, water-fill, and ADMM.
3. **α mask payload:** train NeRV and VQ-VAE/wavelet alternatives in parallel, but promote only if payload and SegNet/PoseNet survive exact archive eval.
4. **Renderer branch:** implement Ω-W-V3 only after sensitivity maps; run IMP as an independent renderer branch; stack only after both have exact archive evidence.
5. **γ coordinator:** run Joint-ADMM, Ballé/hyperprior, arithmetic, and bit optimizer only after at least one α candidate and one β/renderer candidate have Grade A score evidence.
6. **Data-side guardrail:** run MAE-V/SAUG or equivalent augmentation lanes in parallel so codec-only assumptions are tested, not assumed.

Parallelism is allowed only across independent write sets and independent scientific hypotheses. Shared-archive stack experiments wait until component archives are individually graded.

---

## 10. Implementation Readiness Notes — 2026-04-30

These changes improve the path to Grade A:

- PFP16 exact T4 eval has now landed under `experiments/results/lane_g_v3_pfp16/pfp16_a_plus_plus_t4_20260430T1620Z_codex` and changes the score ledger: it is the current Grade A++ frontier at recomputed `1.043987524793892`.
- `OWV3` is now a registered renderer magic with inflate dispatch and synthetic mixed-channel tests. Modal smoke produced a CUDA top-3 Fisher/build packet, but the archive was larger (`912,971` bytes, `+218,897` vs Lane G v3) and has no exact eval. It remains suspicious negative smoke until encoder overhead/config review.
- Lane 12 `.nrv` archive validation is unblocked at the auth-eval whitelist level, and the inflate mask resolver can discover `masks.nrv` from the legacy `masks.mkv` default. The current Lane 12 NeRV `jsonfix40` archive is retired by exact-CUDA regression and may only motivate alpha redesign.
- Remote PFP16 provenance contains invalid legacy parser/adjudication fields (`contest_cuda_score=100.0`, `hard_kill_triggered=true`) and is superseded by `contest_auth_eval.json`.

---

## 11. Bottom Line

The strict contest-grade ledger is much smaller than the historical all-scores inventory. The current authoritative frontier is **Lane G v3 PFP16 = 1.04** (`1.043987524793892` recomputed) with A++ T4 evidence. Most other results are scientifically useful as failure analysis, but not contest-grade enough to support kill/promote decisions until exact CUDA artifact custody is restored.

---

## 12. Exact CUDA/T4 Forensic, Non-Promotable OWV3 Rows

These artifacts have exact CUDA/T4 custody, but are not rank/frontier evidence
because predeclared component or regression gates fired. They are scoped
forensic evidence only.

| Row | Score | Bytes | Archive SHA | Component gate | Allowed use |
|---|---:|---:|---|---|---|
| Paired PFP16 r3 calibration | `1.037045485927815` | `686635` | `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f` | SegNet final-deploy gate fired; same-run calibration only | OWV paired delta baseline, `no_rank_frontier` |
| OWV3 r4 byte-feasible | `1.0378905176070103` | `686557` | `e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec` | SegNet relative ratio `1.003654` > `1.002` | scoped diagnostic, `no_promotion` |
| OWV3 R5 rank-1 | `1.0373951773937642` | `686468` | `16ab95220c8add11b0bc40fb632bc8421f8bb8ad1cfba145f0b6058075237518` | SegNet component gate fired; worse than paired PFP16 by `+0.00034969146594909795` | scoped forensic, `no_rank_frontier` |
| OWV3 R6 rank-1 | `1.0393166493980681` | `686531` | `9f7528bade11bf9cdf3df68f8073d11f196a6d5f48475a8680c21fb58c878c91` | PoseNet relative `1.0213113614240024` > `1.002`; score regressed by `+0.0022711634702530237` vs paired PFP16 | scoped forensic, `no_rank_frontier` |

Interpretation:

- `evidence_grade` in adjudication provenance records hardware/custody quality,
  not promotion eligibility.
- Failed-gate exact CUDA/T4 rows should be labeled
  `A-negative scoped forensic` in paper tooling.
- R7 scalar-threshold selection currently returns zero admissible candidates on
  the existing byte-plan grid; next OWV work requires component-balanced
  PoseNet/SegNet sensitivity or a materially new action rule.
