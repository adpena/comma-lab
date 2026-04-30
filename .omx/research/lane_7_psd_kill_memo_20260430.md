# Lane 7 PSD — Formal Kill Memo (2026-04-30)

**Status**: KILLED for current dispatch cycle. DEFERRED for permanent disposition pending three reactivation criteria (see §4).

**Council vote**: 0 APPROVE / 10 REJECT / 0 ABSTAIN (unanimous; conservative-bias check PASSED).

**Authority**: `.omx/research/council_lane_7_psd_dispatch_review_20260430.md` (full per-voice deliberation).

---

## 1. Verdict + vote breakdown

| Voice | Vote |
|---|---|
| Shannon (LEAD) | REJECT — no R(D) mechanism closes 1.49 → <1.05 gap |
| Dykstra (CO-LEAD) | REJECT — pose constraint violated 11.9× empirically |
| Yousfi | REJECT — PSD bottleneck destroys FastViT-PoseNet luma detail |
| Fridrich | REJECT — PSD aligned with SegNet blind spots, not PoseNet's |
| Contrarian | REJECT — bold APPROVE arguments are pre-emptively underspecified |
| Quantizr (adversarial) | REJECT — leader (0.33) didn't choose PSD; Bayesian evidence against |
| Hotz | REJECT — EV is +$125 per 0.01-score-point; terrible vs Phase 2 |
| Selfcomp | REJECT — PSD lacks PoseNet-aware luma-skip; my 0.38 explicitly avoided |
| MacKay (memorial) | REJECT — MDL net cost +0.187 score points; worse 2-part code |
| Ballé | REJECT — hyperprior amplifies base; apply to Lane G v3, not PSD |

---

## 2. Top 3 rejection reasons (with specific counter-evidence)

### Reason 1 — Empirical historical landing of 1.49 (worse than current 1.05 bar)
- **Evidence tag**: `[contest-CUDA equivalent]` from 2026-04-11
- **Source**: `competition_state.py:131` ("Auth eval 1.49 vs dilated 1.33. Worse.")
- **Detail**: PSD h=64, ep 809: pose=0.01108, seg=0.00532, rate=0.02522. Lane G v3 (current bar) is 1.05. PSD historical = **42% WORSE** than the current bar.
- **Counter-evidence to potential APPROVE**: The 1.38 "breakthrough" in `project_psd_breakthrough.md` required KL distill auxiliary, which is itself on the killed_techniques list at `competition_state.py:126` (PoseNet collapse). The `PSD_STANDARD_ADAPTIVE` profile does NOT include KL distill, so the 1.38 result is not reproducible via this profile.

### Reason 2 — Architectural rate-resolution mismatch with FastViT-PoseNet
- **Evidence tag**: [empirical: 5× PoseNet regression measured 2026-04-11]
- **Source**: `memory/project_psd_auth_eval_verdict.md`, Yousfi/Fridrich council reasoning
- **Detail**: PSD's `PixelUnshuffle(2)` operates at 582×437. SegNet (EfficientNet-B2 stride-2 stem) is aligned with this resolution, hence the empirical 12.8% SegNet improvement. **But PoseNet is FastViT-T12 with attention** at 512×384 with stride-2 stem to 256×192. PSD's half-resolution bottleneck **destroys** the high-frequency luma detail that FastViT's attention layers use for pose-delta prediction. **No training-time stabilizer (boundary_weight, SWA, hard-frame replay) addresses an architectural representation-capacity problem.**

### Reason 3 — Score arithmetic + EV mismatch with current floor
- **Evidence tag**: [prediction] based on score formula derivation
- **Source**: Dykstra Pareto computation, MacKay MDL computation, Hotz EV computation
- **Detail**: PSD's PoseNet floor is √(10·0.011) = 0.332 score points (PoseNet contribution alone). Plus seg (~0.50 with optimistic projection) plus rate (0.63) = **PSD floor ~1.46**. Even with `PSD_STANDARD_ADAPTIVE`'s training improvements optimistically projected, the achievable region does not intersect the {seg ≤ 0.0029, pose ≤ 0.000931, rate ≤ 0.025} feasibility set defined by Lane G v3's anchor.

---

## 3. Reactivation criteria (what would change the decision)

PSD becomes worth retrying IF AND ONLY IF:

1. **A PoseNet-aware luma-skip variant is designed** (Lane PSD-LumaSkip or Lane PSD-Hybrid). This would add a full-resolution luma residual path (similar to Lane LCT's late-stage chroma skip) so the PoseNet-required high-frequency detail is preserved alongside PSD's SegNet-aligned bottleneck. **This is a NEW lane requiring its own council review** — not a re-dispatch of `PSD_STANDARD_ADAPTIVE`.

2. **The current floor moves below 0.50** (from Lane G v3's 1.05 today). At a 0.50 floor, the score arithmetic relaxes enough that PSD's 5× PoseNet regression becomes tolerable IF it is combined with extreme rate savings (PSD + Selfcomp block-FP 1.017 bpw + arithmetic-coded weights = ~12 KB renderer, saving ~4 KB vs current FP4A → -0.0027 score points, comparable to PSD's PoseNet penalty at the new floor).

3. **Phase 2 Lane 19 (SegNet logit-margin) demonstrates SegNet improvements transfer architecture-agnostically.** If logit-margin gives Lane G v3 a 10% SegNet improvement, then PSD's seg-improvement specialty (the 12.8% advantage) becomes redundant and the rate cost of switching architectures becomes unjustifiable.

None of these conditions hold today.

---

## 4. Disposition

| Item | Action |
|---|---|
| `scripts/remote_lane_psd.sh` | **DO NOT CREATE** |
| GPU dispatch | **DO NOT DISPATCH** any PSD run today |
| `competition_state.py:131` | **NO CHANGE** (already correctly lists `psd_architecture` as killed) |
| `src/tac/profiles.py:168` (`PSD_STANDARD_ADAPTIVE`) | **NO CHANGE** (kept for potential future re-evaluation under reactivation criteria) |
| Phase 1 maturity table | **UPDATE**: Lane 7 status changes from "1 (script + watchdog landed but never dispatched)" to **KILLED-DEFERRED** |
| Memory entry | **CREATE**: `project_lane_7_psd_killed_or_deferred_20260430.md` |
| 3-clean-pass adversarial gate | **NOT NEEDED** for this lane (no infrastructure landing); applied prospectively if any reactivation lane is proposed |

---

## 5. Cross-references

- `.omx/research/council_lane_7_psd_dispatch_review_20260430.md` — full per-voice deliberation (THIS memo's source-of-truth)
- `competition_state.py:131` — `psd_architecture` on killed_techniques list
- `memory/project_psd_auth_eval_verdict.md` — 2026-04-11 prior council verdict ("STAY WITH DILATED")
- `memory/project_psd_breakthrough.md` — 1.38 result requiring (now-killed) KL distill
- `memory/project_psd_early_signal.md` — 2026-04-10 ep289 first signal
- `src/tac/profiles.py:168` — `PSD_STANDARD_ADAPTIVE` profile definition
- `src/tac/architectures.py:798` — PSDPostFilter wired in VARIANTS dict
- `feedback_production_hardened_standard_definition_20260430.md` — Lane 7 was Level 1 (now Level 0/KILLED)
- `project_phase1_dispatch_state_corrections_20260429.md` — predicted band [1.10, 1.40] standalone (now empirically rejected)

---

## 6. Conservative-bias check (per CLAUDE.md "Council conduct" rule)

Every REJECT vote was scrutinized for "don't change working code" / "ship what we have" reasoning. **None of the 10 votes used a conservative argument.** All votes cited:
- Empirical historical evidence (1.49 at ep 809 — 5/10 voices)
- Mathematical/geometric reasoning (Shannon R(D), Dykstra Pareto, MacKay MDL — 3 voices)
- Architectural mismatch (Yousfi/Fridrich on PoseNet rate-resolution — 2 voices)
- Bayesian inference from competitor architecture choice (Quantizr — 1 voice)
- Expected-value arithmetic (Hotz: $125/0.01-score-point — 1 voice)
- Multiplicative composition with Phase 2 (Ballé — 1 voice)
- Internal architectural insight from a working 0.38 implementation (Selfcomp — 1 voice)

The unanimity is genuine.
