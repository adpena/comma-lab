# Position — SEAT S1 (Shannon, rate/budget lens) — T3 v7 INCLUSION SYMPOSIUM

STORES CONSULTED: CONVENING_T3_v7_inclusion_symposium_20260708.md (docket + status update) · ORCHESTRATION_LEDGER.md (last 250 lines) · CLAUDE.md non-negotiables · docs/operating_manual_craft_handoff.md · per-item source/artifact cites inline (launch.sh, memos, waterfill numbers as tagged). No sibling position files (BLIND).

**BLIND**: I have not read position_INCL_S* siblings. **Pointer contest-CPU 0.19110 UNMOVED — everything below is MEANS.**

## Operating-within assumption (Fix-7)
I am reasoning inside the frame that **the ONLY authoritative currency is counted `archive.zip` bytes and the expected byte-closed n600 exact row against `upstream/evaluate.py`.** From that frame the rate/budget lens sees each docket item through four sub-ledgers: (a) **counted-byte Δ** (the rate term `25·bytes/37_545_489`), (b) **training-time budget** (the 6.5-day / 3000-ep envelope), (c) **RAM/memory envelope** (128 GiB, single-workload safe-frac 0.85 = 108.8 GiB usable; concurrent-fleet 0.70 = 89.6 GiB), (d) **expected contribution to a lower exact row**. My standing assumption, which frames every call below: **items 3–11 are ALL rate-term-NEUTRAL** — none adds or removes a single counted archive byte. They are training-time, memory, checkpoint-selection, verdict-compute, and resumability items. Their value toward < 0.19110 is therefore always INDIRECT (they enable a faster/safer/better-converged run through which the basis raise + curriculum move the number), never a direct byte-delta. `#363: INFERRED_FROM_DOMAIN_LITERATURE` for the "indirect only" framing; each per-item byte-neutrality is `VERIFIED_VIA_SOURCE_INSPECTION` of the docket/ledger as cited.

Items 1 (basis) and 2 (event mode) are OPERATOR-DECIDED — I note only rate/budget implications, do not relitigate.

---

## Per-item calls (docket items 3–11)

### Item 3 — Micro-batch-pairs → **v7.1-ARM**
Rate: **0 counted bytes**; it is a batched-twin routing change, training-time only. Budget: pure UPSIDE — 2–4× wall-clock reduction (docket L26/L56). But the batch-size regrouping perturbs gradient-noise structure ⇒ **trajectory-affecting**; the logit-adjust leg is bit-exact (offset None = byte-identical) but the micro-batch grouping itself is NOT proven trajectory-equivalent, and waterfill B is pinned=1 UNMEASURED until an uncontended n600 curve. It cannot be IN-v7 without corrupting the sealed baseline's trajectory provenance. Correct class = built + flag-flip-ready + A/B gate named.
- Evidence: ledger "D15 micro-batch routing LANDED (bd6219a0a) … NOT default-on — the n600 trajectory A/B remains the inclusion evidence (waterfill B pinned 1 UNMEASURED)". `#363: VERIFIED_VIA_SOURCE_INSPECTION`.
- Dissent: none. From budget alone it is the single largest wall-clock lever, but wall-clock is lexicographically SECONDARY to trajectory integrity — do not IN-v7 it for speed.

### Item 4 — Safe-compile certified regions → **v7.1-ARM**
Rate: **0 counted bytes** (compile = identical bytes, faster dispatch). Budget: hosc region 1.41× certified, default-OFF byte-identical 0.0 (docket L4/L60). The evidence gate is the run-1 stop-time whole-step bench (D17), which does not exist pre-launch. Byte-identity qualifies it structurally, but the SPEEDUP is unproven at whole-step until the bench ⇒ arm it, flip on the stop-time bench.
- Evidence: ledger "safe-compile v2 LANDED … live `_act` wired flag-flip default-OFF (byte-identical verified 0.0) … GPU re-cert + whole-step bench → run-1 stop checklist". `#363: VERIFIED_VIA_SOURCE_INSPECTION`.
- Dissent: none.

### Item 5 — D16 Metal kernels → **REGISTERED-duty-to-measure**
Rate: **0 counted bytes**. Budget: 2.3–3.9× MEASURED, bit-identical, N=5 deterministic — real. **But the consuming loss term (margin-map / curvelet) is default-OFF in the v7 sealed config**, so the kernel accelerates NOTHING in v7 as-launched: it banks **zero v7 wall-clock**. Speedup with no active consumer is not an IN-v7 budget benefit; it is insurance that fires when a consuming term arms.
- Evidence: docket L64 "speedup is real but the consuming loss term is default-off in v7"; ledger "margin-map + curvelet = documented evidence-based no-gos (VJP wall, no hot term)". `#363: VERIFIED_VIA_SOURCE_INSPECTION`.
- Dissent: I would NOT class this v7.1-ARM (S3 may) — a v7.1 arm implies an A/B on the CURRENT config; there is no term to A/B here. Trigger-registered is the honest class. Minor lens disagreement, defer to S3's whole-step-bench framing if it keeps a megakernel path alive.

### Item 6 — #330 verdict memory reclaim → **IN-v7** (the enabling item, my primary concern)
Rate: **0 counted bytes** — MEASURED bit-identical d_seg/d_pose parent-vs-child (4ba4058e1). This is the item that FUNDS the basis raise. Quantified: in-process verdict RATCHETS **+4.6 GiB** into the parent per n600 verdict; killpg subprocess holds parent at **+0.0 GiB**. The basis raise consumed **+3.93 GiB** of envelope (peak 71.54 GiB). So the reclaim (4.6) **> the basis demand (3.93)** → net **+0.67 GiB headroom preserved**, and — decisively — it removes a per-verdict RATCHET that, over a 3000-ep run with periodic n600 verdicts, is the #205-class OOM signature (verdict spike kills the run before checkpoint). Under the concurrent-fleet envelope (89.6 GiB usable → 18.06 GiB headroom over the 71.54 peak) the un-reclaimed ratchet erodes headroom monotonically; #330 protects that envelope specifically. Score-neutral by bit-identity + budget-enabling = textbook IN-v7.
- **IN-v7 precondition (storage waterfall):** the ~7 GiB transient npz per verdict is an SSD hop, NOT RAM — it must clear the storage-waterfall preflight (SSD tier ≥ ~10 GiB free) or it converts an RAM problem into a disk-full stall. Flag as a launch-checklist precondition, not a blocker.
- Evidence: docket L65–68; ledger "#330 verdict memory reclaim LANDED (4ba4058e1): … killpg subprocess holds parent at +0.0, child bit-identical d_seg/d_pose … Caveat: ~7 GiB transient npz per n600 verdict (SSD hop)". `#363: VERIFIED_VIA_EMPIRICAL_ANCHOR` (measured GiB deltas). Envelope arithmetic re-derived (108.8/89.6 usable, 37.26/18.06 headroom, 0.67 net) `#363: VERIFIED_VIA_SOURCE_INSPECTION` (MEMORY.md L51 policy + docket L52/L79).

### Item 7 — Adaptive-ε → **REGISTERED-duty-to-measure**
Rate: **0 counted bytes**. Budget: NEUTRAL (default-off insurance). Built + byte-identity-OFF proven, but the pre-registered n600 A/B NEVER RAN — superseded by v6's λ_eik=0.01 fixed redesign; the run-1 launch carries no viscosity flags and eikonal is stable ~0.0084 @ep67 with no re-entry. Its failure mode is STRUCTURALLY ABSENT in v7, so a v7.1 arm (which implies a live-config A/B) is the wrong class; register with a telemetry trigger.
- Evidence: ledger "Item-7 evidence check (adaptive-ε #320) DONE by orchestrator … REGISTERED-duty-to-measure, trigger = eikonal re-entry signature". `#363: VERIFIED_VIA_SOURCE_INSPECTION`. Underlying equation `adaptive_eps_cfl_edge_tracking_v1` remains `#363: ASSUMED_AWAITING_VERIFICATION`.
- Dissent: none.

### Item 8 — R-7 finishers (β2-window rewarmup + Polyak finisher EMA) → **REGISTERED-duty-to-measure** (with a sizing FLAG)
Rate: **0 counted bytes** — BOTH legs are optimizer-side / checkpoint-side, never archive bytes. Polyak = a uniform tail-mean checkpoint CANDIDATE (EMA shadow is never replaced; byte-close picks the winner at stop) → a rate-FREE, trajectory-FREE stop-time duty-to-measure. β2-window rewarmup = optimizer trajectory-affecting, A/B owed, β2 law INFERRED/PROVISIONAL. Neither has budget cost.
- **S1 SIZING FLAG (my assigned concern):** the Polyak `start_epoch=0` default **arms the averager from the run START, not the finishing window** — a uniform mean over the whole 3000-ep trajectory would be dominated by pre-convergence weights and would **corrupt the tail-mean** the finisher is supposed to produce. This is a config-sizing defect, not a mechanism defect. **If the council elevates Polyak to IN-v7 as the shipped checkpoint, the IN-v7 precondition is: `start_epoch` sized to the finishing window (the Muon-quench / post-ep726 tail), NOT 0.** As-built default-0, it must NOT be IN-v7. Because that sizing is unset and the byte-close pick is itself the measurement, REGISTERED-duty-to-measure is the honest class today.
- Evidence: ledger "R-7 FINISHERS LANDED … PolyakTailAverager = uniform tail mean … an unmeasured stop-time duty-to-measure, honestly labeled … Named residuals: start_epoch=0 default arms from run START (operator must size); … β2 law INFERRED/PROVISIONAL". `#363: VERIFIED_VIA_SOURCE_INSPECTION`; β2 horizon `#363: INFERRED_FROM_DOMAIN_LITERATURE`.
- Dissent: I flag, I do not block. Rate/budget has no objection to either lever; the objection is exclusively the tail-mean corruption under unsized start_epoch. Surface it to S2/S4 as an IN-v7 precondition if they elevate.

### Item 9 — Event-gates fired-state persistence (canonical resume registry) → **IN-v7**
Rate: **0 counted bytes**. Budget: NEUTRAL — pure resumability correctness (crash-resume bit-equality tested; legacy run-1 emits {} → byte-identical sidecar). Zero-cost insurance that directly protects the 6.5-day budget from a crash-loss re-spend. Hardening, not a lever; trivially IN.
- Evidence: ledger "CANONICAL RESUME REGISTRY LANDED (2b7332f4b/…) … crash-resume bit-identity vs uninterrupted … → symposium item 9 SATISFIED (IN-v7, hardening)". `#363: VERIFIED_VIA_SOURCE_INSPECTION`.
- Dissent: none. From the budget lens this is the highest-value zero-cost item on the docket: it converts a crash from a full 7.4-day re-spend into a resume.

### Item 10 — GPU-verdict hybrid → **REGISTERED-with-trigger**
Rate: **0 counted bytes**. Budget: a potential verdict-compute speedup, BUT gated on the D1 stop-time agreement probe (the GPU verdict must bit-agree with the CPU authority). It touches the AUTHORITY path — and MPS/GPU is NEVER a score authority per the non-negotiable; a GPU verdict that has not passed the agreement probe cannot be trusted to compute the number the seal depends on. No v7 budget benefit until the probe passes. Register, trigger = D1 stop-time agreement probe.
- Evidence: docket L36/L78 "evidence-gated on the stop-time agreement probe; likely registered-with-trigger". `#363: VERIFIED_VIA_SOURCE_INSPECTION`.
- Dissent: none — and I'd resist any push to arm it earlier: the authority path is exactly where a silent GPU/CPU drift becomes a fake score.

### Item 11 — fp16 cf-feats → **REGISTERED-duty-to-measure** (competes-but-not-needed)
Rate: **0 counted bytes** — the coordinate-Fourier features are deterministic (rule-118 FREE, generated at decode, never stored in archive), so fp16-vs-fp32 changes NO counted byte either way. Memory: POSITIVE — fp16 could roughly halve the ~41 GiB cf_mx_cache, freeing ~large envelope. **But the numerics change (fp16 features → different render → different d_seg) ⇒ trajectory-affecting ⇒ cannot be IN-v7 without an A/B.** The S1×S3 joint question the docket asks — "is there room after the 71.54 GiB peak?" — I answer directly: **YES, room exists.** Single-workload headroom after peak = **37.26 GiB**; even concurrent-fleet = **18.06 GiB**. The basis raise was already ADMITTED at 71.54 and #330 preserves +0.67 GiB net. **fp16 cf-feats is therefore NOT REQUIRED to fit v7** — it is memory insurance for a FUTURE larger basis, not a v7 precondition. Spending its trajectory-A/B risk now buys nothing the envelope needs. Register it against a future basis raise that actually exhausts the 37 GiB headroom, pending a fresh waterfill + A/B.
- Evidence: docket L79–81 "the basis raise already consumed +3.93 GiB … fp16-feats' memory-for-capacity trade must be re-waterfilled AGAINST that … likely REGISTERED pending a fresh waterfill". cf_mx_cache ~41 GiB from MEMORY.md L51. `#363: VERIFIED_VIA_SOURCE_INSPECTION` (docket + memory policy); the "fp16 halves the cache" magnitude `#363: INFERRED_FROM_DOMAIN_LITERATURE` (fp16 = half fp32 width; exact freed GiB unmeasured).
- Dissent: none — and I explicitly rule it does NOT compete with the basis raise for a scarce envelope, because the envelope is NOT scarce post-#330 (37 GiB slack). The "they compete" docket framing is true only under a future bigger basis; today they don't.

---

## Closing rate/budget table

| # | Item | Counted-byte Δ | Wall-clock budget | RAM envelope | S1 class |
|---|------|---------------|-------------------|--------------|----------|
| 3 | Micro-batch-pairs | **0** | −50–75% (2–4×) IF armed | neutral | **v7.1-ARM** |
| 4 | Safe-compile regions | **0** | −29% hosc (1.41×) IF armed | neutral | **v7.1-ARM** |
| 5 | D16 Metal kernels | **0** | **0 in v7** (consumer term OFF) | neutral | **REGISTERED** |
| 6 | #330 verdict reclaim | **0** (bit-identical) | protects budget (no OOM re-spend) | **+4.6 GiB reclaim > +3.93 basis; +0.67 net; kills per-verdict ratchet** | **IN-v7** (precond: SSD ≥~10 GiB for 7 GiB npz) |
| 7 | Adaptive-ε | **0** | neutral | neutral (default-off) | **REGISTERED** (trig: eikonal re-entry) |
| 8 | R-7 finishers (β2 + Polyak) | **0** | neutral | neutral | **REGISTERED** (FLAG: Polyak start_epoch=0 corrupts tail-mean; size before any IN-v7 elevation) |
| 9 | Resume registry | **0** | protects budget (crash → resume, not 7.4d re-spend) | neutral (legacy {} byte-identical) | **IN-v7** (hardening) |
| 10 | GPU-verdict hybrid | **0** | 0 until agreement probe | neutral | **REGISTERED** (trig: D1 stop-time probe) |
| 11 | fp16 cf-feats | **0** (feats free/rule-118) | neutral | frees ~½ of ~41 GiB cache, but **NOT NEEDED** (37 GiB slack) | **REGISTERED** (trig: future basis raise that exhausts headroom) |

**S1 headline:** the rate term is UNCHANGED by every item 3–11 — none moves a counted byte. Two items are IN-v7 for BUDGET-PROTECTION reasons only (#6 memory, #9 crash-resume); the rest are wall-clock arms or trigger-registered insurance. **The exact row < 0.19110 will be moved by the basis raise (item 1) + curriculum, NOT by anything on the 3–11 docket** — say so plainly. Envelope is comfortable (37.26 GiB single-workload slack post-basis, post-#330); no item 3–11 forces a memory trade-off in v7 as sealed.

**Envelope funding chain (the one load-bearing arithmetic):** basis raise spends +3.93 GiB → #330 reclaims the +4.6 GiB per-verdict ratchet → net +0.67 GiB preserved AND the #205-class verdict-spike OOM is structurally removed. This is why #6 is IN-v7 and not merely nice-to-have: it is the item that makes item 1's admitted 71.54 GiB peak survivable across 3000 ep of periodic n600 verdicts.
