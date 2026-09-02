# ddm_qxr1 MAIN adjudication — QXO1 realization row: DISTORTION-REFUSED, born-identical by construction; joint-preimage HOLD closed by arithmetic

Date: 2026-09-02. Owner: MAIN. Sources: `/Volumes/APDataStore/pact/ddm_qxr1_qxo1_born_realization_binding/{SCORER_RESULT.json,BINDING_RESULT.json,FIRE_ORDER.json}` (retained, sha-pinned in the receipts) + arm memo `ddm_qxr1_qxo1_born_realization_binding` (charter falsifier pre-registered 2026-09-02).

## 1. RESULT (advisory n600, `[macOS-CPU advisory]`, score_claim=false)

The QXO1 129,309 B archive realized through the current QBT renderer:

- d_seg **0.17077701144748264** (20,145,676 / 117,964,800 sites)
- d_pose **115.83747038015008** (pose term 34.03490419850628)
- rate **0.08610155536927486** · advisory S **51.19870689862382**
- Verdict: **DISTORTION-REFUSED**, scope INSTANCE (exact archive + current receiver binding). Elapsed 236.6 s, $0, all payloads retained (20 render chunks + decoded field + results; AP free 42.2 GB post-run).

## 2. MECHANISM — identical to BR2 by construction, and the row was DERIVABLE

BINDING_RESULT proved before the fire: renderer-consumed state (model / latent_meta / latents) is **byte-identical** to the born object, and `qxo_section_7_pose_stream` / `qxo_section_8` are **never consumed** by the renderer. Deterministic decode ⇒ realized frames = the born render ⇒ distortion identical to BR2 (0.1708 / 115.84) with only the rate term differing. **Lesson (measured, mine):** when a binding proves byte-identity of ALL consumed inputs, the scorer row is derivable — derive before firing. Cost here was small (237 s local) but the class matters.

## 3. FALSIFIER — NOT met; BR2 prior CONFIRMED exactly

Pre-registered: d_seg ≤ 0.01 ∧ pose ≤ 1.25e-4 would have opened the first byte-feasible distortion path. Measured: 17.1× over on seg, ~9.3e5× over on pose absolute budget ([[m110]]).

## 4. THE JOINT-PREIMAGE HOLD IS CLOSED BY ARITHMETIC — DO-NOT-FIRE

qxr1's HOLD-CONDITIONAL successor (a new receiver consuming sections 7–8 "without flat painting or uncounted learned state") cannot meet its own trigger on THIS core:

- Sub-0.12 at 129,309 B allows seg+pose ≤ 0.12 − 0.08610 = **0.03390**.
- The core's partition field itself differs from exact at **1,669,798 sites** (qx3, decoder PERFECT) ⇒ seg term ≥ 100·(1,669,798/117,964,800) = **1.4155** under PERFECT realization — **41.8× the whole distortion budget** before any realization loss or pose cost.
- Making the field exact costs **+486,311 B** (qx3 cheapest counted closure) = 3.5× over the byte cap.

No receiver engineering changes either number. DO-NOT-FIRE recorded for the joint-preimage builder on the current core.

## 5. ROUTING

- **The QX line is closed at every measured joint**: envelope census (qx1) · event forms (qx4 FORMULATION-CLOSED) · conditioning closure (qx3) · cross-transfer (xov1 5.406×) · realization (this row, born-identical). The born/QBT family's ONLY live route is **ddm_qbr1's optimization-vs-capacity discriminator** — this adverse row is an INPUT to MAIN's burn fire-decision (per the qbr1 charter §4) and STRENGTHENS the case for firing it: envelope engineering is exhausted; only the model itself can move the field.
- **SCMDL #1374 PARKED-pending-candidate**: jcb1's fail-closed refusal (memo landed, commit cherry-picked from the serializer fallback bundle 9a9701e711) found the three-candidate roster already closed (dds1 ~613 B ceiling vs 47,603 B packet; jbp1 exact 177,052 B refusal) and the charter's XOV1 hash pins absent from rxc1 receipts — two recall failures at MY charter-write time ([[m122]] recurrence; spawned while the corrections index was 2.6 d stale with its rebuild still running). Bounded-reset infrastructure is NOT fired: no live candidate roster ⇒ no imminent exact row it feeds.
- Fleet: wx1 (divergent body-scope audit) + qbr1 (burn prep) remain live and own both surviving heads; refilling to cap with make-work would violate [[m42]].

## DEAD-ENDS

- Joint-preimage receiver on the QXO1 core (arithmetic above) — reactivation ONLY with a core whose decoded field error ≤ ~0.0003 at ≤137,986 B total.
- Scorer-firing a run whose consumed inputs are proven byte-identical to a scored ancestor (derive instead).
