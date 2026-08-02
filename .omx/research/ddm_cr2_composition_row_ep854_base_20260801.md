# ddm_cr2 (#827) — THE COMPOSITION ROW is built and byte-closed. The recorded blocker STILL BINDS, but only its second half; the recorded *bridge* was wrong on three counts.

**Arm:** ddm_cr2 · 2026-08-01 · axis `[macOS-CPU frozen-PoseNet advisory]` · `score_claim=false`
`promotable=false` · exact contest pointer **UNMOVED**.

**STORES CONSULTED:** `.omx/state/canonical_task_status.jsonl` (#827 row, line 391) ·
`.omx/state/current_focus.md` (the named-blocker paragraph, lines 43-48) ·
`.omx/state/canonical_frontier_pointer.json` · commit `5ea9cd3f0a` (ddm_pw1) ·
commit `6f95b30568` (MAIN's exact row) · `ddm_ep2_receipt.json` ·
the live chain `ddm_v4c_resolve.py` / `ddm_v4d_resolve.py` /
`ddm_v4d_build_composed_archive.py` / `inflate_runner_v4d.py` /
`stage_v4d_realized_gate.sh`. **Deliberately not loaded:** the gc14/gc16/su2 memo
bodies (cited only for the fire-order they set) and the burn window telemetry.

---

## 1. The recorded blocker, verbatim

From `.omx/state/current_focus.md:43-48` (the `#827` ledger row carries
`blockers: []` and defers to this text):

> **🎯 THE COMPOSITION ROW (#827) — the largest live number.** ep854 DOMINATES the seg base of
> `gr1_cell_drop50` (d_seg 0.004310379 @ 359,221 B — the BASE constant of the S=0.9640 v4d archive) by
> **−0.035996 S on seg+rate**, byte-closed, on a verified-identical surface. **NAMED BLOCKER:** the v4d
> pose payload was solved against gr1's RENDERS, so a base swap ships corrections fitted to different
> pixels — MEASURABLE IN ONE n600 EVAL, not assumable. Bridge is mechanical (`pb1_p5` exposes
> `--seg-archive`). Order per gc16/su2: rate parent FIRST, then re-solve pose.

The dispatch seed compressed this to *"a pose re-solve"*. That compression is
**not wrong but under-specified**: the blocker is a *render-coupling* of an
already-solved pose payload, and its stated remedy is a **measurement**, not a
re-solve. The distinction decides the whole unit — a measurement is one gate
run; a re-solve is a scorer campaign.

## 2. Does pw1 dissolve it? **No.** (MEASURED at source)

`ddm_pw1` (`5ea9cd3f0a`) replaced two saturated search menus (`_refine_dim0`'s
±0.048 bound, `_beta_select`'s 3-entry `BETA_MAGS` with sign forced from yaw)
with self-terminating Swann brackets. That improves the pose solve's *search
quality*. It does not touch *which base the pose is solved against*, so the
coupling mechanism is untouched. **The blocker's mechanism survives pw1
unchanged.**

## 3. The recorded *bridge* was wrong on three counts (this is the finding)

| claim | status | evidence |
|---|---|---|
| "`pb1_p5` exposes `--seg-archive`" — so the bridge is mechanical | **TRUE but IRRELEVANT** | `tools/pb1_p5_byte_close_and_eval.py:206` does expose it. But `5ea9cd3f0a` established the `pb1_*` family is **not the live pose chain**; it feeds no v4d archive. |
| the live chain can take a different seg base | **FALSE (was)** | `ddm_v4c_resolve.py:63` `BASES` is a hardcoded 2-entry dict; `ddm_v4d_build_composed_archive.py:38` `BASE` was a module constant with no flag. |
| ep854 is drop-in for that base slot | **FALSE** | the two archives are **different ZIP grammars** — ep854 is `ddm_tr1_runtime_archive.v1` (`manifest.json` + monolithic `state/tr1.ddt1`); the v4d composer consumes `ddm_pfs1_composed_archive.v3_warp` (5 exploded members incl. `state/pose_warp.stp`). `build_oracle`'s own docstring asserts *"both Knee-A and cell_drop50 are the v3_warp sectioned grammar"* — ep854 is neither. |

So the row was blocked by a **grammar gap that no artifact named**, not only by
the pose coupling. That gap is now closed (§4).

## 4. What was built

**`tools/ddm_cr2_transcode_tr1_base_to_v3warp.py`** (new) — converts an endpoint
`ddm_tr1_runtime_archive.v1` into a drop-in `v3_warp` base. The packet grammars
are **identical** (both `ddm_tr1_four_section_packet.v1`, same `section_order`,
same `section_consumers`); only the ZIP layout and the token codec differ.

**`experiments/ddm_v4d_build_composed_archive.py`** (patched, ~10 lines) — the
seg base is now `--base-archive` / `--base-label` instead of a module constant.
This pays the debt on the **existing live surface** rather than adding a new one.

### Controls (every one MEASURED, none asserted)

1. **r7 canary** — re-encoding gr1's own token grid reproduces its shipped
   `state/tokens.dr7t` **byte-identically** (346,478 B). The coder in use *is*
   the coder those bytes came from.
2. **Decode trustworthiness** — `_encode_tokens(decode(ep854 tokens))`
   reproduces ep854's shipped tokens section **byte-for-byte** (355,182 B). The
   decode cannot be silently wrong.
3. **r7 round-trip** — lossless on the ep854 grid (`array_equal`).
4. **Default-path regression** — rebuilding with no new flags reproduces the
   live post-pw1 archive **byte-identically** (`sha 0ef9ff7129461f7318f8`).
5. **Packet identity** — the packet the receiver rebuilds from the composed
   archive is **byte-identical to ep854's own** (`sha f78334bfdd3daff3`, which
   is also ep854's manifest `packet_sha256`). **Therefore d_seg is EXACTLY
   ep854's measured n600 value — not an estimate, not a transfer.**
6. **#417 parse-back** — `ddm_v4d_verify_decode.py` A/B/C/D all pass on the
   composed archive (consumption bijection, field bit-exactness, independent
   compose recompute with the two-plane and beta paths exercised).

## 5. The row

| | live pw1 (MAIN MEASURED) | composed cr2_ep854 |
|---|---:|---:|
| archive.zip | 360,323 B | **285,529 B** |
| d_seg | 0.00431179 → 0.4311790 S | 0.003943024 → **0.3943024 S** |
| rate | 0.2399243 S | **0.1901220 S** |
| pose | 0.2765058 S (d_pose 0.0076455) | **UNMEASURED — transplant** |
| S | 0.9476091 | **0.8609302 predicted, if pose holds** |

`sha256 6edf45fa5052949d…` · gap denominator **0.7754681** (`0.9476091 − 0.172141`).

- d_seg delta **−0.0368766 S** = **−4.755 % of gap**
- rate delta **−0.0498023 S** (−74,794 B) = **−6.422 % of gap**
- **seg+rate delta −0.0866789 S = −11.178 % of gap**

This is **2.41× the −0.035996 S the row was registered at**, because the
registered number compared ep854 *in the packet-native token codec* against
gr1 *in the r7 codec*. At matched codec ep854 wins far harder.

### Decomposition (no double-count)

- At **matched r7 codec**, ep854's token field is **74,973 B smaller** than
  gr1's. This is a **CONTENT** win: ep854's zeroth-order token entropy is
  **2.3644 bits/code** vs gr1's **3.4565** (despite ep854 having *more*
  non-zero cells, 0.8158 vs 0.6983). Later training lowered d_seg **and** rate
  together.
- The separate **−83,677 B codec** delta (packet-native → r7 on ep854) belongs
  to the **ep2 line's own archives**, not to this row. It is not counted here.
- Residual **+179 B** = the transcode provenance blob in the manifest
  (0.0001192 S, 0.015 % of gap) — recoverable, deliberately kept.

**Corroboration:** ep2's own receipt records `counted_ledger_bytes = 275,005`
for ep854 while shipping 360,331 B. My r7-coded seg sections total 275,464 B
(+459 B of that ledger). The byte ledger was approximately right that ~275 KB
was reachable; the archive.zip now *realizes* it. This is a concrete instance
of the focus doc's own finding that `total_counted_bytes` and `archive.zip`
had drifted apart.

## 6. The blocker that REMAINS, and its pre-registered falsifier

The pose payload (per-pair two-plane warp, photometric `(a,b)`, `s_t` index,
`beta_idx`) was fitted against **gr1's** rendered `frame_1`. The composed
archive renders **ep854's** `frame_1`. d_pose is therefore **UNMEASURED**.

Pre-registered, before the gate runs:

- **Break-even pose contribution 0.3631847 S ⇒ break-even d_pose 0.0131903.**
- The transplant may degrade d_pose by up to **1.725×** before the swap turns
  net-negative.
- **If measured d_pose > 0.0131903 the swap is NET-NEGATIVE** and the pose must
  be re-solved against ep854 first. That re-solve is blocked at
  `ddm_v4c_resolve.py:63` (hardcoded `BASES`), which I deliberately did **not**
  fix — it is only needed on the negative branch.

`verdict_scope: INSTANCE` — nothing here kills any family; the open branch is
named.

## 7. STAGED for MAIN — I did not self-fire

`experiments/stage_v4d_realized_gate.sh:3` forbids self-firing and MAIN owns the
single n600 scorer slot (verified idle at 2026-08-01, no scorer process running).

```bash
bash experiments/stage_v4d_realized_gate.sh cpu cr2_ep854
```

(~11 min; archive `v4d_composed_cr2_ep854_archive.zip` and `inflate_runner_v4d.py`
are both already in place under `/Volumes/VertigoDataTier/pact/ddm_v4d_20260731`.)

**Prediction to check the gate against:** d_seg **0.003943024 exactly**
(packet-identity, not a forecast — if the gate reports anything else, the
receiver path, not the base, is at fault). rate **0.1901220**. S **0.8609302 if
and only if d_pose holds at 0.0076455**; the measured d_pose is the answer #827
was registered to obtain.

## 8. My own round-1 adversarial review (findings on my own work)

- **Assumed-key trace:** the `--dim0-offset 31.546875` and `pw1/final_pw1.jsonl`
  inputs were read from the pw1 build receipt, not recalled; the verifier
  independently reports `A_dim0_offset: 31.546875`. My *first* identity-guard
  run used the wrong JSONL and offset and produced a mismatching sha — I caught
  that it was a bad test, not a regression, and re-ran with the receipt's exact
  arguments before claiming byte-identity.
- **Class vs instance:** I fixed the *build* site's hardcoded base (the class:
  "seg base is a module constant"). I did **not** fix `ddm_v4c_resolve.py:63`.
  Stated as open, not hidden.
- **Would the tests pass if the code were broken?** The `#417` verifier checks
  only the **pose** fields — it would pass on a wrong seg base. I identified
  that gap and closed it with the independent packet-identity check (§4.5),
  which is an equality against an artifact I did not produce.
- **Latent defect found in the existing surface, not fixed:**
  `manifest["tokens_sha256"]` and `manifest["tr1_packet_sha256"]` in the
  v3_warp manifest match **neither** the stored `state/tokens.dr7t` nor the
  packet-tokens section nor the rebuilt packet — they are **stale provenance
  strings inherited from an ancestor archive**, never re-stamped when gr1
  applied cell_drop50. They are **inert** (the receiver at
  `inflate_runner_v4d.py:128-136` never reads them), so this is not a decode
  risk, but any future integrity check that trusts them would be reading a
  fiction. The transcoder stamps them correctly (`sha256_of_stored_zip_member`,
  recorded in `manifest["cr2_transcode"]["sha_convention"]`); the pre-existing
  archives are untouched.

## 9. Receipts

- `/Volumes/VertigoDataTier/pact/ddm_cr2_20260801/ddm_cr2_receipt.json`
- `/Volumes/VertigoDataTier/pact/ddm_cr2_20260801/ep854_v3warp_base_archive.zip` (283,636 B, `sha fd50925899b22c7c…`)
- `/Volumes/VertigoDataTier/pact/ddm_v4d_20260731/v4d_composed_cr2_ep854_archive.zip` (285,529 B, `sha 6edf45fa5052949d…`)
- `/Volumes/VertigoDataTier/pact/ddm_v4d_20260731/v4d_composed_cr2_ep854_build_receipt.json`
