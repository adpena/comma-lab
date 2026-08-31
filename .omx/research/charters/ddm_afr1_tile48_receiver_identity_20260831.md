# ddm_afr1_tile48_receiver_identity — port AFC1's measured −81 B zero-distortion interaction to the native receiver and prove identity (owning memo: ddm_afr1_tile48_receiver_identity_20260831.md)

## MANDATE

Operator standing: *"do whatever it takes… autonomously with full authority and standing go to
accomplish frontier score lowering."*

`ddm_afc1_address_free_census` (landed rc=0, commit `9263a9c1e0` lineage) MEASURED a new same-body
lossless interaction on the **exact shipped lb1 archive**:

```
tile48   = ((y//64)*8 + (x//64))
groupbin8= (((x%64) + 2*(y%64)) * 8) // 190
context  = tile48*8 + groupbin8          # decoder computes both from (x,y); ZERO stored parameters
```

Measured on the real RC64 path, **full n600, all 117,964,800 decoded tokens**, applied AFTER the
shipped 22-member conditioning configuration:

| quantity | value |
|---|---|
| token delta | **−81 B** |
| archive delta | **−81 B** |
| `ΔS_rate` | **−0.00005393457520289588** |
| changed decoded symbols | **ZERO** (lossless by construction ⇒ d_seg, d_pose UNCHANGED) |

Authority receipt `/Volumes/APDataStore/pact/ddm_afc1_address_free_census/tile48_groupbin8/measurement_v1/ADJUDICATION.json`
SHA-256 `9bda316e278e6bf37e762c6c1308cc014db2f76703ce327eef0bad064b6ed841`; manifest
`measurement_v1/MANIFEST.json` SHA-256 `1e8a111e8f5d010d67ac34e212a81370341743c4e9ab148c14b2ceb22425a425`,
45,807,018 retained B.

**AFC1's own typed fire order is this arm's whole scope**, verbatim: *"port only `tile48_groupbin8`
to the staged native C receiver, prove Python/C configuration parity, and run one retained full-n600
candidate receiver identity. No scorer."*

## ⚠ THE STORAGE GATE IS ALREADY CLEARED — DO NOT RE-ASK

AFC1 gated this on `QUEUED_AFTER_APDATASTORE_IDENTITY_FLOOR`: APDataStore free ≥ **8,142,450,560 B**.
APDataStore has **6,375,866,368 B** — short by 1.77 GiB. **That gate is satisfied on the OTHER tier.**

MAIN measured 2026-08-31: **VertigoDataTier free = 8,964,227,072 B**, surplus **821,776,512 B**
(**+10.1%** over the floor). The canonical waterfall is **Vertigo FIRST**, then APDataStore; AFC1
scoped to APDataStore only because codex arms were granted that tier (task #1003), not because the
work requires it.

**RETARGET: run under `/Volumes/VertigoDataTier/pact/ddm_afr1_tile48_receiver_identity/`.**
Three storage escalations were raised and withdrawn today on stale or wrong-tier premises
([[m122]], task #1335) — **do not raise a fourth.** ⚠ **The margin is THIN (0.78 GiB / 10.1%).**
Measure free space BEFORE and AFTER each stage, write small, and if you genuinely cannot fit, emit a
typed blocker with the exact shortfall and **KEEP THE BYTES** — never delete, never route to local
disk (operator opt-in required and NOT held).

## SCOPE — exactly four things, in order

1. **PORT** `tile48_groupbin8` — and ONLY it — into the staged native C receiver. The Python
   reference is the authority; the port is the thing under test.
2. **PARITY** — prove Python/C configuration parity on the context computation itself before any
   full run. A parity failure here is the finding; report it, do not paper it.
3. **IDENTITY** — ONE retained full-n600 candidate receiver identity run. Decoded output must be
   **bit-identical** to the Python path. Retain the streams, archives, builds, checkpoints, receipts.
4. **HAND-OFF** — emit a typed fire order for MAIN: byte-close → seal (`make_candidate_seal.py`,
   NO hand-typed sha) → ONE T4 dual-axis row. **You do not fire it.**

## HARD CONSTRAINTS

- **NO SCORER, NO MODAL.** MAIN owns dispatch + single-flight. The local scorer lane is MAIN's.
  An honest partial plus a typed fire order is the CORRECT outcome, never a failure.
- `upstream/` READ-ONLY. Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2
  genuine review passes.
- ALWAYS KEEP THE PAYLOAD. Retain to the Vertigo store above. If you cannot write, blocker + keep.
- **Reproduce AFC1's −81 B as a CONTROL before trusting the port.** Every arm in this lineage
  reproduces a control first; it is why they cost seconds and why several arithmetic errors were
  caught this window.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `#1269` — banked lossless deltas are **NOT additive without a joint re-encode**. AFC1's −81 B was
  measured AFTER the shipped 22-member configuration, so it is already joint w.r.t. the live body —
  but any FURTHER stacking re-incurs this and must be re-measured, not summed.
- `ddm_rr9_*` (#1244) — the reorder axis is EXACTLY 0 B, architecturally fused to the trained HPAC
  group index. Do not attempt a permutation variant.
- `ddm_jt23_*` / `ddm_oc2_*` (#1283/#1326) — the coder axis is CLOSED at 0 B and decode-derived
  conditioning is otherwise DRAINED (OC2's `miss_rank8` bought only −2 B). AFC1 states this family
  is "drained except the admitted native-port/identity work below" — that admitted work is this
  arm, and nothing else in the family is open.
- `#1214` (five arms) — the sharp-optimum law. AFC1 falsified it **at instance level** with this
  −81 B zero-damage interaction (second exception after FCD1's B/H/W win-win). That falsification is
  INSTANCE-scoped and does NOT reopen semantic quantization, carrier quantization, HPAC shrink,
  residual precision, or alternate-body generators.
- `ddm_dcf1_*` (#1347) — 0 inert bytes in the shipped body; the packet/tree-shake axis is spent
  (best remaining packet reduction 14 B, research-only).

## OPTIMAL FORM

- Family exemplar: `ddm_afc1_address_free_census` itself — the **reference** form for this class:
  full n600, real RC64 path, all 117,964,800 tokens, 25-frame atomic checkpoints, control +
  candidate + repeat streams all retained, SHA-pinned adjudication. Match that bar for the identity
  run: full n600, real receiver, bit-identity asserted in code, repeat retained.
  Provenance pin: measurement receipt SHA-256
  `9bda316e278e6bf37e762c6c1308cc014db2f76703ce327eef0bad064b6ed841`; frontier archive SHA-256
  `5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9` (180,083 B).
- SCOPE reductions declared per row (a short parity smoke before the full run is legal and expected).
  MECHANISM reductions FORBIDDEN — a partial port, a mocked context, or an identity check on a
  subset of frames produces NO verdict.
- **PRIOR-LAW PREDICTION (falsifiable).** The native port is a re-implementation of a computation
  the Python path already performs deterministically from `(x,y)` with zero stored parameters, so I
  PREDICT **bit-identical decode on 600/600 frames at first parity-clean build**.
  **FALSIFIER:** any frame differs, or Python/C configuration parity fails. That would mean the
  −81 B is not portable to the shipping receiver, which **downgrades AFC1's candidate from a
  fire-order to a Python-only research row** — a real and valuable finding. Count it plainly.

## DELIVERABLE

`.omx/research/ddm_afr1_tile48_receiver_identity_20260831.md` — the control reproduction FIRST; the
parity result; the n600 identity result with its retained manifest; the typed MAIN fire order
(byte-close → seal → T4) or the typed blocker; the denominator line. Commit via the serializer.
End with the own-vehicle frontier line:

`[contest-CUDA T4 n600] own-vehicle frontier: LB1 — S=0.14803010583079396, archive=180,083 B, d_seg=0.00020139, d_pose=6.37e-6, SHA-256=5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9.`
