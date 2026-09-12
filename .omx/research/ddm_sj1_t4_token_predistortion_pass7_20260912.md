# ddm_sj1 PASS 7 on the move-48 field — SEALED, awaiting MAIN's fire

Tokens: `[no-triality] [p0-ledger-ok]`

Lane `ddm_sj1_t4_token_predistortion_pass7_20260912`. Axis `[macOS-CPU advisory, jg1 instrument,
DALI GT lineage]` for the seg and pose legs; bytes EXACT. **No score is claimed here.** Only
`upstream/evaluate.py` on the shipped bytes is a score, and MAIN fires.

Seal `/Volumes/VertigoDataTier/pact/ddm_sj1_pass7/SEAL_ddm_sj1_token_predistortion_pass7_contest_cuda.json`
(file sha `1ab02bdd6d93c8e7…`, seal sha `0f9bf5cceaf312d6…`, **SEAL_VALID**). Archive **179,153 B**
sha `73e41a6620bd4ea3aaf236eff9de46391857907527358e8eb40ded0925a1c214`.

---

## 1. The candidate, in one table

| leg | value | how |
|---|---:|---|
| seg | **−1.87528e−04** | 12,196 → 12,127 flipped cells, MEASURED on the candidate's OWN parse-back decode |
| rate | **+2.79661e−05** | +42 B, EXACT container delta from the subset's own real re-encode |
| pose | **+9.72977e−05** | RESOLVED 4.543569e−06 vs base 4.586763e−06, after the carrier re-solve |
| **net** | **−6.49157e−05** | 3.25× the −2e−05 bar |

Projected S **0.1363177011168108**. Carrying the local pose DELTA onto the T4 print instead of the
local pose VALUE gives **0.13632010144620363** (net −6.2515e−05); the spread is **2.400e−06**, the
pose-print class the last four packets each measured. Both clear the bar by more than 3×.

Subset: **42 of 130 edited pairs, 69 of 221 cells, 67 tokens.** Price **5.0149 bits/token** against a
**10.4889** break-even — margin **2.092×**. The carrier re-solve moved **232 coordinates over 42
pairs at ZERO byte cost** (pass 6's cost +8 B).

---

## 2. The reach, decomposed before the number was read

**221 cells / 218 tokens / 130 pairs = 1.812 % of the 12,196-cell residual.**

The charter's prior and mine disagreed, and the object split the difference:

| prior | cells | measured / prior |
|---|---:|---:|
| charter — same-field decay, ≈1 % | 120 | **1.84×** |
| this arm — object change on 161 re-rendered pairs | 340 | **0.65×** |

The charter read move 48's field as SAME-FIELD because no pair has been re-rendered *since* pass 6.
That is true, and it is not the question the search asks: the search's baseline is pass 6's INPUT
field (move 42), and pass 6's own admitted edits re-rendered 161 pairs relative to it. MEASURED
before the pass ran: 161 of 600 planes differ, 439 do not, and 90 of the stale ones carry the 154
positions pass 6 FOUND and its Lagrange sweep DROPPED.

| column | pairs | cells | cells/pair |
|---|---:|---:|---:|
| FRESH — re-rendered by pass 6's own repairs | 161 | **66** | **0.410** |
| STALE — carryover, pass 6's dropped positions | 90 | 155 | — |
| STALE — new, on pairs whose render did not move | 349 | **0** | **0.000** |

**The law holds a second time, and exactly: not one new repair exists on a pair whose render did not
move** (`repair_family_exhausts_per_object_20260910`). And F11, pre-registered with the position
list written to disk before any reach number existed: **154 of 154 re-found, fraction 1.000000, zero
pairs disagreeing** — striding (5 shards) and batch (8) matched pass 6 exactly, so those pairs
presented a byte-identical search.

### The new measurement this pass adds

Pass 6's fresh pairs re-opened at **1.156 cells/pair**; pass 7's at **0.410** — **2.82× lower**. The
difference is what re-rendered them. Pass 6's fresh pairs were moved by sister arms' **argmax-NEUTRAL**
edits; pass 7's were moved by **this arm's own repairs**. A repair CONSUMES the local slack; a neutral
edit only JOSTLES it. Label: MEASURED (two passes, one object each), not yet a law — n=2.

---

## 3. The rate leg had to be re-instrumented, and the old one was measurably wrong

This is the largest thing pass 7 found, and it was found by a gate, not by inspection.

`experiments/ddm_sj1_pass5_price.py` prices through `ddm_tc1_mixer_codec.SharedMixer` over the
FreeCorrector/HPAC rows. That was the shipped coder for moves 36–43. On move 48 it re-encodes the
pointer's OWN field to a **118,929 B** stream where the archive ships **118,896 B** — **+33 B and a
different sha** — with two independent processes agreeing byte-for-byte. Its own gate 2 refused it
(`CONTROL IDENTITY FAILED`). A deterministic instrument that does not reproduce the object is not
noise; it is a different coder.

Four pointer moves this arm had not absorbed are the cause:

| move | what changed | effect on the body (RX1 sections, m43 → m48) |
|---|---|---|
| 44 rlc5 | raw `TC1M` rider → counted `RLC1` rider | carrier 18,610 → 18,450 B |
| 45 pc3 | CAP1 predictor refit | — |
| 47 hpr1 | HPAC prior retrained | hpac 11,911 → 11,629 B |
| 46/48 | frame-embedding even rounding INSIDE the receiver | tail 119,969 → 119,056 B (same field, new prior) |

`semantic` is BYTE-IDENTICAL across all of them, which is why the render instrument was untouched and
the price was not.

**The cure is `experiments/ddm_sj1_rlc1_price.py` (commit `65e55bd25`).** It re-implements no coder:
it drives the RECEIVER'S OWN `decode_production_tokens` loop on the pointer tree's runtime with the
true symbols injected at `NativeDecoder.decode`, exactly as `experiments/ddm_hpr1_shape_price.py`
does, and the only delta is that the FIELD is a parameter (hpr1 prices a MODEL change on a fixed
field; this arm prices a FIELD change on a fixed model). hpr1's rail was deliberately NOT edited: it
binds its `producer` sha into every INPUTS.json and encoder-state receipt, so an edit would refuse
every in-flight dpi1/hpr1 resume ([[binding_hash_whole_module_kills_checkpoints_20260909]]).

Its credibility gate is in code and fail-closed: `encode --field control` must repack to the
pointer's archive byte-identically before any candidate price is returned. **MEASURED: two
independent processes both repack to `d830edd3…` at 179,111 B exactly**, and the first-order ideal
118,895.707 B sits 0.29 B from the realized 118,896.

**Law:** the rider is a POINTER-OWNED object. The old `split_tail`/`build_tail` pair would have
rebuilt the tail as `TC1M + weights + stream`, silently reverting move 44's counted-rider cure inside
a tail that still parses. The rider now travels VERBATIM and its length comes from the receiver's own
reader, never re-derived here.

---

## 4. Frame 0 — measured inside the admission, NOT adopted, and this time on economics

Swept the **47 pose-bound dropped pairs**. The adopt stage reports 14 pairs, +12 B selector, ΔS
−4.03e−05 **standing alone**. Inside the admission it loses:

| variant | pairs kept | cells | selector Δ | base bytes | score_modelled |
|---|---:|---:|---:|---:|---:|
| no frame 0 | 42 | 69 | 0 B | 179,111 | **0.1363174850994821** |
| frame 0, fixed-point check | 51 | 86 | +12 B | 179,123 | 0.13631857699107786 |

**Δ = +1.09e−06 — the frame-0 variant is WORSE.** And the fixed point **BREAKS**: 5 of the 14 adopted
pairs (376, 416, 455, 472, 501) fall outside the kept set.

Why the standalone −4.03e−05 does not transfer: it is measured on pairs the sweep DROPS, whose
shipped plane is the live row's, so its pose credit is a BASE-configuration credit, not this
candidate's. Composing the two inside one admission is what prices it honestly.

Pass 6 declined frame 0 for a BUILD reason (this arm's `close` has no selector splice) with the
economics 0.075× of a bar in favour. Pass 7 declines it because it LOSES. The build blocker still
stands and is still worth naming for a successor field; it is no longer the binding reason.

---

## 5. Gates, each MEASURED on the shipped bytes

`F1` flips_before 12,196 EXACTLY · `F2` 130 pairs edited = 130 claimed, every other plane
byte-identical to move 48's · `F3` d_seg_after = (12,196 − 221)/117,964,800 exactly · **`F4`
12,127 predicted = 12,127 measured on the candidate's own parse-back `0.raw`, zero cells
disagreeing** · `F5` twin encodes byte-identical across independent PROCESSES for all three fields ·
`F6` control repacks the pointer byte-identically, twice · `F7` pose base on move 48's own
configuration, pm2 gate ratio 0.99929 · `F8` admitted on the RESOLVED pose, never the stale
9.488294e−05 (20.7× the base before the re-solve) · `F9` subset priced by its own real encode, +42 B,
never the ledger sum (41.760) · `F10` all 88 dropped pairs ship move 48's plane, fail-closed ·
`F11` 154/154, fraction 1.000000 · `F12` the 64 B RLC1 rider carried verbatim · `F13` close identity
control reproduces the staged body at `0683aa83`.

Parse-back: `pin_check` PASS through the candidate tree's own `inflate.py`, decoded field matches the
admitted field, 890.08 s. Public smoke: candidate and frontier BOTH `REACHED_TOKEN_DECODE` at
240.02 s and both reach the CUDA gate. Decode wall clock: **inherited** from move 48's own
`t4_direct` leg (1,023 s on Tesla T4 against a 1,260 s ceiling).

### Two things the seal caught that inspection would not have

1. **`patch_inflate_pins` rewrites the two archive pins inside `inflate.py` but does not touch
   `MANIFEST.sha256`**, so the candidate tree listed the POINTER's `inflate.py` hash against its own
   patched bytes. Move 48's shipped tree is self-consistent (0 mismatches, checked); the candidate
   had exactly 1. Amended, receipt `candidate/MANIFEST_AMENDMENT.json`. **Note for the arm: pass 6's
   candidate tree carries the same inconsistency and would fail this check today.**
2. The first smoke receipt predated that amendment, so its tree sha had drifted and the seal refused
   the receipt as not naming the runtime being sealed. Re-run on the final tree.

The candidate tree differs from move 48's in exactly `archive.zip`, `inflate.py`, `MANIFEST.sha256`.

### A harness gap, found by its own control

Both `parseback` and `public-smoke` first died with `KeyError: 'RLC1_GEOMETRY_LIBRARY'` — **on the
FRONTIER's own shipped bytes as well as on the candidate's**. `inflate.sh:76` exports that library
after compiling `runtime/rlc1_geometry.c`; `ddm_sj1_joint_admission.py` runs the receiver's Python
path directly and does not. The frontier leg failing identically is what proved it was the harness,
not the candidate. Cured by building the `.so` with inflate.sh's exact command
(`cc -O3 -std=c11 -shared -fPIC`) and exporting it. No check was weakened.

---

## 6. Custody

- Store `/Volumes/VertigoDataTier/pact/ddm_sj1_pass7/` — archive, seal, admission, fields, codes,
  parse-back result + `0.raw` (3.66 GB), seg leg, all pre-registrations.
- Second copy `/Volumes/APDataStore/pact/ddm_sj1_t4_token_predistortion_pass7_20260912/custody_pass7/`
  — archive (sha verified `73e41a6620bd4ea3…`), seal (sha verified `1ab02bdd6d93c8e7…`), admitted
  field, resolved codes, and the five pre-registration/verdict receipts.

## 7. What this does NOT claim

No score of any kind: every S here is a PROJECTION on measured legs. No CPU-axis claim. No transfer
of pass 6's 5.2537 bits/token to this body — the price is per object, and move 48 recoded the tail
under a refit prior. The 2.82× fresh-rate drop is n=2, not a law. The 2.400e−06 local-vs-T4 pose
spread is carried, not absorbed; the exact row decides.

## 8. Next from here

1. **The family is NOT closed on this field** — 1.812 % is above the arm's own 1 %/pass rule, and
   3.25× the bar was admitted. A pass 8 on the successor field is warranted; expect the carryover
   column to shrink (pass 7 ships 69 of the 154 it inherited) and the fresh column to follow the
   0.410/pair rate over whatever pairs the next move re-renders.
2. **Backfill the RLC1 rail into the arm's older tooling** — `ddm_sj1_pass5_price.py` is now
   documented as the wrong coder for any body from move 44 on; it still refuses correctly, but a
   successor should not have to re-discover that.
3. **The MANIFEST/pins gap is a class**, not an instance: any arm that calls `patch_inflate_pins` and
   then seals will hit it. Worth a guard.
4. **Frame-0 selector splice** remains a named build item, now with a measured reason NOT to spend on
   it for this field.

<!-- # FORMALIZATION_PENDING: packet input; the equations leg is written by tools/pointer_move_packet.py --equations-leg at harvest, on the exact row -->

Own-vehicle frontier (unchanged by this arm — MAIN fires):
**S 0.13638261682704697 @ 179,111 B [contest-CUDA T4 n600]** (move 48).
