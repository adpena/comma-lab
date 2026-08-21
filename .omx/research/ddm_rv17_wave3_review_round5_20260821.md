# ddm_rv17 — wave-3 round 5 (re-score): all five cures HOLD · round CLEAN · counter 0/3 → 1/3

**Verdict: CLEAN.** Every ddm_fc2 cure verified at source with my own instruments. No new finding
survives. **I also confirm both of the arm's corrections of my round-4 memo, and the first is mine.**

---

## FIRST — my own error, owned plainly

**Correction (a) is CONFIRMED and it is MINE.** My round-4 instrument was
`stat -f '%SB' -t '%Y-%m-%dT%H:%M:%SZ'`. macOS `stat` renders the time in the **local** zone; the
trailing `Z` was a literal I supplied. Measured:

```
birth epoch 1787283740
  my round-4 memo published : 2026-08-20T22:42:20Z     <- WRONG label
  true UTC                  : 2026-08-21T03:42:20Z
  true local                : 2026-08-20T22:42:20-0500
```

Every absolute timestamp in round 4 is off by five hours and carries a timezone the instrument never
produced. **The finding is unaffected** — the 25-minute gap is a difference of two same-zone stamps
and is zone-independent, as is "born 25 minutes later." The delta was always the load-bearing part.

This is my twelfth corrected output this wave, and it is the most pointed one: **it is the exact
genus I turned into a law.** I wrote that an instrument's own units are part of the claim, and then
stamped a UTC suffix on a local reading. A law you write does not exempt you from it. Round 4's
timestamps are withdrawn as absolutes; its deltas and its verdict stand.

**Correction (b) is CONFIRMED.** My round-4 memo said nonexistent dirs "all return `PAYLOAD_ONLY`."
Overstated — the asymmetric one-missing case was already refused pre-fix; only **both**-missing fell
through. I carried the claim under my name, so I own the overstatement. The residual it described
was real but narrower than I wrote it.

---

## THE CURES

| # | cure | verified at source | verdict |
|---|---|---|---|
| **F14** | append-only DISPOSITION receipt + `tools/register_prereg.py` | original prereg sha `bd0b293c7441c074…` **unchanged**; `FS3_DROP_FALSIFIER.DISPOSITION.json` present beside it; registrar exists | **HOLDS** |
| **F15** | withdrawn label reaches the code | live key is now `carrier_DERIVED_extrapolated_leg2` (`:377`); the old string survives **only as its own withdrawal notice** — `"carrier_MEASURED_leg2 -- WITHDRAWN as overclaimed (rv17 wave-3 W3-F7)"` at `:58` / `:105` | **HOLDS** |
| **F16** | pin RHS must be a literal | `:172` `if not isinstance(value, _ast.Constant) or type(value.value) not in (str, int)` — a side-effecting RHS can no longer normalise away | **HOLDS** |
| **F17** | exact-path keying on both tests | `:230` `k not in PAYLOAD_FILES` (was basename) **and** `:234` collects every path whose basename is `inflate.py`. `lib/inflate.py` now lands in `unexpected` **and** is AST-checked — both halves of the asymmetry closed | **HOLDS** |
| **F18** | tri-state + observability | `:248` `inflate_body_identical is True` with the reason in-line ("None means the check did not run") | **HOLDS** — see adjudication |

**F14's cure is better than the cure I proposed.** I asked for a v2 registration. `register_prereg.py`
supplies write-once **git-committed birth copies**, which makes the next in-place edit *detectable*
rather than merely *disclosed*. And its MUTATED control reproduces my exact F14 signature via
`json_key_diff` — the control proves the detector fires on the real defect, not on a synthetic one.
The population sweep (25/25 prereg-like files had no birth copy; 3 show `mtime > birthtime`, routed
to #1179) turns one instance into a measured class. That is bug → class → family done properly.

**F15's cure found a surface I missed.** My memo named two emitters. The arm found a **third** (the
`compose_legs` docstring) and executed a real-CLI control. My enumeration was incomplete; I only ever
grepped for the assignment shape. It also swept for the same shape elsewhere — 6 queued in
`road_undriv_bulk_field.py`, 4 candidates cleared as genuine.

**F16's deliberate divergence from the charter is correct and I endorse it.** I wrote "string-only."
The arm measured the real pins first, found `ARCHIVE_BYTES` is an **int**, and widened to `(str, int)`
— string-only would have refused the real pair. Measuring the object before writing the rule about it
is the discipline; following my instruction literally would have shipped a false refusal.

---

## ADJUDICATION — does routing the archive-must-differ assertion to the caller satisfy F18?

**Yes. No gap survives.** Three reasons, in order of weight:

1. **The vacuity is now visible in the same object as the claim.** `:257` emits
   `"files_compared": len(set(hb) & set(hc))` **beside** `:262` `"verdict"`. On an empty comparison a
   reader sees `verdict: PAYLOAD_ONLY` next to `files_compared: 0`. My concern was never that the
   module must *refuse* — it was that the strongest-sounding verdict could be emitted from an empty
   set **without the reader knowing**. The denominator now travels with the claim. That is the
   canonical cure for a vacuity defect: report the denominator.
2. **The same-directory refusal was declined for a verifiable reason, and declining was right.** A
   pre-registered control requires same-dir-twice to PASS. Refusing it would have meant **mutating a
   pre-registered control to fit a cure** — the precise defect F14 names. The arm surfaced the
   condition as `base_and_candidate_are_the_same_directory` (`:261`) instead. Choosing observability
   over quietly rewriting a prereg is the F14 lesson applied one round later, and I would have been
   wrong to demand the refusal.
3. **The remainder is converted, not re-recorded.** The archive-must-differ assertion went to the
   caller layer under #1179 with an owner. My own wave-2 law is that a caveat is a debt — pay it or
   convert it, don't re-record it. This one was converted into a tracked item with a home.

The one condition under which the residual becomes live: a consumer that reads `verdict` without
reading `files_compared`. That is a consumer defect, not a module defect, and it is now detectable
from the receipt alone.

---

## Honest state

- **Round CLEAN. Counter 0/3 → 1/3.**
- Seven findings raised this wave (W3-F12 … F18); **five are cured and verified**, two (F12 warrant,
  F13 registry pose leg) remain open with MAIN.
- **The substance still has not moved** in 36 rounds: no wrong score, pin, digest, mis-scoped receipt,
  or unverifiable archive claim. The terminal verdict — TASK #1176 REFUSED on the measured pose leg —
  is untouched by every finding and every cure above.
- **What changed this round is the direction of the corrections.** For four waves I corrected the
  work. This round the work corrected me, twice, and both times it was right. A review arm that can
  only find and never be found is not measuring — it is asserting. Round 4's timestamps were wrong
  under my name and I withdraw them; the finding they carried survives on its zone-independent delta.

**Own-vehicle frontier: S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600]**
