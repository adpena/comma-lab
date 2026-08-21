# ddm_rv17 — fc3 re-score: 3 of 4 items CLEAN, one finding survives. Counter 1/3 → 0/3.

## W3-F20 (MED) — the #1181 re-test is described as routed and is in no queue

| field | value |
|---|---|
| **claim** | the seeded re-test of the prefix-derived overstatement factors was "routed to #1181" |
| **denominator first** | the fc3 memo **exists** at the cited path, 17,737 bytes — so this is a real absence, not a vacuous read |
| **measured** | `grep -c 1181` → fc3 memo **0**, canonical task ledger **0**. A repo-wide `grep -rl` over `.omx/research/` and `.omx/state/` returns only three unrelated 2026-06 nerv/snerv JSONs. |
| **what IS honest** | the magnitudes are correctly scoped. `n=150 prefix`, `prefix-based`, `PREFIX-BASED` all appear in the memo. Nothing is mis-quoted as a population factor today. |
| **why it still matters** | the `+9,934 B = +0.0066 S` correction rests on prefix-derived factors (1.046× / 1.056× / 1.034×), and the re-test that would convert them into population factors **has no home**. It will not happen unless someone remembers — which is the failure mode the ledger exists to prevent. |
| **genus** | W3-F19, one round later. A residue described as routed, present in neither the memo that describes it nor the ledger that would hold it. |
| **cure** | open the #1181 row, or name the real number if it was filed under a different id. |

**The asymmetry is the useful part.** Two rounds, two routing claims: `#1180` verified in the ledger,
`#1181` is in nothing. So the routing discipline is **real but not yet systematic** — it holds when
someone does it and there is no gate that makes it hold. That is a stronger statement than either
row alone, and it is only visible because I checked both the same way.

## What is CLEAN

**S3 — MUTATED-BENIGN confirmed, and the arithmetic is the strong leg.** `25 × 353808 / 37545489 =
0.2355862` **exactly**. This is decisive on its own, ahead of the timestamps: a back-fitted number
does not land on the canonical rate formula evaluated at the archive's true byte count. The 23-second
ordering corroborates it; the exact re-derivation carries it. I note the arm labelled its `stat`
timestamps honestly per my round-5 correction — the arm absorbed a correction that was about my error,
not its own.

**S1 — I agree with the refusal to register, and I would have refused harder.** The registrar writes
git-committed **birth** copies. S1's current bytes are **post**-mutation. Registering them would put
post-mutation bytes into git under the name "birth" — converting *"we know this was mutated"* into
*"here are its birth bytes."* That is not a gap in write-once; it is write-once **working**. A
registrar that accepted them would be a fake implementation of its own guarantee, and the
DISPOSITION-receipt path is the only honest disposition for a prereg whose birth bytes are gone.

**Item 3 — the assertion GATES, and I nearly filed a false finding here.** My first instrument
(`grep "raise .*[Tt]reatment"`) saw only the assignment at `:394` and would have supported "it only
populates a report dict." It could not see the mechanism: the function **raises `PoseLegError`
internally** at two points inside its body. Calling it *is* the gate; the assignment captures the
report incidentally. It also sits on the right path — inside the branch where both runtimes were
supplied, immediately after *"The comparator proved nothing ELSE moved. Now prove the payload DID."*
**CLEAN.** I record the near-miss because it is my round-8 trap exactly: an absence claim resting on
an instrument that could not see the thing. I checked before claiming, which is the only reason this
is a clean row and not my thirteenth correction.

**Item 2 relabel — reached 8 sites**, matching the corrected 6→8 census. Scoping honest, per above.

## What I did NOT verify

**Item 4 (the de-vacuified control) is unverified.** I ran out of budget before exercising it. I am
not scoring it clean by omission — it is simply unchecked, and it should be checked before any seal.

## Honest state

- **Counter 1/3 → 0/3.** Nine findings this wave; eight cured and verified, one open (W3-F20).
- **The substance has never moved** in 40 rounds. No wrong score, pin, digest, mis-scoped receipt, or
  unverifiable archive claim. Terminal verdict untouched.
- The wave's single genus is now fully characterised, including its **recurrence rate**: an obligation
  that stops one surface short recurs whenever the last surface is reached by memory rather than by a
  gate. Both F19 and F20 are that, one round apart.

**Own-vehicle frontier: S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600]**
