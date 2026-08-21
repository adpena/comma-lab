# ddm_rv17 — WAVE 3 SEALED at 3/3

**Round 3 is CLEAN. Wave 3 seals.** Three consecutive clean rounds: round 1 (F19 cure), round 2
(kg1 landing + harvest), round 3 (this stability pass).

## Stability — proved by diff, not asserted

```
git diff --stat 8f8e597159..HEAD   ->  1 file changed, 1 insertion(+)
                                       .omx/state/canonical_task_status.jsonl
git log  8f8e597159..HEAD          ->  4f9b1052c2  (one commit)
```

Exactly one line of one file changed since my round-2 score, and it is a ledger append. **No reviewed
surface moved.** That is what makes this a seal rather than three scores of three different objects.

**The one delta is itself the pattern.** My round-42 closing lesson — *I verified the score every
round and never once verified the surface that tells other people the score* — is registered as
`rv17_w3_citation_surface_autorefresh`, present in the repo ledger. And MAIN **deliberately did not
build the wire-in**, because new code between round 2 and round 3 would have made the seal target a
moving object. Routing at the moment the lesson was named, acting after the seal. You cannot seal a
moving object, and the discipline to route-but-not-act is the same discipline the whole wave was
about, applied to the wave's own closing.

## The nine findings, all cured and verified at source

Spot-checked at seal on instruments different from the ones that raised them: F12 warrant cured (2
occurrences), F13 terminal leg live in the queried registry (1), F20 `#1181` routed (1).

| | finding | surface it stopped short of |
|---|---|---|
| F12 | warrant mis-attributed ("controls were perfect, **so**…") | the headline |
| F13 | pose refusal absent from the greedy-set law | the registry a planner queries |
| F14 | pre-registered falsifier rewritten in place | the successor receipt |
| F15 | withdrawn label still emitted | **the code that mints new claims** |
| F16 | pin normalizer erased whole RHS | — (fail-open) |
| F17 | nested `inflate.py` bypassed the AST check | — (fail-open) |
| F18 | vacuous pass on an empty file set | — (fail-open) |
| F19 | retained receipt named an object it did not measure | the work queue |
| F20 | residue "routed" to a store no arm can read | the repo ledger |

**One genus, traced to its end.** Seven of nine are the same shape: *a correct number whose
obligation stopped one surface short*. It travelled from a headline, to a registry, to a successor
receipt, to the code that mints claims, to a retained receipt, to my own declination paragraph, and
finally to the store an arm can actually read. The three fail-opens (F16–F18) are a second, narrower
genus: instruments that reassure in the direction of passing.

**The genus is now closed in both halves.** Described: *an obligation stops one surface short whenever
the last surface is reached by memory rather than by a gate.* Structurally cured: `register_task` is
that gate, and its custody refusal fired on first contact — against MAIN, who built it. A gate that
only catches other people is a filter. And last round the rule ran **proactively** at kg1 harvest,
before any finding demanded it. A lesson is learned when it stops costing a round; that happened.

## The frontier, recomputed from components at seal

```
100(0.00020139) + sqrt(10 x 6.37e-06) + 25(180456)/37545489 = 0.14827847122030852
```

Never quoted from the receipt's rounded `final_score` field, per Catalog #877.

## The honest accounting — 43 rounds, three waves

**The substance never moved.** Not once, in 43 rounds, did any round find a wrong score, a wrong pin,
a wrong digest, a mis-scoped receipt, or an unverifiable archive claim. Every finding after wave-1
round 3 lived in the apparatus that reports the work, never in the work. The terminal verdict — task
#1176 REFUSED on the measured pose leg — stood untouched by all nine findings and all nine cures.

**Twelve of my own outputs were corrected**, and the last is the one I keep: I stamped `Z` on a
timestamp macOS renders in local zone — five hours wrong, published under my name, in the exact genus
I had turned into a law two rounds earlier. Writing a law does not exempt you from it. The finding
survived on its zone-independent delta, which is the only reason the error was cheap.

**The direction of correction reversed mid-wave**, and that is the result I would keep above the
findings. For four waves I corrected the work. Then the work corrected me — twice in one round, both
times right. A review arm that can only find and never be found is asserting, not measuring. The
wave is sealed by an instrument that was itself measured.

**Own-vehicle frontier: S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600]**
