# Consumer fix — the first-measurement fire path validates the same effective tree as `--seal`

`# FORMALIZATION_PENDING: a consumer-fix adjudication for one frozen-contract row. It changes no
definition, no digest, and no term of the score; it records why a consumer of the pr18 definition
moved. Nothing here is a law to formalize.`

**Axis: `[source + exact digest arithmetic]`. No score claim, no promotion claim, no fire ordered by
this memo. Frontier line: composition S 0.1371383667388406 @ 180,246 B [contest-CUDA T4 n600]
(move 45) — unmoved by this row.**

## The defect (MEASURED, re-derived here from the primary artifacts)

`tools/fire_modal_auth_eval.py::_first_measurement_main` validated the **raw** runtime tree:

```python
_pf_require(not validate_tree(runtime), "PREFIRE_NON_TIMING_GATE_REFUSED", "runtime upload validator refused")
```

while the `--seal` fire path at the same file's line ~1015 had, since it was written, sanitized the
macOS metadata litter first and then validated the tree **minus** exactly what the sanitize stage
removed (or, on a dry run, would remove):

```python
litter = sanitize_litter(runtime_dir, apply=not args.dry_run)
refusals = validate_tree(runtime_dir, skip=frozenset(litter))
```

ntb2's move-46 intent stages its runtime on `/Volumes/APDataStore` (ExFAT), where macOS re-creates
AppleDouble `._*` stubs the instant anything writes. I counted them myself on the intent's own
runtime dir (`/Volumes/APDataStore/pact/ddm_ntb2_proof45/frame_even/candidate_runtime`): **94**
litter files. `validate_runtime_upload_file` refuses a dot-prefixed name outright, so the raw-tree
validator refused a valid intent. The receipt is on disk:
`/Volumes/VertigoDataTier/pact/ddm_ntb2_first_measurement/run1/PREFIRE_REFUSAL_953c56d80ec846249edfbfcc5c3df328.json`
— `PREFIRE_NON_TIMING_GATE_REFUSED: runtime upload validator refused`, 2026-09-11T19:14:38Z.

MAIN's commit `d1ed3674b45d1b04d02ef1bdf07dae3a1d7ac4ab` gives the first-measurement path the seal
path's two lines and names the first problems in the refusal text. The next dry run then refused for
a second, correct reason — the frozen contract pins that file's blob:
`PREFIRE_CONTRACT_DRIFT_REFUSED: live file differs from committed blob: tools/fire_modal_auth_eval.py`
(receipt `PREFIRE_REFUSAL_17d893c82e7a42158b49423f8e424b3e.json`, 19:16:32Z). **That refusal is the
contract working.** This row is what answers it.

## The parity argument — measured, not asserted

1. **The skipped set is exactly the set that provably cannot ship.** `sanitize_litter` matches only
   `.DS_Store` and the `._` prefix. Independently, `tac.candidate_seal.runtime_digest_skip_reason`
   refuses **any** dot-prefixed path with "the upload validator refuses it, so it can never ship".
   I checked all 94 of ntb2's stubs against that function: **every one returns a skip reason.**
2. **Skipping them cannot move a pinned identity.** The intent pins
   `candidate.runtime.sha256 = bdd370bcb0b005d0…`; I re-measured `measure_runtime_digest` on the tree
   **with all 94 stubs present** and it is **byte-identical** to the pin (51 files in the digest, 94
   outside it). The litter was never inside the runtime digest, the receiver digest, or the pr18
   behavior digest. The fire path was refusing on files no custody statement ever counted.
3. **A dry run still mutates nothing.** `sanitize_litter(runtime, apply=not args.dry_run)` — a
   rehearsal REPORTS the litter and the validator skips it; only a real fire removes it. This matters
   because these fires are aimed at runtime dirs whose hashes another agent already recorded.
4. **The validator's real refusals are unchanged.** `validate_tree(..., skip=…)` skips only the exact
   relative paths that came back from the sanitize stage; every other file still goes through
   `validate_runtime_upload_file`. Nothing else was relaxed, and the refusal now names the first
   problems instead of stating that something, somewhere, refused.

## Attacking it

*Could `skip` be used to wave through a real offender?* Only a path the sanitize stage returned, and
that stage matches two literal patterns which the independent custody rule already calls unshippable.
*Could the removal change what the evaluator decodes?* The removed files are not in the archive, not
in the runtime digest, and not readable by `inflate.sh`; the T4 upload drops them regardless — which
is precisely why the seal path has removed them for as long as it has existed.

## What this row does NOT do

No definition change (`definition_change: false`). The pr18 behavior-digest definition
(`d84cab3b…`) and the pr14 risk-digest definition are untouched; this row re-pins the implementation
manifest so the frozen contract names the fire tool's new blob, and records the consumer that moved.
Appending it invalidates in-flight intents (pr17 NO-GRACE): ntb2's intent must be re-emitted against
the new latest row before a fire.
