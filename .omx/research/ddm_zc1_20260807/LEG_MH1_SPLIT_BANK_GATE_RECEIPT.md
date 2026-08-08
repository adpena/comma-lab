# ZC1 Leg Receipt: MH1 Split-Bank Gate Per Receipt

Exit: BLOCKED-with-named-blocker.
Named blocker: `NO_ACTIVE_SPLIT_BANK_EXPORT_SURFACE_RECEIVER_REJECTS_EXTENSION_SLOTS`.
Axis: apparatus/source read; no scorer run.
Score claim: false.
Promotion eligible: false.
Verdict scope: FORMULATION, split-bank consumption guard for the current runtime export/receiver surface.

## RECALL EVIDENCE

Searches performed:

- `rg -n "split.bank|split-bank|banked|consumer row|parse-back|member_bytes|extension_slots|active_member" .omx/research src tools`
- Targeted read of `.omx/research/ddm_mh1_month_harvest_20260803.md`.
- Targeted read of `src/tac/optimization/ddm_runtime_exporter.py`.
- Targeted read of `src/tac/optimization/ddm_runtime_receiver.py`.

Found beyond the charter seed:

- MH1's split-bank warning is substantive: banked claims must be checked per receipt, not by headline claim.
- The runtime exporter has an `EXTENSION_SLOTS` structure, but every active member is currently unset.
- The runtime receiver's expected extension slots also have no active member, and the receiver rejects active extension slots in this build.
- No current producer schema was found that supplies receipt path, member bytes, member hash, parse-back equality, and consumer row id for each banked section.

What this changed:

- The charter asked for editor/tests if a smallest-correct wire existed. It does not currently exist on the receiver surface.
- Patching a guard without a real active producer and receiver contract would only make a formal check, not real split-bank consumption.

## Verdict

Blocked. The split-bank gate is required before composed export, but ZC1 found no active split-bank export surface to wire safely.

Minimum non-fake schema for the first producer:

- `receipt_path`
- `member_bytes`
- `member_sha256`
- `parseback_equal`
- `consumer_row_id`

The receiver must then assert these fields for each active banked section and fail closed on missing or false parse-back evidence.

## Follow-On Disposition

QUEUED-WITH-A-FIRE-ORDER:

1. Wait for or build the first real split-bank producer that has active banked members.
2. Add the fields above to the manifest/receipt surface.
3. Add receiver and manifest tests that reject active banks missing proof.
4. Only then consume MH1's split-bank gate as apparatus closure.

No `.py` edits were made in ZC1 because the safe producer/receiver surface was absent.

Own-vehicle frontier line: unchanged, `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
