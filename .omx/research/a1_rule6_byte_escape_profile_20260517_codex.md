# A1 Rule #6 byte-escape profile

This is a planning ledger, not a score claim. It profiles the current A1 archive sections before any Rule #6 byte-only bolt-on is treated as live frontier work.

## Authority

- score_claim: false
- promotion_eligible: false
- ready_for_exact_eval_dispatch: false
- contest_axis: null

## Source

- archive: `submissions/a1/archive.zip`
- archive bytes: `178262`
- archive sha256: `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`
- member `x` bytes: `178162`
- member sha256: `8e664385af0a25ec98bd02d97b697fbf0d2bb3c2d954f5aa5c95b5131330a243`

## Section Map

- header bytes: `4`
- decoder section total: `162168`
- decoder blob bytes: `162164`
- latent blob bytes: `15387`
- sidecar blob bytes: `607`

## Latent Raw-LZMA Sweep

- source filter roundtrip exact: `True`
- raw latent bytes: `16912`
- valid candidates: `750`
- invalid candidates: `300`
- best bytes: `15387`
- best delta vs source bytes: `0`
- best filter: `dict=4096 lc=3 lp=0 pb=0`

## Sidecar Runtime Formats

- current sidecar bytes: `607`
- decoded valid corrections: `597`
- no-op pairs: `3`
- entropy floor estimate bytes: `603`
- gap to entropy floor estimate bytes: `4`
- 600-byte runtime format fits current choices: `False`
- max encoded choice value: `445`
- usable minimum runtime-supported sidecar bytes: `607`
- delta vs current sidecar bytes: `0`

## Conclusion

- classification: `saturated_byte_only_current_runtime`
- best supported delta bytes without runtime change: `0`
- rate-term delta if component distances unchanged: `0`

Do not retread generic arithmetic over A1 latent or sidecar bytes. Move Rule #6 to component-changing bolt-ons, per-section byte-consumption proofs, or a new runtime grammar before claiming score movement.
