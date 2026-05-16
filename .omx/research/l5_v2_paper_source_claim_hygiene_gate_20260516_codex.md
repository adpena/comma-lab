# L5-v2 Paper Source Claim Hygiene Gate - 2026-05-16

## Scope

L5-v2 / Time-Traveler paper, source, and claim-boundary hygiene.

## Change

Added a focused regression guard in
`src/tac/tests/test_l5_paper_fidelity_claim_hygiene.py` that requires the
current L5-v2 source-basis, architecture, and campaign ledgers to keep primary
source URLs/DOIs and claim blockers together.

The guard pins:

- official comma challenge and public PR95/PR101/PR106 URLs;
- HNeRV, DCVC-RT, TeCoNeRV, Slepian-Wolf, and Wyner-Ziv primary sources;
- retrieved-date provenance;
- planning-prior wording;
- paired CPU/CUDA exact-eval wording before score claims.

## Result

The first run failed because the L5-v2 source-basis ledger used generic
CPU/CUDA axis wording but did not explicitly require paired CPU/CUDA exact-eval
custody. The ledger now says that directly.

## Verification

- `.venv/bin/python -m ruff check src/tac/tests/test_l5_paper_fidelity_claim_hygiene.py`
- `PYTHONPATH=src:. .venv/bin/pytest src/tac/tests/test_l5_paper_fidelity_claim_hygiene.py -q`
