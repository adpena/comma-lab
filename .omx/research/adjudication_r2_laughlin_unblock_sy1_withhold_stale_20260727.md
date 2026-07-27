# MAIN ADJUDICATION (2026-07-27): R2 Laughlin-quanta pricing UNBLOCKED — the sy1 withhold is STALE

**Ruling.** sy1 (`codex_findings_ddm_sy1_pantheon_synergy_20260725_codex.md` finding 2 + FEED-603-sy1 row
"RD1 supplies 162 uint8 histograms → PREMISE_FALSIFIED") withheld R2 (Laughlin-style matched-quanta
quantization) on the ground that RD1 custody carried "162 typed dual rows, not 162 uint8 histograms; the
per-dimension absolute-step histogram is explicitly absent." That ground is FALSIFIED BY DIRECT CUSTODY
INSPECTION: the ev1 receipt
`.omx/research/ddm_ev1_campaign_evidence_joins_20260724T191623Z/ddm_ev1_campaign_evidence_join_receipt.json`
carries, per RD1 bucket row, the key `receiver_uint8_abs_step_histogram` as a 256-bin array (verified by
structural walk 2026-07-27: rd1_evidence/bucket_rows[*]/receiver_uint8_abs_step_histogram, list len=256,
plus a top-level histogram_coder record), and ev1's findings state "RD1 has 162/162 exclusive byte homes
and 162/162 exact uint8 histograms." sy1 read the older RD1 receipt, not ev1's join — a staleness-at-
consumption instance of the named confound class (freshness at CONSUMPTION; input-hash lineage).

**Consequences.** (1) R2 is PRICEABLE: the per-dimension absolute-step histograms exist with coder custody
— histogram-matched quantization enters the composed-set races (sy1 §S5-Q gains an R2 arm alongside
Q8/C3/v5, same coordinates/seed/byte accounting). (2) rd1's 162 dual-cell backfill may consume the ev1
histograms directly. (3) sy1's verdict_scope stands corrected at INSTANCE (its RD1-receipt read), the
matched-quanta family OPEN as it said — only the "histogram absent" premise is struck. (4) Uniform int8
remains a control, not a law.

STORES CONSULTED: ev1 receipt (structural walk, this ruling) · ev1 findings L18 · sy1 findings finding-2 ·
FEED-603-sy1 · feedback_sy1 memo §2/§S5-Q · fable eureka A2 (a381cd5166) · staleness-confound law.
Bar honesty: no score claim; consumers are pricing chains. Distance to bar unchanged.
