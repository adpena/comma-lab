# ddm_ec2 verdict CORRECTION (MAIN charter error) — 2026-08-12

**The error (MAIN's, at charter time):** the ec2 admission bar was written as
"complete package < 116,716 B" — but 116,716 B is the cl1 control's TOKENS-ONLY
payload (terminal.range.bin). The control's COMPLETE package is tokens 116,716 +
model 15,088 (terminal.model.bin.xz, measured on disk) + framing 20 =
**131,824 B**. Same wrong-denominator genus as #936.

**Corrected comparison (same retained objects, sha ea70b730…):**
| | tokens | model | coords | framing | complete |
|---|---:|---:|---:|---:|---:|
| cl1 control | 116,716 | 15,088 | 0 | 20 | **131,824** |
| ec2 event-conditioned | 116,436 | 15,168 | 413 | 20 | **132,037** |

**True delta: +213 B (not +15,321).** The FORMULATION_CLOSED sign SURVIVES but
the magnitude was wrong 72×, and the mechanism reading flips: sparse-event
conditioning produced the FIRST measured full-scale learned-context token WIN
(−280 B tokens, 0.24%) — it lost only because its 413 counted coordinate bytes
+ 80 B model growth ate the win. Scope: FORMULATION (counted-coordinate
variant), NOT the family.

**Refined live hypothesis (routes to jo1):** DERIVED-site conditioning — sites
computed from the DECODED previous partition (topology-unstable loci), 0
counted bytes, rule-118 free. Needs only >80 B token savings to win; the
counted variant measured 280 B on video-derived sites. ec2's "free coordinates
closed" verdict was about EC1's video-derived sites specifically and does not
cover decoder-derived site sets.

Law update (sparse×learned-prior): admission arithmetic must compare COMPLETE
package vs COMPLETE package — never a package vs a section. Payloads retained;
no re-run needed for this correction.
