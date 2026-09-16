**PARTIAL: byte controls passed; the joint exchange probe remains unimplemented and unmeasured.** The common contract requires scorer-slot ownership; jrd1 had no assignment while pd5 held the host claim. The scorer phase is queued.

| Exchange row | Measured pairs | Median / IQR S/B | Median / IQR bits |
|---|---:|---|---|
| Field-only | 0/24 | Unmeasured | Unmeasured |
| Renderer-only | 0/24 | Unmeasured | Unmeasured |
| Joint | 0/24 | Unmeasured | Unmeasured |

**Joint/single ratio:** undefined. **Global collateral:** unmeasured, 0/600 pairs. **Gate verdict:** UNMEASURED; no scientific family is closed.

**MEASURED:** Twin full-tail encodes reproduced the 119,097-byte stream and exact 179,332-byte base archive. Six renderer byte controls produced changes from **−4 to +29 B**; these were instrumentation checks, not gradient-selected descent. All 14 control archives passed repeat/parse-back checks. Retained **164,677,657 B across 728 files**, below the 3 GiB cap.

Complete results, provenance, payload paths and SHA-256s:

- [Receipt](/Users/adpena/Projects/pact/.omx/research/ddm_jrd1_20260916/RECEIPT.md) and [final handoff](/Users/adpena/Projects/pact/.omx/research/ddm_jrd1_20260916/HANDOFF.json).
- [Retention manifest](/Users/adpena/Projects/pact/.omx/research/ddm_jrd1_20260916/RETENTION_MANIFEST.json), SHA-256 `3aa4b68667ed24997163e58266648d53c696eb554aa77007f62f19b145211a3c`.
- [Verified commit bundle](/Users/adpena/Projects/pact/.omx/research/ddm_jrd1_20260916/serializer/20260916T011030.702168Z-87158/intended-commit.bundle), SHA-256 `f89a4d046a6bf39151caed97e092a46f42f077e6acc41ac9cd7b8ee8df4294f3`.

**Serializer rc 17:** sandbox Git-object writes were denied. Fallback commit `b72db5de43bdc1d73dec6fb33aefcb69618235ec` is verified but **not landed**; the format-patch is retained beside the bundle. The shared index is unchanged.

No scorer, pose solve, training, burn, Modal, fire, or packet ran. Upstream, PR/sealed trees, receiver, contract code, shipped parameters, pd1–pd5 sources, and Vertigo remained untouched. Mutated payload copies stayed under the owned APDataStore directory; nothing was deleted or moved. Two code-review passes and boundary checks passed. The frontier did not move.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN:** land the verified bundle and post-attempt receipts. **Consumer store:** `.omx/research/ddm_jrd1_20260916/serializer/`. **Fire trigger:** Git-write-capable checkout and successful bundle/file re-verification.
- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN, executor jrd1 successor:** implement and measure the complete three-row probe, including all-600 collateral. **Consumer store:** `/Volumes/APDataStore/pact/ddm_jrd1/`. **Fire trigger:** explicit scorer-slot assignment after competing owners release it, current input-pin verification, and resolution of zero-byte exchange and knee-tolerance definitions. See [fire order](/Users/adpena/Projects/pact/.omx/research/ddm_jrd1_20260916/FIRE_ORDER.md).

**LIVE-HYPOTHESES**

- Joint field/renderer compensation remains plausible: the prior single-axis closures do not measure simultaneous compensation.
- Byte-releasing int4 changes exist; whether any preserve or improve scorer outputs remains untested.

**DEAD-ENDS**

- Assuming an int4 step costs zero bytes is invalid: actual archive changes ranged from −4 to +29 B.
- The older RC1S pricing loader is incompatible with the current SM1S representation.
- No scientific joint-descent path was closed by these byte controls.

composition S 0.13620226906030858 @ 179,332 B [contest-CUDA T4 n600] (move 52)