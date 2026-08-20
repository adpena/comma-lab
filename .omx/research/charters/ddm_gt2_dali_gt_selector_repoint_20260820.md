# ddm_gt2 — finish the DALI-GT cure: repoint the pair SELECTOR, declare the lineage

## MANDATE

Task #1142 is the unwired-cure P0. `ddm_na10` measured that our GT pose table built from
PyAV-decoded frames disagrees with the DALI-decoded contest lineage by **0.887×–1,627× per pair**
(seg 1.43×).

**Part of the cure has already landed — do NOT redo it.** Commit `809199d24f23b3298f8407574870ede35c0f7874`
repointed `qs1.GT_POSE` and the `mt1` defaults to the DALI table (`gt_first6_dali_n600.npy`,
materialized from `gt_cache_dali.pt`; MSE vs PyAV = 1.406151e-04, reproducing the pi2/na10 additive
constant C to 5 digits = table+ordering verified); 9 importer consumers inherit through `qs1`, and
the PyAV table survives as `GT_POSE_PYAV_ADVISORY` for lineage-labeled comparisons. `po1` was left
untouched deliberately (round-local custody GT, not the shared table) — respect that.

**What remains is the live head:** the pair SELECTOR that ps135b/ps1u rank by (`top_mass_pairs`)
is still PyAV-ranked — **Spearman 0.122 vs DALI, top-30 overlap 1/30** — so we are choosing WHICH
pairs to attack from a ranking essentially uncorrelated with the shipping objective. Plus the 11
undeclared `GT_LINEAGE_OK` sites and the missing gate.

Close it: repoint the selector, declare the lineage everywhere it is load-bearing, re-run the
$0 legs, and leave a gate that refuses a silent PyAV-GT consumption.

## THE WORK, in fire order

1. **Repoint the pair selector.** Find every producer/consumer of `top_mass_pairs` (and any
   sibling ranking used to pick pairs for solves) and rebuild the ranking from the DALI GT table.
   Report the new top-30 and the overlap with the old one. If a consumer's result was materially
   selected by the old ranking, SAY SO next to that result — do not silently re-rank history.
2. **The 11 `GT_LINEAGE_OK` declarations.** na10 identified sites that consume a GT table without
   declaring which lineage. Add the declaration at each site. A declaration is only honest if the
   site actually reads the lineage it names — verify, don't annotate.
3. **Two-landing gate.** A warn-only preflight check that refuses a GT-table consumption with no
   lineage declaration. It MUST ship with an EXECUTED positive control (a deliberately undeclared
   consumption → rc≠0, output pasted). No gate without its red run.
4. **$0 re-runs unblocked by (1)+(2):** `qs3` unlock and `sq2 R8` (~4 h, $0, carrier re-solve).
   Run what fits your budget; leave a precise fire-order for what does not, with the exact command.
5. **ps135b / ps1u**: prepare the DALI re-run to READY (the $0.16 T4 leg is MAIN's to fire — do
   not dispatch). Deliver a sealed, exact command line.

## WHAT WOULD MAKE THIS ARM WRONG

If the selector repoint turns out to be a no-op — because every live consumer already reads DALI,
or because the two rankings agree on the pairs anyone actually solved — then the P0 framing is
stale and the honest deliverable is "the cure was already consumed, here is the proof." That is a
GOOD outcome. Measure before you rewire; the 1/30 overlap figure is inherited and must be
re-derived at source before you act on it.

## HARD CONSTRAINTS

- `upstream/` is READ-ONLY. Do NOT touch `submissions/robust_current/jg5_sub015_runtime/`.
- **Do not fire any Modal dispatch.** MAIN owns dispatch; a live T4 row is in flight
  (`fc-01M0FZKTSY9ZRH2TEX27TZACKP`) — do not touch `/Volumes/APDataStore/pact/ddm_rr8/`.
- `.py` edits: 2 genuine review-tracker passes; commit via `tools/commit_autosha.sh`.
- Detached launches ONLY via `tools/launch_detached_process.py`. Long legs: keep them resumable.
- ALWAYS KEEP THE PAYLOAD: any run that materializes a GT table or a ranking persists the bytes
  (SSD tier) with sha256 + byte count in the result JSON — never scalars alone.
- Negative verdicts carry `verdict_scope:` at the narrowest supported level.

## OPTIMAL FORM

- **Family reference:** the canonical wrong-objective cure at its landed form — repoint the
  CONSUMER, declare the lineage at the read site, and gate the undeclared read (the #351 scope
  extension pattern from `ddm_sp2`, which class-protected exactly this genus). SCOPE reductions
  permitted (a subset of the $0 re-runs, stated per row). MECHANISM reductions FORBIDDEN: no
  annotation-only "declaration" that the code does not honor; no gate without an executed red run;
  no re-ranking of a historical result without a note next to that result.
- **Provenance pins:** the landed qs1/mt1 repoint commit `809199d24f23b3298f8407574870ede35c0f7874` ·
  na10 commits `3225e3a880477505b46bde04780faf71198567d4` (COMPLETE, 24 verdicts re-graded),
  `7ec4b62c04f1def8ef7cc81f8aa22fb35531f0ee` (re-grade table + fail-closed GT-lineage gate ask),
  `a65fab150c110930b5908596069dff8e244531c9` (ps135b/ps1u REOPENED) · sp2 class-protect commits
  `69ce2b3c0af0a361554d48dfeb799424311cab75` (GT decode-lineage warn-only scope extension over
  catalog 351) and `98f24b337902f3260a876b0f70724531640b52e3` · the DALI GT artifact
  `gt_first6_dali_n600.npy` / `gt_cache_dali.pt` — locate and pin path + sha256 in your memo; if
  no sha-pinned artifact exists, that absence IS finding #1.
- **PRIOR-LAW PREDICTION (derived, falsifiable):** given `809199d24f` already repointed qs1 + 9
  importers, the standing law predicts the residue is the objects qs1 does NOT mediate. Predict the
  SELECTOR is still PyAV-ranked and repointing it changes the top-30 by >50%, and that ≥2 sites
  outside the qs1 import graph still read PyAV undeclared. FALSIFIER: if the selector already reads
  DALI (or the top-30 overlap is >80%) and ≤1 site is undeclared, then `809199d24f` closed the row
  and the honest deliverable is that proof — report it and close #1142 instead of manufacturing work.

## DELIVERABLE

`.omx/research/ddm_gt2_dali_gt_selector_repoint_20260820.md` — the measured consumer census
(site · lineage-before · lineage-after · commit), the old-vs-new top-30 with overlap, the gate +
its executed positive control, per-row results of the $0 re-runs, and a MAIN fire-order for the
T4 legs. End with the own-vehicle frontier line.
