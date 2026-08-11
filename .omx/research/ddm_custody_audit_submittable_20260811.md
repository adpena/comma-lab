# Submittable-archive custody audit + full mirror (operator 08-11 "no signal lost... submittable archive preserved")

## Verdict: ALL load-bearing objects verified at their pins; full mirror landed on the second tier.

MEASURED [byte/custody apparatus, scorer-free]:
- **lc2 packet COMPLETE + VERIFIED**: archive.zip 187,226 B sha f154f0ab… (matches the contest-CUDA
  row pin) + inflate.sh/py + receiver/carrier/hpac modules + runtime-dependencies.json with
  constriction declared (rr3 #1008 present). Location: ddm_lc2_20260810/submission_modal_stage/.
- **cp135 packet COMPLETE + VERIFIED ×3 copies**: adapted_runtime/archive.zip 186,252 B sha
  6eb1a3b7… (the effective-frontier row) + winner candidate + archive.repeat.zip repeat-identity
  pair (split_brotli_per_section_opt_cap1_metadata__rc64).
- **hy1 capstone head VERIFIED**: C1 solved tokens 117,964,800 B sha 2b0bdf… + F26 HPAC wire
  114,717 B sha 9def0a4b… (both match hy1's pins).
- **ps135 solve retention CONFIRMED payload-law compliant**: every pass retains input_archive.zip
  (= prior pass's selected archive) + 600 archive_variants + receipt.
- **HAZARD FOUND: VertigoDataTier at 99% (24 Gi free).** Sufficient for the solve's remaining
  passes (~0.5 GB) but NOT safe for terminal T1+ builds. ROUTING DECISION: all new heavy stores
  route to /Volumes/APDataStore/pact/ until Vertigo is certified-or-blocked cleaned post-terminal.
- **MIRROR LANDED**: /Volumes/APDataStore/pact/submittable_custody_mirror_20260811/ (5.1 GB,
  MIRROR_MANIFEST.json, key shas re-verified ON the mirror). Refresh policy: re-rsync leg_a at
  the solve terminal.

No pointer moved; this is custody, not mechanism evidence.
