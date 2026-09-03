# ddm_g8s — gen-8 compress.py single-run re-proof: PASS (2026-09-03)

## Verdict

The edited `submissions/semantic_joint_ctxmix/compress.py` (--repeats default 1, commit
f20b5e4baf) rebuilt the shipped afr1 archive from the pinned base in ONE run and the
SHA-256 pin gate passed. The operator-requested two-run removal loses nothing: the pin
plus the per-stage in-memory determinism repeats carried the full correctness burden.

## The receipt

- status PASS, determinism mode `single_rebuild_sha_pinned`
- final archive sha256 `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`,
  180,002 bytes — EXACTLY the shipped/pinned afr1 bytes (re-verified independently by
  MAIN with shasum on the delivered `retained/archive.zip`)
- wall clock 4,140.9 s ≈ 69 min for the complete five-stage replay (vs ~2.1 h under the
  old mandatory two-run default from the g8c proof: 3,495.1 + 3,499.4 s) — the halving
  is realized
- base archive: `/Volumes/APDataStore/pact/ddm_rc1/candidate_runtime_composed/archive.zip`
- store: `/Volumes/VertigoDataTier/pact/ddm_g8s_single_run_reproof/store_v2` (APFS —
  the first scaffold on ExFAT died to AppleDouble sidecars, #1122 genus; payloads
  RETAINED: run_1 stage outputs + delivered archive.zip + inputs)
- launch: counter 728, pid 87082, canonical tools/launch_detached_process.py; receipt
  G8S_SINGLE_RUN_REPROOF_DONE rc=0

## Docs-only delta note (binding scope of this proof)

The run launched from the tree at commit f20b5e4baf and verified the tree manifest at
startup. During the run, README.md + MANIFEST.sha256 changed (ae61287766 — the
operator-requested local-process disclosure + post-submission TODO; docs only). Neither
file is on the compress path; `compress.py` itself is byte-identical between the proven
and final trees. The proof therefore binds the shipping `compress.py` as-is.

## Consequence

README's reproduction claim ("One run takes about an hour of CPU... refuses unless the
rebuilt archive matches the pinned SHA-256 exactly") is now MEASURED on the exact
shipping script: 69 min, refusal gate exercised, byte-exact output. Packet remains
PREPARED-HOLD; the operator's one-line confirm is the only publish gate.
