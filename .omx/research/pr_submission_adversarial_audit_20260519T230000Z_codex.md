# PR Submission Adversarial Audit (Re-Fire 2026-05-19T23:00Z)

## Summary verdict
- 51 claims audited
- 25 VERIFIED / 23 INCORRECT / 2 HALLUCINATION / 1 UNVERIFIABLE
- Final: NOT_SAFE_TO_PR
- Critical issues list:
  - Author attribution is scrambled in both public-facing files. PR95, PR98, PR100, and PR101 are assigned to the wrong authors in multiple places.
  - The current local `submission_dir` runtime tree is not the runtime tree used by the paired auth-eval JSONs or mirrored by commit `462f84cdd`; `src/codec_sidecar.py` exists locally but is absent from that commit.
  - The README full-score verification command cannot run as written: `archive.zip` extracts only member `x`, not `inflate.sh`.
  - The PR body and README overclaim a global Brotli wrapper and inherited FP11 wrapper from PR101. The FEC6 selector is appended outside PR101's Brotli-coded source payload.
  - `archive_manifest.json` contains a hallucinated arithmetic-coded latent-residual innovation and an incorrect exact rate-term value.
  - `pre_submission_compliance_check.py --contest-final --strict` does not pass against the packet and current public text.

## Section A: Author attribution per claim
| File | Claim | Truth (gh api verified) | Status | Recommended fix |
|---|---|---|---|---|
| `PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md:23` | PR95 decoder substrate established by `@SajayR`. | `/opt/homebrew/bin/gh api .../pulls/95` returns PR95 user `AaronLeslie138`, title `hnerv_muon submission (0.20)`. | INCORRECT | Credit PR95 / `hnerv_muon` HNeRV decoder origin to `@AaronLeslie138`; note PR101 reuses the same model bytes. |
| `PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md:23` | PR98 medal-class iteration by `@AaronLeslie138`. | PR98 user is `EthanYangTW`, title `hnerv_muon_finetuned_from_pr95 (0.1963)`. | INCORRECT | Credit PR98 to `@EthanYangTW`. |
| `PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md:23` | PR100 medal-class iteration by `@EthanYangTW`. | PR100 user is `BradyMeighan`, title `hnerv_lc_v2 submission (0.1954)`. | INCORRECT | Credit PR100 / `hnerv_lc_v2` to `@BradyMeighan`. |
| `PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md:23` | PR101 substrate by `@SajayR`. | PR101 user is `SajayR`, title `add hnerv ft microcodec submission`. | VERIFIED | Keep, but separate PR101 microcodec from PR95 decoder origin. |
| `PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md:23` | PR102 by `@EthanYangTW`. | PR102 user is `EthanYangTW`, title `hnerv_lc_v2_scale095_rplus1 submission (0.19538 CPU)`. | VERIFIED | Keep. |
| `PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md:23` | PR103 by `@rem2`. | PR103 user is `rem2`, title `hnerv_lc_ac submission (0.19)`. | VERIFIED | Keep. |
| `PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md:96` | PR108 closure attributed to Yassine's closure decision. | PR108 user is `andrei-minca`; issue comment closure text was written by `YassineYousfi`. | VERIFIED | Phrase as "PR108 by `@andrei-minca`; maintainer closure comment by `@YassineYousfi`." |
| `README.md:17` | Every HNeRV decoder line is verbatim from `@SajayR`'s reference implementation. | `src/model.py` SHA-256 is identical across PR95, PR98, PR101, and this packet; PR95 author is `@AaronLeslie138`. | INCORRECT | Say byte-identical to the PR95 `@AaronLeslie138` decoder and to the copy included in PR101. |
| `README.md:19` | PR95 is `@SajayR`. | PR95 user is `AaronLeslie138`. | INCORRECT | Replace with `@AaronLeslie138`. |
| `README.md:20` | PR98 is `@AaronLeslie138`. | PR98 user is `EthanYangTW`. | INCORRECT | Replace with `@EthanYangTW`. |
| `README.md:21` | PR100 and PR102 are both `@EthanYangTW`. | PR100 user is `BradyMeighan`; PR102 user is `EthanYangTW`. | INCORRECT | Split PR100 and PR102 attribution. |
| `README.md:22` | PR101 is `@BradyMeighan`. | PR101 user is `SajayR`. | INCORRECT | Replace with `@SajayR`. |
| `README.md:23` | PR103 is `@rem2`. | PR103 user is `rem2`. | VERIFIED | Keep. |

## Section B: Score / byte / number / formula claims
| File | Claim | Verification method | Result | Status |
|---|---|---|---|---|
| All three target files | Candidate `archive.zip` is 178,517 bytes, SHA-256 `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`. | `/usr/bin/shasum -a 256 .../submission_dir/archive.zip`; `zipinfo -v`. | Exact match. | VERIFIED |
| PR body / README / manifest | ZIP has single member `x`, stored, 178,417 bytes. | `/opt/homebrew/bin/python3 -c 'import zipfile...' archive.zip`; `zipinfo -v`. | `x 178417 178417`; compression method stored. | VERIFIED |
| PR body / README / manifest | PR101 source archive is 178,258 bytes, SHA-256 `b83bf348...`. | `/usr/bin/shasum -a 256 experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip`; `wc -c`. | Exact match. | VERIFIED |
| Goal H4 / manifest | `source_pr101_payload` is 178,158 bytes. | `.omx/research/pr101_fec6_packetir_identity_proof_20260519_codex.md`. | Section `source_pr101_payload` length is 178,158. | VERIFIED |
| Goal H4 / manifest | FEC6 selector payload is 249 bytes; fixed-Huffman bitstream is 243 bytes / 1,944 bits. | PacketIR identity proof and manifest lines 79-87. | `selector_fec6_payload` 249; `selector_fec6_fixed_huffman_bitstream` 243; `selector_code_bits_total` 1944. | VERIFIED |
| PR body / README / manifest | 600 pairs and naive 4-bit selector cost is 300 bytes. | Arithmetic: `600 * 4 / 8`. | 300 bytes. | VERIFIED |
| PR body / README | 249-byte selector wire format is "~3.24 bits/pair". | Arithmetic: `249 * 8 / 600`; `1944 / 600`. | 249 wire bytes = 3.32 bits/pair; 3.24 bits/pair applies only to the 243-byte bitstream excluding 6-byte header. | INCORRECT |
| `archive_manifest.json:97` | Exact `rate_term_unrounded` is `0.11886708796066302`. | `python3 Decimal`: `25 * 178517 / 37545489`. | Exact value is `0.11886714273451066...`. | INCORRECT |
| PR body / README | Approximate rate term `25 * 178517 / 37545489 ~= 0.118867`. | Same Decimal calculation. | Rounds to `0.118867`. | VERIFIED |
| PR body / README / manifest | Archive delta vs PR101 is +259 bytes and rate delta is about +0.000172. | `178517 - 178258`; `25 * 259 / 37545489`. | +259 bytes; +0.00017245746885864238. | VERIFIED |
| PR body / README / manifest | Candidate CPU score is `0.192051 [contest-CPU]`. | `experiments/results/modal_auth_eval_paired_20260519/cpu/contest_auth_eval.json`. | `score_recomputed_from_components = 0.1920513168811056`; archive SHA matches. | VERIFIED |
| PR body / README | Candidate CUDA score is `0.226210 [contest-CUDA T4]`. | `experiments/results/modal_auth_eval_paired_20260519/cuda/contest_auth_eval.json` and older `modal_auth_eval/archive_6bae0201fb08/contest_auth_eval.json`. | `score_recomputed_from_components = 0.22621002169349796`; GPU model `Tesla T4`. | VERIFIED |
| PR body / README / manifest | PR101 GOLD CPU score is `0.192845 [contest-CPU]`. | `/opt/homebrew/bin/gh api .../issues/101/comments`; local public-comment scorecard. | Public CPU comment components recompute to `0.1928450127024255`. | VERIFIED |
| PR body / README | CPU improvement is `-0.000794`. | `0.1920513168811056 - 0.1928450127024255`. | `-0.0007936958213199`, rounds to `-0.000794`. | VERIFIED |
| PR body / README | Net improvement is `-0.000622` after subtracting selector rate cost from headline `-0.000794`. | Compare candidate CPU score to PR101 CPU score and separate rate delta. | Actual net score delta vs PR101 is already `-0.000794`; subtracting rate again double-counts the rate penalty. | INCORRECT |
| PR body / README | CUDA result is "canonical paired Modal A100 auth_eval" or final verification used Vast.ai T4. | CUDA JSON provenance and commit `462f84cdd` message. | Auth eval was Modal `Tesla T4`; no A100 or Vast.ai T4 verification is evidenced for this score. | INCORRECT |
| `PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md:107` | `pre_submission_compliance_check.py --contest-final` passes except hosted URL / paired CUDA host attachment. | Ran `.venv/bin/python scripts/pre_submission_compliance_check.py --contest-final --strict ...`. | Exit code 1. Failures include CPU threshold, runtime-tree mismatch, manifest member table, report SHA/size mentions, source reproduce binding, CUDA label scan, dispatch terminal claim, raw Modal call id. | INCORRECT |

## Section C: Feature attribution chain
| Feature | Currently claimed source | Empirical source | Status | Recommended attribution |
|---|---|---|---|---|
| HNeRV decoder `src/model.py` | PR95 decoder and model lines attributed to `@SajayR` in PR body / README. | Model SHA `e63b04...` is byte-identical in PR95 `@AaronLeslie138`, PR98, PR101, and this packet. PR95 is the origin in the verified PR chain. | INCORRECT | "HNeRV decoder originally from PR95 `@AaronLeslie138`; reused byte-identically by PR101 `@SajayR` and this packet." |
| PR101 microcodec substrate | `@SajayR` PR101. | PR101 user `SajayR`; PR101 source includes `hnerv_ft_microcodec/src/codec.py` with compact decoder/latent sidecar/Huffman code. | VERIFIED | Keep. |
| PR100 latent-correction sidecar / `schema.py` pattern | Assigned to `@EthanYangTW` in PR body and README. | PR100 user `@BradyMeighan`; source files are `hnerv_lc_v2/inflate.py`, `sidecar.py`, `schema.py`, `hnerv_model.py`. | INCORRECT | Credit `hnerv_lc_v2` and sidecar/schema pattern to `@BradyMeighan`; PR102 is `@EthanYangTW` retune. |
| FES1/FEC2/FEC3/FEC5/FEC6 selector framework | Claimed as ours / new bolt-on. | `rg` over PR100, PR101, PR102, PR103 source dirs and all PR101 intake `source/submissions` finds no `FES*`, `FEC*`, `frame_selector`, or selector framework tokens; target packet contains them. | VERIFIED | Keep as local FEC/FES lineage, but do not imply it existed in PR101. |
| PR101 selector baseline | PR body / README say PR101 has no per-pair selector. | PR101 `inflate.py` is 2,073 bytes and has no selector framework; source grep is negative. | VERIFIED | Keep. |
| K=16 selector compared against "PR101 GOLD's K=8" | Linked source comment at `inflate.py:40` says "vs PR101 GOLD's K=8". | PR101 has no selector and no K=8 selector. K=8 is an internal predecessor (`FEC5_FIXED_K8_MODE_IDS`), not PR101. | INCORRECT | Say "vs internal FEC5 K=8 predecessor; PR101 has no selector." |
| Fixed-Huffman selector layer | Claimed new layer, sister to PR101 canonical Huffman primitive. | Target `inflate.py` implements `FEC6_FIXED_K16_CODE_BITS` and decoder; PR101 implements canonical Huffman only for latent sidecar. | VERIFIED | Keep, with 243-byte bitstream / 249-byte wire distinction. |
| Canonical Huffman for latent sidecar | Claimed inherited from PR101. | PR101 `src/codec.py` defines `decode_canonical_huffman` and `decode_canonical_huffman_all`; local `src/codec_sidecar.py` is a split/refactor of that logic. | VERIFIED | Keep, but bind it to latent sidecar only. |
| Brotli q=11 "outer" wrapping state-dict + latent + selector inside `x` | Claimed in PR body, README, and manifest. | PR101 source payload contains Brotli streams; FEC6 wrapper is `FP11 + source_len + source_payload + selector_len + selector_payload`. The selector is outside PR101's Brotli-coded source payload; ZIP stores `x` uncompressed. | HALLUCINATION | Replace with: "PR101 uses Brotli-coded decoder/sidecar streams; FEC6 selector is appended as fixed-Huffman payload outside those streams." |
| FP11 wrapper / magic-byte format | Claimed inherited unchanged from PR101. | PR101 member first bytes are not `FP11`; FEC6 member first bytes are `46503131` (`FP11`) and wrapper parsing lives in target `inflate.py`. | INCORRECT | Treat the `FP11` wrapper as local FEC6 packet grammar, not inherited PR101 grammar. |
| Arithmetic-coded latent residuals | `archive_manifest.json:104` claims a small arithmetic-coded residual stream. | PR103 `hnerv_lc_ac/inflate.py` imports `constriction` and uses `RangeDecoder`; this packet has no `constriction`, range coder, ANS, or arithmetic coder. | HALLUCINATION | Delete `innovation_4_arithmetic_coded_latent_residuals`; explicitly state no PR103 arithmetic-coder inheritance. |
| No scorer weights at inflate time | PR body / README claim no scorer weights loaded at inflate time. | Target runtime imports `torch`, `brotli`, `codec`, `frame_selector`, `model`; no scorer/model weights from PoseNet/SegNet are loaded. | VERIFIED | Keep. |

## Section D: Permalinks + path references
| Reference | Resolvable | Status | Fix |
|---|---:|---|---|
| Commit `462f84cdd62154e056de804de3b87a30a451b2e7` in `adpena/comma-lab`. | Yes: `/opt/homebrew/bin/gh api repos/adpena/comma-lab/commits/462f84cdd...` returns the commit. | VERIFIED | Keep the commit only if it matches the final packet files. |
| Permalinks for `inflate.sh`, `inflate.py`, `src/frame_selector.py`, `src/model.py`, and README at `462f84cdd`. | Yes: GitHub contents API resolves these files at that ref. | VERIFIED | Keep after any final source-sync commit. |
| `src/codec_sidecar.py` references in README / manifest. | No at `462f84cdd`: GitHub contents API returns 404 for `src/codec_sidecar.py`. | INCORRECT | Either link a newer commit containing `codec_sidecar.py`, or revert public text/runtime references to the monolithic `src/codec.py` actually present at `462f84cdd`. |
| "Selector and codec source are mirrored at commit `462f84cdd`" for the current local `submission_dir`. | Partially: remote `src/codec.py` is 17,108 bytes; local target `src/codec.py` is 6,107 bytes plus local `src/codec_sidecar.py` 12,158 bytes. | INCORRECT | Re-sync source and auth-eval runtime, then pin a new commit; do not claim current local split runtime is mirrored at `462f84cdd`. |
| README `src/frame_selector.py` says "selector stream in `0.bin`". | No: ZIP member is `x`; PacketIR sections name the selector inside member `x`. | INCORRECT | Replace `0.bin` with `x` / archive member payload. |
| README full score verification command unzips archive to `/tmp/archive_dir` then runs `/tmp/archive_dir/inflate.sh`. | No: `archive.zip` contains only `x`; `inflate.sh` is not inside the archive. | INCORRECT | Stage runtime tree separately from `archive.zip`, then call `bash <runtime>/inflate.sh /tmp/archive_dir /tmp/inflate_out /tmp/list.txt`. |
| Hosted archive URL placeholder. | No: `<HOSTED_URL_PLACEHOLDER>` is intentionally not a real URL. | UNVERIFIABLE | Do not submit until replaced with a hosted archive URL whose SHA-256 is verified. |
| PR body / README runtime-tree lists omit `src/codec_sidecar.py` while local runtime imports it. | No for current local packet. | INCORRECT | Include `src/codec_sidecar.py` in runtime lists, source links, dependency manifests, and auth-eval custody, or use the monolithic runtime consistently. |
| PR body says `src/codec.py` parses renderer state-dict, latent sidecar, and selector sections out of `x`. | No for current local packet: `inflate.py` parses `FP11` wrapper and selector; `src/codec.py` parses only the PR101 source payload after wrapper removal. | INCORRECT | Attribute wrapper/selector parsing to `inflate.py`; state-dict/latent parsing to `src/codec.py` / `src/codec_sidecar.py`. |

## Section E: Other hallucinations or unsupported claims

H1 failed. The author chain must be rewritten before publication. The verified chain is PR95 `@AaronLeslie138`, PR98 `@EthanYangTW`, PR100 `@BradyMeighan`, PR101 `@SajayR`, PR102 `@EthanYangTW`, PR103 `@rem2`, PR108 by `@andrei-minca` with a maintainer closure comment by `@YassineYousfi`.

H2 and H9 passed for selector ownership. Grep over PR100/101/102/103 and over all PR101 intake `source/submissions` found no FES/FEC/frame-selector framework. The selector framework is local work. Keep that claim, but remove any wording implying PR101 had K=8 selector machinery.

H3 is mixed. PR100 `@BradyMeighan` did introduce the `hnerv_lc_v2` latent-correction sidecar/schema pattern; PR101 `@SajayR` introduced the compact `hnerv_ft_microcodec` code path with canonical Huffman sidecar handling and compact Brotli stream parsing. The public text currently assigns PR100 to the wrong author and collapses these two steps.

H4 and H5 are mostly verified for archive bytes and scores, except for the manifest's exact rate term, the selector bits-per-pair wording, the net `-0.000622` interpretation, and the Modal A100/Vast.ai T4 claims.

H6 passed negatively: PR103 uses arithmetic/range coding via `constriction`, but this packet does not. The manifest arithmetic-coded latent residual claim is not just unsupported; it contradicts the runtime source.

H7 passed for commit existence, but failed for final source custody. `462f84cdd` exists and resolves, yet it does not mirror the current local split runtime tree. Any PR using the current local files needs a new pinned commit and a fresh matching auth-eval/runtime-tree custody story.

H8 passed for decoder lineage: PR95 contains the HNeRV decoder code used here, and the model file hash is identical across PR95, PR98, PR101, and this packet.

H10 found additional unsafe claims: the README score-verification command is unrunnable as written; `pre_submission_compliance_check.py --contest-final --strict` fails; public text leaks a raw Modal call id; and the generic `archive_manifest.json` lacks the member table expected by the compliance gate even though it records selector-specific fields.

## Section F: Recommended rewrite of 'changes from upstream' section

This submission is a FEC6 selector bolt-on around the public HNeRV lineage.

The HNeRV decoder architecture in `src/model.py` originates in PR95 by `@AaronLeslie138` (`hnerv_muon`). The same decoder file is byte-identical in PR98 by `@EthanYangTW`, PR101 by `@SajayR`, and this packet. The immediate byte substrate for this packet is PR101 by `@SajayR` (`hnerv_ft_microcodec`), whose public CPU-axis score recomputes from evaluator-comment components to `0.1928450127024255 [contest-CPU]` with archive `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e` and 178,258 charged bytes.

Relevant prior lineage: PR98 by `@EthanYangTW` fine-tuned the PR95 HNeRV line. PR100 by `@BradyMeighan` introduced the `hnerv_lc_v2` latent-correction sidecar/schema pattern. PR102 by `@EthanYangTW` retuned that `hnerv_lc_v2` family. PR103 by `@rem2` used `constriction` range coding in `hnerv_lc_ac`; this FEC6 packet does not import or inherit that arithmetic/range coder.

Inherited from PR101: compact decoder and latent payload parsing, Brotli-coded decoder/sidecar streams inside the PR101 source payload, and canonical Huffman decoding for the latent sidecar. The canonical Huffman primitive is inherited only for the latent sidecar.

New in this packet: the archive member `x` wraps the PR101 source payload as `FP11 + source_len + source_pr101_payload + selector_len + selector_payload`. The `FP11` wrapper and appended FEC6 selector are local packet grammar, not inherited PR101 grammar. The selector framework (`FES1` through `FEC6`), the 31-mode transform palette, the K=16 active palette, the fixed 16-symbol selector codebook, and the offline per-pair selector decisions are local FEC6 work. Grep over PR100, PR101, PR102, PR103, and all PR101 intake `source/submissions` found no FES/FEC/frame-selector framework.

Archive and score facts: candidate archive SHA-256 is `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`; `archive.zip` is 178,517 bytes and contains one stored member `x` of 178,417 bytes. PacketIR identity proof splits `x` into a 178,158-byte PR101 source payload plus a 249-byte FEC6 selector payload. The selector payload is a 6-byte header plus a 243-byte fixed-Huffman bitstream; the bitstream is 1,944 bits, or 3.24 bits/pair over 600 pairs. The full selector wire payload is 249 bytes, or 3.32 bits/pair including header. The archive byte delta vs PR101 is +259 bytes, for an added rate term of `25 * 259 / 37545489 = 0.00017245746885864238`.

Measured scores must be kept axis-separated. The candidate CPU auth-eval JSON records `0.1920513168811056 [contest-CPU]` on Linux x86_64 with the same archive SHA. Against PR101's `0.1928450127024255 [contest-CPU]`, the total CPU-axis score delta is `-0.0007936958213199`, reported as `-0.000794`. The candidate CUDA auth-eval JSON records `0.22621002169349796 [contest-CUDA T4]` on Modal `Tesla T4` with the same archive SHA. Do not call this an A100 eval, do not call it Vast.ai verification, and do not combine CPU and CUDA into one leaderboard claim.

Publication caveat: do not claim final `pre_submission_compliance_check.py --contest-final --strict` passage until the hosted URL, source commit, local runtime tree, auth-eval runtime tree, archive manifest member table, terminal dispatch claim, public source binding, and public hygiene scans all pass together. Remove raw Modal call ids from public text; keep them only in private custody artifacts.
