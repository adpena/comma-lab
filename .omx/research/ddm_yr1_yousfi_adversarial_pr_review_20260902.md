# DDM YR1 — Yousfi-style adversarial review of the AFR1 publication packet

Date: 2026-09-02  
Arm: `ddm_yr1_yousfi_adversarial_pr_review_20260902`  
Object reviewed: frozen packet `/Volumes/APDataStore/pact/ddm_pq12/generation_7_afr1/` plus the FULL and TIGHT PR-body source drafts  
Review axis: scorer-free document, custody, receiver, and public-surface review. No scorer, archive mutation, remote dispatch, upload, or publication occurred.

## Executive conclusion

**FINDING ROUND; DO NOT PUBLISH THIS PACKET AS-IS.** The sealed AFR1 object has a strong retained
`[contest-CUDA T4, n600]` receipt and the score arithmetic re-derives exactly, but the proposed
public package fails the likely maintainer review at the policy and packaging surfaces before its
technical result gets a fair hearing. The two decisive facts are:

1. both candidate PR descriptions are agent-drafted source material, while the live policy bans
   agents writing the full public PR description; and
2. the 55-file packet mixes the exact receiver with private review/build matter, while the body
   says a compression script is included and mergeable even though that script explicitly cannot
   run from a bare challenge checkout.

The highest-leverage change is therefore **not another review pass**: the operator must first
decide that publication is policy-compatible and write a new, short PR description from a blank
page in their own words. If that gate clears, MAIN should publish a fresh, minimized packet
generation—not modify frozen generation 7—with one external archive, an inline report block, the
smallest byte-identical receiver closure, and direct permanent links to concise lineage and rebuild
evidence in the source repository.

The retained exact result itself remains:

`0.020139 + 0.007981227975693965 + 25*180002/37545489 = 0.14797617125559104`.

Here `0.020139` and `0.007981227975693965` are score contributions (not raw distortions); the raw
distortions are `d_seg=0.00020139` and `d_pose=6.37e-6`. This review did not re-run the evaluator.
The authority source is the retained T4 report (`report.txt:17-23`) and the archive/runtime identity
was independently rehashed in PQ14 (`ddm_pq14_drive_pr_review_round1_20260902.md:12-24`).

## §SHIP-SET — disposition of every frozen-packet file

Definitions used here:

- `SHIP-IN-PR`: uploaded/linked as the PR attachment or pasted into the PR body.
- `REPO-SIDE`: kept in a public source/challenge repository and linked when material; not attached
  as PR-review bulk.
- `INTERNAL-ONLY`: custody, build, review, superseded, or private provenance material that should
  not enter the challenge PR.

This is a classification of **all 55 regular files found under the frozen root**, not an instruction
to mutate that root.

| # | Frozen packet path | Disposition | Reason |
|---:|---|---|---|
| 1 | `._.compress_py_pre_ce1_superseded_20260902` | INTERNAL-ONLY | AppleDouble metadata for a superseded private artifact; no receiver or review value. |
| 2 | `._BORROWED_SUBSTRATE_ACCOUNTING.md` | INTERNAL-ONLY | AppleDouble metadata; never source or evidence. |
| 3 | `._COMPRESS.md` | INTERNAL-ONLY | AppleDouble metadata; never source or evidence. |
| 4 | `._README.md` | INTERNAL-ONLY | AppleDouble metadata; never source or evidence. |
| 5 | `._compress.py` | INTERNAL-ONLY | AppleDouble metadata; never executable source. |
| 6 | `._report.txt` | INTERNAL-ONLY | AppleDouble metadata; never evaluator output. |
| 7 | `.compress_py_pre_ce1_superseded_20260902` | INTERNAL-ONLY | Deliberately superseded 42 KB compression entry point; retaining it publicly creates an ambiguity, not reproducibility. |
| 8 | `BORROWED_SUBSTRATE_ACCOUNTING.md` | INTERNAL-ONLY | 51 KB append-only internal ledger with preserved stale generations and `.omx` receipts; replace publicly with a concise lineage/credit note. |
| 9 | `COMPRESS.md` | REPO-SIDE | Useful rebuild contract beside the real research-repository entry point; it explicitly belongs in that repository, not a bare challenge submission. |
| 10 | `FX5_BUILD_MANIFEST.json` | INTERNAL-ONLY | Build-only, no-score receipt with absolute SSD paths and candidate-family internals (`FX5_BUILD_MANIFEST.json:2-6,31-35`). |
| 11 | `LICENSE` | INTERNAL-ONLY | Duplicate packet copy; the destination repository's license governs the contribution. Preserve third-party attribution separately and narrowly. |
| 12 | `MANIFEST.sha256` | INTERNAL-ONLY | Current authority manifest includes the internal FX5 receipt as a runtime row (`MANIFEST.sha256:4-13`); create a clean public receiver manifest after identity proof. |
| 13 | `README.md` | INTERNAL-ONLY | Packet-control document: it says PREPARED HOLD and names internal receipts (`README.md:73-82,120-125`); replace with a concise public README. |
| 14 | `THIRD_PARTY_NOTICES.md` | INTERNAL-ONLY | Whole-lab notice inventory is broader than this receiver; create a receiver-scoped public notice if required. |
| 15 | `archive.zip` | SHIP-IN-PR | The sole binary attachment/external asset: 180,002 B, SHA-256 `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`. |
| 16 | `archive_manifest.json` | INTERNAL-ONLY | It binds the archive to the current 38-row content tree, including the internal FX5 row; regenerate after the public receiver set is frozen. |
| 17 | `compress.py` | REPO-SIDE | Real deterministic rebuild orchestrator, but only in comma-lab with `src/tac`, `experiments`, and retained pinned inputs; direct-link its evaluated commit. |
| 18 | `cpr1/carrier_codec.py` | REPO-SIDE | Exact receiver dependency; keep in the public receiver tree. |
| 19 | `cpr1/ddm_mp2_semantic_receiver.py` | REPO-SIDE | Exact receiver dependency; keep in the public receiver tree. |
| 20 | `cpr1/hpac_integer.py` | REPO-SIDE | Exact receiver dependency; keep in the public receiver tree. |
| 21 | `cpr1/hpac_integer_sparse.py` | REPO-SIDE | Part of the sealed CPR1 receiver source; do not prune without a new identity proof. |
| 22 | `cpr1/inflate.py` | REPO-SIDE | Exact renderer/receiver implementation imported by the runtime. |
| 23 | `cpr1/integer_model_io.py` | REPO-SIDE | Exact CPR1 model-deserialization dependency. |
| 24 | `inflate.py` | REPO-SIDE | Challenge entry point; pins archive SHA/bytes and invokes the F26 receiver. |
| 25 | `inflate.sh` | REPO-SIDE | Required challenge entry point; dependency bootstrap and compiler behavior need the fixes in F6. |
| 26 | `report.txt` | SHIP-IN-PR | Inline only: paste the verbatim seven-line evaluator results block; do not commit/upload this extended 84-line packet report (`pull_request_template.md:9-13`). |
| 27 | `runtime/__init__.py` | REPO-SIDE | Package marker in the exact sealed receiver tree. |
| 28 | `runtime/baseline.py` | REPO-SIDE | Imported by renderer-weight decode; exact receiver dependency. |
| 29 | `runtime/bits.py` | REPO-SIDE | Imported by residual and renderer-weight decoders. |
| 30 | `runtime/carrier_repack.py` | REPO-SIDE | Imported by F26 inflate, residual archive, and coefficient decode. |
| 31 | `runtime/compensation_overlay.py` | REPO-SIDE | Imported by F26 inflate and residual archive. |
| 32 | `runtime/ddm_wc1_advisory_runtime.py` | REPO-SIDE | Dynamically imported on live F26 paths (`runtime/f26_inflate.py:530,591,624`); the name “advisory” is not evidence that it is removable. |
| 33 | `runtime/dx2_cabac_coefficients.py` | REPO-SIDE | Dynamically selected by residual-archive decoding (`runtime/residual_archive.py:203`). |
| 34 | `runtime/entropy/__init__.py` | REPO-SIDE | Package marker in the exact sealed receiver tree. |
| 35 | `runtime/entropy/adaptive_ans.py` | REPO-SIDE | Imported by renderer-weight codec. |
| 36 | `runtime/entropy/coefficient_ar1_codec.py` | REPO-SIDE | Imported by residual-archive decode. |
| 37 | `runtime/entropy/coefficient_predictor.py` | REPO-SIDE | Imported by the coefficient codec. |
| 38 | `runtime/entropy/rc64.py` | REPO-SIDE | Native decoder wrapper imported by residual-archive decode. |
| 39 | `runtime/entropy/rc64_backend.c` | REPO-SIDE | Load-bearing native range-decoder backend compiled by `inflate.sh`. |
| 40 | `runtime/entropy/renderer_weight_codec.py` | REPO-SIDE | Imported by both F26 inflate and residual archive. |
| 41 | `runtime/f26_corrector_native.c` | REPO-SIDE | Optional speed path with an explicit Python fallback; part of the evaluated tree. |
| 42 | `runtime/f26_hpac_native.c` | REPO-SIDE | Optional HPAC native path; keep until the evaluated public closure is re-proved. |
| 43 | `runtime/f26_hpac_native.py` | REPO-SIDE | Dynamically imported when native HPAC is selected (`runtime/f26_inflate.py:569`). |
| 44 | `runtime/f26_inflate.py` | REPO-SIDE | Core receiver orchestration. |
| 45 | `runtime/frame0_selector.py` | REPO-SIDE | Imported by F26 inflate, carrier repack, and residual archive. |
| 46 | `runtime/free_corrector.py` | REPO-SIDE | Member of the corrector fallback/import closure. |
| 47 | `runtime/fx1_logistic_mixer_corrector.py` | REPO-SIDE | Imported by FX2 and the native-corrector fallback closure. |
| 48 | `runtime/fx2_model_axis_corrector.py` | REPO-SIDE | Imported by the shipped free-corrector path. |
| 49 | `runtime/hpac_inference.py` | REPO-SIDE | Imported by F26 inflate for deterministic CUDA configuration. |
| 50 | `runtime/ihs2.py` | REPO-SIDE | Imported by residual-archive decode. |
| 51 | `runtime/ihs2_gate_a.py` | REPO-SIDE | Dynamically imported by IHS2 v3 decode (`runtime/ihs2.py:339`). |
| 52 | `runtime/native_free_corrector.py` | REPO-SIDE | Imports and validates the Python corrector family as native fallback/reference. |
| 53 | `runtime/residual_archive.py` | REPO-SIDE | Core payload parser and production-token decoder. |
| 54 | `runtime/rr4_free_corrector.py` | REPO-SIDE | Imported by FX1, FX2, free-corrector, and native-corrector paths. |
| 55 | `runtime/rr5_arith_basis.py` | REPO-SIDE | Dynamically selected by residual-archive decoding (`runtime/residual_archive.py:196`). |

Counts: **2 SHIP-IN-PR, 39 REPO-SIDE, 14 INTERNAL-ONLY**. “REPO-SIDE” is conservative for the
37-file receiver code closure: removing a seemingly unused file is not authorized by static naming
alone. The public manifest may shrink only after a fresh, retained, byte-identical inflation proof.

## §COMPARISON — relevant public PR attachment and repository sets

The earlier prediction was: successful/relevant submissions would not attach `BORROWED`, `COMPRESS`,
or review scaffolds. The retained record gives a more precise result.

| PR | Observed public state and archive surface | Repository-side set | What it says about AFR1 |
|---|---|---|---|
| #130 | Eval-bot accepted `[contest-CUDA]`; archive external, report inline. | 13 files: `LINEAGE_AND_CITATIONS.md`, `MANIFEST.sha256`, `README.md`, `carrier_codec.py`, `compress.sh`, `hpac_integer.py`, `hpac_integer_sparse.py`, `inflate.py`, `inflate.sh`, `integer_model_io.py`, `repack_carrier.py`, `test_carrier_codec.py`, `verification.json`. | Refutes “no analysis/lineage docs in the repo”; supports one concise lineage document and no attachment bulk. |
| #133 | Eval-bot accepted `[contest-CUDA]`; archive external, report inline. | 11 files: `CREDITS.md`, `LICENSE`, `MANIFEST.sha256`, `README.md`, `carrier_codec.py`, `hpac_integer.py`, `hpac_integer_sparse.py`, `inflate.py`, `inflate.sh`, `integer_model_io.py`, `verification.json`; no compression script. | Strong precedent for answering “No” on compression while shipping a credible receiver and concise credits. |
| #135 | Leaderboard row `0.162`; archive external, report inline. | 29 files: body/README/archive pin, `compress.sh`, CPR1, receiver runtime, and `verify_submission.py`. | Shows a larger runtime tree can be acceptable, but its policy remediation was precision and demonstrated human work, not more prose (`YOUSFI_REVIEW_CHECKLIST.md:58-70`). |
| #136 | Closed under the LLM policy; archive external, report inline. | 22 files: README, compress/inflate pair, and 18 training/source files. | More source did not rescue policy failure; retained review also found a missing dependency path and wrong-module reproduction references (`ddm_hx1_pr_wave_harvest_20260817.md:346-350`). |
| #138 | Later closed with the live-policy link; archive external, report inline. | 28 files: README/archive pin/compress plus CPR1 and runtime. | A detailed mechanism and large source set did not cure policy/repro defects; its compression script referenced an absent verifier and only copied an archive (`ddm_hx1_pr_wave_harvest_20260817.md:338-345`). |
| AFR1 frozen | Not published; archive and report currently mixed into a 55-file private packet. | 37 exact receiver-code files plus 18 documentation/build/archive/metadata files. | Larger than every comparison set and includes six AppleDouble files, a superseded compressor, a 51 KB internal ledger, and an absolute-path build manifest. |
| AFR1 recommended | One external `archive.zip` link/upload; seven-line report block inline. | Exact receiver closure; clean archive/runtime manifest; concise README + lineage/credits + receiver-scoped notices; comma-lab permalinks for the real rebuild apparatus. | Matches the observed boundary: thin attachment surface, enough public evidence to inspect, no private scaffolding. |

**Prediction outcome:** `FOLDED/PARTIALLY REFUTED`. No retained comparison PR attached separate
analysis documents; the template itself says the archive is uploaded and the report is copied inline
(`upstream/.github/pull_request_template.md:5-13`). But #130 and #133 did carry concise lineage/credit
documents repository-side. The useful law is therefore: **do not attach internal analysis; do publish
small, directly relevant, permanent evidence beside the source.**

## §FINDINGS — ranked adversarial findings

### F1 — BLOCKER — the proposed public body is prohibited agent-authored output

- **Observation:** Both FULL and TIGHT explicitly call themselves source material whose final public
  text must be operator-authored (`PR_BODY_FINAL_DRAFT.md:135-145`;
  `PR_BODY_FINAL_DRAFT_TIGHT.md:88-103`). The live policy bans agents writing the full PR
  description/public comments, and Yousfi closed four PRs with the policy link; the accepted #135
  remediation demanded visible human work and precision (`YOUSFI_REVIEW_CHECKLIST.md:42-70`).
- **Concrete fix:** The operator decides eligibility after personally reading the code and policy,
  then writes a fresh, short template response in their own words. Do not edit either agent draft
  into a “final” body and do not paste its long innovation prose.
- **Owner:** `OPERATOR`.

### F2 — HIGH — the public surface is an internal packet dump, not a reviewable submission

- **Observation:** The frozen root contains 55 files, including six `._*` AppleDouble files, a
  superseded 42 KB compressor, a 51 KB multi-generation accounting ledger, and private build/control
  documents. The live template requests an archive link, report content, runtime choice, compression
  yes/no, and comments—nothing resembling an evidence-folder upload
  (`upstream/.github/pull_request_template.md:1-23`). Yousfi repeatedly asks that the repository remain
  lightweight (`YOUSFI_REVIEW_CHECKLIST.md:72-82`).
- **Concrete fix:** Build a new public generation from the §SHIP-SET boundary. Never mutate frozen
  generation 7. Keep internal review/custody records internal; replace them with concise public docs.
- **Owner:** `MAIN`.

### F3 — HIGH — “Yes to both” overstates what a challenge-repo reviewer can run

- **Observation:** FULL says the compression script is included and should be merged
  (`PR_BODY_FINAL_DRAFT.md:47-54`); TIGHT repeats it (`PR_BODY_FINAL_DRAFT_TIGHT.md:33-40`). But the
  packet's own compression contract says `compress.py` runs from the research repository and “Copied
  into a bare submission directory it will not import” (`COMPRESS.md:69-79`). The code inserts the
  parent `src` path and imports `tac`, then calls `experiments` scripts (`compress.py:112-125`). It also
  requires a retained pinned base that the maintainer does not receive.
- **Concrete fix:** In the operator-authored body, answer **No** to inclusion/merge unless MAIN first
  produces a genuinely self-contained challenge-repo compressor. Link the exact public comma-lab
  commit and describe it as an independently inspectable rebuild receipt, not an included PR script.
- **Owner:** `OPERATOR`.

### F4 — HIGH — the exact runtime manifest is contaminated by a private build receipt

- **Observation:** `MANIFEST.sha256` says its 38 rows are the evaluated runtime and makes
  `FX5_BUILD_MANIFEST.json` row 1 (`MANIFEST.sha256:1-13`). That JSON is explicitly a no-measurement
  build artifact and exposes absolute SSD paths (`FX5_BUILD_MANIFEST.json:2-6,31-35`). Omitting it from
  the PR breaks the advertised 38-row tree identity; shipping it leaks irrelevant private state.
- **Concrete fix:** In a new packet generation, exclude FX5 from the public set and generate a new
  clean runtime manifest/content-tree receipt. Before claiming equivalence, inflate the same sealed
  archive through the minimized tree and retain proof that all 3,662,409,600 output bytes equal the
  authority raw hash. Do not infer equivalence merely because the JSON appears unimported.
- **Owner:** `MAIN`.

### F5 — HIGH — public evidence is either dangling or pointed at private internals

- **Observation:** The body cites `BORROWED_SUBSTRATE_ACCOUNTING.md` as a bare filename
  (`PR_BODY_FINAL_DRAFT.md:62-66`), gives three runtime digests without a direct public manifest link
  (`PR_BODY_FINAL_DRAFT.md:128-133`), and otherwise links only the repository root. The packet README
  names internal receipts and an `.omx` memo (`README.md:73-82`). A reviewer cannot follow these claims
  from the PR to a stable, reviewable object.
- **Concrete fix:** Create a concise public lineage/credit note and clean receiver manifest, commit
  them with the exact receiver source, and use commit-pinned URLs. Keep `.omx`, SSD paths, task IDs,
  internal review counters, and superseded narratives out of public prose.
- **Owner:** `MAIN`.

### F6 — HIGH — the evaluated decoder assumes network package installation and an unguarded compiler

- **Observation:** If Brotli 1.2.0 is absent, `inflate.sh` requires `uv` and performs a network install
  (`inflate.sh:16-30`). It then unconditionally invokes `${CC:-cc}` for the range-decoder backend
  (`inflate.sh:32-35`), unlike the later corrector build, which has a Python fallback
  (`inflate.sh:37-53`). The authority row proves the retained T4 environment, not a cold/no-network
  challenge runner.
- **Concrete fix:** MAIN must either close the dependency set inside the submission or obtain a
  clean-run receipt on the declared GitHub T4 runner with the exact dependency/network assumptions
  made explicit and a fail-closed compiler preflight. Any source change requires a fresh runtime
  identity and full byte-identical output proof before the old score may be associated with it.
- **Owner:** `MAIN`.

### F7 — MEDIUM — the “exact score” wording outruns the precision of the printed report

- **Observation:** The evaluator prints raw distortions to eight decimal places and displays `0.15`;
  the packet then calls `0.14797617125559104` exact (`report.txt:17-26`;
  `PR_BODY_FINAL_DRAFT.md:27-32`). The packet itself correctly calculates a possible
  `3.63296497868841e-6` absolute error from printed-component rounding (`README.md:32-34`).
- **Concrete fix:** Say “score recomputed from the evaluator's printed components” and preserve the
  verbatim `0.15` line. Reserve “exact” for archive bytes, hashes, and arithmetic conditional on those
  printed components.
- **Owner:** `OPERATOR`.

### F8 — MEDIUM — GPU-required prose conflicts with silent CPU fallback

- **Observation:** The body says GPU is required and selects `linux-nvidia-t4`
  (`PR_BODY_FINAL_DRAFT.md:38-45`), but `inflate.py` silently selects CPU when CUDA is unavailable
  (`inflate.py:54-63`). The CPU axis has no AFR1 authority row and the packet explicitly refuses to
  transfer a predecessor score (`report.txt:56-61`).
- **Concrete fix:** Fail closed with a clear CUDA-required error for the public receiver, or measure
  and document the CPU path as a separate axis. Do not allow an accidental CPU fallback to look like
  the reviewed object.
- **Owner:** `MAIN`.

### F9 — MEDIUM — the body is optimized for completeness of the internal record, not maintainer comprehension

- **Observation:** FULL devotes lines 62-103 to lineage, 23 moves, mechanism families, and concurrent
  development; TIGHT still spans 104 lines. Yousfi's accepted remediation asks for baseline, precise
  changed file/function, resulting score, optional negative, and optional LLM setup, and explicitly
  rejects verbosity in place of precision (`YOUSFI_REVIEW_CHECKLIST.md:58-70`).
- **Concrete fix:** The operator-authored body should use the literal live template and four short
  factual units: inherited baseline; changed mechanism/file; measured score and axis; one limitations/
  attribution paragraph with permanent links. Move the 23-move ledger to repository evidence.
- **Owner:** `OPERATOR`.

### F10 — LOW — the projected timing ceiling creates avoidable review surface

- **Observation:** The measured charged total is 621.632 s, comfortably under 1,800 s, but public
  prose adds a projected cold-cache ceiling while the packet report describes an 822–1302 s projected
  residual window (`report.txt:43-54`; `PR_BODY_FINAL_DRAFT.md:42-45`). The projection is not needed
  to establish the measured pass.
- **Concrete fix:** State only measured inflate/evaluate/total times, hardware, and the 30-minute
  threshold. Keep projection methodology in internal custody unless a maintainer asks.
- **Owner:** `OPERATOR`.

## §MAINTAINER-WALKTHROUGH — likely first-pass experience

This is an evidence-grounded simulation, not a fabricated quotation.

1. **Policy check comes first.** A maintainer in Yousfi's position would likely recognize the dense,
   polished, exhaustive body as the exact risk the live LLM policy addresses. The retained census
   shows policy-only closures and says this demand closes a PR on sight
   (`YOUSFI_REVIEW_CHECKLIST.md:42-70,186-200`). The invisible “operator note” does not make the visible
   prose human-authored. The single likely first question, stated as an inference rather than a quote,
   is: **did the submitter personally read/write most of this code and personally write this PR body?**
2. **The score and archive identity are legible.** If the policy gate clears, the seven-line report,
   180,002-byte archive, archive hash, T4 axis, and 621.632 s measured total are enough to decide that
   this is a serious row. The numeric audit already found the archive/hash/components internally
   consistent (`ddm_pq14_drive_pr_review_round1_20260902.md:12-24`).
3. **The checkout then becomes confusing.** Six AppleDouble files, a superseded compressor, a large
   append-only accounting file, private SSD paths, and packet HOLD language signal that the author
   uploaded a lab handoff rather than selected a public interface. This is where trust starts to fall.
4. **Inflation is plausible on the known T4, but not environment-closed.** The retained authority run
   is real. On an independent cold runner, however, the maintainer must have exactly Brotli 1.2.0 or
   working `uv` plus network, and must have `cc`. The first native compile has no fallback. A
   maintainer would reasonably ask for a clean runner proof before treating 578.935 s as portable.
5. **The compression answer fails the literal checkout test.** The PR says “Yes to both,” but the
   submitted copy says it cannot import from the submission directory. The honest technical fact is
   stronger but different: comma-lab has a deterministic, retained-input reconstruction chain. That
   should be a direct source-repository link, not a claim that the challenge PR includes a working
   compressor.
6. **Attribution is better than most submissions but over-delivered in the wrong place.** #130 and
   #133 show that concise lineage/credits beside source are normal. The 51 KB multi-generation ledger
   is useful internal evidence, not a maintainer interface. A two-page public lineage note with direct
   links would preserve the honesty while removing machine-generated bulk.
7. **Would he be able to run it?** On the retained T4-like environment with network/`uv`, Brotli 1.2,
   compiler, CUDA, and the exact 37 receiver-code files: likely yes, because that object already ran.
   From the proposed PR as a clean, minimal, independently reviewable checkout: not yet proven. The
   missing proof is packaging/environment closure, not another score.

## §VERDICT

**Round outcome: FINDING. Clean-pass counter remains 0.** Nothing was published or mutated. The
packet's measured row is not rejected; its public interface is.

| Maintainer-seat compliance question | Verdict |
|---|---|
| Does the current public-designated set inflate/evaluate without our repo checkout? | **NOT PROVEN.** The receiver files can travel with the challenge checkout, but Brotli/network/`uv`/`cc` closure has only been proved on the retained authority environment. |
| Are the live template questions covered? | **Structurally yes, substantively no.** Name, archive, report, GPU, compression, competitiveness/description, and comments are present; the compression “Yes to both” answer is misleading for a bare checkout. |
| Is the 30-minute claim legible? | **YES on the measured T4 axis.** 578.935 s inflate + 42.696 s evaluate = 621.632 s, below 1,800 s. The extra projection is unnecessary. |
| Can every public claim be checked from what the maintainer holds? | **NO.** Archive identity/report arithmetic are checkable; the 23-move, lineage, rebuild, and runtime-tree claims point to bare filenames, internal receipts, or a repository root instead of commit-pinned public evidence. |

The single highest-leverage change is: **OPERATOR writes a new precise PR body from scratch after
personally resolving the live LLM-policy eligibility question.** Until that happens, no amount of
manifest polishing should be represented as publication readiness.

After that gate, MAIN's next useful unit is a fresh minimal receiver packet with clean public
manifests and a retained byte-identical inflation receipt. The correct target is not “make generation
7 look cleaner”; it is “prove that the exact archive plus the public-only receiver set emits the same
3,662,409,600 output bytes.”

### Follow-on dispositions

- `FOLDED` — AFR1 CPU authority row. Owner: `OPERATOR`; consumer store:
  `.omx/research/ddm_pq1_submission_packet_prep_20260815/CPU_AXIS_SEALED_FIRE_ORDER.json`; fire trigger:
  only a new explicit operator or governing-policy demand for an AFR1 CPU authority row, after unique
  lane claim. This review found no new trigger.
- `QUEUED-WITH-FIRE-ORDER 1` — publication-policy eligibility and original body. Owner: `OPERATOR`;
  consumer store: operator-controlled final PR draft beside the PQ1 publication packet; fire trigger:
  the operator elects to pursue publication, personally reads the live policy and relevant submitted
  code, and confirms compliance in their own words.
- `QUEUED-WITH-FIRE-ORDER 2` — public-only packet generation. Owner: `MAIN`; consumer store:
  `/Volumes/APDataStore/pact/ddm_pq12/generation_8_afr1_public_minimal/`; fire trigger: fire-order 1 is
  positively resolved and the operator authorizes preparation (not publication). Create a new root;
  never edit frozen generation 7.
- `QUEUED-WITH-FIRE-ORDER 3` — clean-run dependency and byte-identity closure. Owner: `MAIN`;
  consumer store: `/Volumes/APDataStore/pact/ddm_pq12/generation_8_afr1_public_minimal/receipts/`;
  fire trigger: fire-order 2 has a clean manifest, sufficient SSD preflight passes, a unique runtime
  lane is claimed if remote execution is needed, and the exact archive/runtime inputs are pinned.
  Retain the full raw payload or a certified lossless cold-store object plus its SHA; no scorer run is
  needed for an identity-only receiver change.
- `QUEUED-WITH-FIRE-ORDER 4` — final public wording and upload. Owner: `OPERATOR`; consumer store: the
  operator-selected public PR; fire trigger: fire-order 3 proves byte identity and dependency closure,
  direct commit-pinned evidence links resolve, and the operator explicitly authorizes the external
  state changes. This arm grants no such authorization.

## RECALL EVIDENCE

The review recalled the full packet/publication corpus before judging it, then resolved claims against
the current authority surfaces:

Searches executed (bounded to the relevant surfaces):

- content search for `yousfi|official pr|publication packet|pr body|coding-agents|llm policy|ship-set|archive attachment`
  over `.omx/research/`, `CANONICAL_RESEARCH_INDEX*`, and the `sub015_DAG_*` FEED blocks;
- `.venv/bin/python tools/list_canonical_equations.py --json`, narrowed to equation IDs/names containing
  `score|archive|rate|runtime|compression`;
- content search for `submission|archive|inflate|runtime|manifest|public|reproduc` over design/SPEC
  surfaces, including `canonical_submission_pipeline_specification_memo_20260526.md`; and
- exact task/ledger search for `1111|1363|1381|1382|publication` over `.omx/state` ledgers and the
  harvested `NEXT_IF_RESUMED` queue.

- Governing contracts: `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`,
  `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`, this arm's charter, and
  `.omx/tmp/codex_runs/_common_contract.md`.
- Current template and policy: `upstream/.github/pull_request_template.md`; retained live README at
  `.omx/research/ddm_pq7_pr_engineering_20260820/_census_raw/upstream_README_live.md:31-42,120-140`.
- Maintainer behavior: `.omx/research/ddm_pq7_pr_engineering_20260820/YOUSFI_REVIEW_CHECKLIST.md`, a
  bounded census of 76 Yousfi comments from 81 PRs (`:7-17`).
- Prior review: `.omx/research/ddm_pq14_drive_pr_review_round1_20260902.md`, including independent
  arithmetic/hash checks and the live-commit visibility check (`:10-24`).
- PR comparison custody: retained PR #130/#133/#135/#136/#138 bodies, diffs, manifests, and archive
  intake under `.omx/research/ddm_pq7_pr_engineering_20260820/_census_raw/`,
  `/Volumes/APDataStore/pact/ddm_hx1/`, and `/Volumes/APDataStore/pact/ddm_pi135/`; evidence grades and
  known reproduction defects are summarized in `ddm_hx1_pr_wave_harvest_20260817.md:108-145,338-352`.
- Exact object: every one of the 55 frozen packet files was enumerated; receiver imports/dynamic imports
  were traced; `report.txt`, both manifests, both PR-body variants, README/COMPRESS/accounting, and the
  shell/Python entry points were line-read. No file under the frozen root was changed.

What appeared beyond the charter's named seeds, and what it changed:

- Canonical equation `modal_dispatch_runtime_tree_hash_local_vs_worker_parity_v1` requires the local
  projected receiver tree and worker tree to match. This changed a tempting “drop obviously inert FX5
  JSON” recommendation into F4's stricter **new manifest plus byte-identical receiver proof**.
- Canonical equation `pr95_family_l42_lazy_brotli_auto_install_bootstrap_v1` records lazy Brotli install
  as a known runtime-closure pattern. It prevented an over-broad claim that network bootstrap is itself
  novel or automatically invalid; F6 instead asks whether this exact declared runner is dependency-
  closed and independently reproducible.
- The proposed canonical submission-pipeline spec explicitly models `runtime_dep_closure`, archive/
  runtime manifests, source self-containment, and a distinct public attribution layer
  (`canonical_submission_pipeline_specification_memo_20260526.md:162-205,357-375`). It is a design
  document, not proof that those helpers are live, but it reinforced the attachment/repository/internal
  split used in §SHIP-SET.
- The DAG records the already-learned distinction that leaderboard admission and merge are separate,
  with refactoring/overlap controlling merge (`sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md:19366-19377`).
  This kept the verdict from conflating a runnable leaderboard row with merge-ready compression source.
- The harvested PQ13 queue already routes the coding-agents decision through task #1363 into #1111
  (`.omx/state/codex_arm_queue.next_if_resumed.jsonl:410`), and the operator P0 ledger keeps publication
  behind an explicit one-line confirm (`.omx/state/operator_p0_ledger.jsonl:540`). This review therefore
  **FOLDS** into those owners rather than manufacturing a duplicate task or treating doc cleanup as
  publication authorization.

Bounded absence statement: in the retained comparison scope of PRs #130, #133, #135, #136, and #138,
I did not find separate analysis documents attached to the PR. I did find concise lineage/credit files
committed repository-side in #130 and #133, which is why the stronger “never ship analysis docs” law
is not adopted.

`[contest-CUDA T4 n600] own-vehicle frontier: AFR1 — S=0.14797617125559104, archive=180,002 B, d_seg=0.00020139, d_pose=6.37e-6, SHA-256=cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25; frontier unchanged by this scorer-free review.`
