# Executive Review of PR #110 – `hnerv_fec6_fixed_huffman_k16`

## Verdict

**Verdict:** **Approve with minor edits**.  The submission follows the contest
contract, provides exact archive custody (SHA‑256, byte count and member list),
clearly distinguishes between CPU and CUDA evidence, discloses training
compute, and isolates the runtime from broader research infrastructure.  The
proposed FEC6 + fixed‑Huffman selector is a narrow improvement over the PR #101
baseline and yields a modest CPU‑axis gain (≈0.00079 absolute improvement) at
the cost of 259 additional bytes【294212394795766†L354-L361】.  The PR does not
overclaim broad advancements and explicitly notes that further progress
requires changing deeper layers of the compression stack【58280996536521†L220-L236】.

The submission would benefit from tightening the PR body, moving long
explanatory sections into linked documents, and clarifying the tone in a few
places.  None of these issues block approval; they are suggestions to
minimize reviewer friction.

## Top risks (ordered by severity)

1. **Long and context‑heavy PR body.**  The body mixes the essential
   submission facts with several paragraphs of research context.  While the
   context is valuable, maintainers might find it distracting.  The source
   map already clarifies that broader research lives in `comma‑lab` and `tac`
   and is not part of this runtime【58280996536521†L220-L240】.  Moving the
   long discussion into a referenced document and keeping the PR body laser
   focused on the submission packet will reduce cognitive load.

2. **Ambiguity around training compute disclosure.**  The body states that
   an A100 was used for one‑shot per‑video training【294212394795766†L264-L266】
   but does not provide the exact training command, epochs or a hash of the
   trained weights.  Since the contest only evaluates archive bytes and
   runtime, the training compute is advisory, but a curious reviewer may
   wonder how reproducible the training is.  Suggest linking to a runbook in
   `comma‑lab` or noting that training scripts are not part of the submission.

3. **Potentially misleading language about “saturation.”**  The PR asserts
   that within the tested HNeRV family there is no deliverable rate‑term
   gain from post‑hoc recompression and implies the archive is entropy
   saturated【294212394795766†L354-L361】.  This is a reasonable observation
   based on experiments, but the wording could be softened so it doesn’t
   discourage others from exploring alternative entropy coders.  Clarify that
   saturation is observed for the specific archive bytes in PR #110 and the
   immediate lineage, not a general statement about all HNeRV variants.

4. **Overly specific internal revision details.**  The body references
   internal commit hashes (`b392343d758…`) and prior working directories.  The
   commit on `comma‑lab` is helpful for reproducibility【294212394795766†L360-L364】,
   but internal file paths should be removed or moved to a separate audit
   document.  Exposing internal paths may confuse maintainers and is not
   necessary for contest compliance.

5. **Implicit reliance on external research docs.**  The PR links to the
   `full_stack_source_map.md` and candidate inventories in `comma‑lab`【294212394795766†L323-L326】.
   Those documents are not part of the runtime and are very long.  A reviewer
   might worry they need to read them to understand the submission.  Make it
   clear that these links are optional background and remove them from the
   main bullet list.

6. **Minor tone issues.**  Phrases like “current top merged submission” or
   “medalist submissions cluster” imply standings that could change rapidly.
   Maintain a neutral, factual tone: state that PR #101 is the maintainer‑awarded
   gold submission and provide the delta without comparative rhetoric.  This
   will keep the PR from aging poorly if leaderboard positions change.

## Recommended edits to the PR body

1. **Condense the overview.**  Start with a short summary: “This submission
   extends the PR #101 HNeRV packet with a 31‑mode FEC6 frame selector and a
   fixed‑Huffman code over the selector indices.  It yields a
   0.000794‑point CPU‑axis improvement at +259 bytes.”  Then list the key
   evidential facts (archive location, SHA‑256, size, CPU/CUDA scores,
   runtime tree, dependencies, training hardware) in a compact bullet list.

2. **Move research context to a separate document.**  Create a short
   comment or commit message linking to `comma‑lab/docs/full_stack_source_map.md`
   and the candidate inventory.  Remove the long paragraphs about future
   directions from the PR body and instead mention that the source map
   contains background for interested readers【58280996536521†L220-L240】.

3. **Remove internal path disclosures.**  Drop references to internal `.omx` paths or
   local revisions.  If pointing to a reproducible commit, provide only the
   public repository and commit hash (e.g. `comma‑lab` commit
   `b392343d758aba0…`)【294212394795766†L360-L364】.

4. **Clarify the “saturation” statement.**  Rephrase as: “For the tested
   HNeRV variants, we found no material score reduction from recompressing the
   exported archive bytes using general‑purpose compressors.  This suggests
   that further score reductions will likely require changing the emitted
   representation, training priors, or entropy model rather than post‑hoc
   recompression.”【58280996536521†L220-L236】

5. **Simplify training compute disclosure.**  State: “Training was run one
   shot per video on an A100.  Training scripts and datasets live in
   `comma‑lab` and are not part of this submission.”  Remove any wording
   implying that the training hardware affects evaluation.

6. **Reduce comparative language.**  Replace “current top merged submission”
   with “maintainer‑awarded gold submission (PR #101)” and similar terms.  Do
   not reference awards or medals unless necessary to define the baseline.

## Recommended edits to accompanying README/report/manifest

1. **Manifest completeness.**  Ensure the `archive_manifest.json` lists the single
   member `x` with its size and compression method (`ZIP_STORED` / 0) and
   includes the SHA‑256 of the archive【294212394795766†L235-L244】.  Include the
   version of `torch`, `numpy` and `brotli` used.

2. **Reproducibility instructions.**  Provide exact commands for running
   `upstream/evaluate.py` on both CPU and CUDA, including the commit hash of
   the upstream repository, the device (`--device cpu` or `cuda`), and the
   number of threads【294212394795766†L252-L259】.  Include the commands used to
   compute the SHA‑256 and size of the archive (e.g. `sha256sum archive.zip`).

3. **Training provenance (optional).**  If including training details in the
   report, provide the training script name, hyperparameters, dataset summary,
   and commit hash.  If those details are confidential, state that they live
   in `comma‑lab` and will be shared privately upon request.

4. **Tone and terminology.**  Use the term “submission” rather than
   “packet.”  Avoid abbreviations like “FEC6” without an initial definition;
   expand it as “frame‑entropy coding with six bits of redundancy” upon first
   use.  Spell out hardware names clearly (e.g. “NVIDIA A100”).

## Tone and positioning advice

* **Be concise and factual.**  Lead with measurable facts: size, SHA, scores,
  runtime files, dependencies.  Reserve discussions of research philosophy for
  separate documents or personal blogs.

* **Acknowledge limitations.**  Point out that the improvement is small and
  that HNeRV remains the control arm.  Stress that the community is invited
  to challenge the FEC6 approach and explore other architectures.

* **Avoid leaderboard language.**  Focus on absolute numbers instead of
  relative rankings.  Mention that PR #101 is the maintainer‑awarded gold
  baseline【294212394795766†L355-L357】 and provide the delta; avoid words like
  “medalist” or “frontier,” which can sound subjective.

* **Credit upstream contributors.**  Continue to credit the authors of
  previous PRs and the maintainers but avoid implying that the lineage
  automatically guarantees superiority.  Each variant still needs to be
  re‑evaluated on the same axis.

## What not to add to the PR body

* **Do not embed long research summaries.**  Keep the PR body to one
  page.  Place long context in a linked document.

* **Do not reference internal file system paths or private directories.**

* **Do not suggest that training hardware influences evaluation.**

* **Do not claim saturation beyond your measured data.**
