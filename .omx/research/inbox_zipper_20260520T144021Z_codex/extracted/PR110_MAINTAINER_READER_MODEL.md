# Maintainer Reader Model for PR #110

Understanding how an upstream maintainer approaches a contest submission
helps tailor the PR for maximum clarity and minimum friction.  The following
model is based on typical reviewer behavior and the contest’s submission
contract.

## What maintainers care about

1. **Contract compliance.**  The evaluator expects `archive.zip` and a
   deterministic `inflate.sh` that inflates the archive into a runtime tree and
   then calls `upstream/evaluate.py`.  Any uncharged sidecars or hidden
   dependencies are immediate grounds for refusal.  Maintainers will check
   that all score‑affecting files are within the archive and that the runtime
   does not fetch anything over the network or import private repos.

2. **Reproducibility.**  They want to reproduce the score by running a
   single command.  Providing the SHA‑256 of the archive, the exact size and
   member list, the commands used to compute the score, and the hardware
   configuration reduces guesswork.  External links should be stable.

3. **Conciseness.**  Reviewers have limited time.  A concise PR body that
   highlights the novelty and evidence without requiring them to read a
   research treatise will be appreciated.  Long discussions about
   meta‑research can be distracting.

4. **Tone and clarity.**  Overly promotional language can trigger scepticism.
   Conversely, underselling the work may make it hard to appreciate real
   improvements.  A neutral tone that acknowledges both strengths and
   limitations signals maturity and builds trust.

5. **Attribution and lineage.**  The contest community values openness.
   Properly crediting prior submissions (e.g. PR #95, #101) while clearly
   distinguishing what is new in PR #110 helps maintainers understand where
   the improvement comes from and reduces confusion about duplicate code.

## Possible sources of friction

1. **Dense prose and research digressions.**  The long paragraphs about
   entropy saturation, future directions and candidate inventories may make
   maintainers feel that they need to read multiple documents to understand
   a narrow submission.  Keeping the PR body lean avoids this friction.

2. **Implicit assumptions about scorer behavior.**  Statements about
   saturation or compression limits could be seen as overclaiming.  Reviewers
   might worry that the authors are discouraging others from exploring
   alternative entropy models.  Softening such claims reduces tension.

3. **Missing evaluation provenance.**  Without an explicit evaluator commit
   hash and command, maintainers may worry that scores were obtained with a
   modified evaluator.  Adding these details is low‑cost and builds trust.

4. **Internal paths and private commits.**  References to local directories or
   private commit hashes that are not publicly accessible can confuse
   reviewers and raise questions about reproducibility.  Keeping the PR free
   of such references eliminates this friction.

## How to make the PR maximally reviewer‑friendly

* **Lead with the novelty.**  Start with a one‑paragraph summary that
  describes the FEC6 selector, the K=16 active palette, the fixed Huffman
  code, and the resulting score improvement.  Provide the absolute scores
  and the byte delta relative to the baseline.

* **Use bullet lists for evidence.**  Summarize the archive location, SHA,
  size, runtime files, dependencies, evaluation commands, and training
  hardware in a structured list.  This allows a reviewer to quickly tick off
  compliance items.

* **Explicitly state non‑dependencies.**  Make it obvious that the runtime
  does not import `comma‑lab` or `tac`【294212394795766†L323-L326】, and that the
  linked research documents are optional context【58280996536521†L220-L240】.

* **Credit previous work without implying dominance.**  Mention that
  PR #95 provided the decoder and PR #101 provided the microcodec, but
  emphasize that PR #110 is self‑contained and only reuses code where
  necessary.  Avoid language like “current top submission”; instead, say
  “maintainer‑awarded gold submission PR #101.”

* **Close with next steps.**  A short note saying “Future research directions
  and supporting materials are available in the source map” signals that
  additional context exists without burdening the reviewer.
