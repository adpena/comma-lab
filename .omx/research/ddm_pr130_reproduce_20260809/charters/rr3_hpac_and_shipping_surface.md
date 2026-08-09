# ddm_rr3 — recursive adversarial review, LAYER 3: HPAC + the SHIPPING surface

**Operator directive 2026-08-09:** *"The Recursive adversarial Review must check every step of every
stage of everything, recursive fractal of our port to upstream PR one thirty."*

Fresh eyes. **This is the highest-stakes arm: it carries the one contest-critical question nobody has
answered.**

## THE OBJECT

`BASE = PR130 CPR1 S = 0.172141297491896447` `[contest-CUDA, DALI GT, n600]`, archive 191,052 B sha
`0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`.
Intake (READ-ONLY): `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo`,
`SOURCE_REPO_HEAD = e34f31bc4969042c0051ac81aa3c56884419a231`.
Ledger under audit: `.omx/research/ddm_pr130_reproduce_20260809/OFF_THE_SHELF_VS_PORTED.md` (3856788c96).

**HPAC is THE rate axis.** Measured leave-one-out marginals on the reproduced archive:
tokens 116,980 B = **61.23%** → 0.0778922 S · semantic 36,580 B = 19.15% · pose 23,384 B = 12.24% ·
hpac 15,092 B = 7.90% · ZIP overhead 104 B. Superadditivity gap −20 B (0.0267%) ⇒ marginals additive.
Joint LZMA of the 3 model sections is 224 B WORSE than 3 separate streams (a free win, unclaimed).

## THE CONTEST-CRITICAL QUESTION — answer this FIRST

`encode-tokens` required `constriction==0.5.0` (a Rust/numpy arithmetic-coding lib). We installed it
locally. **Does the DECODE path — the shipped `inflate.sh`/`inflate.py` that the contest runtime runs
— also need `constriction`?**

If YES, then per CLAUDE.md lesson L4 as amended (dep cap DELETED; binding constraints are
(a) it installs/imports in the contest runtime inside the 30-min decode budget, (b) deterministic
decode, (c) rule-118: no video-derived data smuggled as code or as a "dependency"):
- Is it DECLARED in the runtime tree's dependency set?
- Does it SELF-INSTALL fail-closed (the e4 brotli precedent), and has that bootstrap been PROVEN by a
  bare-venv smoke — not assumed from host site-packages (the **r5 lesson: prove the bootstrap, never
  assume the host**)?
- Does it fit the 30-min budget? Note `timeout-minutes: 30` is the WHOLE CI job (task #835), and
  `evaluate.py:64` sums `rglob('*')` over `videos/` for the rate denominator (Catalog #812), NOT the
  constant 37,545,489.
This is a SHIPPING RISK, not a curiosity. Report it as the headline whichever way it resolves.

## YOUR SCOPE — every step, element-deep

1. **`code/hpac_integer.py`** — the integer-lattice masked-autoregressive model. The constructor guard
   `channels * weight_bound * activation_bound + 32768 >= 2**24` (64·127·127+32768 = 1,065,024 <
   16,777,216) keeping accumulations exact in fp32. The `probability_table` int16 quantization at 1/8
   logit resolution. The sheared-wavefront mask (`patch_group_mask`, `offset = col - center +
   delta*(row - center)`, mask-A `offset<0` / mask-B `offset<=0`; `(1+delta)·patch − delta = 190`
   groups/frame). Verify each claim at source; correct anything MAIN got wrong.
2. **`code/hpac_self_compress.py`** — learned per-output-channel `bit_depth`. What exactly is learned,
   how is it discretized, and does the shipped `hpac.bin.xz` (15,164 B, sha `ef8bb9d5…`) round-trip?
3. **`code/pack_hpac_self_compress.py`** + **`extract_integer_hpac_archive.py`** — hb2 fixed a
   round-trip failure here. Re-audit the FIX adversarially: is it correct, or does it merely make the
   symptom go away? (Fixes are unreviewed new code.)
4. **`code/codec_hpac_integer.py`** — the arithmetic codec. MAIN verified device is load-bearing at 8
   sites (`:58,:66,:70,:105,:108,:119,:174,:206`) with `.cpu()` only at `:29,:81,:122` for numpy
   marshalling. Extend: what is `logit_hash_encode` (we got
   `33fd711b305efb12ab9f7363c1404229996b79b8e27896f20ed910e98f105f75`), is there a matching DECODE
   hash, and does the codec verify encode/decode agreement anywhere?
5. **The assembly surface** — `rebuild_submission_hpac.py`, `build_submission_archive.py`,
   `compress.sh`. What ZIP semantics (stored vs deflated, member order, timestamps) make the rebuild
   byte-identical? Is that reproducibility fragile to anything (locale, zipfile version, filesystem
   order)?
6. **`residuals()`** — `output[1:] = (tokens[1:] - tokens[:-1]) % NUM_CLASSES`, temporal delta in the
   class ring mod 5. Confirm; note whether it is label-field-specific (it matters for #978: our
   frontier is LATENT tokens, theirs is dense SEMANTIC).

## ROUND-1 FINDINGS — build past, do not re-derive

- F1: `reproduce.sh` reads BANKED artifacts — byte-identity closes ASSEMBLY only.
- F3: `encode-tokens` consumed the BANKED `hpac_selfcompress_l1_fastbits_e60.pt`
  (`canonical_hpac_checkpoint()` default; `HPAC_CHECKPOINT` unset) ⇒ the byte-identity validates the
  CODEC PORT, not our 60-epoch hpac training.
- MEASURED: `tokens.bin` 116,980 B sha `948379872ff81a4e5d948ec301c143be00ebd0033544c8abdfb4af0f4c4a15eb`
  byte-identical on Metal, 600 frames, 1,057 s, rc=0. hpac train 59.18 s/epoch.

## OPTIMAL FORM

- **Reference form:** source-complete read of all five HPAC/codec modules + the assembly path, PLUS a
  real bare-venv dependency-closure test for the decode path (the r5 lesson — prove the bootstrap).
- **SCOPE reductions (legal):** static reading where execution needs Metal (arms have no device);
  CPU-only venv bootstrap smokes ARE in scope and are the reference form for the dependency question.
- **MECHANISM reductions (declare TOY-BRACKET):** answering the constriction question by reading an
  import list instead of running a clean-env import; accepting hb2's fix because tests pass without
  reading what it changed; asserting ZIP determinism without checking the writer's settings.
- **Provenance pins:** intake `e34f31bc4969042c0051ac81aa3c56884419a231`; ledger 3856788c96;
  `hpac.bin.xz` sha `ef8bb9d59bdd3916fb77713c11cdcb85e029f01d80b82472a40ab28f7e56a9ee` / 15,164 B.

## NON-NEGOTIABLES

- Intake READ-ONLY; never edit in place, never `git add` inside it.
- MPS/MLX never score authority; no Metal device available to you.
- No number without a locatable receipt; ABSENT is honest, restating is not.
- verdict_scope on every negative. Denominators on every count.
- Commit via `tools/subagent_commit_serializer.py`, POST-EDIT `--expected-content-sha256` per file,
  tags `[no-triality] [p0-ledger-ok]`. NO Claude/AI attribution, no `Co-Authored-By`.
- `REVIEW_GATE_OVERRIDE=1` FORBIDDEN with `.py`; fine for `.md`/`.json`.

## DELIVERABLE

`.omx/research/ddm_pr130_reproduce_20260809/RR3_HPAC_SHIPPING_AUDIT.md` — **the constriction decode
verdict as §1** (with the bare-venv receipt or an explicit blocker), then per-element table, ranked
findings with falsifiers, and "could not check / why."
