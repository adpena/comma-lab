# Second fresh-reader verdict on submissions/mrs6 (Opus reader, no context, 2026-09-16 ~21:10Z) — recorded by MAIN (condensed from the reader's message; headings MAIN's, findings the reader's)

**Verdict: YES — would trigger the evaluation workflow as-is.** "The archive checks out byte-for-byte against FORMAT.md, the
failure paths are loud, and the timing is measured on the target device." Single most-improving change: give inflate.py the
prose treatment the README got (2,080 lines, six comments); close second: say in the README what frame 0 of each pair is and
why the seven pixel edits exist.

## Checks that held (reader-verified against FORMAT.md)
14-byte header `<4sBBBBHHH`; section sizes sum to 179,186 (+100 ZIP overhead = 179,286); boundary table 96 B; geometry block
64 B / 17 config values; flag byte 250 = five named bits + bit 32 set and unused; three C files hold structure/tuned constants
only (no per-frame/per-pixel/per-video data); range_decoder.c returns typed error codes; corrector.c refuses a non-correctly-rounded sqrt.

## Documented claims that do NOT hold (fix in FORMAT.md)
1. FORMAT.md §1 says the prior is "int8 weight codes with one fp16 scale per row"; `load_prior_weights` reads a per-row bit
   depth from nibbles (0–15 bits) and `<i2`/`i1` parameters; no fp16 in the prior path (that sentence describes the renderer stream).
2. FORMAT.md says the corrector's smaller families cover "4-neighbour and 8-neighbour spatial agreement"; corrector.c has
   N_CAUSAL = 4 with contexts from two neighbours (`spatial`) and four (`spatial4`); no 8-neighbour context exists.

## Code findings
- `split_payload_sections` names byte 7 `reserved` and uses it as the live flag byte; names byte 6 `table_mode` (the byte that is
  actually reserved) — the two names are backwards vs FORMAT.md.
- `decode_carrier`: bytes 123–138 (16 B) are never read; not mentioned in FORMAT.md; inside the charged member. Flag bit 32 set, unread.
- `PriorPredictors.predict`: `extra` assigned only if family == 1 then used ⇒ UnboundLocalError for any other family; the
  family == 2 branches are unreachable (half-removed feature, no guard).
- Never-validated tags: `magic`/`version` unpacked and unchecked; `restore_renderer_stream` unpacks four header fields and uses
  none; `restore_prior_stream` unpacks nine, uses three, skips one unexplained byte; the selector magic check is circular.
- Inert code: dead xz branch (+ `import lzma`); `decode_report` leftovers (`checkpoint_resume`, `token_cache`,
  `checkpoint_resumed_from_frame`); `validate_rice_parameters` validates nothing; `packed_length` discards `signed_bounds`;
  `IntegerPrior.norm_mode`/`use_norm_gates` unread; `unpack_renderer_weights` returns None on bad magic; `ByteRangeDecoder._byte`
  returns None past the end.
- Runtime patching: `freeze_model_codes` replaces `module.codes`/`model.frame_codes` on live objects (documented, deliberate).
- Unreadable literals: inflate.py 111, 400, 422, 450, 469, 1443, 2012; corrector.c 60–84; geometry.c 11–36.
- Seeds in a decoder (`torch.manual_seed(20260916)`, np seed) with nothing asserting every parameter was overwritten.
- Determinism deliberately off (cudnn.deterministic False) with an honest comment; score tied to one cuDNN selection.
- The archive's own hash compiled into the decoder (strong integrity; also single-payload by construction).
- Frame 0 is not a reconstruction: `127.5 + 64 * carrier` (12-coefficient field from 24×32) plus whole-frame ±1 / roll /
  checkerboard edits on ~25 first frames — the least-explained thing in the submission.
- The 16-digit score carries no report file in the packet (the PR body will carry report.txt verbatim).
- FORMAT.md "11,629 bytes stored, 16,249 after Brotli" reads backwards.
- Missing-dependency table: cc absent → loud rc 1; brotli absent → named SystemExit; numpy/torch absent → bare ImportError
  (README asserts they are present in the evaluator); no CUDA → message, `--device cpu` warns; stale .so only caught by
  missing symbol if inflate.py is run directly; ~3.5 GiB output space not checked; range_decoder.c includes math.h without -lm.

## Earlier complaints, status
Timing on target: FIXED (1,045.8 s + 44.3 s on T4, one run). Two implementations: MOSTLY FIXED (three tiny rules still exist
in both languages: bit reader, winner-takes-slack frequency rule, sqrt power ladder). Unreadable literals: STILL APPLIES.
Header flag bits: FIXED in docs, names swapped in code. Half-removed features: STILL APPLIES (one latent crash). Flattening
artifacts: STILL APPLIES (`bytes.fromhex('')`, hex-string magics, bare strings as docstrings, 300-char lines). Jargon: FIXED in
prose, still in code ("semantic"/"renderer", SLOT_WIDTH, RULE_SHIPPED, "the Lane mixer" at corrector.c:653).
