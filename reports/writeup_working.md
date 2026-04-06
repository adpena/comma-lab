# writeup working notes

## best-writeup objective

The writeup is strongest when it shows the frontier shape, not just the frontier point.

## current honest state

- live floor: `2.12` at `524x394`, `crf34`, `film-grain=22`, `lanczos`, `unsharp=0.35`, explicit `tv/bt709` encode tags, explicit `rgb24(pc)` decode
- nearby failed continuation: `2.21` at `crf35`
- nearby failed reconstruction trim: `2.20` at `unsharp=0.30`
- nearby failed synthesis removal: `3.33` at `film-grain=0`
- nearby failed geometry trim: `2.23` at `522x392`
- nearby upscale win: `2.18` at unchanged bytes with `lanczos`

## strongest new writeup angle

The local frontier is now more useful because the lab found two different kinds of wins:
- a clean same-bytes reconstruction-kernel win (`bicubic -> lanczos`)
- a production-hardening win from making the color contract explicit (`2.18 -> 2.12`)

That gives a clear contrast against the nearby losses and makes the final operating point look deliberate rather than accidental.

## strongest new rigor angle

The writeup should now make the engineering-cleanliness story explicit too:

- the honest payload is now the installed runtime payload under test, not a fuzzier repo-local approximation
- smoke and eval now share the same clean-run assumption
- AV1 + ROI is guarded instead of silently drifting into x265-only logic
- ROI tooling now respects the configured ffmpeg/ffprobe binaries
- ROI placement and postfilter knobs now actually do what they claim
- the flat path now uses explicit bt709/tv tagging and explicit rgb24(pc) decode conversion
- packaging survives invalid `TMPDIR` instead of failing on a stale temp-root path
