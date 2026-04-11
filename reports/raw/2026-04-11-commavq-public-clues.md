# commavq public clues

## official leaderboard state

Source: `commaai/commavq` README and `comma.ai/leaderboard`, accessed 2026-04-11.

- top public lossless result: `3.4x`
- method label: `self-compressing neural network`
- next cluster:
  - `2.9x` arithmetic coding with GPT
  - `2.7x` arithmetic coding with GPT
  - `2.6x` arithmetic coding with GPT
- classical cluster:
  - `2.3x` / `2.2x` `zpaq`
  - `1.6x` `lzma`

## public discussion clues

### Brady Wynn writeup

Source: `https://bradywynn.github.io/comma/`, dated 2025-04-08.

- says the provided GPT is around `614MB`
- says decode/compress time was impractical with the provided model
- reports a working approach using a **smaller transformer**
- key modeling choice: **predict all 128 tokens of the next frame at once**
- arithmetic coding over that model reportedly reached about `2.9x`

### Reddit PSA thread

Source: `r/Comma_ai`, 2026-01-09.

- public claim: the provided GPT was not what the best submissions used
- public reply from `YourSuperheroine` frames the challenge as a novel real-world problem and implies arithmetic coding itself was not the hard part

### Reddit solution thread

Source: `r/Comma_ai`, 2026-01-30.

- reports a `5.3M` parameter transformer
- `8-bit` quantization
- `constriction` entropy coding
- key modeling choice again: **predict all 128 tokens of the next frame at once**
- final officially checked result corrected to `2.7x`

## implication for this repo

- token-by-token use of the provided GPT is still a valid measurement lane, but it is probably not the fastest route to a competitive full submission
- the strongest public architecture clue is:
  - **next-frame prediction over all 128 tokens**
  - then arithmetic coding
- the repo should treat this as the highest-EV post-GPT-scoring modeling pivot once the current arithmetic coder gap is understood
