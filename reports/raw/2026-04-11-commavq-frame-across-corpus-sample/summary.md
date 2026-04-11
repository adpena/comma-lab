# frame-across-corpus sample

## result

- segments: `32`
- frame indices: `1200`
- original bytes: `9830400`

### canonical frame-order stream
- archive bytes: `7990531`
- ratio: `1.2302561619496877`

### token-derived bucketed frame-order stream
- archive bytes: `7990531`
- ratio: `1.2302561619496877`
- delta bytes: `0`

## interpretation

- this exact frame-across-corpus formulation is much weaker than the clip-global stream
- simple frame-order bucketing does not help on top of it
- current evidence still favors:
  - one global clip stream
  - stronger grouped prediction
  - minimal resets
