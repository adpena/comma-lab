# hold-rate bucketing sample

## result

- record count: `64`
- canonical global stream:
  - archive bytes: `6472014`
  - ratio: `3.0378179033605304`
- hold-rate bucketed global stream:
  - archive bytes: `6472014`
  - ratio: `3.0378179033605304`
- delta bytes: `0`

## interpretation

- hold-rate bucketing is exactly neutral on this real sample
- cheap token-derived bucketing still has no measured advantage over one canonical global stream
- the strongest systems direction remains:
  - one global stream
  - fewer resets
  - stronger grouped prediction
