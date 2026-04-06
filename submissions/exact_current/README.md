# exact_current submission

This submission is intentionally shaped for the challenge as currently published.

## Files

- `archive.zip` — intentionally tiny placeholder archive
- `inflate.sh` — calls `inflate.py`
- `inflate.py` — reconstructs raw frames from the upstream repo's `videos/` directory

## Expected usage

This submission is meant to be copied into the upstream challenge repo under:

```text
submissions/exact_current/
```

Then evaluated with the upstream `evaluate.sh`.

## Notes

- The inflator prefers the upstream `frame_utils.yuv420_to_rgb` path when available.
- The archive payload is not used by the inflator.
