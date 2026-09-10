# FastContextMixer parent review

Reviewed experiments/ddm_tc4_fast.py after its author's review and real34-frame
controls. Integer transition history stores a lower endpoint only once both
adjacent classes have been decoded. It checks both newly-known neighbour pairs;
query masking excludes row y itself, later rows, and unread groups. Selection
of highest block and highest bit uses exact integer arithmetic, including
remainder0 and63. The plane supplies ordered upper/lower classes. Row lengths
cannot exceed64 and queries only use lengths1..63 from earlier columns in the
same block. Nonavailable run and above-bin sentinels match the reference.

Scratch resets at begin_frame and is absent from end-frame snapshots by design;
all persistent old mixer and new KT counts remain in inherited state. The old
mixer enforces exact group sequence before scratch updates. Map observations
cannot reach their own coding rows. New weights and class-major order are
unchanged. Validation exercises full coding/observe API, both sentinels and
actual original frequencies/mixed float32 rows, with a fresh-object snapshot
roundtrip and independent process resume. Thirty-four sampled real frames are
implementation controls, not n600 runtime or compression authority.

No correctness finding in this pass. Ruff clean. Export deliberately slices
only the generic imports/algorithm/class before validate, removing all research
CLI imports, absolute paths and control-only functions; the receiver review
must inspect that literal port. Full public n600 identity remains required.
