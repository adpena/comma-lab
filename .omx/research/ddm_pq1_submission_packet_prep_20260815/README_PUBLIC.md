# AFR1 packet generation 7

This directory is a frozen, not-yet-published contest packet for
`ddm_afr1_tile48_groupbin8`.

## Exact object

| Field | Value |
|---|---|
| Archive | `archive.zip`, 180,002 bytes |
| SHA-256 | `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` |
| Member | one stored member `p`, 179,902 bytes, SHA-256 `cf1afed8542e9dbd274b52ef14cc844a42cf2d659efceecf33edc3ab59c2edac` |
| Runtime | 38 enumerated files |
| Runtime tree | `6cdfa27dd1e9b46fc2bbbe88774c78d95ed3605fee7a15ba3861f96e24041e58` |
| Portable content tree | `4856087f5f857c83f045736db1db18d41667eb98942b25242422ab636a797c84` |
| Runtime files digest | `b2638b491371fd0961382b99f1dfacb42b2b22ae37c28ee4306f7e0ae1b32ffc` |

The packet must be judged as archive plus enumerated receiver tree. Reusing the
archive with a different receiver is a different object.

## Exact measured result

`[contest-CUDA] Tesla T4, n600`

```
seg   = 0.020139
pose  = 0.007981227975693965
rate  = 0.11985594327989708
S     = 0.14797617125559104
```

The evaluator displays `0.15` after rounding to two decimals. The claim is the
score recomputed from components. Its worst-case error from the eight-decimal
printed components is `3.63296497868841e-06`.

AFR1 is competitive on the claimed axis: it is below the public PR #135 result
of `0.162` on `[contest-CUDA]`. It is not a claim of a CPU score.

## What changed after the prior packet freeze

The generation-6 packet froze at rc2: 180,456 bytes and
`S=0.14827847122030852`. Five lossless coder/container pointer moves followed:

| State | Bytes | Exact CUDA S |
|---|---:|---:|
| fx5 e1 | 180,386 | 0.14823186109359 |
| dx2 | 180,368 | 0.14821987563243377 |
| gb1 | 180,215 | 0.14811799921260607 |
| lb1 | 180,083 | 0.14803010583079396 |
| AFR1 | 180,002 | 0.14797617125559104 |

AFR1 is 454 bytes smaller than rc2. Their measured CUDA raw outputs are
byte-identical, so both distortion legs are unchanged and the full score delta
is the rate term. The changes add no learned artifact and do not alter the
borrow/own classifications in `BORROWED_SUBSTRATE_ACCOUNTING.md` section 11.

## Decode budget

The authority run measured 578.9354022370001 seconds for inflation and
42.69640948199992 seconds for evaluation. The conservative charged total is
621.631811719 seconds. This passes the projected 822-second cold-cache residual
ceiling by 200.36818828100002 seconds. The residual window is explicitly a
projection; the two component timings are measured.

## CPU boundary

`[contest-CPU]` is **RECORD-WITH-REASON** for AFR1. No exact AFR1 CPU score was
run, and no older CPU row is inherited. The reason is measured on this lineage:
the CPU axis was 0.0432 worse on identical bytes, with pose about 21 times
worse, so it cannot become the effective frontier. This is an axis disposition,
not a CPU score claim.

## Reproduction boundary

The end-to-end `compress.py` VERIFIED result is scoped to packet generation 3,
the bytes on which it ran. AFR1 was not re-run through that entry point.
`compress.py` therefore fails closed by exact AFR1 SHA before doing work rather
than pretending the older recipe can reproduce it.

What was verified for AFR1 is separate and concrete: deterministic encoder
repeat, native/Python receiver identity across all 600 pairs, exact archive and
runtime hashes, a tolerance-zero packet-target seal, and an exact T4 authority
row on these bytes.

## Attribution

The learned semantic renderer and pose-carrier vehicle descends from PR #130
and PR #135, with PR #133 transitively in that ancestry. We do not claim that
learned vehicle. Our work is the decision and representation layer over it:
joint edit admission, lossless coder/container transforms, receiver assembly,
and the custody apparatus. The shipped accounting file gives the per-section
classification, prior-art disclosures, and receipts.

The public source repository is <https://github.com/adpena/comma-lab>. The
evaluated source commit is
`1c9fbbf58716eb0f26bcdf2a91e3c89d0e4efdde`; the archive and runtime hashes
above provide the deterministic binding. Public visibility of this exact commit
has not been re-verified by this offline freeze step.

## Verify locally

From this directory:

```bash
sha256sum archive.zip
sha256sum -c MANIFEST.sha256
python3 compress.py --expected-archive-sha256 cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25
```

The last command is expected to refuse with the named AFR1 missing-stage list;
that refusal is the honest reproduction result for this entry point.

## Publication state

**PREPARED HOLD, NOT PUBLISHED.** There is no hosted archive URL in this packet.
No PR, upload, host action, or public comment occurred during the freeze. The
repository operator must personally decide policy compatibility and write all
public-facing text before any publication action.
