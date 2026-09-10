# TC3 move40 memo and seal-preparation audit

Bounded read-only narrative, arithmetic and custody review while the full
public receiver runs. No seal, native build, measurement, scorer or source
mutation was executed. Only this review note was written.

Reviewed snapshots:

| File | SHA-256 |
| --- | --- |
| `.omx/research/ddm_tc3_lane_predictor_receiver_seal_20260910.md` | `aca4aa835bb35f3dcdef2729b973737de1ef8d302e4e9e614e6b5bb1d305660a` |
| `.omx/research/ddm_tc3_20260910/SEAL_ARGV.json` | `f622a429adf55cd37e98f8b05c93e8d40d6bebd68250e07f2748fa279ee85150` |
| `.omx/research/ddm_tc3_20260910/SEAL_PREPARATION.json` | `4b17ac5e0ecc1f80bfda5f5256f8dc4139f5391c35d0c25870657c149eec06bb` |

## Verified narrative and arithmetic

No numeric or verdict-scope correction was found. The memo matches the
independently audited move40 fit/encode/container receipts: A saves 79 actual
archive bytes, B saves 37, and B is 42 bytes worse than A. Raw, rider, archive,
fitting-denominator and historical-ratio distinctions are preserved. The
registered A bound and the B analogy are explicitly separated, and neither
is presented as a universal or integer-code ceiling. The 37-byte stopping
rule remains an operational FORMULATION outcome.

The baseline receipt is current move40, archive 180,233 bytes with SHA
`986d536b31ed1079c517dadea73ba33daf018c53692a2b2fbbf8d6244dfe9857`.
Its reported components recompute to S 0.13763861019288715. Direct rate
arithmetic gives `-79 * 25 / 37,545,489 = -0.00005260285729665154`, as in the
memo, and projected candidate S 0.13758600733559048. The previous measured
audit's subtraction of the two rounded binary floats produces a difference
about 1e-17 away; that is floating-point rounding, not a conflicting gain.

The two-frame 103-array resume claim, equal checkpoint SHA, repeat recovery,
structured smoke's zero problems, equation output and inherited Ruff finding
counts are supported by their current retained receipts. They are correctly
distinguished from the still-pending full public n600 output identity. The
historical move39 recovery is now explicitly CLOSED/SUPERSEDED.

## Seal preparation

The prepared argument vector names only current move40 runtime, output,
baseline receipt, public smoke and retained evidence paths. Its expected
candidate archive SHA is
`299a8201662c8a407881a63214d944d0c8da25bf244ca4af034ecb730f5a7936`,
matching staged A at 180,154 bytes. The admission value is exactly the
negative 30-byte rate floor. Declaring the future contest-CUDA axis does not
claim a measured TC3 score; the notes preserve MAIN-only firing, explicit
same-host/CUDA boundaries and full public identity before execution.

`SEAL_PREPARATION.json` says NOT_EXECUTED. At this review snapshot the only
missing retained-path argument is the intentionally pending
`move40/PUBLIC_IDENTITY.json`. No stale move39 archive/field/candidate digest
or move39 evidence path occurs in the current vector. The historical vector
is separately retained and is not a current seal instruction.

## Original inventory omission

An independent complete visible-file census found 48 files in LIVE, 47 in
owned `source_runtime`, and 49 in `candidate_runtime`. Every one of the 47
shared source files is byte-identical to LIVE. The only LIVE-to-owned-source
omission is root `MANIFEST.sha256`, 4,301 bytes, SHA
`795469939275bbbd1cb624fff85f252b3a3c3b5a1bbd5dab592517b34c9ea3c9`.
The manifest has 46 rows and already disagrees with LIVE's `inflate.py` hash.

The original manifest is used by `compress.py:1520`'s source verifier, called
at `compress.py:2232`. It is not read by the literal public shell, public
`inflate.py`, runtime Python or runtime C. The public source-checkpoint digest
in `runtime/f26_inflate.py:195` covers `runtime/` and `cpr1/`, not that root
inventory. This omission therefore changes provenance inventory and the
full runtime-tree digest, but not the receiver's computation or any counted
archive payload. The copied `compress.py` is not claimed as a working TC3
encoder reproduction route.

Exactly six paths differ between owned source and candidate; **seven paths**
differ between original LIVE and candidate when the omitted manifest is
counted. Those seven are `MANIFEST.sha256` removed; `archive.zip`, `inflate.py`,
`inflate.sh` and `runtime/residual_archive.py` changed; and
`runtime/tc3_geometry.py`, `runtime/tc3_mixer.py` added.

The reviewed memo's six-path sentence explicitly refers to owned source and
is true. It should additionally state the bounded original-inventory omission
so readers do not infer full original-tree identity. Parent retained the
original inventory as `move40/SOURCE_MANIFEST.sha256` and the actual comparison
as `move40/SOURCE_COPY_CENSUS.json`, outside the frozen runtime. That preserves
the extra provenance without invalidating the running candidate's binding.
No receiver-source correction or rerun of the current public proof is
indicated by this metadata exclusion. A full-tree identity claim would be
incorrect and must not replace the precise file comparison.

## RECALL EVIDENCE

This continuation joins `move40_measured_comparison_review.md`, current fit,
encode/container/candidate receipts, public-control/smoke/equation/Ruff receipts,
baseline authority receipt, and the charter's receiver-only scope. For the new
inventory issue, searched the entire original receiver source with
`MANIFEST|sha256sum|shasum|rglob|iterdir|os.walk|glob(` and inspected the encoder
manifest verifier, public source-digest roots, shell and Python entrypoints,
and seal runtime-digest implementation. That established the precise
encoder-inventory dependency and excluded an unobserved public manifest read
within the inspected source. No new scientific claim or mechanism is added.

Disposition: FOLDED into the parent's active final memo and current public
proof/one-seal harvest; no independent queue row or extra measurement.

LIVE-HYPOTHESES: A's current measured byte saving still awaits full literal
public output identity; its causal implementation and passing partial controls
make survival plausible, without proving n600 identity.

DEAD-ENDS: rerunning public inflation solely because stale encoder inventory
was omitted is unsupported by the inspected execution path. Claiming full
original runtime-tree identity is also closed by the explicit manifest
difference. Historical move39 receiver evidence cannot replace move40 proof.
