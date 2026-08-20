# compress.py — what it rebuilds, and what it structurally cannot

This is the compression-side entry point for the submission. It is included because the challenge
asks for one and because a submission you cannot inspect is a submission you cannot trust. It is
**not** a one-command rebuild of the shipped `archive.zip`, and this file says exactly why.

## The honest headline

`compress.py` **cannot rebuild the `archive.zip` in this submission**
(`df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080`, 180,456 B). Run it against
that archive and it refuses by name, before doing any work, and prints the build stages it does
not express and the script that really performs each one.

That refusal is deliberate. The alternative — a script that runs, produces different bytes, and
reports success — is worse than no script.

## What it does express, exactly

The archive's payload member has eight sections. This entry point rebuilds **one** of them:

- **Rebuilt: the token stream.** It replays the shipped decode order, re-encodes the token field
  under the free decode-time probability corrector, splices the new stream into the member, and
  repacks the archive.
- **Carried through byte-identically: the other seven sections.** By construction, not by
  assumption.
- **Gated: `verify`.** The rebuilt archive is hashed and compared to the expected sha256 and byte
  count. A mismatch exits non-zero. There is no "close enough".

For a candidate inside that grammar the byte-close is genuine and end-to-end.

## What it does not express, and who does

The shipped candidate's chain re-decides content inside sections this script only copies:

| Stage not expressed here | The script that actually performs it |
|---|---|
| Segmentation token edit solve over 573 pairs (writes the semantic stream) | `ddm_jg3_joint_solve.py` |
| Splice of those edits into the base body | the jg4 composition step |
| Joint edit-admission waterfill; 455 of 573 edits admitted | `ddm_jg5_pose_resolve_on_edited_renders.py` |
| Pose-carrier re-solve against the candidate's own renders, and the archive rebuild that re-encodes the carrier stream | `ddm_up3_carrier_splice.py::build_archive`, solver `ddm_br1_pose_basis_reorientation.py::gn_solve_pair` |

No configuration file closes that gap. The missing pieces are missing **stages**, not missing
options, and the script says so rather than suggesting a flag that cannot help.

## Stage A is documented, not re-run

Training the checkpoint from raw video is days of GPU compute. `compress.py provenance` emits the
lineage instead: the stage scripts, their arguments, the corrector module, and an input manifest
with a sha256 for every input. What is *verifiable* is everything downstream of the checkpoint —
which is the part that determines the archive bytes.

## Requirements

- **Python 3.11+** (uses `hashlib.file_digest`). No third-party packages in this file: its only
  imports outside the standard library are from the `tac` research package (below).
- **It runs from the research repository, not from this submission directory.** It resolves three
  things by repository layout: `<repo>/src` for `tac.candidate_seal`, and
  `<repo>/experiments/ddm_rr2_encoder_byteclose.py` and `…/ddm_rr2_receiver_close.py`, which it
  shells out to. Copied into a bare submission directory it will not import.
- Inputs are supplied by `--inputs-json` or environment variables; run
  `python3 compress.py --emit-inputs-template` to print the manifest, including the expected
  sha256 of each pinned input. There are no local-path defaults.

## Usage

```bash
# Print the input manifest with every pinned sha256. Works anywhere.
python3 compress.py --emit-inputs-template

# Full chain for a candidate inside the grammar.
python3 compress.py --stage all --store ./build \
    --inputs-json inputs.json \
    --expected-archive-sha256 <sha> --expected-archive-bytes <n>

# Verify only.
python3 compress.py --stage verify --store ./build \
    --expected-archive-sha256 <sha> --expected-archive-bytes <n>
```

Stages: `provenance`, `encode`, `build`, `verify`, `decode`, `all` (`all` runs everything except
`decode`). `--resume` continues bit-faithfully from a retained frame-boundary checkpoint.

## Known gaps

1. **The shipped candidate is outside the grammar** — the four stages above. Closing this needs the
   four named builders wired behind one entry point; that work is not done.
2. **The expressibility registry is hand-maintained.** A future candidate outside the grammar that
   nobody has registered will fall through to a less precise message.
3. **No tests.** The refusal and happy paths were verified by running them, not by a suite.
