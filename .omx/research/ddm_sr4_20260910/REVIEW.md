# ddm_sr4 operational review

Pass 1: the hardlink action hashes both complete data forks, rejects changed identity and writers, writes and fsyncs a certificate before replacing the source directory entry, then proves both paths share the destination inode. Historical hashes only select candidates. The full retained payload is never discarded. Stage receipts permit resuming by checking existing inode equality.

Pass 2: destination and source are on the same filesystem for hardlinks; the explicitly live directories, upstream components, and symlink aliases are rejected. Source metadata differences (mtime/inode) are recorded rather than promised identical; content identity is the invariant. Failed checks retain bytes and journal the blocker. The independent move-42 pair passed on the actual files.

Move review: reuse tac.artifact_moved.move_with_manifest, whose copy_verified rereads source and hashes destination before retiring source. Nine exact older advisory raws are allowlisted; GB1 identity raws are excluded because their memo binds a seal. Destination budget is checked per file with an 80 GiB post-copy floor. Existing MOVED destination claims and open descriptors refuse. Source memo/hash/identity are pinned. Interrupted copies remain; resume uses a fresh attempt path. The restoration command copies the preserved data fork; it does not claim the missing BO2/RD2 archive runtime can regenerate it. No archive or runtime is moved.

Syntax check: embedded Python in both shell files compiled. A keyword syntax error in the draft move script was corrected before any launch. No Python source files or scientific production functions changed.

No score, scorer, training, GPU, upstream, PR, staged-index, or pointer operation is authorized by these scripts. Checkpoint is kept in this arm directory because the charter prohibits edits to .omx/state; serializer-managed state is used only by the explicitly required serializer.

Final move review: interrupted-copy and resume events also enter the shared arm ledger; all copy exceptions retain bytes and record the blocker. Destination stat identity is captured after the canonical full destination hash. The bounded filename census is checked by the unchanged canonical validator (16 file records, 2 canonical legacy-directory exclusions, 0 violations); each certificate is SHA-pinned and decoded from those same pinned bytes before constructing incoming-target exclusions. The slower canonical discovery completed independently: the same 16 certificate paths, zero violations. Embedded Python syntax passed before move execution.

Final custody verifier review: requires 9 completed moves, checks every retained hardlink name against the full-hash destination inode/size/mtime binding, checks moved destination identities captured after full hashing, and validates the 16 old plus 9 new MOVED manifests using canonical metadata/header checks. It fails if either reserve target is missed. This is verification of actual mutations, not a synthetic test. Shell and embedded Python syntax passed.

Actual final audit: corrected move-receipt field normalization (path versus source), reran successfully on 25 real hardlink pairs, 9 real copied raws, and 25 MOVED certificates. Both reserve assertions passed. No mutation was retried or changed by this parser correction.
