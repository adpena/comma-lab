# TC4 receiver independent review 2 — clean 1/2

Reviewer: `/root/tc4_review`; 2026-09-10. Reviewed final `experiments/ddm_tc4_receiver.py` SHA-256 `5671759babebbe326ba9c797fe003dd1640037a06da0f4f38bee0f80d0cd0a6c`. No receiver or scorer launched by this reviewer.

**CLEAN.** Both pass-1 findings are fixed. Selection itself is now source-bound, and recovered staging checks the mask plus actual selected archive hash and size. Completed public identity is scoped to the requested work name, so `public_fresh2` cannot return the first run's receipt. A further recovery gap found during the fix review is also closed: after validating a completed primary work receipt, resume republishes the canonical root receipt, covering a crash between those two writes.

Reviewed the final binding, staging, smoke, partial decode, completed decode and resume branches. The public smoke validator demands the actual candidate/control runtime identities and exact CUDA gate message; unrelated RuntimeErrors cannot pass. Full identity requires the literal public report and exact retained source raw bytes plus the pinned 600-frame field. Native builds are retained and source/argv/library verified on reuse; frame checkpoints preserve nested TC4 counts/config with the decoder and corrector state. This is a clean code review, not an execution or score verdict. One further clean pass is required.
