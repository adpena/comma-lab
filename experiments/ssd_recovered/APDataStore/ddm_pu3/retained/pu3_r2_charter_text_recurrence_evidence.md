## YOUR SURFACE -- a MEASURED truncation nobody uncapped
Task #850 measured: the pose Gauss-Newton solve is HARD-CAPPED at 2-3 relinearizations with NO convergence test, and it is STILL DESCENDING 13-23% per iteration when it stops. That is a cap masquerading as a floor -- the exact genus this repo names "first attempt wall-clock is not a family verdict" and "caps are a genus, trajectory-stop them".
A sister case is already proven: an uncapped seg solve (#935) was run after the same discovery.
Your job: uncap the pose solve, run it to an actual convergence criterion, and measure what it reaches.
