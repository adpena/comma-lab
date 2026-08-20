# ra3 prediction, recorded BEFORE the realised rows landed

written_utc: (see file mtime) -- deliberately written while the sweep was at pair ~150/600.

## What I predict, and why

The trust region is ANCHORED AT THE INCUMBENT (mu -> infinity gives u = 0, i.e. exactly ra2's
projection) and the incumbent is in the per-pair candidate set. So realised acceptance CANNOT
return a worse row than 15.211x. "Beats 15.211x" is therefore guaranteed by construction and is
NOT evidence about the family. I say this before seeing the number so it cannot be read as an
excuse afterwards.

The informative threshold is the BAR, not the incumbent:
  advisory bar 1.0651x (ra2's credit convention) -- needs a 14.28x improvement over the incumbent
  T4 bar (ratio-transfer) 1.3185x               -- needs an 11.54x improvement

## My honest expectation

The first-order model predicts d_pose falling from 3.217e-3 (mu=100, essentially the incumbent)
to 3.416e-6 (mu=1e-4) -- i.e. it predicts clearing the bar by ~40x. I do NOT expect that.

jc1 MEASURED that a linear pose model used as a DESIGNER over-states its control authority by
134x-1,065x, and that the error is a function of whether the model chose the step, not of the
step's size. My steps are much smaller than jc1's (1.0004x-2.32x the incumbent step, vs jc1's
3.4x-43.9x of coefficient RMS), so I expect the model to behave better -- but "better than
1,065x wrong" is a low bar.

Point prediction: realised accepted ratio in the range 2x-12x, i.e. an improvement over the
incumbent of roughly 1.3x-8x, still MISSING the advisory bar by ~2x-11x.
Probability I assign to actually clearing the advisory bar: LOW, ~10-15%.

## What each outcome means

  A. ratio < 15.211x but > bar   -> the family's last reformulation is measured and does not
                                    reach. The pre-registered falsifier does NOT fire as written,
                                    but its threshold is structurally passable, so the honest
                                    verdict is still closure. FLAG AS PRE-REGISTRATION DEVIATION.
  B. ratio <= bar                -> the family LIVES; emit a sealed fire-order for MAIN.
  C. ratio == 15.211x exactly    -> no pair improved; falsifier fires exactly as written.

## Decision rule for a SECOND Gauss-Newton round (recorded before the histogram exists)

J_i was measured at the SHIPPED coefficients, but every candidate sits ~13.5% of coefficient RMS
away, so the model extrapolates. A second round would re-measure J at the projection point
(~500 s) and re-solve. Whether that is indicated is decided by the accepted-slot histogram:

  * accepted mu concentrates AGGRESSIVE (mu <= 0.1)  -> the trust region is NOT binding, the
    model is being believed and is paying; re-linearising at the true operating point is the
    indicated next move.
  * accepted mu concentrates CONSERVATIVE (mu >= 10, or the projection wins outright) -> the
    NONLINEARITY is binding, not the linearisation point; a better Jacobian will not rescue it
    and round 2 is not indicated.

## My own attack on this method, recorded before the result

1. PER-PAIR REALISED ACCEPTANCE IS SELECTION ON THE SCORED SET. I choose mu per pair by the
   measured error on the same 600 pairs I then report. In a general ML setting that is cheating.
   Here it is not: the contest scores exactly these 600 pairs, the archive is legitimately fit to
   this video (rule 118 counts the video-derived payload), and -- decisively -- the choice needs
   NO side information, because it is baked into the stored z_i. It costs zero bytes.
   BUT it can cost RATE indirectly: a per-pair mixture across mu values may be less smooth along
   the frame axis than a single-mu solution, and the coefficient stream is DELTA-coded along
   exactly that axis. That is measured by the rate check on the accepted candidate, not assumed.

2. THE T4 COLUMN IS A RATIO-TRANSFER AND rn1 UNDERMINES IT. rn1 measured the advisory pose
   instrument ~18-21x optimistic in LEVEL against contest-CUDA. My within-axis advisory ratio is
   sound; the T4 bar assumes the d_pose RATIO transfers across axes, which nobody has tested.
   Lead with the advisory miss; label the T4 column a transfer.

3. d_seg NEEDS NO MEASUREMENT AND THAT IS NOW A PROOF, NOT A TALLY. upstream/modules.py:108 is
   ``x = x[:, -1, ...]  # Use only last frame``: SegNet reads ONLY frame_1. The carrier renders
   ONLY frame_0. d_seg is invariant to ANY carrier edit STRUCTURALLY. ra2 called this "MEASURED
   identically on 4 treatments"; it is stronger than that and no future carrier arm need re-measure it.
