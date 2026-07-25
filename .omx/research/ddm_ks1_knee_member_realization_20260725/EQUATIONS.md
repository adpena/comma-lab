# DDM KS1 knee-member realization — canonical equations leg

All measured terms below are `[macOS-CPU frozen-scorer advisory]`; they are not
contest scores.

## Receiver-closed score functional

For exact packet bytes `B`, realized SegNet error `d_seg`, and frozen PoseNet
error `d_pose`:

`S(B,d_seg,d_pose) = 100*d_seg + sqrt(10*d_pose) + 25*B/37,545,489`.

The fresh RD1-knee endpoint is:

`S(130093, 0.07051923116048177, 36.618184751411334) =
26.274425245267324`.

## Card §7 rate-window line

At the card's fixed pose-class budget:

`d_seg_max(B) = (0.173931 - 25*B/37,545,489)/100`.

Therefore:

| Candidate | B | measured d_seg | d_seg_max(B) | measured endpoint under line |
|---|---:|---:|---:|---|
| W_seg | 130,870 | 0.024124510023328993 | 0.0008679003880490144 | false |
| W_joint step50 | 130,101 | 0.06974277072482639 | 0.0008730208433985238 | false |
| RD1 knee / W_joint initial | 130,093 | 0.07051923116048177 | 0.0008730741121147737 | false |

The filter also admits a state when descent from it can plausibly reach the
line. That reachability is not re-labelled as measured closure: W_seg retains
the card's presumptive reachability; step50 and its knee ancestor remain
indeterminate with respect to terminal closure.

## Lexicographic decision

Among admitted/reachable starts, minimize measured n600 d_seg. The knee costs
777 fewer packet bytes than W_seg but has

`0.07051923116048177 - 0.024124510023328993 =
0.046394721137152775`

more d_seg. Thus its byte saving cannot overturn the lexicographic first
quantity:

`PRESUMPTIVE START CONFIRMED W_seg`.

No resealer is invoked because there is no displacement.
