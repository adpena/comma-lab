# Canonical equations draft — DDM V10 Fisher G2CS1 + transported events

**Status:** DERIVED from registered scorer laws and MEASURED advisory anchors. No equation or score
claim is promoted by this isolated branch.

## E1 — exact action and rate price

For final archive length `B`, source bytes `N=37,545,489`, and frozen evaluator distances,

`S(B) = 100 D_seg + sqrt(10 D_pose) + 25 B/N`.

All numeric instantiations here are `[macOS-CPU frozen-scorer advisory]`, `score_claim=false`.

## E2 — Fisher/head candidate acquisition

For an erroneous scorer cell with target class `c`, realized class `c'`, target top-two margin `m`,
and registered exact rank-4 head normal `n_cc' = ||w_c-w_c'||₂`,

`d_flip = |m| / n_cc'`,

`κ_F(m) = 1/2 sech²(m/2)`,

`A = 1[c != c'] · κ_F(m) · w_band(|m|) / max(d_flip, 10^-3)`.

`w_band` is 4 on `[0,.1)`, 2 on `[.1,.5)`, 1 on `[.5,1)`, and .25 thereafter. The reciprocal
orders cheapest exact-head flips first. `A` proposes semantic candidates only; it is not an
admission verdict and makes no unmeasured inner-Jacobian claim.

## E3 — receiver-semantic correction operators

For normalized horizontal coordinate `x_hat ∈ [-1,1]`, Road-mask displacement coefficients `a_q`,

`Δy(x_hat) = round(Σ_{q=0}^3 a_q x_hat^q)`,

`M'_Road(y,x) = M_Road(y - Δy(x_hat), x)`.

For event `e=(t0,role,action,shape,bbox,L,gx,gy)` and the sole counted Pose6 code `p_t`,

`Δx_t = round((p_t[0]-p_t0[0]) gx / 16)`,

`Δy_t = round((p_t[1]-p_t0[1]) gy / 16)` for `t0 <= t < t0+L`.

The generic receiver translates and rasterizes the bbox primitive, then unions it for birth or
subtracts it for death before canonical class paint. The packet cannot encode pixel coordinates as a
list, RGB, scorer state, or target labels. Lane G2CS1 remains `a'_{t,j,q}=a_{t,j,q}+fp32(δ)` followed
by generic LBND2 rerasterization.

## E4 — exact measured marginal admission

Let current and proposed receivers differ by one semantic candidate. Replay every complete canonical
scorer batch touched by that candidate and update the full-window totals. With `P=pair_count·384·512`,

`G_dist = 100(E_current-E_proposed)/P + sqrt(10 Dpose_current) - sqrt(10 Dpose_proposed)`.

For exact marginal archive bytes `ΔB`, admit iff

`E_proposed < E_current`,

`Dpose_proposed <= Dpose_base + ε_pose`, and

`G_dist > 25 ΔB/N`.

Here `ε_pose=0`. This is a sequential measured greedy search across a mechanism-diverse inventory,
not a closed-form optimizer. Full bridge rungs are remeasured after selection.

## E5 — nested exact-byte ladder and stop

For requested added budgets `b ∈ {0,5120,15360,40960,102400}`, select the longest admitted nested
sequence whose receiver-closed archive obeys `B_b-B_0 <= b`. Exact byte-identical archives share a
deterministic bridge result. `b-(B_b-B_0)>0` is reported as unspent budget; it is not filled by an
unmeasured correction.

The n64 knee is the first nonzero rung: `ΔDseg/ΔB = 2.050904929047e-6` per byte at `ΔB=1,353`.
All later requested rungs are identical. n256 has no admitted nonzero rung.

## E6 — scoped falsifier

The preregistered near-200 KB instance falsifier is

`F = [B >= 180,000] ∧ [D_seg > .00116] ∧ [measured marginal plateau]`.

It is not triggered: the largest measured totals are 53,021 B and 72,397 B. Instead the earlier
condition `V = [candidate inventory exhausted] ∧ [D_seg > .00116]` holds. Therefore only this finite
G2CS1-c3 / Road-cubic / Lane-Movable-bbox INSTANCE vocabulary is falsified. The broader structured
carrier and transported-event families remain open.
