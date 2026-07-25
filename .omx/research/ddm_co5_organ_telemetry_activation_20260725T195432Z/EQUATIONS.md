# DDM CO5 equation disposition

UTC: 2026-07-25T19:54:32Z  
Status: `NO_NEW_LAW_VALIDATED`

The four recalled designs remain mathematically well-formed, but CT1 × EV1
does not supply the matched authority needed to validate any of them:

1. Pontryagin/Bellman residual

   \[
   r_t^\lambda =
   \lambda_t-\left(\partial_{x_t}L_t+
   J_{x_{t+1},x_t}^{\mathsf T}\lambda_{t+1}\right).
   \]

   Required but absent: ordered adjacent campaign costates and realized
   transition-Jacobian terms.

2. M34 dual consistency

   \[
   z_{t,d} =
   \frac{\lambda^{\mathrm{organ}}_{t,d}
   -\lambda^{\mathrm{M34}}_{t,d}}
   {\sigma_{t,d}}.
   \]

   Required but absent: same-state, same-dimension duals in matched units and
   a measured uncertainty band.

3. Compression progress per effort

   \[
   \rho_{[t_0,t_1]} =
   \frac{S(x_{t_0})-S(x_{t_1})}
   {(T_{t_1}-T_{t_0})/3600}.
   \]

   Required but absent: two exact receiver-realized N600 campaign endpoints,
   matching counted archive identities, and their wall-clock interval. The
   CT1 batch-local trace is not a legal numerator.

4. Regret-bounded duty allocation

   \[
   U_i(n)=\widehat{\rho}_i+
   c\sqrt{\frac{\log n}{N_i}}.
   \]

   Required but absent: active measured progress/effort, typed fired-duty
   outcomes, and a preregistered confidence/exploration constant.

No canonical-equations registry row or empirical anchor is appended. A failed
backtest is scoped to the stopped-v5 CT1 telemetry joined with EV1; the four
design families remain open.

