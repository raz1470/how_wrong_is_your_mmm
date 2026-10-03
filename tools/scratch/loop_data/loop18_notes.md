
### Loop 18 — is the 20-channel bias collapse caused by blackout-month collisions? (targets: channel-count scaling)

Hypothesis from loop 17: only 3 months (>4 wk) are eligible for a blackout, so 20 channels pile ~6.7 per month. Added `min_blackout_month_weeks` (default None = old behaviour) to `cross_month_blackout_roundrobin_schedule` / `roundrobin_plus_edge_layer`; set to 4 so 12 months are eligible (dark weeks capped at 3 in 4-wk months). Script `loop18.py`, 4 seeds, mrw=4.

| channels | eligible blackout months | variance | bias % | id | cost % |
|---|---|---|---|---|---|
| 10 | 3 (old) | 0.169 | 18.0 | 15.6 | 1.69 |
| 10 | 12 | 0.152 | 26.3 | 18.1 | 1.22 |
| 20 | 3 (old) | 0.210 | 70.7 | 18.1 | 1.69 |
| 20 | 12 | 0.199 | 68.7 | 20.4 | 1.19 |

**Hypothesis rejected.** Spreading blackouts over 12 months doesn't fix 20-channel bias (70.7 -> 68.7) and hurts 10-channel bias/id (shorter effective blackout, 3 wk in 4-wk months); it only lowers cost. Keep the default (None).
**Likelier cause:** the package itself warns "156 observations across 20 channels (7.8 per channel) ... estimates tend to be very wide and unstable even before collinearity". Unphased bias at 20 channels is already 104%, so this is a sample-size limit of the fit, not a design flaw.
**Next up (loop 19):** 20 channels with longer history (e.g. 208-260 obs, >=10/channel) to confirm the design recovers when the fit is adequately powered.
