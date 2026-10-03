
### Loop 20 — 10 channels, 3 years, winner vs ALL default strategies (4 seeds)

Script `loop20_allstrat.py`; raw `loop_data/loop20_allstrat_seeds*_raw.csv`. 52wk warm-up + 156wk plan; built-in levers applied across the full plan, winner phased per year. Sorted by bias.

| strategy | variance | bias % | identifiability | cost % | max spike |
|---|---|---|---|---|---|
| Blackout (dark=4, prob=1.0) | 0.031 | 3.268 | 1.24 | 15.031 | 13.45x |
| Blackout (dark=3, prob=0.8) | 0.037 | 4.164 | 1.946 | 9.17 | 13.82x |
| +/-80% (edge, balanced) | 0.06 | 5.998 | 9.435 | 2.572 | 2.81x |
| +/-60% (edge, balanced) | 0.078 | 7.749 | 14.847 | 1.36 | 2.19x |
| +/-80% (seesaw) | 0.073 | 8.456 | 16.266 | 1.964 | 2.28x |
| +/-60% (seesaw) | 0.094 | 11.281 | 22.752 | 1.059 | 1.90x |
| Blackout (dark=1) | 0.075 | 11.441 | 14.494 | 1.623 | 1.78x |
| +/-40% (edge, balanced) | 0.109 | 11.59 | 23.182 | 0.584 | 1.70x |
| +/-80% (uniform) | 0.103 | 11.624 | 20.7 | 0.751 | 2.89x |
| roundrobin_plus_edge_layer | 0.092 | 12.946 | 8.822 | 1.692 | 2.69x |
| +/-60% (uniform) | 0.131 | 14.865 | 27.832 | 0.399 | 2.16x |
| +/-40% (seesaw) | 0.129 | 16.388 | 30.403 | 0.46 | 1.56x |
| +/-40% (uniform) | 0.171 | 20.749 | 35.712 | 0.173 | 1.67x |
| +/-20% (edge, balanced) | 0.171 | 23.558 | 35.975 | 0.145 | 1.32x |
| +/-20% (seesaw) | 0.192 | 26.552 | 38.959 | 0.115 | 1.27x |
| +/-20% (uniform) | 0.22 | 34.37 | 42.983 | 0.045 | 1.30x |
| unphased | 0.252 | 49.006 | 45.418 | 0.0 | 1.00x |

**Read:** the winner is NOT dominant at 3 years. It is the best client-suitable option on identifiability among low-cost designs (8.8 vs 9.4 for +/-80% edge, 14.8 for +/-60% edge), but +/-60% edge-balanced beats it on variance (0.078 vs 0.092), bias (7.7 vs 12.9), cost (1.36 vs 1.69%) and spike (2.19x vs 2.69x), losing only on identifiability (14.8 vs 8.8). +/-80% edge-balanced beats it on variance/bias with similar identifiability but higher cost (2.57%). Blackout wins rigor outright but at 9-15% cost and ~13x single-week spikes. Winner's real case: best identifiability per unit of cost/spike; needs a decision on whether identifiability matters more than bias/variance.
**Caveat:** 4 seeds only; bias sd ~2 at 3yr so gaps under ~4 pts are not significant.
