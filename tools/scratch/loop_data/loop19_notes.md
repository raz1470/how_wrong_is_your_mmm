
### Loop 19 — final robustness pass + new default (2026-09-19)

`min_recipient_weeks` default changed 2 -> 4 in `design.py` (loop 15). Final pass: `roundrobin_plus_edge_layer` (t=4, edge_cap=15%, mrw=4), 8 seeds, 52wk warm-up history, phased every year, canonical calendar. Script `loop19_final.py`, raw `loop_data/loop19_final_n{5,10}_raw.csv`.

| channels | years | lever | variance | bias % (sd, max) | identifiability | cost %/yr | spike mean (max) |
|---|---|---|---|---|---|---|---|
| 5 | 1 | Unphased | 0.396 | 66.4 (11.4, 79.0) | 43.7 | 0 | 1.0x |
| 5 | 1 | winner | 0.155 | 11.8 (4.7, 20.0) | 13.0 | 1.73 | 2.46x (2.68) |
| 5 | 3 | Unphased | 0.271 | 19.6 (7.7, 32.7) | 45.1 | 0 | 1.0x |
| 5 | 3 | winner | 0.086 | 5.3 (1.9, 8.2) | 9.7 | 1.70 | 2.56x (2.68) |
| 10 | 1 | Unphased | 0.432 | 102.9 (13.7, 125.5) | 41.3 | 0 | 1.0x |
| 10 | 1 | winner | 0.181 | 26.9 (9.7, 44.2) | 13.0 | 1.77 | 2.50x (2.79) |
| 10 | 3 | Unphased | 0.252 | 50.9 (6.6, 64.3) | 43.4 | 0 | 1.0x |
| 10 | 3 | winner | 0.090 | 12.1 (1.9, 14.7) | 8.8 | 1.70 | 2.63x (2.79) |

**Verdict: winner is robust.** Beats Unphased on every metric in all 4 settings across 8 seeds; worst-case spike 2.79x (well under Blackout's 6.5x); annual cost flat at ~1.7%. Bias is the noisy metric at 1 year (10ch sd 9.7, max 44%) and tightens sharply at 3 years (sd 1.9). Bias falls with more years and fewer channels (5ch/3yr: 5.3%).
Caveat: bias is unphased-relative, so absolute values depend on the scenario's demand process; 1-year 10-channel bias (~27%) is the weakest cell.
**Open / not done:** promote design into `_phaser.py` (nothing wired in yet); `edge_cap_pct` should ship as a configurable knob (loop 11).
