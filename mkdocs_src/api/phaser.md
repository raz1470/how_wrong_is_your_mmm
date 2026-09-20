# BudgetPhaser, Blackout, Redistribute and MonthStep

`BudgetPhaser` recommends the weekly spend phasing needed to reduce
marginal-return uncertainty, while preserving monthly totals exactly.
`Blackout` is one of the deviation shapes it accepts per channel — a hard
on/off switch instead of a continuous weekly range. `Redistribute` is the
third: a round-robin blackout whose freed budget moves to a recipient
month, with an edge layer on top. It preserves annual totals exactly, but
not monthly ones. `MonthStep` is the fourth: every whole month's spend is
scaled up or down by a fixed step, with signs from orthogonal Hadamard
columns across channels; it also preserves annual totals, not monthly ones.

::: how_wrong_is_your_mmm.BudgetPhaser

::: how_wrong_is_your_mmm.Blackout

::: how_wrong_is_your_mmm.Redistribute

::: how_wrong_is_your_mmm.MonthStep
