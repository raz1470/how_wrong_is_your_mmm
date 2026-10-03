# BudgetPhaser, Blackout, Redistribute and MonthStep

`BudgetPhaser` recommends the weekly spend phasing needed to reduce
marginal-return uncertainty. Each channel takes one of four shapes:

- a continuous weekly range (a float), which keeps every month's total;
- `Blackout`, a hard on/off switch, which also keeps every month's total;
- `Redistribute`, a dark month whose freed budget moves to another month,
  with an optional peak month and a light weekly nudge on top;
- `MonthStep`, every whole month scaled up or down by a fixed step, with
  signs from orthogonal Hadamard columns across channels.

`Redistribute` and `MonthStep` keep each channel's annual total, not its
monthly ones.

::: how_wrong_is_your_mmm.BudgetPhaser

::: how_wrong_is_your_mmm.Blackout

::: how_wrong_is_your_mmm.Redistribute

::: how_wrong_is_your_mmm.MonthStep
