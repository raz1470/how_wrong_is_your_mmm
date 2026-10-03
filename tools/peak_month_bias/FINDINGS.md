# Peak month bias at 15 channels

Peak month makes bias worse than the unphased plan at 15 channels on the
article's original dataset (41.3% mean absolute bias against 35.9%, 200
draws). On three other simulated datasets it improves bias. Two checks on
why, both run by `peak_checks.py` at 200 draws.

## Result

Peak month does not reduce bias at 15 channels. It moves it between
channels. The original dataset is the one where the unphased plan's bias
happened to be low, so the same Peak month result reads as a loss.

| Dataset | Unphased | Peak month |
|---|---|---|
| Original (6, 12, 13) | 35.9 | 41.3 |
| (106, 112, 113) | 63.8 | 43.0 |
| (206, 212, 213) | 56.6 | 51.6 |
| (306, 312, 313) | 50.6 | 41.9 |

Mean absolute bias %, 15 channels (100 draws on the other three datasets).
Peak month lands between 41 and 52 on
every dataset. Unphased ranges from 36 to 64.

## Check 1: the funding cut against demand

The idea was that cutting the other months to fund the peak leaves spend
more aligned with demand. It does not.

- Each channel's correlation with demand falls under Peak month, from
  about 0.65 to about 0.41, and by a similar amount on every channel.
- The change in a channel's bias is unrelated to how high demand is in its
  peak month (correlation -0.03 across 3,000 draw-channel pairs).

## Check 2: month clashes

With 15 channels and 12 months, channels 13 to 15 peak in the same month
as channels 1 to 3 on every draw. Each pair also shares a response curve,
because the scenario cycles four curve types and 12 is a multiple of 4.

- Within a clashing pair bias moves from one channel to the other. Channel
  1 goes from 22% to 110% while Channel 13 goes from 51% to -22%.
- With the clash removed (channels 13 to 15 left unphased), bias on the 12
  peaking channels falls from 34.0% to 23.2%. Channel 1 drops to 43% and
  Channel 5, which never clashed, from 94% to 49%.
- The three unphased channels then absorb it: their bias rises from 43.5%
  to 98.0%. Across all 15 the result is 38.1%, still no better than
  unphased.

So clashes explain which channels are hit, not the total.

## Where the gains on other datasets come from

On the nine channels that never clash, Peak month leaves bias about where
it was on all four datasets (41 to 44, 45 to 47, 40 to 48, 46 to 41). Its
gains on the other three datasets come from the six clashing channels,
whose unphased bias was very high there (up to 336% on one channel) and is
pulled back toward the average.

## What this means

- Peak month is aimed at saturation. At 15 channels it should not be
  expected to help bias. Dark month and Combined are the strategies that
  do.
- No package change follows from this. The pairing of same-curve channels
  in one month is a property of the test scenario, not of real plans.
- Not checked: whether spreading peaks so that no two channels share a
  month (for example over two years) would change the total.
