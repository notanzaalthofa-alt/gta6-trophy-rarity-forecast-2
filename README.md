# GTA VI Trophy Rarity Forecast

Forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, the
percent of the players Exophase tracks who will have earned it at the case's
target, as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy.

Entry for the make.fun bounty
[`gta6-trophy-rarity-forecast-01`](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).
`main.py` (pure Python standard library) reads each case file in
`$MAKEFUN_INPUT_DIR` (default `/input`) and writes one `<id>.out` per
`<id>.in` to `$MAKEFUN_OUTPUT_DIR` (default `/output`).

## The model

A trophy's earn rate is **earners / players-tracked**, and the two move for
different reasons: earners rise as the current players progress through the
game, while players-tracked rises as new players sync their profiles — which
*dilutes* every rate, because a fresh player holds almost no trophies. So the
model fits the two series separately, each as a power law in time since
launch, and divides:

```
log P(t)   = aP + bP * log(1 + t)      players tracked
log E_i(t) = a_i + b_i * log(1 + t)    earners of trophy i
```

`t` is hours since 2026-11-19T00:00Z, fitted by least squares over the case's
history reads, then evaluated at the target. Fitting earners and players
separately (instead of fitting the rate directly) keeps the dilution visible:
when the tracked base jumps at a refresh, the players fit steepens and the
rate falls, even though earners keep growing.

Two safeguards keep the extrapolation honest:

- **Shrinkage.** Rare trophies have few earners, so their log-earner fit is
  noisy. Each trophy's slope is shrunk toward the players slope (earners
  grow at least as fast as the tracked base) in proportion to how many reads
  actually show earners for it.
- **A dilution prior for a stale page.** When the whole history is flat (the
  page has not refreshed since launch night), the fits say "nothing moves",
  but at launch that flatness means a refresh is overdue: the tracked base
  is about to jump and dilute every rate. In that case the median is pulled
  down by the factor a jump of the tracked base would cause, assuming the
  newcomers hold each trophy at 30% of the current rate (and at the full
  rate for a trophy everybody already holds).

## The spread

Quantiles are the median shifted by the normal scores
(-1.645, -0.674, 0, 0.674, 1.645) times a logit spread that combines three
terms in quadrature:

- the fit residual (how well the trophy's earner trend fits its history),
- a horizon term, `0.5 * sqrt(horizon in days)`,
- a refresh term, `0.4 * log(expected jump of the tracked base)`, plus an
  extra 0.4 when the history is flat (a refresh is likely but its size is
  unknown).

The spread is what makes the forecast honest: the pinball loss on the logit
scale charges a lot for a low quantile set too high when the rate drops
below it, and little for a low quantile set too low when the rate stays up,
so the 5% quantile sits well below the median whenever a refresh could
dilute the rate.

## Validation

- The program runs in ~0.06 s per case (limit 8 s) and is deterministic.
- On the kit's practice cases (`p00`, `p01`, whose histories are the flat
  launch-night page), every output is valid: 55 rows, one per trophy, five
  quantiles each in [0, 100] and non-decreasing, 3,164 bytes (limit
  65,536). The dilution prior pulls every median to 0.65x its last rate,
  and the 5% quantile to ~0.2x, so the forecast expects the refresh the
  practice cases straddle.
- Sensitivity check against the persistence baseline (which scores 1.51 on
  the practice): under two plausible refresh profiles (newcomers earning
  each trophy at 30% of the current rate, or only the easy trophies), this
  forecast's pinball loss is 0.39 / 1.00 versus persistence's 0.95 / 1.85.
  It only loses to persistence if no refresh happens at all — which the
  bounty states does happen on launch night.

## License

MIT. See `LICENSE`.
