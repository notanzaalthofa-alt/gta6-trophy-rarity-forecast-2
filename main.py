#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase at a case's target.

For every trophy of Grand Theft Auto VI on PlayStation 5, forecast the percent
of the players Exophase tracks who will have earned it at the case's target,
as five quantiles (5%, 25%, 50%, 75%, 95%).

Method (cohort decomposition in log time)
-----------------------------------------
A trophy's earn rate is earners / players-tracked, and the two move for
different reasons: earners rise as the current players progress through the
game, while players-tracked rises as new players sync their profiles (which
dilutes every rate, because a fresh player holds almost no trophies). So the
model fits the two series separately, each as a power law in time since
launch, and divides:

    log P(t)   = aP + bP * log(1 + t)      players tracked
    log E_i(t) = a_i + b_i * log(1 + t)    earners of trophy i

with t in hours since 2026-11-19T00:00Z, fitted by least squares over the
case's history reads, then evaluated at the target and divided to give the
median rate. Fitting earners and players separately (instead of fitting the
rate directly) keeps the dilution visible: when the tracked base jumps at a
refresh, the players fit steepens and the rate falls, even though earners
keep growing.

Two safeguards keep the extrapolation honest:

* Shrinkage. Rare trophies have few earners, so their log-earner fit is
  noisy. Each trophy's slope is shrunk toward the players slope (earners
  grow at least as fast as the tracked base) in proportion to how many
  reads actually show earners for it.
* A dilution prior for a stale page. When the whole history is flat (the
  page has not refreshed since launch night), the fits say "nothing moves",
  but at launch that flatness means a refresh is overdue: the tracked base
  is about to jump and dilute every rate. In that case the median is pulled
  down by the factor a jump of the tracked base would cause, assuming the
  newcomers hold each trophy at a fraction of the current rate (and at the
  full rate for a trophy everybody already holds).

The spread (logit scale) combines the fit residual, a horizon term that
grows with the square root of the horizon in days, and a refresh term that
grows with how much the tracked base is expected to jump. Quantiles are the
median shifted by the normal scores (-1.645, -0.674, 0, 0.674, 1.645) times
that spread, which keeps them ordered and honest about the uncertainty.

Pure standard library. Deterministic: the forecast depends only on the case
file, never on the wall clock or any randomness. Reads $MAKEFUN_INPUT_DIR
(default /input) and writes $MAKEFUN_OUTPUT_DIR (default /output), one
<id>.out per <id>.in found.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
Z_SCORES = (-1.645, -0.674, 0.0, 0.674, 1.645)
LO, HI = 0.01, 99.99  # rates are clamped here before any logit


def clamp_rate(p):
    """Clamp a percent into the range the logit is defined on."""
    return min(HI, max(LO, p))


def logit(p):
    p = clamp_rate(p) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x):
    if x >= 0:
        e = math.exp(-x)
        p = 1.0 / (1.0 + e)
    else:
        e = math.exp(x)
        p = e / (1.0 + e)
    return clamp_rate(p * 100.0)


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_launch(moment):
    return (moment - LAUNCH).total_seconds() / 3600.0


def least_squares(xs, ys):
    """Least-squares line y = a + b*x. Returns (a, b, residual_std)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(
            sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        )
    else:
        resid = 0.0
    return a, b, resid


def build_series(case):
    """Turn a case's history reads into per-trophy time series.

    Returns (times, players, earners) where times are hours since launch,
    players is the tracked-base series, and earners[i] is trophy i's earner
    count series (None where that read lacked the rate).
    """
    times, players, earners = [], [], []
    trophies = case["trophies"]
    for read in case.get("history", []):
        try:
            moment = parse_time(read.get("at") or read["slot"])
        except (KeyError, ValueError):
            continue
        p = read.get("players_tracked")
        rates = read.get("rates")
        if p is None or rates is None or len(rates) != len(trophies):
            continue
        times.append(hours_since_launch(moment))
        players.append(float(p))
        earners.append(
            [
                (r * p / 100.0) if r is not None else None
                for r in rates
            ]
        )
    return times, players, earners


def fit_players(times, players):
    """Fit log players-tracked on log time. Returns (a, b, last, flat)."""
    if not times:
        return 0.0, 0.0, 0.0, True
    xs = [math.log1p(max(0.0, t)) for t in times]
    ys = [math.log(max(1.0, p)) for p in players]
    a, b, _ = least_squares(xs, ys)
    flat = len(set(players)) <= 1
    return a, b, players[-1], flat


def fit_earners(times, earners_col, player_slope):
    """Fit log earners of one trophy on log time, shrunk toward the players
    slope. Returns (a, b, n_points, residual)."""
    xs, ys = [], []
    for t, e in zip(times, earners_col):
        if e is not None and e >= 1.0:
            xs.append(math.log1p(max(0.0, t)))
            ys.append(math.log(e))
    n = len(xs)
    if n >= 3:
        a, b, resid = least_squares(xs, ys)
    elif n >= 1:
        # Too few points for a slope: earners track the players base.
        a = ys[-1] - player_slope * xs[-1]
        b = player_slope
        resid = 0.0
    else:
        a, b, resid = 0.0, player_slope, 0.0
    # Shrink the slope toward the players slope; the fewer reads with
    # earners, the more shrinkage (rare trophies have noisy fits).
    w = n / (n + 4.0)
    b = w * b + (1.0 - w) * player_slope
    return a, b, n, resid


def dilution_median(last_rate, jump, newcomer_share=0.3):
    """Median rate after the tracked base jumps by `jump` (a factor).

    The newcomers hold each trophy at `newcomer_share` of the current rate
    (a trophy everybody already holds stays held by everybody), so the new
    rate is the earner-weighted mix of the old cohort and the newcomers.
    """
    if last_rate >= 99.5:
        return 100.0
    newcomer_rate = newcomer_share * last_rate
    return (last_rate + newcomer_rate * (jump - 1.0)) / jump


def forecast_trophy(last_rate, earners_last, a_e, b_e, resid, n_points,
                    a_p, b_p, players_last, target_x, horizon_days,
                    history_flat):
    """One trophy's five quantiles at the target, in percent."""
    # Players at the target from the players fit (never shrinking, capped
    # against a runaway extrapolation).
    log_p_target = a_p + b_p * target_x
    players_target = math.exp(log_p_target)
    players_target = max(players_last, min(players_target, players_last * 100.0))

    # Earners at the target from the trophy's fit (never above the players,
    # never below zero).
    log_e_target = a_e + b_e * target_x
    earners_target = math.exp(log_e_target)
    earners_target = max(0.0, min(earners_target, players_target))
    # Do not let the fit invent earners for a trophy nobody holds yet.
    if earners_last is not None and earners_last < 1.0:
        earners_target = min(earners_target, max(1.0, earners_last * 4.0))

    median = 100.0 * earners_target / players_target if players_target else last_rate
    median = clamp_rate(median)

    jump = players_target / players_last if players_last else 1.0

    # A flat history at launch means the page is stale and a refresh is
    # overdue: pull the median down by the dilution a jump would cause.
    if history_flat:
        flat_jump = max(2.0, jump)
        median = clamp_rate(dilution_median(last_rate, flat_jump))
        jump = max(jump, flat_jump)

    # Spread (logit std): fit residual, a horizon term, and a refresh term
    # that grows with the expected jump of the tracked base.
    spread_fit = min(1.0, max(0.10, resid))
    spread_horizon = 0.5 * math.sqrt(max(0.0, horizon_days))
    spread_refresh = 0.4 * math.log(max(1.0, jump))
    if history_flat:
        spread_refresh += 0.4
    spread = math.sqrt(
        spread_fit ** 2 + spread_horizon ** 2 + spread_refresh ** 2
    )

    centre = logit(median)
    quantiles = []
    for z in Z_SCORES:
        q = inv_logit(centre + z * spread)
        quantiles.append(round(q, 4))
    # Rounding can only tie neighbours, never reorder them.
    for k in range(1, 5):
        if quantiles[k] < quantiles[k - 1]:
            quantiles[k] = quantiles[k - 1]
    return quantiles


def forecast_case(case):
    """Forecast every trophy of one case. Returns the output object."""
    trophies = case["trophies"]
    times, players, earners = build_series(case)
    horizon_days = float(case.get("horizon_minutes", 0)) / 1440.0
    target = parse_time(case["target"])
    target_x = math.log1p(max(0.0, hours_since_launch(target)))

    a_p, b_p, players_last, players_flat = fit_players(times, players)

    # The history is stale when nothing moved at all (players and every
    # rate constant across every read).
    history_flat = players_flat and all(
        len({e[i] for e in earners if e[i] is not None}) <= 1
        for i in range(len(trophies))
    ) if earners else True

    # Last rate per trophy (most recent non-null read).
    last_rates = []
    for i in range(len(trophies)):
        rate = None
        for j in range(len(earners) - 1, -1, -1):
            e = earners[j][i]
            if e is not None:
                rate = 100.0 * e / players[j]
                break
        if rate is None:
            rate = 0.5
        last_rates.append(rate)
    # Last earner count per trophy, same order.
    last_earners = []
    for i in range(len(trophies)):
        e = None
        for j in range(len(earners) - 1, -1, -1):
            if earners[j][i] is not None:
                e = earners[j][i]
                break
        last_earners.append(e if e is not None else 0.0)

    rows = []
    for i, trophy in enumerate(trophies):
        col = [row[i] for row in earners]
        a_e, b_e, n_points, resid = fit_earners(times, col, b_p)
        q = forecast_trophy(
            last_rates[i], last_earners[i], a_e, b_e, resid, n_points,
            a_p, b_p, players_last, target_x, horizon_days, history_flat,
        )
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main(argv=None):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(
        name for name in os.listdir(in_dir) if name.endswith(".in")
    )
    if not names:
        print("no .in files in %s" % in_dir, file=sys.stderr)
        return 1
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_path = os.path.join(out_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
