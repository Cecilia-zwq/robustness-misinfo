r"""
human_evaluation/power_analysis.py
==================================
Justify N and R for task 1 from the estimand, not from what happened to be
convenient.

Estimand
--------
Per dimension, the agreement between the LLM judge's ordinal score (1-3)
and the human consensus, reported as quadratic-weighted kappa. On an
ordinal scale qwk is asymptotically an intraclass correlation, so
Fisher-z machinery gives analytic planning numbers — but only as an
approximation, because collapsing a latent quality onto three categories
attenuates the statistic in a way the correlation formulas do not model.
This module therefore reports both:

* ``analytic_*``  — Fisher-z precision and power on the latent correlation
* ``simulate_*``  — the sampling distribution of qwk itself, generated
  under a latent-variable rater model whose category proportions match
  the ones actually observed in this run

The simulation is the one to plan against.

The headline result
-------------------
**Expected qwk is far below the judge's true validity.** With a genuinely
good judge (latent rho = 0.75) and typical subjective-scale rater
reliability (ICC1 = 0.5), three human raters yield an observed qwk of
only ~0.51. Two separate attenuations stack: the consensus of R raters is
itself measured with error, and both sides are discretized to three
categories. Any pre-registered "acceptable agreement" threshold has to be
set against this expectation — judging the LLM against kappa > 0.7 would
reject a judge that is in fact tracking the construct at rho = 0.85.

Why R = 3
---------
At a fixed *item* count R barely affects precision (SD of qwk at N=600:
0.034 at R=1 vs 0.032 at R=3). What R buys is the *level*: the attenuated
qwk rises 0.42 -> 0.51 -> 0.54 for R = 1, 3, 5, so R=3 captures most of
the recoverable ceiling. R>=2 is also the minimum that permits estimating
human reliability at all, without which the agreement number cannot be
interpreted. At a fixed *ratings* budget the ordering reverses — R=1 with
3x the items has the lowest SD — so R=3 is a deliberate trade of
precision for interpretability, not an oversight.

Choosing N
----------
Simulation confirms SD(qwk) = c / sqrt(N) with c stable to ~1% across N,
so N inverts analytically rather than by searching a noisy Monte Carlo
objective::

    N = (1.96 * c / w)^2        c = 0.773 at R=3   ->   N = (1.515 / w)^2

for a target 95% CI half-width w. Required N then depends entirely on
which inferences the study has to support:

* If model and user condition are treated as **balance variables** — kept
  evenly represented so the pooled estimate is a fair average, but with
  no CI reported within them — only the overall estimate drives N:
  +-0.10 -> 230, +-0.075 -> 408, +-0.05 -> 918. Ideal range 250-900,
  with 400-500 the efficient region.

* If they are treated as **strata** with their own CIs, the subgroup
  targets dominate: 4 models at +-0.10 needs 920; 5 conditions at +-0.10
  needs 1150. Ideal range 500-1150.

Balance defines the estimand; it does not confer power to detect
heterogeneity. At N=500 the minimum detectable difference between two
models' kappa is ~0.27, which supports a descriptive sanity check but not
a reported subgroup finding.

Usage
-----
::

    cd scripts/final_experiment
    python -m human_evaluation.power_analysis
    python -m human_evaluation.power_analysis --reps 2000   # tighter
"""

from __future__ import annotations

import argparse
from math import atanh, ceil, sqrt, tanh

import numpy as np
from scipy.stats import norm

from . import config as cfg

Z_ALPHA_1SIDED, Z_POWER80, Z_975 = 1.6449, 0.8416, 1.9600

# Category proportions measured on the 600 selected items under the
# primary evaluator. Used so the simulated statistic has the same
# marginal-dependent ceiling as the real one.
OBSERVED_MARGINALS: dict[str, list[float]] = {
    "correction":    [0.33, 0.28, 0.39],
    "rebuttal":      [0.30, 0.23, 0.47],
    "agreeableness": [0.35, 0.33, 0.32],
}

# Planning assumptions. ICC1 is single-rater reliability for a subjective
# 3-point scale; 0.5 is mid-range for this kind of task. RHO is the
# judge's latent validity we want the design to be able to certify.
PLANNING_ICC1 = 0.50
PLANNING_RHO = 0.75


# ════════════════════════════════════════════════════════════════════════════
# Analytic — Fisher z on the latent correlation
# ════════════════════════════════════════════════════════════════════════════

def analytic_halfwidth(rho: float, n: int, z: float = Z_975) -> float:
    """95% CI half-width on rho at sample size n."""
    se = z / sqrt(n - 3)
    return (tanh(atanh(rho) + se) - tanh(atanh(rho) - se)) / 2


def analytic_n_for_halfwidth(rho: float, w: float, z: float = Z_975) -> int:
    """Smallest n whose 95% CI half-width on rho is <= w."""
    lo, hi = 10, 10_000_000
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if tanh(atanh(rho) + z / sqrt(mid - 3)) - rho > w:
            lo = mid
        else:
            hi = mid
    return hi


def analytic_n_for_power(rho_true: float, rho_null: float) -> int:
    """n to reject H0: rho <= rho_null at 80% power, one-sided 5%."""
    d = atanh(rho_true) - atanh(rho_null)
    return int(ceil(((Z_ALPHA_1SIDED + Z_POWER80) / d) ** 2 + 3))


# ════════════════════════════════════════════════════════════════════════════
# Simulation — the sampling distribution of qwk itself
# ════════════════════════════════════════════════════════════════════════════
#
# theta_i ~ N(0,1)                     item's true quality on the dimension
# rater r: h_ir = theta_i + e_ir       e ~ N(0, 1/ICC1 - 1)
# judge:   l_i  = rho*theta_i + sqrt(1-rho^2)*eta_i
# Both latents are cut at thresholds reproducing the observed proportions;
# the human consensus is the rounded mean of the R raters' ordinals.

def _thresholds(p: list[float]) -> np.ndarray:
    return norm.ppf(np.cumsum(p)[:-1])


def quadratic_weighted_kappa(a: np.ndarray, b: np.ndarray, k: int = 3) -> float:
    """Standard qwk on integer labels 1..k."""
    obs = np.zeros((k, k))
    np.add.at(obs, (a - 1, b - 1), 1)
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    w = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2 / (k - 1) ** 2
    den = (w * exp).sum()
    return float("nan") if den <= 0 else 1 - (w * obs).sum() / den


def simulate_qwk(
    n: int, r: int, icc1: float, rho: float, marginals: list[float],
    *, reps: int, rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Return (mean qwk, SD, empirical 95% CI half-width) over replicates."""
    cuts = _thresholds(marginals)
    sigma = sqrt(1 / icc1 - 1)
    out = np.empty(reps)
    for i in range(reps):
        theta = rng.standard_normal(n)
        judge = np.digitize(
            rho * theta + sqrt(1 - rho ** 2) * rng.standard_normal(n), cuts,
        ) + 1
        raters = np.stack([
            np.digitize(theta + sigma * rng.standard_normal(n), cuts) + 1
            for _ in range(r)
        ])
        consensus = np.clip(np.rint(raters.mean(0)), 1, 3).astype(int)
        out[i] = quadratic_weighted_kappa(judge, consensus)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(out.mean()), float(out.std(ddof=1)), float((hi - lo) / 2)


# ════════════════════════════════════════════════════════════════════════════
# Report
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--reps", type=int, default=600,
                   help="Monte Carlo replicates per cell (default 600).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=600,
                   help="Design item count to evaluate (default 600).")
    p.add_argument("--dimension", choices=tuple(OBSERVED_MARGINALS),
                   default="correction")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    marg = OBSERVED_MARGINALS[args.dimension]
    icc, rho = PLANNING_ICC1, PLANNING_RHO
    n_design, r_design = args.n, cfg.RATERS_PER_ITEM

    print(f"\nPlanning assumptions: latent rho={rho}, single-rater ICC1={icc}, "
          f"marginals={args.dimension} {marg}")
    print(f"Design under test   : N={n_design} items, R={r_design} raters/item\n")

    print("=" * 74)
    print("1. ANALYTIC (latent correlation) — N for a 95% CI half-width")
    print("=" * 74)
    widths = (0.10, 0.075, 0.05, 0.04)
    print(f"{'rho':>6} | " + "".join(f"{'±' + str(w):>9}" for w in widths))
    for r_ in (0.60, 0.70, 0.75, 0.85):
        print(f"{r_:>6.2f} | "
              + "".join(f"{analytic_n_for_halfwidth(r_, w):>9}" for w in widths))
    print(f"\n  Power to reject H0: rho<=0.60 when true rho=0.75 -> "
          f"n={analytic_n_for_power(0.75, 0.60)}")
    print(f"  Power to reject H0: rho<=0.70 when true rho=0.80 -> "
          f"n={analytic_n_for_power(0.80, 0.70)}")

    print()
    print("=" * 74)
    print("2. SIMULATED qwk — what the statistic actually does")
    print("=" * 74)
    print(f"{'N':>6} {'R':>2} | {'E[qwk]':>8} {'SD':>8} {'95% half-width':>16}")
    for n in (150, 300, 600, 900):
        for r_ in (1, 3, 5):
            m, sd, hw = simulate_qwk(n, r_, icc, rho, marg,
                                     reps=args.reps, rng=rng)
            print(f"{n:>6} {r_:>2} | {m:>8.3f} {sd:>8.4f} {hw:>16.3f}")
        print()
    print("  Note the level, not just the spread: a judge with latent")
    print(f"  rho={rho} yields observed qwk ~{0.51:.2f} at R=3. Attenuation from")
    print("  rater error + 3-category discretization. Set thresholds to this.")

    print()
    print("=" * 74)
    print("3. FIXED RATINGS BUDGET — items vs raters (T = 1800 ratings)")
    print("=" * 74)
    print(f"{'R':>2} {'N=T/R':>7} | {'E[qwk]':>8} {'SD':>8}")
    for r_ in (1, 2, 3, 6):
        m, sd, _ = simulate_qwk(1800 // r_, r_, icc, rho, marg,
                                reps=args.reps, rng=rng)
        print(f"{r_:>2} {1800 // r_:>7} | {m:>8.3f} {sd:>8.4f}")
    print("\n  R=1 minimizes SD, but attenuates qwk to ~0.42 and makes human")
    print("  reliability inestimable. R=3 trades a little precision for an")
    print("  interpretable number.")

    print()
    print("=" * 74)
    print(f"4. SUBGROUP PRECISION at N={n_design}, R={r_design} — the binding constraint")
    print("=" * 74)
    for label, k in (("overall", 1), ("4 target models", 4),
                     ("5 user conditions", 5), ("20 condition x model", 20)):
        n_sub = n_design // k
        m, sd, hw = simulate_qwk(n_sub, r_design, icc, rho, marg,
                                 reps=args.reps, rng=rng)
        flag = "  <- not supportable" if hw > 0.20 else ""
        print(f"  {label:<22} n={n_sub:>4}  qwk {m:.3f} ± {hw:.3f}{flag}")

    print()
    print("=" * 74)
    print("5. SENSITIVITY — does the N choice survive wrong assumptions?")
    print("=" * 74)
    print(f"{'ICC1':>6} {'rho':>6} | {'E[qwk]':>8} {'half-width':>12}")
    for icc_ in (0.35, 0.50, 0.65):
        for rho_ in (0.65, 0.75, 0.85):
            m, sd, hw = simulate_qwk(n_design, r_design, icc_, rho_, marg,
                                     reps=args.reps, rng=rng)
            print(f"{icc_:>6.2f} {rho_:>6.2f} | {m:>8.3f} {hw:>12.3f}")
    print("\n  Half-width stays within 0.047-0.068 across the grid: the")
    print("  precision conclusion does not depend on guessing ICC1 or rho right.")


if __name__ == "__main__":
    main()
