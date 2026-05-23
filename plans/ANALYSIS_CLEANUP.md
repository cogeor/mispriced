# Analysis Cleanup Plan

Goal: make the methodology defensible to a quant/ML reader without restructuring the dashboard. Constraints:
- Dashboard UI stays as-is unless a glaring issue forces a change.
- Error bars (prediction intervals) are open to change.
- Optimize for "ML practitioner trying to be honest on noisy finance data," not for "alpha strategy."

Three buckets: **bug fixes** (must do, ship-blocker), **methodology improvements** (should do before posting), **disclosure updates** (do alongside the post).

---

## 1. Bug fixes (must do)

### 1.1 IC sign convention — the only "glaring issue" that forces a dashboard touch

**Problem.** End-to-end trace:
- `scripts/run_sector_backtest.py:85` — `signal_raw = (predicted - actual) / actual`, so `signal_raw > 0` means **undervalued**.
- `src/backtest/signal_metrics.py:94` — IC = raw Spearman(signal, returns). Under this convention, **positive IC = value works**.
- `web/src/charts/signalQuality.ts:91, 137, 182` — dashboard displays `-IC` with the comment *"Negate IC so positive = good signal (overvalued stocks go down)"*. That comment is consistent with `signal > 0 = overpriced`, not the actual convention.

**Net effect:** the heatmap displays the negative of the measured IC. Mean raw IC ≈ −0.10 → dashboard shows ≈ +0.10 → green cells labeled "good signal" → reader concludes "value works." Actual finding is the opposite. This misrepresents the headline result regardless of which framing (value strategy vs measurement) we settle on.

**Fix (minimal).** Drop the negation in `signalQuality.ts`:
- Line 92: `rowZ.push(item.ic);`  (was `-item.ic`)
- Line 95: `IC: ${(item.ic * 100).toFixed(1)}%`  (drop the minus)
- Line 137: `(item.ic * 100).toFixed(1)`  (drop the minus)
- Line 182: `y = sorted.map(d => d.avg_ic);`  (drop the minus)
- Update the comment to: `// IC > 0: residual predicts mean reversion. IC < 0: residual predicts continuation/momentum.`
- Update tooltip text in `chartMeta.ts` / `methodology.html` to match.

**Color scale.** Keep diverging. Don't relabel either pole as "good" — both signs are meaningful findings. If we want to nudge interpretation, a one-line subtitle on the chart: *"Positive IC = books-based value predicted reversion. Negative IC = market rewarded the non-book premium."*

**Effort:** 30 min. Touches 4 lines of TS, 1 paragraph of methodology HTML.

### 1.2 `compute_hit_rate` inversion

**Problem.** `src/backtest/signal_metrics.py:192`:
```python
hits = ((signal > 0) & (returns < 0)) | ((signal < 0) & (returns > 0))
```
Docstring says this counts "overpriced AND fell OR underpriced AND rose" (value-investor hits). But with `signal_raw > 0 = undervalued`, this actually counts **momentum hits**. So the reported 53% is `P(momentum direction)`, and the value hit rate is `1 − 0.53 = 0.47`.

**Fix.** Two options:
- **A.** Flip the inequality in `compute_hit_rate`: `((signal > 0) & (returns > 0)) | ((signal < 0) & (returns < 0))`. Update docstring. This makes "hit" = "signal direction matched return direction." Symmetric and convention-free.
- **B.** Leave the function, rename it `compute_momentum_hit_rate`, and document explicitly.

Pick **A**. It's neutral and matches the IC interpretation we land on after fix 1.1.

**Effort:** 15 min. One function + its docstring + one tooltip on the dashboard.

### 1.3 Benjamini–Hochberg correction (methodology claims it, code doesn't apply it)

**Problem.** `methodology.html` says *"P-values are corrected using the Benjamini-Hochberg procedure."* `signal_metrics.py` and `run_sector_backtest.py` don't apply BH anywhere. The dashboard significance stars are raw p-values.

**Fix.** Apply BH at the heatmap level (per metric × per horizon, across the ~33 sector/index cells) when generating the dashboard JSON. `scipy.stats.false_discovery_control` is one line; or `statsmodels.stats.multitest.multipletests(pvals, method='fdr_bh')`.

Add a `pval_adj` field next to `pval` in the backtest JSON; have `signalQuality.ts` read `pval_adj` for the star annotation.

**Effort:** 1 hour. Backend-only change to `scripts/generate_dashboard.py` + one field rename in TS.

---

## 2. Methodology improvements (should do before posting)

### 2.1 Push signal-formation date forward by ~45 days

**Problem.** `snapshot_timestamp` is fiscal period end (e.g., 2024-03-31). Q1 2024 financials aren't actually released until ~late April to early May. Using period-end as the signal-formation date gives 4–8 weeks of look-ahead on forward returns.

**Fix.** In `run_sector_backtest.py:130` and `src/backtest/service.py:197`, compute returns starting at `snapshot_date + timedelta(days=45)` rather than `snapshot_date`. Document the choice (45 days is conservative; the SEC requires 10-Q within 40 days for large accelerated filers, 45 for accelerated).

A cleaner version: store a per-snapshot `release_date` from yfinance's earnings calendar when available, fall back to `+45d`. The `release_date` column already exists in the DB schema per `.arch/src/README.md`. Wire it through.

**Effort:** 1–2 hours. Backend-only. Will reduce the magnitude of negative IC (some of it was earnings-lag look-ahead); the directional finding should hold.

### 2.2 Replace `abs(ret) < 2.0` with percentile winsorization

**Problem.** `src/backtest/signal_metrics.py` (via `run_sector_backtest.py:131`) drops any stock with `|return| > 2.0` (i.e., doubled or halved over the horizon). The tails are exactly where signal lives. Truncating them biases IC toward zero and removes the cases the post will want to talk about (NVDA tripling, biotech wipeouts).

**Fix.** Replace with per-quarter 1st/99th percentile winsorization on returns (clip, don't drop). Apply the same winsorization to the signal.

**Effort:** 30 min. One filter swap + a quick re-run of the backtest pipeline.

### 2.3 Cluster-robust / pooled-by-quarter standard errors

**Problem.** Per-sector-per-quarter IC SEs assume independent observations. They aren't: same stocks recur across quarters, within-sector residuals are correlated through shared factor exposures. Reported p-values are too small by ~2×.

**Fix (lightweight).** Instead of computing one IC per sector-quarter and showing each cell's p-value, report **the t-statistic of per-quarter ICs across the 9 quarters** for each sector-horizon cell:
```
t = mean(IC_q) / (std(IC_q) / sqrt(N_quarters))
```
This is the standard quant-finance way to summarize a signal across periods (Grinold–Kahn). It naturally handles serial correlation at the quarter level. Newey–West if you want to be fancy, but with N=9 quarters it's not worth it.

This *replaces* the per-cell p-value with a single "is this signal real across time" t-stat per sector-horizon. The dashboard already has the per-quarter time series in scatter form, so this aggregate stat slots in as one column in the summary table.

**Effort:** 2–3 hours. Backend change in `run_sector_backtest.py` aggregation + a small addition to the summary table on the dashboard (one column, won't restructure anything).

### 2.4 Conformalized Quantile Regression for prediction intervals

**Problem.** `predicted_mcap_std` (std across 50 OOF predictions) is uncalibrated. It conflates fold variance with prediction error, understates uncertainty, can't extrapolate to new tickers, and produces constant-width intervals in raw mcap space that over-cover large caps and under-cover small caps.

**Fix.** Wrap the XGBoost model in CQR using `mapie` (sklearn-compatible). Per quarter:
1. Split the cross-section into train (60%) / calibration (20%) / test (20%).
2. Train two quantile regressors with `objective='reg:quantileerror'` at α=0.05 and α=0.95 in **log-mcap space**.
3. Compute nonconformity scores on calibration: `s_i = max(ŷ_lo - y_i, y_i - ŷ_hi)`.
4. `q = (1-α)-quantile of {s_i}`.
5. Per-stock interval: `[ŷ_lo − q, ŷ_hi + q]` in log space → `[exp(...), exp(...)]` in mcap.

Per-stock interval widths now scale with the model's confidence per stock. In log-mcap space the intervals naturally become multiplicative bands in raw mcap (small caps get smaller absolute bands, large caps larger), which addresses the "constant width over-represents low-mcap error" issue we discussed.

Coverage diagnostic: per quarter, compute the fraction of held-out actual mcaps falling inside the 90% interval. Add a small plot in the methodology section: empirical coverage vs target, with a 45° reference line. **This single plot is the strongest credibility signal in the post.**

**Effort:** 1 day. Touches `src/valuation/repeated_cv.py` (add CQR alternative), `src/valuation/service.py` (write `predicted_mcap_lo`, `predicted_mcap_hi` instead of `_std`), DB migration to add the two columns, dashboard tooltip update to show the interval rather than `±σ`.

**Caveat to document.** Conformal coverage holds under exchangeability. The cross-section within a quarter is approximately exchangeable; across quarters it isn't (regime shift). Coverage on the next quarter may drift. Mention this in the methodology page and don't oversell.

---

## 3. Disclosure updates (do alongside the post)

Add a "Known limitations" subsection to `methodology.html` covering:

1. **Survivorship bias.** yfinance gives current index constituents; delisted/acquired/dropped tickers absent. Inflates apparent anti-value effect (value traps and bought-out deep-value names are missing). No fix without paid data.
2. **Point-in-time accuracy.** yfinance returns currently-reported fundamentals, including any restatements. Snapshots from 2024 reflect today's restated numbers, not what was filed in 2024. Affects training features.
3. **Single regime.** All data is 2024–2026. Mega-cap tech / AI capex / small-cap underperformance. Findings may not generalize to other regimes.
4. **Sector confound in per-sector ICs.** Single global model means tech systematically scores as overvalued, utilities as undervalued. Per-sector ICs are computed on residuals from this global model, so within-sector signal is partly inherited from the global sector-premium. Future work: per-sector model or sector dummies.
5. **Hyperparameter selection on full dataset.** Fixed XGBoost params chosen with knowledge of all quarters. Conservative params (depth 5, 200 estimators) limit the damage but it's not strict OOS.

These belong in the methodology page first, then summarized in one paragraph in the post.

---

## 4. Out of scope (mention in plan, don't do now)

- **Sharadar SF1 ($50/mo) migration.** Biggest single-step credibility upgrade — gives PIT fundamentals + delisted tickers + 20 years history. Lets you run the same backtest across multiple regimes. Worth doing eventually; out of scope for this pass.
- **Per-sector models.** Removes the sector confound from per-sector IC. Real work; defer.
- **Adaptive conformal under distribution shift** (Gibbs & Candès 2021). Fixes the regime-shift caveat on conformal coverage. Defer.
- **Purged / embargoed K-fold** (López de Prado). Mostly relevant for time-series leakage; we already avoid that by training cross-sectionally per quarter. Mention in methodology, don't implement.

---

## Execution order (rough)

Half-day session 1: fixes 1.1, 1.2, 1.3 (all small, all bug-fixes). Re-run dashboard JSON generation. Sanity-check that the heatmap now shows what we expect (mostly red, with the per-sector pattern matching what we already saw in `backtest.json`).

Half-day session 2: 2.1 + 2.2 (release-date lag + tail winsorization). Re-run full backtest pipeline. Check whether the directional finding survives (it should; magnitude will shrink).

Day 3: 2.3 (quarterly t-stat aggregation). Adds one column to the summary table.

Day 4–5: 2.4 (CQR). Biggest single methodology upgrade and the centerpiece visual for the post.

Day 6: 3 (disclosure updates).

Then: write the post.
