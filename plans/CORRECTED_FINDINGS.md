# Corrected Findings — Analysis Cleanup Regen

Generated: 2026-05-23 (loop 06 of analysis-cleanup).

All "before" numbers are read from
`web/public/backtest.json` as it existed prior to the loop-04/05
regen (snapshotted into `baseline.json` at the start of this loop).
"After" numbers are read from the regenerated
`web/public/backtest.json` produced in Task 3, which now reflects
the 45-day signal-formation lag, 1/99 winsorization, server-side
BH FDR, and per-cohort quarterly IC t-stat.

Source baseline `generated_at`: None.
New `generated_at`: n/a.

## 1. Pooled IC: before vs after (sectors + indices, n_obs-weighted)

| Metric | Horizon | Old (sectors) | New (sectors) | Δ | Old (indices) | New (indices) | Δ |
|---|---|---|---|---|---|---|---|
| raw | 10d | -0.0978 | -0.0462 | +0.0516 | -0.1073 | -0.0562 | +0.0510 |
| raw | 30d | -0.1100 | -0.0778 | +0.0322 | -0.1294 | -0.1147 | +0.0147 |
| raw | 60d | -0.1214 | -0.0811 | +0.0403 | -0.1130 | -0.0806 | +0.0324 |
| raw | 90d | -0.1334 | -0.0610 | +0.0724 | -0.1254 | -0.0374 | +0.0879 |
| residual | 10d | -0.0641 | -0.0373 | +0.0268 | -0.0931 | -0.0460 | +0.0471 |
| residual | 30d | -0.0891 | -0.0750 | +0.0142 | -0.1125 | -0.1010 | +0.0115 |
| residual | 60d | -0.0852 | -0.0657 | +0.0195 | -0.0888 | -0.0736 | +0.0151 |
| residual | 90d | -0.0939 | -0.0300 | +0.0639 | -0.0996 | -0.0301 | +0.0694 |

## 2. Headline sign check

| Metric | Horizon | Old (all) | New (all) | Sign flipped? |
|---|---|---|---|---|
| raw | 10d | -0.1018 | -0.0504 | no |
| raw | 30d | -0.1182 | -0.0933 | no |
| raw | 60d | -0.1179 | -0.0809 | no |
| raw | 90d | -0.1300 | -0.0511 | no |
| residual | 10d | -0.0764 | -0.0409 | no |
| residual | 30d | -0.0990 | -0.0859 | no |
| residual | 60d | -0.0867 | -0.0690 | no |
| residual | 90d | -0.0963 | -0.0301 | no |

Total sign flips across 8 cells: **0**.

## 3. Significant-cell counts per (metric, horizon)

"Old raw" = baseline `pval < 0.05`. "New raw" = post-regen
`pval < 0.05`. "New BH-adj" = post-regen `pval_adj < 0.05`.
Counts are out of the total cells in each cohort (~11 sectors,
~11 indices).

### 3a. Sector heatmap

| Metric | Horizon | Total | Old raw sig | New raw sig | New BH-adj sig |
|---|---|---|---|---|---|
| raw | 10d | 11 | 5 | 0 | 0 |
| raw | 30d | 11 | 4 | 2 | 1 |
| raw | 60d | 11 | 5 | 1 | 0 |
| raw | 90d | 11 | 7 | 1 | 0 |
| residual | 10d | 11 | 3 | 0 | 0 |
| residual | 30d | 11 | 3 | 2 | 1 |
| residual | 60d | 11 | 4 | 1 | 0 |
| residual | 90d | 11 | 4 | 0 | 0 |

### 3b. Index heatmap

| Metric | Horizon | Total | Old raw sig | New raw sig | New BH-adj sig |
|---|---|---|---|---|---|
| raw | 10d | 9 | 1 | 0 | 0 |
| raw | 30d | 9 | 2 | 2 | 1 |
| raw | 60d | 9 | 2 | 1 | 0 |
| raw | 90d | 9 | 3 | 0 | 0 |
| residual | 10d | 9 | 1 | 1 | 0 |
| residual | 30d | 9 | 2 | 1 | 0 |
| residual | 60d | 9 | 2 | 1 | 0 |
| residual | 90d | 9 | 3 | 0 | 0 |

## 4. Hit-rate deltas (loop 02 flipped the inequality)

Loop 02 changed `compute_hit_rate` from counting momentum hits
to counting directional agreement. If lag/winsorization changed
the underlying cohort negligibly, `new_hit ≈ 1 - old_hit` should
hold within a few percentage points. Deviations beyond that are
attributable to the lag + winsorization changing which rows enter
the cohort.

| Metric | Horizon | Old sector hit | New sector hit | 1−old | Residual (new − (1−old)) |
|---|---|---|---|---|---|
| raw | 10d | +0.522 | +0.485 | +0.478 | +0.007 |
| raw | 30d | +0.532 | +0.476 | +0.468 | +0.008 |
| raw | 60d | +0.530 | +0.473 | +0.470 | +0.003 |
| raw | 90d | +0.537 | +0.476 | +0.463 | +0.013 |
| residual | 10d | +0.508 | +0.485 | +0.492 | -0.006 |
| residual | 30d | +0.518 | +0.470 | +0.482 | -0.013 |
| residual | 60d | +0.516 | +0.480 | +0.484 | -0.005 |
| residual | 90d | +0.526 | +0.488 | +0.474 | +0.014 |

## 5. Cells crossing |t| ≥ 2 (new field, no baseline equivalent)

The Grinold-Kahn t-stat of per-quarter ICs is a new column from
loop 05. There is no "before" comparison; this section just lists
the cells that clear the conventional |t| ≥ 2 bar.

| Scope | Name | Metric | Horizon | IC | t-stat | n_q | pval | pval_adj |
|---|---|---|---|---|---|---|---|---|
| index | SP600 | raw | 30d | -0.1264 | -8.05 | 8 | +0.0141 | +0.0636 |
| index | SP600 | residual | 30d | -0.1103 | -7.68 | 8 | +0.0538 | +0.1899 |
| index | SP400 | raw | 30d | -0.1466 | -4.72 | 8 | +0.0512 | +0.1152 |
| index | SP400 | residual | 30d | -0.1329 | -4.12 | 8 | +0.0670 | +0.1899 |
| sector | Technology | raw | 10d | -0.0788 | -4.07 | 8 | +0.2584 | +0.4802 |
| sector | Technology | raw | 30d | -0.1530 | -3.85 | 8 | +0.0015 | +0.0170 |
| sector | Basic Materials | raw | 30d | -0.1207 | -3.51 | 8 | +0.2602 | +0.3578 |
| sector | Consumer Cyclical | raw | 30d | -0.1104 | -3.34 | 8 | +0.0728 | +0.2668 |
| sector | Consumer Defensive | residual | 60d | -0.1143 | -3.15 | 8 | +0.3369 | +0.3896 |
| sector | Basic Materials | raw | 10d | -0.1389 | -3.13 | 8 | +0.1350 | +0.4802 |
| sector | Industrials | raw | 30d | -0.1089 | -3.07 | 8 | +0.0495 | +0.2668 |
| index | SP500 | raw | 30d | -0.1268 | -2.95 | 8 | +0.0510 | +0.1152 |
| sector | Industrials | residual | 30d | -0.1159 | -2.91 | 8 | +0.0640 | +0.1947 |
| index | SP600 | raw | 10d | -0.0863 | -2.88 | 8 | +0.0883 | +0.2548 |
| sector | Energy | raw | 60d | -0.0715 | -2.88 | 8 | +0.3079 | +0.4839 |
| sector | Technology | residual | 30d | -0.1522 | -2.83 | 8 | +0.0006 | +0.0064 |
| index | SP500 | residual | 60d | -0.0902 | -2.77 | 8 | +0.4861 | +0.5468 |
| sector | Consumer Defensive | raw | 30d | -0.1084 | -2.74 | 8 | +0.3171 | +0.3875 |
| index | SP500 | residual | 30d | -0.1147 | -2.74 | 8 | +0.1512 | +0.2721 |
| index | Russell1000 | raw | 30d | -0.1145 | -2.62 | 8 | +0.0043 | +0.0390 |
| sector | Consumer Defensive | residual | 90d | -0.0793 | -2.54 | 8 | +0.6150 | +0.6150 |
| index | SP500 | raw | 60d | -0.0936 | -2.52 | 8 | +0.5072 | +0.5706 |
| sector | Energy | raw | 90d | -0.0968 | -2.49 | 8 | +0.2392 | +0.3605 |
| sector | Basic Materials | residual | 30d | -0.1219 | -2.48 | 8 | +0.0885 | +0.1947 |
| sector | Basic Materials | residual | 10d | -0.1286 | -2.46 | 8 | +0.1349 | +0.2969 |

_(truncated; 15 more cells with |t| ≥ 2 omitted.)_

Total cells with |t| ≥ 2: **40** (out of 160 with a finite t-stat).

## 6. What the post should now claim

Direct, post-ready paragraph for the implementer to paste / adapt
into `plans/POST.md`. Numbers come from the regenerated
`backtest.json`; do not paraphrase from memory.

> After fixing the IC sign-display bug, adding a 45-day
> signal-formation lag, winsorizing returns at 1/99, and applying
> Benjamini-Hochberg FDR per (metric, horizon) cohort, the pooled
> Spearman IC of the book-fundamentals residual vs.
> forward returns is **-0.0809** (raw signal,
> 60-day horizon, n-weighted across all sector and index cells).
> The size-residualized signal pools to **-0.0690** at
> the same horizon. At 30 days the raw pooled IC is
> **-0.0933** (residual: **-0.0859**); at 10
> days raw is **-0.0504** (residual: **-0.0409**).

> After BH adjustment across each (metric, horizon) cohort, the
> dashboard reports significance stars on a substantially smaller
> set of cells than the raw p-values would suggest — see Section 3
> for exact counts. The headline directional finding ("value
> residual is anti-predictive of forward returns") DOES still hold
> in the pooled-IC sign sense for the raw signal at 60 days under
> the corrected methodology.

> Sign flips across (metric, horizon) cells: **0** out
> of 8. See Section 2 for the cell-by-cell view.

### Action items for the post draft

1. Replace any pre-correction IC magnitude quoted in the post
   draft with the corresponding cell from Section 1.
2. If a sign flip is listed in Section 2 for the headline
   `(raw, 60d)` cell, the entire "value is dead" framing needs a
   reread before posting.
3. Switch any sentence that currently quotes "hit rate of XX%"
   to the corresponding `New sector hit` value from Section 4,
   and reword it as "fraction of stocks where signal direction
   matched return direction" to match the loop-02 definition.
4. If Section 5 lists fewer than ~5 cells with |t| ≥ 2, the
   "Grinold-Kahn t-stat" sentence in the methodology section
   should explicitly call out that only a handful of cohorts
   clear the conventional 2-sigma bar — do not oversell.

