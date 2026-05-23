# Post Plan

## Positioning

- **Author voice:** ML practitioner with limited finance background, working in the open. Not a quant. Not pitching alpha.
- **Hook:** "I built a fundamentals-only valuation model across 12 global indices to see what I'd find. Here's what the data said — including the parts that contradict textbook value investing."
- **What we're selling:** the *measurement*, the *dashboard*, the *honesty*, and the *open source*. Not a trade.

## Target audiences and venues

| Audience | Venue | Goal | Framing emphasis |
|---|---|---|---|
| Engineering hiring | Hacker News (Show HN), LinkedIn | Portfolio credibility | End-to-end build + live dashboard + methodology rigor |
| ML practitioners | personal blog → HN, /r/MachineLearning | Discussion | What's hard about ML on financial data; conformal intervals; honest backtest |
| Finance-adjacent generalists | Substack / Medium | Reach | The "value is dead" finding, made legible |

One post, written for ML practitioners as the primary reader. The other two audiences inherit naturally from that voice (rigorous but plain, no jargon for jargon's sake).

## Core thesis (one sentence)

> A book-fundamentals-only XGBoost model, applied honestly to 2024–2026 cross-sectional data across 12 global indices, finds that the gap between book-implied fair value and market price is **anti-predictive** of forward returns — consistent with the post-2008 literature on value's decline and the rising importance of intangibles.

## What we can credibly claim

- The decomposition itself is meaningful: every stock's market cap can be split into a book-explainable component and a residual.
- On 2024–2026 data the residual is negatively correlated with forward returns at 10–60 day horizons, across most sectors and indices.
- After residualizing for size, the magnitude roughly halves but the sign holds — so it's not just "small caps lagged."
- The signal direction is consistent with intangibles/growth being the dominant priced factor in this regime.

## What we will not claim

- That this is a tradeable strategy.
- That the result generalizes to other regimes (we only have one).
- That the magnitudes are precise (survivorship + tail truncation + restatement issues bias them).
- That book fundamentals are uninformative — only that they're insufficient.

## Outline (target ~1,000 words for the long version; 250-word HN summary derives from it)

### 1. Hook (1 paragraph)
A single sentence, then a single chart: the IC heatmap (post-fix). Mostly red. Caption: *"Across 11 sectors and 3 horizons, the book-fundamentals residual predicts forward returns — in the opposite direction value investing assumes. This is what one regime says."*

### 2. Who I am, what I built (1 paragraph)
- ML practitioner, no formal finance background.
- Built `mispriced.ch` end-to-end: yfinance ingestion → SQLite → XGBoost with repeated CV → conformal intervals → interactive dashboard across 12 global indices.
- Open source. Methodology page linked. Code linked.

### 3. The model in one paragraph (no math)
- Per quarter, cross-sectional XGBoost regression: log(market cap) on ~10 balance-sheet features.
- No price features. No time series. Each quarter is fit independently.
- Output per stock: a predicted fair market cap and a 90% prediction interval (conformal).
- The residual `(actual − predicted) / actual` is the quantity we then study.

### 4. The finding (2 paragraphs + 1 chart)
- Per-quarter, per-sector Spearman IC of residual vs. forward returns.
- Pooled across 9 quarters, mean IC ≈ [actual number after fixes] at 10–60 day horizons.
- Negative sign: stocks the model calls "cheap on the books" subsequently underperform stocks it calls "expensive."
- Show: heatmap, pooled t-stat, quintile spread chart.

### 5. The "value is dead" reframe — the key explanatory section (2-3 paragraphs)
For ML readers who don't know the finance debate:
- For ~80 years (Graham & Dodd → Fama–French) the dominant academic and practitioner story was: cheap-on-fundamentals stocks outperform expensive ones. Book-to-market was the central factor.
- Since roughly 2008, the value factor has been flat to negative. The dominant explanation: **intangibles**. GAAP expenses R&D, brand-building, software development, and customer acquisition rather than capitalizing them. So a company whose value is mostly intangible (every modern tech company, but also pharma, services, platforms) systematically looks "expensive" on book metrics that don't see its real assets.
- What our model is measuring, in plain ML terms: the residual after fitting on tangible-only fundamentals captures everything the books don't see — intangibles, growth expectations, narrative, sentiment. Its negative IC means that residual carries persistent priced information, and the market reprices toward it rather than away from it.
- Two sentences citing **Lev & Srivastava (2019)** "Explaining the Recent Failure of Value Investing" and the opposing **Arnott, Harvey, Kalesnik, Linnainmaa (2021)** "Reports of Value's Death May Be Greatly Exaggerated." Note that this is contested; we're measurement, not arbitration.

### 6. What I learned about ML on financial data (1-2 paragraphs)
The list of gotchas that will resonate with ML readers:
- **yfinance is restated, not point-in-time.** Pulling "Q1 2024 financials" today gives you whatever's been restated since, not what was filed at the time. There's no version control on fundamentals.
- **Survivorship bias is structural.** Index constituent lists are current. Stocks that delisted, got acquired, or fell out of the index between training quarter and now are simply absent. Most of these are losers; their absence inflates apparent signal in one direction.
- **Earnings lag.** Period-end ≠ release date. Q1 numbers aren't public until late April/May. Using period-end as the signal-formation date is a ~6-week look-ahead.
- **Standard CV gives uncalibrated uncertainty.** Std-across-folds is what most ML projects ship; it's not a probability statement. Conformal prediction (specifically CQR) gives calibrated intervals for free. Show the coverage diagnostic plot — empirical coverage on held-out vs. nominal 90%.
- **Tail truncation is tempting and dangerous.** Dropping `|return| > 200%` cleans the noise but throws out exactly the cases the signal is supposed to predict.

### 7. Caveats (one paragraph, no shrinking from them)
- One regime (2024–2026).
- yfinance survivorship + restatement.
- Single global model not sector-conditioned.
- Conformal coverage holds within a quarter, can drift across quarters.
- Hyperparameters chosen with knowledge of full dataset.

### 8. Close (1 paragraph)
- Live: mispriced.ch
- Code: GitHub link
- Methodology page link
- "Built this to learn; happy to be told what I got wrong."

## Sources to cite

| Topic | Source | Use |
|---|---|---|
| Cross-sectional ML in asset pricing | **Gu, Kelly, Xiu (2020)** "Empirical Asset Pricing via Machine Learning," Review of Financial Studies | Anchor: "this is the textbook approach, scaled to a much larger setting." Establishes credibility for the choice of method. |
| Value's failure & intangibles | **Lev & Srivastava (2019, 2022)** "Explaining the Recent Failure of Value Investing" | The headline interpretation of our finding. |
| Counterargument | **Arnott, Harvey, Kalesnik, Linnainmaa (2021)** "Reports of Value's Death May Be Greatly Exaggerated," Financial Analysts Journal | Show we know the debate isn't settled. |
| Intangibles in returns | **Daniel & Titman (2006)** "Market Reactions to Tangible and Intangible Information" | Earlier evidence that intangible-driven price changes persist. |
| Conformal prediction intervals | **Romano, Patterson, Candès (2019)** "Conformalized Quantile Regression," NeurIPS | The methodology behind our calibrated intervals. |
| Signal evaluation standards | **Grinold & Kahn,** *Active Portfolio Management* (1995/2000) | Reference for IC and t-stat-of-quarterly-ICs. One footnote. |
| Pitfalls in ML for finance | **López de Prado,** *Advances in Financial Machine Learning* (2018) | Footnote on purged k-fold + why we're cross-sectional (avoids most of the time-series traps it warns about). |

Cite these as numbered footnotes, not inline jargon. ML readers may not know who any of them are; one-line gloss each.

## What to *not* put in the post

- Long math. The methodology page has it. The post links.
- Code snippets longer than 5 lines.
- A claim that the dashboard's index-level "X is overpriced by Y%" is a forecast. It's a measurement of fit, not a forecast.
- Disclaimers in caps. One plain sentence ("not investment advice") at the bottom is enough.

## Companion artifacts to land alongside the post

These should be polished before the post drops:

1. **Live dashboard at mispriced.ch.** Title chart on landing should be the corrected IC heatmap with the honest legend.
2. **`/methodology` page.** Updated with the disclosure section from the analysis plan (#3 in `ANALYSIS_CLEANUP.md`).
3. **GitHub README.** Add a "Findings" section near the top with a one-paragraph summary + chart, before the architecture. Link the post.
4. **Conformal coverage diagnostic plot.** One PNG. Lives on the methodology page and in the post.

## Drafting order

1. Finish the analysis cleanup (`ANALYSIS_CLEANUP.md` items 1 + 2.1–2.4).
2. Regenerate dashboard JSON. Eyeball the heatmap; confirm finding direction matches what we want to write.
3. Write section 5 first (the value-is-dead reframe). This is the heart of the post; if it doesn't sing, the rest doesn't matter.
4. Write section 6 (ML-on-finance gotchas). Easiest section, lots of concrete material.
5. Write sections 1–4 around them.
6. Write section 7–8 last.
7. Cut 20%.

## Where to post, in what order

1. **Day of:** publish on personal blog / Substack. Tweet/post with the IC heatmap as the lead image.
2. **+1 day:** Show HN — "Show HN: Mispriced — a fundamentals-only valuation model across 12 global indices." Lead with the dashboard URL, link the writeup. HN audience cares about the build + honesty more than the finance claim.
3. **+2 days:** LinkedIn version — shorter, professional framing, emphasis on "ML practitioner takes on finance, here's the methodology." For hiring funnel.
4. **+1 week:** if HN traction, cross-post to /r/MachineLearning with a "what I learned" framing focused on conformal + the data-quality lessons.

Skip /r/quant, /r/algotrading, /r/wallstreetbets. Wrong audience for this framing.
