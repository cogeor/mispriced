/**
 * Metadata content for chart info tooltips.
 *
 * Each entry is keyed by the Plotly container element ID and provides
 * Data, Axes, and Insights sections shown on hover/tap of the "?" icon.
 */

import { IC_COPY } from './config';

export interface ChartMeta {
    data: string;
    axes: string;
    insights: string;
}

export const CHART_META: Record<string, ChartMeta> = {
    valuationChart: {
        data: 'Each dot is one stock. Predicted market cap comes from an XGBoost model trained on ~30 fundamental features using 10x5 repeated cross-validation. Only stocks with market cap > $100M and at least one financial metric are included.',
        axes: 'X-axis: model-predicted market cap (log scale). Y-axis: actual market cap (log scale). Dots on the diagonal are fairly valued; above = underpriced, below = overpriced.',
        insights: 'Tight clustering around the diagonal indicates good model fit. Outliers far from the line represent the strongest mispricing signals. Color encodes mispricing magnitude.',
    },

    sectorChart: {
        data: 'Average mispricing per GICS sector for the selected quarter, computed as (predicted - actual) / actual. Error bars show one standard deviation across stocks in each sector.',
        axes: 'X-axis: sector name. Y-axis: average mispricing (%). Positive (green) = sector is overpriced on average; negative (red) = underpriced.',
        insights: 'Sectors with small error bars have more consistent signals. Large bars suggest high within-sector dispersion. Compare across quarters to spot persistent sector tilts.',
    },

    indexChart: {
        data: 'Market-cap-weighted average mispricing per index, including a "Global" aggregate. Uncertainty is propagated as sigma_index = sqrt(sum(w_i^2 * sigma_i^2)).',
        axes: 'X-axis: index name. Y-axis: weighted average mispricing (%). Error bars show propagated uncertainty. Green = overpriced; red = underpriced.',
        insights: 'Compare indices to spot relative valuation across markets. The "Global" bar gives an overall market sentiment. Small error bars indicate higher conviction.',
    },

    sectorTimeSeriesChart: {
        data: 'Sector-level average mispricing over all available quarters. Each line is one sector. Click the legend to toggle sectors on/off; double-click to isolate one.',
        axes: 'X-axis: quarter date. Y-axis: average sector mispricing (%). Each line is a sector colored by its legend entry.',
        insights: 'Identify sectors with persistent over/underpricing trends. Mean-reverting sectors may offer cyclical opportunities. Diverging sectors suggest structural shifts.',
    },

    indexTimeSeriesChart: {
        data: 'Index-level weighted average mispricing over all available quarters. Each line is one index. Click legend to toggle; double-click to isolate.',
        axes: 'X-axis: quarter date. Y-axis: weighted average index mispricing (%). Each line is an index colored by its legend entry.',
        insights: 'Track how regional markets move relative to each other over time. Convergence suggests global sentiment alignment; divergence indicates local factors.',
    },

    icSectorChart: {
        data: 'Information Coefficient (Spearman rank correlation) between mispricing signal and forward returns, grouped by sector and horizon. Stars indicate statistical significance after Benjamini-Hochberg correction.',
        axes: 'Rows: sectors. Columns: forward return horizons (10, 30, 60, 90 days). Cell color: IC value (blue = positive, red = negative). Annotations: IC% and significance stars.',
        insights: `${IC_COPY.tooltipPhrase} Look for cells with consistent sign across horizons. Stars (p<0.05) indicate reliable signal.`,
    },

    icSectorDecayChart: {
        data: 'Average IC across all sectors at each horizon, showing how signal strength evolves over time. Error bands show standard deviation across sectors.',
        axes: 'X-axis: forward return horizon (trading days). Y-axis: average IC (Spearman correlation). Shaded area: +/- 1 standard deviation.',
        insights: `${IC_COPY.tooltipPhrase} Signal typically decays at longer horizons as prices converge to fair value. Peak |IC| horizon suggests optimal holding period.`,
    },

    icIndexChart: {
        data: 'Same as Sector IC but grouped by index instead of sector. Shows whether the signal works better in certain markets.',
        axes: 'Rows: indices. Columns: forward return horizons (10, 30, 60, 90 days). Cell color: IC value. Annotations: IC% and significance stars.',
        insights: `${IC_COPY.tooltipPhrase} Compare IC across markets to identify where the model has best predictive power.`,
    },

    icIndexDecayChart: {
        data: 'Average IC across all indices at each horizon, showing signal decay by market. Error bands show cross-index dispersion.',
        axes: 'X-axis: forward return horizon (trading days). Y-axis: average IC. Shaded area: +/- 1 standard deviation.',
        insights: `${IC_COPY.tooltipPhrase} Faster decay in efficient markets (e.g., US) vs slower decay in less covered markets.`,
    },

    uncertaintyChart: {
        data: 'Each dot is one stock. Model uncertainty (CV prediction std) plotted against absolute prediction error. Helps assess whether the model "knows what it doesn\'t know".',
        axes: 'X-axis: relative std deviation of predictions across CV folds (%). Y-axis: absolute prediction error (%). Diagonal = well-calibrated uncertainty.',
        insights: 'Points near the diagonal mean the model\'s uncertainty matches actual error — a sign of good calibration. High-uncertainty outliers should be treated with caution.',
    },

    sizePremiumChart: {
        data: 'The beta coefficient of the log-quadratic size correction model per quarter. Captures how much small-cap stocks are systematically mispriced differently from large-caps.',
        axes: 'X-axis: quarter date. Y-axis: size premium beta (coefficient on log(market_cap) in the correction model). Each dot is one quarter.',
        insights: 'Negative beta means small caps tend to be underpriced relative to large caps. Track stability over time — large swings suggest regime changes in size factor.',
    },
};
