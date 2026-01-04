import type { BacktestItem } from './types';

export function formatPct(val: number, decimals = 1): string {
    return (val * 100).toFixed(decimals) + '%';
}

export function formatCurrency(val: number): string {
    return '$' + val.toLocaleString();
}

export function formatQuarter(dateStr: string): string {
    if (!dateStr) return '';
    const date = new Date(dateStr);
    const month = date.getMonth();
    const quarter = Math.floor(month / 3) + 1;
    const year = date.getFullYear();
    return `Q${quarter} ${year}`;
}

export function formatMcap(b: number): string {
    if (b >= 1000) return (b / 1000).toFixed(1) + 'T';
    if (b >= 1) return b.toFixed(1) + 'B';
    return (b * 1000).toFixed(0) + 'M';
}

export function getSignificanceStars(pval: number): string {
    if (pval < 5e-8) return '★★★';
    if (pval < 5e-4) return '★★';
    if (pval < 0.05) return '★';
    return '';
}

/**
 * Apply Benjamini-Hochberg FDR correction to p-values.
 * Returns a new array with adjusted p-values in the same order.
 */
export function benjaminiHochberg(pvals: number[]): number[] {
    const n = pvals.length;
    if (n === 0) return [];

    // Create array of {index, pval} and sort by pval ascending
    const indexed = pvals.map((p, i) => ({ idx: i, pval: p }));
    indexed.sort((a, b) => a.pval - b.pval);

    // Calculate adjusted p-values: p_adj = p * n / rank
    const adjusted = indexed.map((item, rank) => ({
        idx: item.idx,
        adjPval: Math.min(item.pval * n / (rank + 1), 1)
    }));

    // Ensure monotonicity: work backwards, taking cumulative minimum
    for (let i = n - 2; i >= 0; i--) {
        adjusted[i].adjPval = Math.min(adjusted[i].adjPval, adjusted[i + 1].adjPval);
    }

    // Restore original order
    const result = new Array<number>(n);
    adjusted.forEach(item => {
        result[item.idx] = item.adjPval;
    });

    return result;
}

/**
 * Apply BH correction to BacktestItem array, returning new array with corrected pvals.
 */
export function applyBHCorrection(items: BacktestItem[]): BacktestItem[] {
    if (items.length === 0) return [];

    const pvals = items.map(d => d.pval);
    const corrected = benjaminiHochberg(pvals);

    return items.map((item, i) => ({
        ...item,
        pval: corrected[i]
    }));
}

export interface AggregateResult {
    data: BacktestItem[];
    quarter: string | null;
}

export function aggregateToSummary(
    tsData: BacktestItem[],
    forQuarter: string | null = null
): AggregateResult {
    if (!tsData || tsData.length === 0) return { data: [], quarter: null };

    // Build quarter counts
    const quarterCounts: Record<string, number> = {};
    tsData.forEach(d => {
        if (d.quarter) {
            quarterCounts[d.quarter] = (quarterCounts[d.quarter] || 0) + 1;
        }
    });

    // Sort quarters by date (most recent first)
    const quarters = Object.entries(quarterCounts)
        .sort((a, b) => b[0].localeCompare(a[0]));

    if (quarters.length === 0) return { data: [], quarter: null };

    // Determine target quarter
    let targetQuarter = forQuarter;

    // If requested quarter has no data (or was not specified), find best available
    if (!targetQuarter || !quarterCounts[targetQuarter]) {
        // Pick the most recent quarter with at least 70% of max count (complete data)
        const maxCount = Math.max(...Object.values(quarterCounts));
        targetQuarter = quarters.find(([, count]) => count >= maxCount * 0.7)?.[0] || quarters[0]?.[0];

        // Log fallback if a specific quarter was requested but not found
        if (forQuarter && forQuarter !== targetQuarter) {
            console.log(`Backtest: No data for ${forQuarter}, showing ${targetQuarter} instead`);
        }
    }

    // Filter to target quarter
    const filteredData = targetQuarter
        ? tsData.filter(d => d.quarter === targetQuarter)
        : tsData;

    const grouped = new Map<string, {
        name: string;
        horizon: number;
        metric: string;
        ics: number[];
        pvals: number[];
        spreads: number[];
        hit_rates: number[];
        n_obs: number;
    }>();

    filteredData.forEach(d => {
        const metric = d.metric || 'raw';
        const key = `${d.name}_${d.horizon}_${metric}`;
        if (!grouped.has(key)) {
            grouped.set(key, {
                name: d.name,
                horizon: d.horizon,
                metric,
                ics: [],
                pvals: [],
                spreads: [],
                hit_rates: [],
                n_obs: 0
            });
        }
        const g = grouped.get(key)!;
        g.ics.push(d.ic);
        g.pvals.push(d.pval);
        if (d.spread !== undefined) g.spreads.push(d.spread);
        if (d.hit_rate !== undefined) g.hit_rates.push(d.hit_rate);
        g.n_obs += d.n_obs;
    });

    const results = Array.from(grouped.values()).map(g => ({
        name: g.name,
        horizon: g.horizon,
        metric: g.metric,
        ic: g.ics.reduce((a, b) => a + b, 0) / g.ics.length,
        pval: g.pvals.reduce((a, b) => a + b, 0) / g.pvals.length,
        n_obs: g.n_obs,
        spread: g.spreads.length > 0 ? g.spreads.reduce((a, b) => a + b, 0) / g.spreads.length : 0,
        hit_rate: g.hit_rates.length > 0 ? g.hit_rates.reduce((a, b) => a + b, 0) / g.hit_rates.length : 0
    }));

    return { data: results, quarter: targetQuarter ?? null };
}
