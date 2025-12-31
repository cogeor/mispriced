import { BacktestItem } from './types';
export function formatPct(val, decimals = 1) {
    return (val * 100).toFixed(decimals) + '%';
}
export function formatCurrency(val) {
    return '$' + val.toLocaleString();
}
export function aggregateToSummary(tsData) {
    if (!tsData || tsData.length === 0)
        return [];
    const grouped = new Map();
    tsData.forEach(d => {
        const metric = d.metric || 'raw';
        const key = `${d.name}_${d.horizon}_${metric}`;
        if (!grouped.has(key)) {
            grouped.set(key, {
                name: d.name, horizon: d.horizon, metric,
                ics: [], pvals: [], n_obs: 0
            });
        }
        const g = grouped.get(key);
        g.ics.push(d.ic);
        g.pvals.push(d.pval);
        g.n_obs += d.n_obs;
    });
    return Array.from(grouped.values()).map(g => ({
        name: g.name,
        horizon: g.horizon,
        metric: g.metric,
        ic: g.ics.reduce((a, b) => a + b, 0) / g.ics.length,
        pval: g.pvals.reduce((a, b) => a + b, 0) / g.pvals.length,
        n_obs: g.n_obs,
        // Fill required fields with defaults if not aggregated
        spread: 0,
        hit_rate: 0
    }));
}
//# sourceMappingURL=utils.js.map