import './style.css';
import { DashboardData, ScatterPoint, BacktestItem } from './types';
import { formatCurrency, formatPct, aggregateToSummary } from './utils';
import { renderValuationMap } from './charts/valuationMap';
import { renderIndexChart, renderIndexTimeSeries } from './charts/indexCharts';
import { buildICHeatmap, renderSignalDecay } from './charts/signalQuality';
import { renderSectorChart, renderUncertaintyChart, renderSizePremiumChart } from './charts/miscCharts';
let dashboardData = null;
let mispricingMode = 'raw';
let colorBy = 'sector';
async function init() {
    try {
        const response = await fetch('/dashboard_data.json');
        if (!response.ok)
            throw new Error('Failed to load data');
        dashboardData = await response.json();
        if (dashboardData) {
            renderDashboard();
            setupEventListeners();
        }
    }
    catch (e) {
        console.error(e);
        document.body.innerHTML = `<div class="p-10 text-red-500">Error loading dashboard data: ${e}</div>`;
    }
}
function renderDashboard() {
    if (!dashboardData)
        return;
    updateStats();
    updateCharts();
}
function updateStats() {
    if (!dashboardData)
        return;
    const s = dashboardData.stats;
    setText('totalTickers', s.total_tickers.toLocaleString());
    setText('totalMcap', s.total_actual_mcap_t.toFixed(1));
    setText('generatedAt', new Date(dashboardData.generated_at).toLocaleString());
    setText('quarterDate', s.quarter_date);
    setText('metricDesc', s.metric_description);
    setText('undervaluedCount', s.undervalued_count.toLocaleString());
    const total = s.total_tickers || 1;
    setText('undervaluedPct', ((s.undervalued_count / total) * 100).toFixed(0));
    setText('indicesTracked', s.indices_tracked.toLocaleString());
    // Median/Avg depend on current collection of mispricing in scatter_data?
    // Or precomputed stats are global?
    // The stats in JSON are precomputed likely on 'raw'.
    // If I switch to sizeNeutral, I should recompute median from scatter_data.
    const key = mispricingMode === 'sizeNeutral' ? 'residualMispricing' : 'mispricing';
    const values = dashboardData.scatter_data.map(d => d[key]);
    values.sort((a, b) => a - b);
    const median = values[Math.floor(values.length / 2)];
    const avg = values.reduce((a, b) => a + b, 0) / values.length;
    setText('medianMispricing', (median > 0 ? '+' : '') + (median * 100).toFixed(1) + '%');
    setText('avgMispricing', (avg * 100).toFixed(1));
    const label = mispricingMode === 'sizeNeutral' ? '(Size-Neutral)' : '(Raw)';
    setText('medianMispricingLabel', label);
}
function updateCharts() {
    if (!dashboardData)
        return;
    const metricKey = mispricingMode === 'sizeNeutral' ? 'residualMispricing' : 'mispricing';
    const sortKey = metricKey; // Same
    // 1. Valuation Map
    renderValuationMap(dashboardData.scatter_data, 'valuationChart', colorBy, metricKey);
    // 2. Sector Breakdown (Static usually, but re-render if needed)
    renderSectorChart(dashboardData.sector_breakdown, 'sectorChart');
    // 3. Index Charts
    renderIndexChart(dashboardData.index_chart_data, 'indexChart', metricKey);
    renderIndexTimeSeries(dashboardData.index_timeseries, 'indexTimeSeriesChart', metricKey);
    // 4. Signal Quality
    // Need to aggregate data based on metric
    // Python 'sector_ts' has 'metric' field.
    const backtestMetricKey = mispricingMode === 'sizeNeutral' ? 'residual' : 'raw';
    // Aggregate on the fly or use pre-aggregated?
    // Python provided 'sector_ts'. Let's use it.
    const sectorStats = aggregateToSummary(dashboardData.backtest_data.sector_ts);
    buildICHeatmap(sectorStats, 'icSectorChart', backtestMetricKey);
    const indexStats = aggregateToSummary(dashboardData.backtest_data.index_ts);
    buildICHeatmap(indexStats, 'icIndexChart', backtestMetricKey);
    // Signal Quality Table
    renderSignalQualityTable(sectorStats, indexStats, backtestMetricKey);
    // 5. Signal Decay
    // Global Horizon data might be specific to Metric?
    // JSON 'horizon' -> list of {horizon, avg_ic, ...}
    // Probably raw? Python export logic: 'horizon': aggregated globally.
    // Let's assume it's raw. If we want residual, we need to re-aggregate sector_ts by horizon for that metric.
    const decayData = recomputeDecay(dashboardData.backtest_data.sector_ts, backtestMetricKey);
    renderSignalDecay(decayData, 'icHorizonChart');
    // 6. Uncertainty
    renderUncertaintyChart(dashboardData.scatter_data, 'uncertaintyChart', metricKey);
    // 7. Size Premium
    renderSizePremiumChart(dashboardData.size_coefficients, 'sizePremiumChart');
    // 8. Tables
    renderTopTables(dashboardData.scatter_data, sortKey);
}
function recomputeDecay(ts, metric) {
    const filtered = ts.filter(d => (d.metric || 'raw') === metric);
    const byHorizon = new Map();
    filtered.forEach(d => {
        if (!byHorizon.has(d.horizon))
            byHorizon.set(d.horizon, []);
        byHorizon.get(d.horizon).push(d.ic);
    });
    return Array.from(byHorizon.entries()).map(([h, ics]) => ({
        horizon: h,
        avg_ic: ics.reduce((a, b) => a + b, 0) / ics.length
    }));
}
function renderSignalQualityTable(sectorStats, indexStats, metric) {
    const el = document.getElementById('signalQualityTable');
    if (!el)
        return;
    const filteredS = sectorStats.filter(d => d.metric === metric);
    const filteredI = indexStats.filter(d => d.metric === metric);
    // Top 15 by Abs IC
    const all = [
        ...filteredS.map(d => ({ ...d, type: 'Sector' })),
        ...filteredI.map(d => ({ ...d, type: 'Index' }))
    ];
    all.sort((a, b) => Math.abs(b.ic) - Math.abs(a.ic));
    el.innerHTML = all.slice(0, 15).map(d => {
        const sig = d.pval < 0.05 ? '*' : (d.pval < 0.10 ? '~' : '');
        const icClass = d.ic > 0 ? 'text-green-400' : 'text-red-400';
        return `
            <tr class="border-b border-gray-800 hover:bg-gray-800/50 transition-colors">
                <td class="py-2 px-1">
                    <span class="text-xs text-gray-500 block">${d['type']}</span>
                    <span class="font-medium text-gray-200">${d.name}</span>
                </td>
                <td class="py-2 text-right text-gray-500 text-xs">${d.n_obs}</td>
                <td class="py-2 text-right">${d.horizon}d</td>
                <td class="py-2 text-right ${icClass}">${(d.ic * 100).toFixed(2)}%${sig}</td>
                <td class="py-2 text-right text-gray-400">${(d.spread * 100).toFixed(1)}%</td>
                <td class="py-2 text-right text-gray-400">${(d.hit_rate * 100).toFixed(0)}%</td>
            </tr>
        `;
    }).join('');
}
function renderTopTables(data, key) {
    const sorted = [...data].sort((a, b) => a[key] - b[key]);
    const topUndervalued = sorted.slice(0, 10);
    const topOvervalued = sorted.slice(-10).reverse();
    renderTableRows('undervaluedTable', topUndervalued, key);
    renderTableRows('overvaluedTable', topOvervalued, key);
}
function renderTableRows(id, rows, key) {
    const el = document.getElementById(id);
    if (!el)
        return;
    el.innerHTML = rows.map(r => `
        <tr class="border-b border-gray-800 hover:bg-gray-800/50 transition-colors cursor-pointer" onclick="window.open('https://www.google.com/finance/quote/${r.ticker}:NYSE', '_blank')">
            <td class="py-3 font-medium text-blue-400">${r.ticker}</td>
            <td class="py-3 text-gray-300 max-w-[150px] truncate" title="${r.company}">${r.company}</td>
            <td class="py-3 text-right ${r[key] < 0 ? 'text-green-400' : 'text-red-400'} font-bold">
                ${(r[key] * 100).toFixed(1)}%
            </td>
            <td class="py-3 text-right text-gray-400">$${r.actual.toFixed(1)}B</td>
        </tr>
    `).join('');
}
function setText(id, val) {
    const el = document.getElementById(id);
    if (el)
        el.textContent = val;
}
function setupEventListeners() {
    document.getElementById('modeRaw')?.addEventListener('click', () => setMode('raw'));
    document.getElementById('modeSizeNeutral')?.addEventListener('click', () => setMode('sizeNeutral'));
    document.getElementById('colorSector')?.addEventListener('click', () => setColor('sector'));
    document.getElementById('colorMispricing')?.addEventListener('click', () => setColor('mispricing'));
}
function setMode(mode) {
    mispricingMode = mode;
    document.getElementById('modeRaw')?.classList.toggle('active', mode === 'raw');
    document.getElementById('modeSizeNeutral')?.classList.toggle('active', mode === 'sizeNeutral');
    document.getElementById('mispricingModeLabel').textContent = mode === 'raw' ? '(raw)' : '(size-neutral)';
    updateStats();
    updateCharts();
}
function setColor(c) {
    colorBy = c;
    document.getElementById('colorSector')?.classList.toggle('active', c === 'sector');
    document.getElementById('colorMispricing')?.classList.toggle('active', c === 'mispricing');
    const metricKey = mispricingMode === 'sizeNeutral' ? 'residualMispricing' : 'mispricing';
    if (dashboardData)
        renderValuationMap(dashboardData.scatter_data, 'valuationChart', colorBy, metricKey);
}
// Start
init();
//# sourceMappingURL=main.js.map