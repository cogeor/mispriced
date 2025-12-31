// Simplified Vanilla JS Dashboard Logic

let dashboardData = null;
let mispricingMode = 'raw';
let colorBy = 'sector';

// --- Utils ---
function formatPct(val, decimals = 1) {
    return (val * 100).toFixed(decimals) + '%';
}

function formatCurrency(val) {
    return '$' + val.toLocaleString();
}

function aggregateToSummary(tsData) {
    if (!tsData || tsData.length === 0) return [];

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
        spread: 0,
        hit_rate: 0
    }));
}

// --- Charts (Plotly Wrappers) ---

const PLOT_BG_COLOR = 'rgba(0,0,0,0)';
const PAPER_BG_COLOR = 'rgba(0,0,0,0)';
const TEXT_COLOR = '#9ca3af'; // gray-400
const GRID_COLOR = '#374151'; // gray-700 opacity
const HOVER_BG = '#1f2937';
const HOVER_BORDER = '#374151';

function renderValuationMap(data, elementId, colorBy, metricKey) {
    const el = document.getElementById(elementId);
    if (!el) return;

    const isResidual = metricKey === 'residualMispricing';
    // Helper to get displayed predicted value based on mode
    const getPredicted = (d) => isResidual ? d.actual * (1 + (d.residualMispricing || 0)) : d.predicted;

    const x = data.map(d => d.actual);
    const y = data.map(d => getPredicted(d));
    const text = data.map(d => {
        const mVal = d[metricKey];
        const mPct = (mVal * 100).toFixed(1) + '%';
        return `<b>${d.company} (${d.ticker})</b><br>${d.sector}<br><br>Actual: $${d.actual.toFixed(2)}B<br>Predicted: $${getPredicted(d).toFixed(2)}B<br>Mispricing: <b>${mPct}</b>`;
    });

    const customdata = data.map(d => [d.ticker, d.sector, d.company, d[metricKey]]);

    let markerColor;
    let showScale = false;

    if (colorBy === 'sector') {
        markerColor = data.map(d => d.sectorColor);
    } else {
        const mValues = data.map(d => d[metricKey]);
        markerColor = mValues;
        showScale = true;
    }

    const trace = {
        x: x,
        y: y,
        mode: 'markers',
        type: 'scatter',
        text: text,
        customdata: customdata,
        hoverinfo: 'text',
        marker: {
            size: 8,
            color: markerColor,
            opacity: 0.8,
            line: { width: 1, color: '#111827' }
        }
    };

    if (showScale) {
        trace.marker.colorscale = 'RdYlGn';
        trace.marker.reversescale = true;
        trace.marker.cmin = -0.5;
        trace.marker.cmax = 0.5;
        trace.marker.colorbar = {
            title: 'Mispricing',
            thickness: 12,
            tickformat: '.0%',
            tickfont: { color: TEXT_COLOR },
            titlefont: { color: TEXT_COLOR }
        };
    }

    const minVal = Math.min(Math.min(...x), Math.min(...y)) * 0.8;
    const maxVal = Math.max(Math.max(...x), Math.max(...y)) * 2;

    const layout = {
        margin: { t: 20, r: 20, b: 50, l: 60 },
        paper_bgcolor: PAPER_BG_COLOR,
        plot_bgcolor: PLOT_BG_COLOR,
        xaxis: {
            title: 'Actual Market Cap ($B)',
            type: 'log',
            gridcolor: GRID_COLOR,
            tickfont: { color: TEXT_COLOR },
            titlefont: { color: TEXT_COLOR },
            zerolinecolor: GRID_COLOR
        },
        yaxis: {
            title: 'Predicted Market Cap ($B)',
            type: 'log',
            gridcolor: GRID_COLOR,
            tickfont: { color: TEXT_COLOR },
            titlefont: { color: TEXT_COLOR },
            zerolinecolor: GRID_COLOR
        },
        shapes: [
            {
                type: 'line',
                x0: minVal, y0: minVal,
                x1: maxVal, y1: maxVal,
                line: { color: '#4b5563', width: 2, dash: 'dot' }
            }
        ],
        hoverlabel: { bgcolor: HOVER_BG, bordercolor: HOVER_BORDER, font: { color: '#f3f4f6' } },
        showlegend: false
    };

    const config = { responsive: true, displayModeBar: false };
    Plotly.newPlot(elementId, [trace], layout, config);

    // Explicitly add click handler
    if (el.on) {
        el.removeAllListeners && el.removeAllListeners('plotly_click');
        el.on('plotly_click', (data) => {
            const pt = data.points[0];
            const ticker = pt.customdata[0];
            window.open(`https://www.google.com/finance/quote/${ticker}:NYSE`, '_blank');
        });
    }
}

function renderIndexChart(data, elementId, metricKey) {
    const el = document.getElementById(elementId);
    if (!el) return;

    const validData = data.filter(d => d.count > 0);
    validData.sort((a, b) => b[metricKey] - a[metricKey]);

    const x = validData.map(d => d.index);
    const y = validData.map(d => d[metricKey]);
    // Enhanced tooltip
    const text = validData.map(d => `<b>${d.index}</b><br>Mispricing: ${(d[metricKey] * 100).toFixed(1)}%<br>Count: ${d.count}`);

    // Gradient logic hard to do in vanilla, stick to solids
    const colors = validData.map(d => d[metricKey] > 0 ? '#10b981' : '#ef4444'); // green-500, red-500

    const trace = {
        x: x, y: y, type: 'bar', text: text, hoverinfo: 'text', marker: { color: colors, line: { width: 0 } }
    };

    const layout = {
        margin: { t: 10, r: 10, b: 100, l: 50 },
        paper_bgcolor: PAPER_BG_COLOR,
        plot_bgcolor: PLOT_BG_COLOR,
        xaxis: { tickfont: { color: TEXT_COLOR }, gridcolor: GRID_COLOR, tickangle: -45 },
        yaxis: { title: 'Mispricing', tickformat: '.0%', tickfont: { color: TEXT_COLOR }, gridcolor: GRID_COLOR },
        hoverlabel: { bgcolor: HOVER_BG, bordercolor: HOVER_BORDER, font: { color: '#f3f4f6' } }
    };

    Plotly.newPlot(elementId, [trace], layout, { responsive: true, displayModeBar: false });
}

function renderIndexTimeSeries(data, elementId, metricKey) {
    const el = document.getElementById(elementId);
    if (!el) return;

    if (metricKey === 'residualMispricing' && data.length > 0 && data[0].residualMispricing === undefined) {
        el.innerHTML = '<div class="absolute inset-0 flex items-center justify-center text-gray-500">Residual Time Series not available</div>';
        return;
    }

    const statsByIndex = new Map();
    data.forEach(d => {
        if (!statsByIndex.has(d.index)) statsByIndex.set(d.index, { x: [], y: [] });
        const val = metricKey === 'residualMispricing' ? (d.residualMispricing || 0) : d.mispricing;
        statsByIndex.get(d.index).x.push(d.date);
        statsByIndex.get(d.index).y.push(val);
    });

    const traces = [];
    statsByIndex.forEach((series, indexName) => {
        traces.push({
            name: indexName, x: series.x, y: series.y, type: 'scatter', mode: 'lines+markers', line: { width: 2 }
        });
    });

    const layout = {
        margin: { t: 20, r: 20, b: 40, l: 50 },
        paper_bgcolor: PAPER_BG_COLOR,
        plot_bgcolor: PLOT_BG_COLOR,
        xaxis: { tickfont: { color: TEXT_COLOR }, gridcolor: GRID_COLOR, showgrid: false },
        yaxis: { tickformat: '.0%', tickfont: { color: TEXT_COLOR }, gridcolor: GRID_COLOR },
        legend: { orientation: 'h', y: 1.1, font: { color: TEXT_COLOR } },
        hoverlabel: { bgcolor: HOVER_BG, bordercolor: HOVER_BORDER, font: { color: '#f3f4f6' } }
    };

    Plotly.newPlot(elementId, traces, layout, { responsive: true, displayModeBar: false });
}

function buildICHeatmap(summaryData, elementId, filterMetric) {
    const el = document.getElementById(elementId);
    if (!el) return;

    const filtered = summaryData.filter(d => (d.metric || 'raw') === filterMetric);
    if (filtered.length === 0) {
        el.innerHTML = '<div class="flex items-center justify-center h-full text-gray-500">No Data</div>';
        return;
    }

    const horizons = Array.from(new Set(filtered.map(d => d.horizon))).sort((a, b) => a - b);
    const names = Array.from(new Set(filtered.map(d => d.name))).sort();

    const z = [];
    const text = [];

    names.forEach(name => {
        const rowZ = [];
        const rowText = [];
        horizons.forEach(h => {
            const item = filtered.find(d => d.name === name && d.horizon === h);
            if (item) {
                rowZ.push(item.ic);
                const sig = item.pval < 0.05 ? '★' : '';
                const txt = `<b>${item.name} (${h}d)</b><br>IC: ${(item.ic * 100).toFixed(1)}%${sig}<br>N: ${item.n_obs}<br>p-val: ${item.pval.toFixed(3)}`;
                rowText.push(txt);
            } else {
                rowZ.push(NaN);
                rowText.push('N/A');
            }
        });
        z.push(rowZ);
        text.push(rowText);
    });

    const trace = {
        z: z, x: horizons.map(h => `${h}d`), y: names, text: text, type: 'heatmap', hoverinfo: 'text',
        colorscale: 'RdBu', zmin: -0.2, zmax: 0.2,
        colorbar: { title: 'IC', thickness: 12, tickformat: '.0%', tickfont: { color: TEXT_COLOR }, titlefont: { color: TEXT_COLOR } },
        xgap: 1, ygap: 1
    };

    const layout = {
        margin: { t: 10, r: 10, b: 40, l: 150 },
        paper_bgcolor: PAPER_BG_COLOR,
        plot_bgcolor: PLOT_BG_COLOR,
        xaxis: { title: 'Horizon', tickfont: { color: TEXT_COLOR }, side: 'bottom' },
        yaxis: { tickfont: { color: TEXT_COLOR }, automargin: true },
        hoverlabel: { bgcolor: HOVER_BG, bordercolor: HOVER_BORDER, font: { color: '#f3f4f6' } }
    };

    Plotly.newPlot(elementId, [trace], layout, { responsive: true, displayModeBar: false });
}

function renderSignalDecay(horizonData, elementId) {
    const el = document.getElementById(elementId);
    if (!el) return;
    if (!horizonData || horizonData.length === 0) return;

    const sorted = [...horizonData].sort((a, b) => a.horizon - b.horizon);
    const x = sorted.map(d => d.horizon);
    const y = sorted.map(d => d.avg_ic);

    const trace = {
        x: x, y: y, type: 'scatter', mode: 'lines+markers',
        line: { color: '#3b82f6', width: 3, shape: 'spline' },
        marker: { size: 10, color: '#1d4ed8', line: { width: 2, color: '#3b82f6' } }
    };

    const layout = {
        margin: { t: 20, r: 20, b: 50, l: 60 },
        paper_bgcolor: PAPER_BG_COLOR,
        plot_bgcolor: PLOT_BG_COLOR,
        xaxis: { title: 'Horizon (Days)', tickfont: { color: TEXT_COLOR }, gridcolor: GRID_COLOR },
        yaxis: { title: 'Average IC', tickformat: '.1%', tickfont: { color: TEXT_COLOR }, gridcolor: GRID_COLOR, zerolinecolor: GRID_COLOR },
        hoverlabel: { bgcolor: HOVER_BG, bordercolor: HOVER_BORDER, font: { color: '#f3f4f6' } }
    };

    Plotly.newPlot(elementId, [trace], layout, { responsive: true, displayModeBar: false });
}

function renderSectorChart(scatterData, elementId, metricKey) {
    const el = document.getElementById(elementId);
    if (!el) return;

    // Dynamically aggregate from scatterData respecting metricKey
    // Logic port from dashboard.html renderSectorChartWithMode
    const sectors = [...new Set(scatterData.map(d => d.sector))];

    // Compute aggregations
    // We want Average Mispricing used for Y axis
    const data = sectors.map(sector => {
        const items = scatterData.filter(d => d.sector === sector);
        const count = items.length;
        // Average mispricing for this sector using current metric
        const total = items.reduce((sum, d) => sum + (d[metricKey] || 0), 0);
        const avg = total / count;
        // Color used in dashboard.html was constant per sector? 
        // Or based on avg mispricing?
        // dashboard.html line 461 used 'sectorColors[sector]'.
        // app.js seems to carry 'sectorColor' on each data item. We can pick the first one.
        const color = items[0] ? items[0].sectorColor : '#9ca3af';
        return { sector, avgMispricing: avg, count, color };
    });

    data.sort((a, b) => b.avgMispricing - a.avgMispricing);

    const x = data.map(d => d.sector);
    const y = data.map(d => d.avgMispricing);
    const colors = data.map(d => d.color);

    // Tooltip should show count as well
    const text = data.map(d => `<b>${d.sector}</b><br>Avg Mispricing: ${(d.avgMispricing * 100).toFixed(1)}%<br>Count: ${d.count}`);

    const trace = { x: x, y: y, type: 'bar', marker: { color: colors, opacity: 0.8 }, text: text, hoverinfo: 'text' };

    const layout = {
        margin: { t: 20, r: 20, b: 100, l: 50 },
        paper_bgcolor: PAPER_BG_COLOR,
        plot_bgcolor: PLOT_BG_COLOR,
        xaxis: { tickangle: -45, tickfont: { color: TEXT_COLOR } },
        yaxis: { title: 'Avg Mispricing', tickformat: '.0%', tickfont: { color: TEXT_COLOR }, gridcolor: GRID_COLOR },
        hoverlabel: { bgcolor: HOVER_BG, bordercolor: HOVER_BORDER, font: { color: '#f3f4f6' } }
    };

    Plotly.newPlot(elementId, [trace], layout, { responsive: true, displayModeBar: false });
}

function renderUncertaintyChart(data, elementId, metricKey) {
    const el = document.getElementById(elementId);
    if (!el) return;

    // Filter outliers for clean plot, matching dashboard.html logic
    const uncertaintyData = data.filter(d => d.relStd < 1.0 && Math.abs(d[metricKey]) < 1.5);

    const x = uncertaintyData.map(d => d.relStd * 100);
    const y = uncertaintyData.map(d => Math.abs(d[metricKey]) * 100);

    // Calculate R Correlation
    const n = x.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
    for (let i = 0; i < n; i++) {
        sumX += x[i];
        sumY += y[i];
        sumXY += x[i] * y[i];
        sumX2 += x[i] * x[i];
        sumY2 += y[i] * y[i];
    }
    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    const corr = denominator !== 0 ? numerator / denominator : 0;

    const text = uncertaintyData.map(d => `<b>${d.ticker}</b><br>Unc: ${d.relStd.toFixed(1)}%<br>Err: ${Math.abs(d[metricKey]).toFixed(1)}%`);
    const colors = uncertaintyData.map(d => d.sectorColor);

    const trace = {
        x: x, y: y, mode: 'markers', type: 'scatter', text: text, hoverinfo: 'text',
        marker: { size: 5, opacity: 0.6, color: '#60a5fa' }
    };

    // Line y=x
    const traceLine = {
        x: [0, 80], y: [0, 80], mode: 'lines', type: 'scatter',
        line: { color: 'rgba(255,255,255,0.3)', dash: 'dash', width: 1 }, hoverinfo: 'skip'
    };

    const layout = {
        margin: { t: 20, r: 20, b: 50, l: 60 },
        paper_bgcolor: PAPER_BG_COLOR,
        plot_bgcolor: PLOT_BG_COLOR,
        xaxis: { title: 'Model Uncertainty (Rel. Std Dev %)', ticksuffix: '%', tickfont: { color: TEXT_COLOR }, gridcolor: GRID_COLOR },
        yaxis: { title: 'Abs. Prediction Error (%)', ticksuffix: '%', tickfont: { color: TEXT_COLOR }, gridcolor: GRID_COLOR },
        hoverlabel: { bgcolor: HOVER_BG, bordercolor: HOVER_BORDER, font: { color: '#f3f4f6' } },
        showlegend: false,
        annotations: [{
            x: 0.98, y: 0.98, xref: 'paper', yref: 'paper',
            text: `ρ = ${corr.toFixed(3)}`, showarrow: false,
            font: { color: '#8b919a', size: 12 },
            bgcolor: 'rgba(26, 29, 36, 0.8)', borderpad: 4
        }]
    };

    Plotly.newPlot(elementId, [trace, traceLine], layout, { responsive: true, displayModeBar: false });
}

function renderSizePremiumChart(data, elementId) {
    const el = document.getElementById(elementId);
    if (!el) return;
    if (!data || data.length === 0) {
        el.innerHTML = '<div class="flex items-center justify-center h-full text-gray-500">No Size Premium Data</div>';
        return;
    }
    const x = data.map(d => d.quarter);
    // Use slope instead of beta, matching JSON structure
    const y = data.map(d => (d.slope || d.beta || 0) * 100);

    const trace = { x: x, y: y, type: 'scatter', mode: 'lines+markers', line: { color: '#fbbf24', width: 2 } };

    const layout = {
        margin: { t: 20, r: 20, b: 50, l: 60 },
        paper_bgcolor: PAPER_BG_COLOR,
        plot_bgcolor: PLOT_BG_COLOR,
        xaxis: { tickfont: { color: TEXT_COLOR }, gridcolor: GRID_COLOR },
        yaxis: { title: 'Size Elasticity (Beta)', ticksuffix: '%', tickfont: { color: TEXT_COLOR }, gridcolor: GRID_COLOR },
        shapes: [{ type: 'line', x0: x[0], x1: x[x.length - 1], y0: 0, y1: 0, line: { color: '#6b7280', width: 1, dash: 'dash' } }],
        hoverlabel: { bgcolor: HOVER_BG, bordercolor: HOVER_BORDER, font: { color: '#f3f4f6' } }
    };
    Plotly.newPlot(elementId, [trace], layout, { responsive: true, displayModeBar: false });
}


// --- Main Application State ---

async function init() {
    try {
        // Assume hosted relatively 
        const response = await fetch('public/dashboard_data.json');
        if (!response.ok) throw new Error(`HTTP ${response.status} loading data`);
        dashboardData = await response.json();

        if (dashboardData) {
            setupEventListeners(); // Setup listeners *before* initial render
            renderDashboard();
            updateStats(); // Initial stats
        }
    } catch (e) {
        console.error(e);
        document.getElementById('app').innerHTML = `
            <div class="flex justify-center items-center h-screen">
                <div class="text-center p-8 bg-red-900/20 border border-red-500/50 rounded-lg max-w-lg">
                    <h2 class="text-2xl font-bold text-red-500 mb-2">Error Loading Data</h2>
                    <p class="text-gray-300 mb-4">${e.message}</p>
                    <p class="text-sm text-gray-500">Ensure 'dashboard_data.json' exists in the 'public' folder and you are running the server.</p>
                </div>
            </div>`;
    }
}

function renderDashboard() {
    if (!dashboardData) return;
    updateStats();
    updateCharts();
}

function updateStats() {
    if (!dashboardData) return;

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

    // Compute dynamic stats based on mode
    const key = mispricingMode === 'sizeNeutral' ? 'residualMispricing' : 'mispricing';
    const values = dashboardData.scatter_data.map(d => d[key]);
    values.sort((a, b) => a - b);
    const median = values[Math.floor(values.length / 2)];
    const avg = values.reduce((a, b) => a + b, 0) / values.length;

    setText('medianMispricing', (median > 0 ? '+' : '') + (median * 100).toFixed(1) + '%');
    setText('avgMispricing', (avg * 100).toFixed(1));

    // Update label next to Title
    setText('mispricingModeLabel', mispricingMode === 'raw' ? '(raw)' : '(size-neutral)');
}

function updateCharts() {
    if (!dashboardData) return;

    const metricKey = mispricingMode === 'sizeNeutral' ? 'residualMispricing' : 'mispricing';
    const sortKey = metricKey;

    renderValuationMap(dashboardData.scatter_data, 'valuationChart', colorBy, metricKey);
    renderSectorChart(dashboardData.scatter_data, 'sectorChart', metricKey);
    renderIndexChart(dashboardData.index_chart_data, 'indexChart', metricKey);
    renderIndexTimeSeries(dashboardData.index_timeseries, 'indexTimeSeriesChart', metricKey);

    const backtestMetricKey = mispricingMode === 'sizeNeutral' ? 'residual' : 'raw';
    const sectorStats = aggregateToSummary(dashboardData.backtest_data.sector_ts);
    buildICHeatmap(sectorStats, 'icSectorChart', backtestMetricKey);

    const indexStats = aggregateToSummary(dashboardData.backtest_data.index_ts);
    buildICHeatmap(indexStats, 'icIndexChart', backtestMetricKey);

    renderSignalQualityTable(sectorStats, indexStats, backtestMetricKey);

    const decayData = recomputeDecay(dashboardData.backtest_data.sector_ts, backtestMetricKey);
    renderSignalDecay(decayData, 'icHorizonChart');

    renderUncertaintyChart(dashboardData.scatter_data, 'uncertaintyChart', metricKey);
    renderSizePremiumChart(dashboardData.size_coefficients, 'sizePremiumChart');
    renderTopTables(dashboardData.scatter_data, sortKey);
}

function recomputeDecay(ts, metric) {
    const filtered = ts.filter(d => (d.metric || 'raw') === metric);
    const byHorizon = new Map();
    filtered.forEach(d => {
        if (!byHorizon.has(d.horizon)) byHorizon.set(d.horizon, []);
        byHorizon.get(d.horizon).push(d.ic);
    });
    return Array.from(byHorizon.entries()).map(([h, ics]) => ({
        horizon: h,
        avg_ic: ics.reduce((a, b) => a + b, 0) / ics.length
    }));
}

function renderSignalQualityTable(sectorStats, indexStats, metric) {
    const el = document.getElementById('signalQualityTable');
    if (!el) return;

    const filteredS = sectorStats.filter(d => d.metric === metric);
    const filteredI = indexStats.filter(d => d.metric === metric);

    const all = [
        ...filteredS.map(d => ({ ...d, type: 'Sector' })),
        ...filteredI.map(d => ({ ...d, type: 'Index' }))
    ];
    all.sort((a, b) => Math.abs(b.ic) - Math.abs(a.ic));

    el.innerHTML = all.slice(0, 15).map(d => {
        const sig = d.pval < 0.05 ? '★' : '';
        const icClass = d.ic > 0 ? 'text-green-400' : 'text-red-400';
        return `
            <tr class="border-b border-gray-800 hover:bg-gray-800/50 transition-colors">
                <td class="py-3 px-4">
                    <span class="text-xs text-gray-500 block mb-0.5">${d.type}</span>
                    <span class="font-medium text-gray-200">${d.name}</span>
                </td>
                <td class="py-3 px-2 text-right text-gray-500 text-xs">${d.n_obs}</td>
                <td class="py-3 px-2 text-right text-gray-400 font-mono">${d.horizon}d</td>
                <td class="py-3 px-2 text-right font-medium ${icClass}">${(d.ic * 100).toFixed(2)}%${sig}</td>
                <td class="py-3 px-2 text-right text-gray-400 font-mono">${(d.spread * 100).toFixed(1)}%</td>
                <td class="py-3 px-4 text-right text-gray-400 font-mono">${(d.hit_rate * 100).toFixed(0)}%</td>
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
    if (!el) return;
    el.innerHTML = rows.map(r => `
        <tr class="border-b border-gray-800 hover:bg-gray-800/50 transition-colors cursor-pointer group" onclick="window.open('https://www.google.com/finance/quote/${r.ticker}:NYSE', '_blank')">
            <td class="py-3 px-5 font-bold text-blue-400 group-hover:text-blue-300 transition-colors">${r.ticker}</td>
            <td class="py-3 px-2 text-gray-300 max-w-[150px] truncate" title="${r.company}">${r.company}</td>
            <td class="py-3 px-2 text-right ${r[key] < 0 ? 'text-green-400' : 'text-red-400'} font-bold">
                ${(r[key] * 100).toFixed(1)}%
            </td>
            <td class="py-3 px-5 text-right text-gray-400 font-mono">$${r.actual.toFixed(1)}B</td>
        </tr>
    `).join('');
}

function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

function setupEventListeners() {
    const modeRaw = document.getElementById('modeRaw');
    const modeSize = document.getElementById('modeSizeNeutral');
    const colorSector = document.getElementById('colorSector');
    const colorMisc = document.getElementById('colorMispricing');

    if (modeRaw) modeRaw.addEventListener('click', () => setMode('raw'));
    if (modeSize) modeSize.addEventListener('click', () => setMode('sizeNeutral'));

    if (colorSector) colorSector.addEventListener('click', () => setColor('sector'));
    if (colorMisc) colorMisc.addEventListener('click', () => setColor('mispricing'));
}

function setMode(mode) {
    mispricingMode = mode;
    const isRaw = mode === 'raw';

    const rawBtn = document.getElementById('modeRaw');
    const sizeBtn = document.getElementById('modeSizeNeutral');

    if (rawBtn && sizeBtn) {
        if (isRaw) {
            rawBtn.classList.add('active');
            sizeBtn.classList.remove('active');
        } else {
            sizeBtn.classList.add('active');
            rawBtn.classList.remove('active');
        }
    }

    updateStats();
    updateCharts();
}

function setColor(c) {
    colorBy = c;

    const sectorBtn = document.getElementById('colorSector');
    const miscBtn = document.getElementById('colorMispricing');

    const activeClass = 'bg-gray-700';
    const activeText = 'text-white';

    if (c === 'sector') {
        // Active
        sectorBtn.classList.add(activeClass, activeText, 'shadow-sm');
        sectorBtn.classList.remove('text-gray-400', 'hover:bg-gray-800');

        // Inactive
        miscBtn.classList.remove(activeClass, activeText, 'shadow-sm');
        miscBtn.classList.add('text-gray-400', 'hover:bg-gray-800');
    } else {
        // Active
        miscBtn.classList.add(activeClass, activeText, 'shadow-sm');
        miscBtn.classList.remove('text-gray-400', 'hover:bg-gray-800');

        // Inactive
        sectorBtn.classList.remove(activeClass, activeText, 'shadow-sm');
        sectorBtn.classList.add('text-gray-400', 'hover:bg-gray-800');
    }

    const metricKey = mispricingMode === 'sizeNeutral' ? 'residualMispricing' : 'mispricing';
    if (dashboardData) renderValuationMap(dashboardData.scatter_data, 'valuationChart', colorBy, metricKey);
}

// Ensure DOM is ready (though script is at bottom)
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
