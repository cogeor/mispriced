import Plotly from 'plotly.js-dist-min';
import { BacktestItem } from '../types';
export function buildICHeatmap(summaryData, elementId, filterMetric) {
    const el = document.getElementById(elementId);
    if (!el)
        return;
    // Filter by metric
    const filtered = summaryData.filter(d => (d.metric || 'raw') === filterMetric);
    if (filtered.length === 0) {
        el.innerHTML = '<div class="flex items-center justify-center h-full text-gray-500">No Data</div>';
        return;
    }
    // Get unique horizons and names
    const horizons = Array.from(new Set(filtered.map(d => d.horizon))).sort((a, b) => a - b);
    const names = Array.from(new Set(filtered.map(d => d.name))).sort();
    // Pivot data for heatmap z-values
    // matrix[y_index][x_index] -> y=name, x=horizon
    const z = [];
    const text = [];
    names.forEach(name => {
        const rowZ = [];
        const rowText = [];
        horizons.forEach(h => {
            const item = filtered.find(d => d.name === name && d.horizon === h);
            if (item) {
                rowZ.push(item.ic);
                const sig = item.pval < 0.05 ? '*' : '';
                const txt = `IC: ${(item.ic * 100).toFixed(1)}%${sig}<br>N: ${item.n_obs}<br>p: ${item.pval.toFixed(3)}`;
                rowText.push(txt);
            }
            else {
                rowZ.push(NaN); // or 0
                rowText.push('N/A');
            }
        });
        z.push(rowZ);
        text.push(rowText);
    });
    const trace = {
        z: z,
        x: horizons.map(h => `${h}d`),
        y: names,
        text: text,
        type: 'heatmap',
        hoverinfo: 'text',
        colorscale: 'RdBu',
        reversescale: false, // Positive IC is good (Blue usually?), Negative bad (Red?)
        // Finance: Green/Red.
        // Standard RdBu: Red negative, Blue positive.
        zmin: -0.2,
        zmax: 0.2,
        colorbar: {
            title: 'IC',
            thickness: 10,
            tickformat: '.1%',
            tickfont: { color: '#8b919a' }
        }
    };
    const layout = {
        margin: { t: 20, r: 20, b: 40, l: 120 }, // Left margin for names
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {
            title: 'Horizon',
            tickfont: { color: '#8b919a' }
        },
        yaxis: {
            tickfont: { color: '#8b919a' },
            automargin: true
        },
        hoverlabel: { bgcolor: '#1a1d24', bordercolor: '#2a2e38', font: { color: '#e8eaed' } }
    };
    const config = { responsive: true, displayModeBar: false };
    Plotly.newPlot(elementId, [trace], layout, config);
}
export function renderSignalDecay(horizonData, elementId) {
    const el = document.getElementById(elementId);
    if (!el)
        return;
    if (!horizonData || horizonData.length === 0)
        return;
    // Sort by horizon
    const sorted = [...horizonData].sort((a, b) => a.horizon - b.horizon);
    const x = sorted.map(d => d.horizon);
    const y = sorted.map(d => d.avg_ic);
    const trace = {
        x: x,
        y: y,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#3b82f6', width: 3 },
        marker: { size: 8 }
    };
    const layout = {
        margin: { t: 20, r: 20, b: 40, l: 50 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {
            title: 'Horizon (Days)',
            tickfont: { color: '#8b919a' },
            gridcolor: '#2a2e38'
        },
        yaxis: {
            title: 'Average IC',
            tickformat: '.1%',
            tickfont: { color: '#8b919a' },
            gridcolor: '#2a2e38'
        },
        hoverlabel: { bgcolor: '#1a1d24', bordercolor: '#2a2e38', font: { color: '#e8eaed' } }
    };
    const config = { responsive: true, displayModeBar: false };
    Plotly.newPlot(elementId, [trace], layout, config);
}
//# sourceMappingURL=signalQuality.js.map