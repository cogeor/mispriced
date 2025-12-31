import Plotly from 'plotly.js-dist-min';
import { SectorBreakdownItem, ScatterPoint, SizeCoefficient } from '../types';
export function renderSectorChart(data, elementId) {
    const el = document.getElementById(elementId);
    if (!el)
        return;
    // Sort by count
    data.sort((a, b) => b.count - a.count);
    const x = data.map(d => d.sector);
    const y = data.map(d => d.count);
    const colors = data.map(d => d.color);
    const trace = {
        x: x,
        y: y,
        type: 'bar',
        marker: { color: colors }
    };
    const layout = {
        margin: { t: 20, r: 20, b: 80, l: 40 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {
            tickangle: 45,
            tickfont: { color: '#8b919a' }
        },
        yaxis: {
            title: 'Count',
            tickfont: { color: '#8b919a' },
            gridcolor: '#2a2e38'
        },
        hoverlabel: { bgcolor: '#1a1d24', bordercolor: '#2a2e38', font: { color: '#e8eaed' } }
    };
    const config = { responsive: true, displayModeBar: false };
    Plotly.newPlot(elementId, [trace], layout, config);
}
export function renderUncertaintyChart(data, elementId, metricKey) {
    const el = document.getElementById(elementId);
    if (!el)
        return;
    // x: Uncertainty (relStd), y: Absolute Error/Mispricing
    const x = data.map(d => d.relStd);
    const y = data.map(d => Math.abs(d[metricKey]));
    const text = data.map(d => `${d.ticker}<br>Unc: ${(d.relStd * 100).toFixed(1)}%<br>Err: ${(Math.abs(d[metricKey]) * 100).toFixed(1)}%`);
    // Color by sector
    const colors = data.map(d => d.sectorColor);
    const trace = {
        x: x,
        y: y,
        mode: 'markers',
        type: 'scatter',
        text: text,
        hoverinfo: 'text',
        marker: { size: 6, opacity: 0.6, color: colors }
    };
    const layout = {
        margin: { t: 20, r: 20, b: 40, l: 60 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {
            title: 'Model Uncertainty (Std Dev / Price)',
            tickformat: '.0%',
            tickfont: { color: '#8b919a' },
            gridcolor: '#2a2e38'
        },
        yaxis: {
            title: 'Absolute Mispricing Magnitude',
            tickformat: '.0%',
            tickfont: { color: '#8b919a' },
            gridcolor: '#2a2e38'
        },
        hoverlabel: { bgcolor: '#1a1d24', bordercolor: '#2a2e38', font: { color: '#e8eaed' } }
    };
    const config = { responsive: true, displayModeBar: false };
    Plotly.newPlot(elementId, [trace], layout, config);
}
export function renderSizePremiumChart(data, elementId) {
    const el = document.getElementById(elementId);
    if (!el)
        return;
    if (!data || data.length === 0) {
        el.innerHTML = '<div class="text-center text-gray-500 pt-10">No Size Premium Data</div>';
        return;
    }
    const x = data.map(d => d.quarter);
    const y = data.map(d => d.beta * 100); // %
    // Error bars? data has t_stat. CI = beta +/- (beta/t_stat)*1.96? Roughly.
    // Or just simple line.
    const trace = {
        x: x,
        y: y,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#c9a227' }
    };
    const layout = {
        margin: { t: 20, r: 20, b: 40, l: 60 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {
            tickfont: { color: '#8b919a' },
            gridcolor: '#2a2e38'
        },
        yaxis: {
            title: 'Size Elasticity (Beta)',
            ticksuffix: '%',
            tickfont: { color: '#8b919a' },
            gridcolor: '#2a2e38'
        },
        // Add a zero line
        shapes: [
            { type: 'line', x0: x[0], x1: x[x.length - 1], y0: 0, y1: 0, line: { color: '#555', width: 1, dash: 'dash' } }
        ],
        hoverlabel: { bgcolor: '#1a1d24', bordercolor: '#2a2e38', font: { color: '#e8eaed' } }
    };
    const config = { responsive: true, displayModeBar: false };
    Plotly.newPlot(elementId, [trace], layout, config);
}
//# sourceMappingURL=miscCharts.js.map