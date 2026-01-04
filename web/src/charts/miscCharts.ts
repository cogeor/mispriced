import Plotly from 'plotly.js-dist-min';
import type { ScatterPoint, SizeCoefficient, MetricKey } from '../types';

/* eslint-disable @typescript-eslint/no-explicit-any */
type PlotData = any;
type PlotLayout = any;

const PLOT_BG_COLOR = 'rgba(0,0,0,0)';
const PAPER_BG_COLOR = 'rgba(0,0,0,0)';
const TEXT_COLOR = '#e5e7eb';
const GRID_COLOR = '#374151';
const HOVER_BG = '#1f2937';
const HOVER_BORDER = '#374151';

export function renderSectorChart(
    scatterData: ScatterPoint[],
    elementId: string,
    metricKey: MetricKey
): void {
    const el = document.getElementById(elementId);
    if (!el) return;

    // Filter out "Unknown" sector
    const filteredData = scatterData.filter(d => d.sector && d.sector !== 'Unknown');
    const sectors = [...new Set(filteredData.map(d => d.sector))];

    // Helper to compute sector stats
    const computeSectorStats = (items: ScatterPoint[], name: string) => {
        // Filter out extreme outliers (> 200% or < -90% mispricing)
        const validItems = items.filter(d => {
            const val = d[metricKey] || 0;
            return val > -0.9 && val < 2.0;
        });
        const count = validItems.length;
        if (count === 0) {
            return { sector: name, avgMispricing: 0, count: items.length, color: '#6b7280', stdError: 0, totalMcap: 0 };
        }
        // Use market-cap weighted average for more robust aggregation
        const totalMcap = validItems.reduce((sum, d) => sum + d.actual, 0);
        const weightedSum = validItems.reduce((sum, d) => sum + (d[metricKey] || 0) * d.actual, 0);
        // Negate so positive = overvalued
        const avg = -(weightedSum / totalMcap);
        // Compute std error for error bars
        const values = validItems.map(d => -(d[metricKey] || 0));
        const variance = values.reduce((sum, v) => sum + Math.pow(v - avg, 2), 0) / count;
        const std = Math.sqrt(variance);
        const se = std / Math.sqrt(count);
        // Color: positive (overvalued) = green, negative (undervalued) = red
        const color = avg > 0 ? '#10b981' : '#ef4444';
        return { sector: name, avgMispricing: avg, count: items.length, color, stdError: se, totalMcap };
    };

    // Compute aggregations for each sector
    const data = sectors.map(sector => {
        const items = filteredData.filter(d => d.sector === sector);
        return computeSectorStats(items, sector);
    });

    // Add Global aggregate
    const globalStats = computeSectorStats(filteredData, 'Global');
    data.push(globalStats);

    // Sort by mispricing value (overvalued to undervalued, i.e., highest to lowest)
    data.sort((a, b) => b.avgMispricing - a.avgMispricing);

    const x = data.map(d => d.sector);
    const y = data.map(d => d.avgMispricing);
    const colors = data.map(d => d.color);
    const errors = data.map(d => d.stdError);

    const text = data.map(d => `<b>${d.sector}</b><br>Avg Mispricing: ${d.avgMispricing > 0 ? '+' : ''}${(d.avgMispricing * 100).toFixed(1)}%<br>Std Error: ±${(d.stdError * 100).toFixed(1)}%<br>Count: ${d.count}`);

    const trace: PlotData = {
        x: x,
        y: y,
        type: 'bar',
        marker: { color: colors, opacity: 0.8 },
        hovertext: text,
        hoverinfo: 'text',
        textposition: 'none',
        error_y: {
            type: 'data',
            array: errors,
            visible: true,
            color: '#9ca3af',
            thickness: 2,
            width: 4
        }
    };

    const layout: PlotLayout = {
        margin: { t: 20, r: 20, b: 120, l: 60 },
        paper_bgcolor: PAPER_BG_COLOR,
        plot_bgcolor: PLOT_BG_COLOR,
        xaxis: { tickangle: -45, tickfont: { color: TEXT_COLOR, size: 11 }, automargin: true },
        yaxis: { title: 'Mispricing', tickformat: '.0%', tickfont: { color: TEXT_COLOR }, titlefont: { color: TEXT_COLOR }, gridcolor: GRID_COLOR },
        hoverlabel: { bgcolor: HOVER_BG, bordercolor: HOVER_BORDER, font: { color: '#f3f4f6' } }
    };

    Plotly.newPlot(elementId, [trace], layout, { responsive: true, displayModeBar: false });
}

export function renderSectorLegend(scatterData: ScatterPoint[]): void {
    const el = document.getElementById('sectorLegend');
    if (!el) return;

    const sectorColors: Record<string, string> = {};
    scatterData.forEach(d => {
        if (!sectorColors[d.sector]) {
            sectorColors[d.sector] = d.sectorColor;
        }
    });

    el.innerHTML = Object.entries(sectorColors).map(([sector, color]) => `
        <div class="legend-item">
            <span class="legend-dot" style="background: ${color};"></span>
            <span>${sector}</span>
        </div>
    `).join('');
}

export function renderUncertaintyChart(
    data: ScatterPoint[],
    elementId: string,
    metricKey: MetricKey
): void {
    const el = document.getElementById(elementId);
    if (!el) return;

    // Filter outliers for clean plot
    const uncertaintyData = data.filter(d => d.relStd < 1.0 && Math.abs(d[metricKey]) < 1.5);

    const x = uncertaintyData.map(d => d.relStd * 100);
    const y = uncertaintyData.map(d => Math.abs(d[metricKey]) * 100);

    const text = uncertaintyData.map(d => {
        const negated = -(d[metricKey] || 0);
        return `<b>${d.ticker}</b><br>Unc: ${(d.relStd * 100).toFixed(1)}%<br>Err: ${Math.abs(d[metricKey] * 100).toFixed(1)}%<br>Mispricing: ${negated > 0 ? '+' : ''}${(negated * 100).toFixed(1)}%`;
    });

    // Color: positive (overvalued) = green, negative (undervalued) = red (after negation)
    const mispricingValues = uncertaintyData.map(d => -(d[metricKey] || 0));
    const colors = mispricingValues.map(v => v > 0 ? '#10b981' : '#ef4444');

    const trace: PlotData = {
        x: x,
        y: y,
        mode: 'markers',
        type: 'scatter',
        text: text,
        hoverinfo: 'text',
        marker: { size: 5, opacity: 0.6, color: colors }
    };

    const layout: PlotLayout = {
        margin: { t: 20, r: 20, b: 50, l: 60 },
        paper_bgcolor: PAPER_BG_COLOR,
        plot_bgcolor: PLOT_BG_COLOR,
        xaxis: { title: 'Model Uncertainty (Rel. Std Dev %)', titlefont: { color: '#f3f4f6' }, ticksuffix: '%', tickfont: { color: TEXT_COLOR }, gridcolor: GRID_COLOR },
        yaxis: { title: 'Abs. Prediction Error (%)', titlefont: { color: '#f3f4f6' }, ticksuffix: '%', tickfont: { color: TEXT_COLOR }, gridcolor: GRID_COLOR },
        hoverlabel: { bgcolor: HOVER_BG, bordercolor: HOVER_BORDER, font: { color: '#f3f4f6' } },
        showlegend: false
    };

    Plotly.newPlot(elementId, [trace], layout, { responsive: true, displayModeBar: false });
}

export function renderSizePremiumChart(data: SizeCoefficient[], elementId: string): void {
    const el = document.getElementById(elementId);
    if (!el) return;

    if (!data || data.length === 0) {
        el.innerHTML = '<div class="flex items-center justify-center h-full text-gray-500">No Size Premium Data</div>';
        return;
    }

    const x = data.map(d => d.quarter);
    // Negate slope so that "Size Premium" is positive (Small > Large)
    const y = data.map(d => -((d.slope ?? d.beta) || 0) * 100);
    const errors = data.map(d => ((d.se ?? d.slopeSE) || 0) * 100);

    const trace: PlotData = {
        x: x,
        y: y,
        type: 'scatter',
        mode: 'markers',
        marker: { color: '#60a5fa', size: 8 },
        error_y: {
            type: 'data',
            array: errors,
            visible: true,
            color: 'rgba(96, 165, 250, 0.5)',
            thickness: 2,
            width: 4
        }
    };

    const layout: PlotLayout = {
        margin: { t: 20, r: 20, b: 50, l: 60 },
        paper_bgcolor: PAPER_BG_COLOR,
        plot_bgcolor: PLOT_BG_COLOR,
        xaxis: { tickfont: { color: TEXT_COLOR }, gridcolor: GRID_COLOR },
        yaxis: { title: 'Size Premium (Beta)', ticksuffix: '%', tickfont: { color: TEXT_COLOR }, titlefont: { color: TEXT_COLOR }, gridcolor: GRID_COLOR },
        shapes: [{
            type: 'line',
            x0: x[0],
            x1: x[x.length - 1],
            y0: 0,
            y1: 0,
            line: { color: '#6b7280', width: 1, dash: 'dash' }
        }],
        hoverlabel: { bgcolor: HOVER_BG, bordercolor: HOVER_BORDER, font: { color: '#f3f4f6' } }
    };

    Plotly.newPlot(elementId, [trace], layout, { responsive: true, displayModeBar: false });
}
