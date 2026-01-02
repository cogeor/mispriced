import Plotly from 'plotly.js-dist-min';
import type { BacktestItem } from '../types';
import { getSignificanceStars, applyBHCorrection } from '../utils';

/* eslint-disable @typescript-eslint/no-explicit-any */
type PlotData = any;
type PlotLayout = any;

const PLOT_BG_COLOR = 'rgba(0,0,0,0)';
const PAPER_BG_COLOR = 'rgba(0,0,0,0)';
const TEXT_COLOR = '#e5e7eb';
const GRID_COLOR = '#374151';
const HOVER_BG = '#1f2937';
const HOVER_BORDER = '#374151';

export function buildICHeatmap(
    summaryData: BacktestItem[],
    elementId: string,
    filterMetric: string,
    _quarter: string | null = null,
    mcapOrder?: Map<string, number>,
    addGlobal: boolean = false
): void {
    const el = document.getElementById(elementId);
    if (!el) return;

    let filtered = summaryData.filter(d => (d.metric || 'raw') === filterMetric);
    if (filtered.length === 0) {
        el.innerHTML = '<div class="flex items-center justify-center h-full text-gray-500">No Data</div>';
        return;
    }

    // Apply Benjamini-Hochberg correction to p-values
    filtered = applyBHCorrection(filtered);

    // Compute Global by averaging across all names for each horizon (weighted by n_obs)
    if (addGlobal) {
        const horizons = Array.from(new Set(filtered.map(d => d.horizon)));
        const globalItems: BacktestItem[] = horizons.map(h => {
            const itemsAtHorizon = filtered.filter(d => d.horizon === h);
            const totalObs = itemsAtHorizon.reduce((sum, d) => sum + d.n_obs, 0);
            const weightedIC = itemsAtHorizon.reduce((sum, d) => sum + d.ic * d.n_obs, 0) / totalObs;
            const weightedHitRate = itemsAtHorizon.reduce((sum, d) => sum + d.hit_rate * d.n_obs, 0) / totalObs;
            // Combine p-values using Fisher's method approximation (use min for simplicity)
            const minPval = Math.min(...itemsAtHorizon.map(d => d.pval));
            return {
                name: 'Global',
                horizon: h,
                ic: weightedIC,
                pval: minPval,
                n_obs: totalObs,
                spread: 0,
                hit_rate: weightedHitRate,
                metric: filterMetric
            };
        });
        filtered = [...filtered, ...globalItems];
    }

    const horizons = Array.from(new Set(filtered.map(d => d.horizon))).sort((a, b) => a - b);

    // Sort names by mcap if provided, otherwise alphabetically
    let names = Array.from(new Set(filtered.map(d => d.name)));
    if (mcapOrder && mcapOrder.size > 0) {
        names.sort((a, b) => (mcapOrder.get(b) || 0) - (mcapOrder.get(a) || 0));
    } else {
        names.sort();
    }

    const z: number[][] = [];
    const text: string[][] = [];

    names.forEach(name => {
        const rowZ: number[] = [];
        const rowText: string[] = [];
        horizons.forEach(h => {
            const item = filtered.find(d => d.name === name && d.horizon === h);
            if (item) {
                // Negate IC so positive = good signal (overvalued stocks go down)
                rowZ.push(-item.ic);
                const sig = item.pval < 0.05 ? '★' : '';
                const hitPct = item.hit_rate ? `${(item.hit_rate * 100).toFixed(1)}%` : 'N/A';
                const txt = `<b>${item.name} (${h}d)</b><br>IC: ${(-item.ic * 100).toFixed(1)}%${sig}<br>Hit Rate: ${hitPct}<br>N: ${item.n_obs}<br>p-val: ${item.pval.toExponential(1)}`;
                rowText.push(txt);
            } else {
                rowZ.push(NaN);
                rowText.push('N/A');
            }
        });
        z.push(rowZ);
        text.push(rowText);
    });

    const trace: PlotData = {
        z: z,
        x: horizons.map(h => `${h}`),
        y: names,
        text: text,
        type: 'heatmap',
        hoverinfo: 'text',
        colorscale: [
            [0, 'rgba(239, 68, 68, 0.9)'],
            [0.5, 'rgba(156, 163, 175, 0.15)'],
            [1, 'rgba(34, 197, 94, 0.9)']
        ],
        zmin: -0.2,
        zmax: 0.2,
        colorbar: {
            title: 'IC',
            thickness: 12,
            tickformat: '.0%',
            tickfont: { color: TEXT_COLOR },
            titlefont: { color: TEXT_COLOR }
        },
        xgap: 1,
        ygap: 1
    };

    // Add annotations for IC values and p-value significance
    const annotations: any[] = [];
    names.forEach((name) => {
        horizons.forEach((h) => {
            const item = filtered.find(d => d.name === name && d.horizon === h);
            if (item) {
                const icPct = (-item.ic * 100).toFixed(1);
                const sig = getSignificanceStars(item.pval);
                const sigText = sig ? '\n' + sig : '';
                annotations.push({
                    x: `${h}`,
                    y: name,
                    text: `${icPct}%${sigText}`,
                    showarrow: false,
                    font: { color: '#ffffff', size: 10 }
                });
            }
        });
    });

    const layout: PlotLayout = {
        margin: { t: 10, r: 10, b: 60, l: 150 },
        paper_bgcolor: PAPER_BG_COLOR,
        plot_bgcolor: PLOT_BG_COLOR,
        xaxis: { title: 'Horizon (Days)', titlefont: { color: '#ffffff' }, tickfont: { color: TEXT_COLOR }, side: 'bottom', showgrid: false, tickvals: horizons.map(h => `${h}`), ticktext: horizons.map(h => `${h}`) },
        yaxis: { tickfont: { color: TEXT_COLOR }, automargin: true, showgrid: false },
        hoverlabel: { bgcolor: HOVER_BG, bordercolor: HOVER_BORDER, font: { color: '#f3f4f6' } },
        annotations: annotations
    };

    Plotly.react(elementId, [trace], layout, { responsive: true, displayModeBar: false });
}

export interface HorizonDecayItem {
    horizon: number;
    avg_ic: number;
    std_error?: number;
}

export function renderSignalDecay(horizonData: HorizonDecayItem[], elementId: string): void {
    const el = document.getElementById(elementId);
    if (!el) return;
    if (!horizonData || horizonData.length === 0) return;

    const sorted = [...horizonData].sort((a, b) => a.horizon - b.horizon);
    const x = sorted.map(d => d.horizon);
    // Negate IC so positive = good signal (overvalued stocks go down)
    const y = sorted.map(d => -d.avg_ic);
    const errors = sorted.map(d => d.std_error || 0);

    const trace: PlotData = {
        x: x,
        y: y,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#3b82f6', width: 3, shape: 'spline' },
        marker: { size: 10, color: '#1d4ed8', line: { width: 2, color: '#3b82f6' } },
        error_y: {
            type: 'data',
            array: errors,
            visible: true,
            color: 'rgba(59, 130, 246, 0.5)',
            thickness: 2,
            width: 4
        }
    };

    const layout: PlotLayout = {
        margin: { t: 20, r: 20, b: 60, l: 70 },
        paper_bgcolor: PAPER_BG_COLOR,
        plot_bgcolor: PLOT_BG_COLOR,
        xaxis: { title: 'Horizon (Days)', titlefont: { color: '#ffffff' }, tickfont: { color: TEXT_COLOR }, gridcolor: GRID_COLOR, tickvals: x, ticktext: x.map(v => `${v}`) },
        yaxis: { title: 'Average IC', titlefont: { color: '#ffffff' }, tickformat: '.1%', tickfont: { color: TEXT_COLOR }, gridcolor: GRID_COLOR, zerolinecolor: GRID_COLOR },
        hoverlabel: { bgcolor: HOVER_BG, bordercolor: HOVER_BORDER, font: { color: '#f3f4f6' } }
    };

    Plotly.react(elementId, [trace], layout, { responsive: true, displayModeBar: false });
}
