import Plotly from 'plotly.js-dist-min';
import type { ScatterPoint, ColorMode, MetricKey } from '../types';

/* eslint-disable @typescript-eslint/no-explicit-any */
type PlotData = any;
type PlotLayout = any;

const PLOT_BG_COLOR = 'rgba(0,0,0,0)';
const PAPER_BG_COLOR = 'rgba(0,0,0,0)';
const TEXT_COLOR = '#e5e7eb';
const GRID_COLOR = '#374151';
const HOVER_BG = '#1f2937';
const HOVER_BORDER = '#374151';

export function renderValuationMap(
    data: ScatterPoint[],
    elementId: string,
    colorBy: ColorMode,
    metricKey: MetricKey
): void {
    const el = document.getElementById(elementId);
    if (!el) return;


    const isResidual = metricKey === 'residualMispricing';
    // Helper to get displayed predicted value based on mode
    const getPredicted = (d: ScatterPoint): number =>
        isResidual ? d.actual * (1 + (d.residualMispricing || 0)) : d.predicted;

    // Swap axes: x = predicted, y = actual
    const x = data.map(d => getPredicted(d));
    const y = data.map(d => d.actual);
    const text = data.map(d => {
        // Negate mispricing so positive = overvalued
        const mVal = -(d[metricKey]);
        const mPct = (mVal > 0 ? '+' : '') + (mVal * 100).toFixed(1) + '%';
        const uncert = ((d.relStd || 0) * 100).toFixed(1) + '%';
        return `<b>${d.company} (${d.ticker})</b><br>${d.sector}<br><br>Predicted: $${getPredicted(d).toFixed(2)}B<br>Actual: $${d.actual.toFixed(2)}B<br>Mispricing: <b>${mPct}</b><br>Uncertainty: ±${uncert}`;
    });

    const customdata = data.map(d => [d.ticker, d.sector, d.company, d[metricKey]]);

    let markerColor: string[] | number[];
    let showScale = false;

    if (colorBy === 'sector') {
        markerColor = data.map(d => d.sectorColor);
    } else {
        // Negate so positive = overvalued
        markerColor = data.map(d => -(d[metricKey]));
        showScale = true;
    }

    const trace: PlotData = {
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
        // After negation: Red (undervalued/negative) to Green (overvalued/positive)
        trace.marker.colorscale = [
            [0, 'rgba(239, 68, 68, 0.9)'],  // Red for negative (undervalued)
            [0.5, 'rgba(156, 163, 175, 0.2)'],
            [1, 'rgba(34, 197, 94, 0.9)']   // Green for positive (overvalued)
        ];
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

    // Use actual market cap (y) values for consistent axis range across modes
    // This ensures axes stay the same when switching between Raw and Size-Neutral
    const actualValues = data.map(d => d.actual).filter(v => v > 0);
    const actualMin = Math.min(...actualValues);
    const actualMax = Math.max(...actualValues);
    const rangeMin = actualMin * 0.5;
    const rangeMax = actualMax * 2;

    const layout: PlotLayout = {
        margin: { t: 20, r: 20, b: 50, l: 60 },
        paper_bgcolor: PAPER_BG_COLOR,
        plot_bgcolor: PLOT_BG_COLOR,
        xaxis: {
            title: 'Predicted Market Cap ($B)',
            type: 'log',
            range: [Math.log10(rangeMin), Math.log10(rangeMax)],
            gridcolor: GRID_COLOR,
            tickfont: { color: TEXT_COLOR },
            titlefont: { color: TEXT_COLOR },
            zerolinecolor: GRID_COLOR
        },
        yaxis: {
            title: 'Actual Market Cap ($B)',
            type: 'log',
            range: [Math.log10(rangeMin), Math.log10(rangeMax)],
            gridcolor: GRID_COLOR,
            tickfont: { color: TEXT_COLOR },
            titlefont: { color: TEXT_COLOR },
            zerolinecolor: GRID_COLOR
        },
        shapes: [
            {
                type: 'line',
                x0: rangeMin, y0: rangeMin,
                x1: rangeMax, y1: rangeMax,
                line: { color: '#4b5563', width: 2, dash: 'dot' }
            }
        ],
        hoverlabel: { bgcolor: HOVER_BG, bordercolor: HOVER_BORDER, font: { color: '#f3f4f6' } },
        showlegend: false
    };

    const config = { responsive: true, displayModeBar: false };

    // Use Plotly.react for fast updates (reuses existing plot if present)
    Plotly.react(elementId, [trace], layout, config);

    // Add click handler only once (check if already attached)
    const plotEl = el as unknown as {
        on: (event: string, handler: (data: unknown) => void) => void;
        _clickHandlerAttached?: boolean;
    };
    if (plotEl.on && !plotEl._clickHandlerAttached) {
        plotEl._clickHandlerAttached = true;
        plotEl.on('plotly_click', (eventData: unknown) => {
            const data = eventData as { points: Array<{ customdata: [string, string, string, number] }> };
            const pt = data.points[0];
            const ticker = pt.customdata[0];
            window.open(`https://www.google.com/finance/quote/${ticker}:NYSE`, '_blank');
        });
    }
}
