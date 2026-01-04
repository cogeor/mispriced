import Plotly from 'plotly.js-dist-min';
import type { IndexChartItem, IndexTimeSeriesPoint, SectorTimeSeriesPoint, ScatterPoint, MetricKey } from '../types';

/* eslint-disable @typescript-eslint/no-explicit-any */
type PlotData = any;
type PlotLayout = any;

const PLOT_BG_COLOR = 'rgba(0,0,0,0)';
const PAPER_BG_COLOR = 'rgba(0,0,0,0)';
const TEXT_COLOR = '#e5e7eb';
const GRID_COLOR = '#374151';
const HOVER_BG = '#1f2937';
const HOVER_BORDER = '#374151';

export function renderIndexChart(
    data: IndexChartItem[],
    scatterData: ScatterPoint[],
    elementId: string,
    metricKey: MetricKey
): void {
    const el = document.getElementById(elementId);
    if (!el) return;

    // Filter out Unknown index and ensure count > 0
    const validData = data.filter(d => d.count > 0 && d.index !== 'Unknown');

    // Compute Global from scatter_data (individual stocks) to avoid index overlap issues
    // This gives the true market-wide aggregate
    const validStocks = scatterData.filter(d => d.actual > 0);
    const totalMcap = validStocks.reduce((sum, d) => sum + d.actual, 0);

    // Only add Global if we have scatter data
    let chartData: IndexChartItem[] = [...validData];
    if (totalMcap > 0) {
        const globalMispricing = validStocks.reduce((sum, d) => sum + (d[metricKey] || 0) * d.actual, 0) / totalMcap;
        const globalCount = validStocks.length;
        const globalItem: IndexChartItem = {
            index: 'Global',
            mispricing: metricKey === 'mispricing' ? globalMispricing : 0,
            residualMispricing: metricKey === 'residualMispricing' ? globalMispricing : 0,
            mispricingPct: (globalMispricing * 100).toFixed(1) + '%',
            residualMispricingPct: (globalMispricing * 100).toFixed(1) + '%',
            color: globalMispricing > 0 ? '#ef4444' : '#10b981',
            status: 'aggregate',
            totalActual: '',
            totalPredicted: '',
            count: globalCount,
            officialCount: globalCount
        };
        chartData = [...validData, globalItem];
    }

    // Negate mispricing so positive = overvalued, then sort highest to lowest
    chartData.sort((a, b) => (-b[metricKey]) - (-a[metricKey]));

    const x = chartData.map(d => d.index);
    // Negate so positive = overvalued
    const y = chartData.map(d => -(d[metricKey]));
    // Compute approximate std error: assume typical std of 30% and use 1/sqrt(n)
    const errors = chartData.map(d => 0.30 / Math.sqrt(d.count));

    const text = chartData.map((d, i) => {
        const negated = -(d[metricKey]);
        return `<b>${d.index}</b><br>Mispricing: ${negated > 0 ? '+' : ''}${(negated * 100).toFixed(1)}%<br>Std Error: ±${(errors[i] * 100).toFixed(1)}%<br>Count: ${d.count}`;
    });

    // Color: positive (overvalued) = green, negative (undervalued) = red
    const colors = chartData.map(d => -(d[metricKey]) > 0 ? '#10b981' : '#ef4444');

    const trace: PlotData = {
        x: x,
        y: y,
        type: 'bar',
        hovertext: text,
        hoverinfo: 'text',
        textposition: 'none',
        marker: { color: colors, line: { width: 0 } },
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
        margin: { t: 10, r: 10, b: 120, l: 60 },
        paper_bgcolor: PAPER_BG_COLOR,
        plot_bgcolor: PLOT_BG_COLOR,
        xaxis: { tickfont: { color: TEXT_COLOR, size: 11 }, gridcolor: GRID_COLOR, tickangle: -45, automargin: true },
        yaxis: { title: 'Mispricing', tickformat: '.0%', tickfont: { color: TEXT_COLOR }, titlefont: { color: TEXT_COLOR }, gridcolor: GRID_COLOR },
        hoverlabel: { bgcolor: HOVER_BG, bordercolor: HOVER_BORDER, font: { color: '#f3f4f6' } }
    };

    Plotly.newPlot(elementId, [trace], layout, { responsive: true, displayModeBar: false });
}

// Color palette for distinct line colors
const INDEX_COLORS = [
    '#3b82f6', // blue
    '#ef4444', // red
    '#10b981', // green
    '#f59e0b', // amber
    '#8b5cf6', // violet
    '#ec4899', // pink
    '#06b6d4', // cyan
    '#f97316', // orange
    '#84cc16', // lime
    '#6366f1', // indigo
    '#14b8a6', // teal
    '#a855f7', // purple
];

export interface TimeSeriesTraceData {
    name: string;
    dates: string[];
    values: number[];
    errors: number[];
    count: number; // for sorting by mcap
    color: string;
}

export function renderIndexTimeSeriesMulti(
    data: IndexTimeSeriesPoint[],
    elementId: string,
    metricKey: MetricKey,
    enabledIndices: Set<string>
): void {
    const el = document.getElementById(elementId);
    if (!el) return;

    if (metricKey === 'residualMispricing' && data.length > 0 && data[0].residualMispricing === undefined) {
        el.innerHTML = '<div class="absolute inset-0 flex items-center justify-center text-gray-500">Residual Time Series not available</div>';
        return;
    }

    // Group by index
    // Negate mispricing so positive = overvalued (consistent with bar charts)
    const statsByIndex = new Map<string, { dates: string[]; values: number[]; counts: number[] }>();
    data.forEach(d => {
        if (!statsByIndex.has(d.index)) statsByIndex.set(d.index, { dates: [], values: [], counts: [] });
        const entry = statsByIndex.get(d.index)!;
        const rawVal = metricKey === 'residualMispricing' ? (d.residualMispricing || 0) : d.mispricing;
        entry.dates.push(d.date);
        entry.values.push(-rawVal); // Negate so positive = overvalued
        entry.counts.push(d.count);
    });

    // Sort indices by total count (proxy for mcap)
    const indexTotalCounts: Record<string, number> = {};
    data.forEach(d => {
        indexTotalCounts[d.index] = (indexTotalCounts[d.index] || 0) + d.count;
    });
    const sortedIndices = Object.entries(indexTotalCounts)
        .sort((a, b) => b[1] - a[1])
        .map(([idx]) => idx);

    const traces: PlotData[] = [];
    sortedIndices.forEach((indexName, i) => {
        const entry = statsByIndex.get(indexName)!;
        // Sort by date
        const sorted = entry.dates.map((d, idx) => ({ date: d, value: entry.values[idx], count: entry.counts[idx] }))
            .sort((a, b) => a.date.localeCompare(b.date));

        const dates = sorted.map(s => s.date);
        const values = sorted.map(s => s.value);
        // Estimate error based on count: assume typical std of 30% and use 1/sqrt(count)
        // Minimum 1% error to ensure visibility
        const errors = sorted.map(s => Math.max(0.01, 0.30 / Math.sqrt(s.count)));

        const color = INDEX_COLORS[i % INDEX_COLORS.length];
        const visible = enabledIndices.has(indexName);

        traces.push({
            name: indexName,
            x: dates,
            y: values,
            type: 'scatter',
            mode: 'lines+markers',
            line: { width: 2, color },
            marker: { size: 8, color },
            visible: visible ? true : 'legendonly',
            error_y: {
                type: 'data',
                array: errors,
                visible: true,
                color: color,
                thickness: 2,
                width: 4
            },
            hovertemplate: `<b>${indexName}</b><br>%{x}<br>Mispricing: %{y:.1%}<br>SE: ±%{error_y.array:.1%}<extra></extra>`
        });
    });

    const layout: PlotLayout = {
        margin: { t: 10, r: 20, b: 40, l: 60 },
        paper_bgcolor: PAPER_BG_COLOR,
        plot_bgcolor: PLOT_BG_COLOR,
        xaxis: { tickfont: { color: TEXT_COLOR }, gridcolor: GRID_COLOR, showgrid: false },
        yaxis: {
            title: 'Mispricing',
            tickformat: '.0%',
            tickfont: { color: TEXT_COLOR },
            titlefont: { color: TEXT_COLOR, size: 12 },
            gridcolor: GRID_COLOR
        },
        legend: {
            orientation: 'h',
            y: 1.15,
            x: 0.5,
            xanchor: 'center',
            font: { color: TEXT_COLOR, size: 11 },
            itemclick: 'toggle',
            itemdoubleclick: 'toggleothers'
        },
        hoverlabel: { bgcolor: HOVER_BG, bordercolor: HOVER_BORDER, font: { color: '#f3f4f6' } }
    };

    Plotly.newPlot(elementId, traces, layout, { responsive: true, displayModeBar: false });
}

// Sector colors - distinct from index colors
const SECTOR_COLORS = [
    '#3b82f6', // blue (Global)
    '#10b981', // green
    '#f59e0b', // amber
    '#ef4444', // red
    '#8b5cf6', // violet
    '#ec4899', // pink
    '#06b6d4', // cyan
    '#f97316', // orange
    '#84cc16', // lime
    '#6366f1', // indigo
    '#14b8a6', // teal
    '#a855f7', // purple
];

export function renderSectorTimeSeriesMulti(
    sectorTimeSeries: SectorTimeSeriesPoint[],
    elementId: string,
    metricKey: MetricKey,
    enabledSectors: Set<string>
): void {
    const el = document.getElementById(elementId);
    if (!el) return;

    // Group by sector
    // Filter out 'Unknown' sector
    const filteredData = sectorTimeSeries.filter(d => d.sector && d.sector !== 'Unknown');

    // Negate mispricing so positive = overvalued (consistent with bar charts)
    const statsBySector = new Map<string, { dates: string[]; values: number[]; counts: number[] }>();
    filteredData.forEach(d => {
        if (!statsBySector.has(d.sector)) statsBySector.set(d.sector, { dates: [], values: [], counts: [] });
        const entry = statsBySector.get(d.sector)!;
        const rawVal = metricKey === 'residualMispricing' ? (d.residualMispricing || 0) : d.mispricing;
        entry.dates.push(d.date);
        entry.values.push(-rawVal); // Negate so positive = overvalued
        entry.counts.push(d.count);
    });

    // Compute Global aggregate
    const dateData = new Map<string, { values: number[]; counts: number[] }>();
    filteredData.forEach(d => {
        const rawVal = metricKey === 'residualMispricing' ? (d.residualMispricing || 0) : d.mispricing;
        if (!dateData.has(d.date)) dateData.set(d.date, { values: [], counts: [] });
        const entry = dateData.get(d.date)!;
        entry.values.push(-rawVal); // Negate so positive = overvalued
        entry.counts.push(d.count);
    });

    const globalDates = Array.from(dateData.keys()).sort();
    const globalValues: number[] = [];
    const globalErrors: number[] = [];

    globalDates.forEach(date => {
        const entry = dateData.get(date)!;
        const totalCount = entry.counts.reduce((a, b) => a + b, 0);
        const weightedSum = entry.values.reduce((sum, v, i) => sum + v * entry.counts[i], 0);
        const avg = weightedSum / totalCount;
        // Minimum 1% error to ensure visibility
        const se = Math.max(0.01, 0.30 / Math.sqrt(totalCount));
        globalValues.push(avg);
        globalErrors.push(se);
    });

    // Sort sectors by total count (proxy for mcap)
    const sectorTotalCounts: Record<string, number> = {};
    filteredData.forEach(d => {
        sectorTotalCounts[d.sector] = (sectorTotalCounts[d.sector] || 0) + d.count;
    });
    const sortedSectors = Object.entries(sectorTotalCounts)
        .sort((a, b) => b[1] - a[1])
        .map(([s]) => s);

    const traces: PlotData[] = [];

    // Add Global trace first
    const globalVisible = enabledSectors.has('Global');
    traces.push({
        name: 'Global',
        x: globalDates,
        y: globalValues,
        type: 'scatter',
        mode: 'lines+markers',
        line: { width: 3, color: SECTOR_COLORS[0] },
        marker: { size: 8, color: SECTOR_COLORS[0] },
        visible: globalVisible ? true : 'legendonly',
        error_y: {
            type: 'data',
            array: globalErrors,
            visible: true,
            color: SECTOR_COLORS[0],
            thickness: 2,
            width: 4
        },
        hovertemplate: `<b>Global</b><br>%{x}<br>Mispricing: %{y:.1%}<br>SE: ±%{error_y.array:.1%}<extra></extra>`
    });

    // Add per-sector traces
    sortedSectors.forEach((sectorName, i) => {
        const entry = statsBySector.get(sectorName)!;
        const sorted = entry.dates.map((d, idx) => ({ date: d, value: entry.values[idx], count: entry.counts[idx] }))
            .sort((a, b) => a.date.localeCompare(b.date));

        const dates = sorted.map(s => s.date);
        const values = sorted.map(s => s.value);
        // Minimum 1% error to ensure visibility
        const errors = sorted.map(s => Math.max(0.01, 0.30 / Math.sqrt(s.count)));

        // Use different colors starting from index 1 (skip first color used for Global)
        const color = SECTOR_COLORS[(i + 1) % SECTOR_COLORS.length];
        const visible = enabledSectors.has(sectorName);

        traces.push({
            name: sectorName,
            x: dates,
            y: values,
            type: 'scatter',
            mode: 'lines+markers',
            line: { width: 2, color },
            marker: { size: 6, color },
            visible: visible ? true : 'legendonly',
            error_y: {
                type: 'data',
                array: errors,
                visible: true,
                color: color,
                thickness: 2,
                width: 4
            },
            hovertemplate: `<b>${sectorName}</b><br>%{x}<br>Mispricing: %{y:.1%}<br>SE: ±%{error_y.array:.1%}<extra></extra>`
        });
    });

    const layout: PlotLayout = {
        margin: { t: 10, r: 20, b: 40, l: 60 },
        paper_bgcolor: PAPER_BG_COLOR,
        plot_bgcolor: PLOT_BG_COLOR,
        xaxis: { tickfont: { color: TEXT_COLOR }, gridcolor: GRID_COLOR, showgrid: false },
        yaxis: {
            title: 'Mispricing',
            tickformat: '.0%',
            tickfont: { color: TEXT_COLOR },
            titlefont: { color: TEXT_COLOR, size: 12 },
            gridcolor: GRID_COLOR
        },
        legend: {
            orientation: 'h',
            y: 1.15,
            x: 0.5,
            xanchor: 'center',
            font: { color: TEXT_COLOR, size: 11 },
            itemclick: 'toggle',
            itemdoubleclick: 'toggleothers'
        },
        hoverlabel: { bgcolor: HOVER_BG, bordercolor: HOVER_BORDER, font: { color: '#f3f4f6' } }
    };

    Plotly.newPlot(elementId, traces, layout, { responsive: true, displayModeBar: false });
}
