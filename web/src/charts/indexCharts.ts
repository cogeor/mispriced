import Plotly from 'plotly.js-dist-min';
import type { IndexChartItem, IndexTimeSeriesPoint } from '../types';

export function renderIndexChart(
    data: IndexChartItem[],
    elementId: string,
    metricKey: 'mispricing' | 'residualMispricing'
) {
    const el = document.getElementById(elementId);
    if (!el) return;

    // Filter out indices with no data
    const validData = data.filter(d => d.count > 0);

    // Sort by mispricing
    validData.sort((a, b) => b[metricKey] - a[metricKey]);

    const x = validData.map(d => d.index);
    const y = validData.map(d => d[metricKey]);
    const text = validData.map(d => {
        const val = d[metricKey];
        return `${d.index}<br>${(val * 100).toFixed(1)}%<br>Count: ${d.count}`;
    });

    const colors = validData.map(d => d[metricKey] > 0 ? '#00a86b' : '#d94545');

    const trace = {
        x: x,
        y: y,
        type: 'bar',
        text: text,
        hoverinfo: 'text',
        marker: {
            color: colors
        }
    };

    const layout = {
        margin: { t: 20, r: 20, b: 60, l: 40 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {
            tickfont: { color: '#8b919a' },
            gridcolor: '#2a2e38'
        },
        yaxis: {
            title: 'Mispricing',
            tickformat: '.0%',
            tickfont: { color: '#8b919a' },
            gridcolor: '#2a2e38'
        },
        hoverlabel: { bgcolor: '#1a1d24', bordercolor: '#2a2e38', font: { color: '#e8eaed' } }
    };

    const config = { responsive: true, displayModeBar: false };
    Plotly.newPlot(elementId, [trace], layout, config);
}

export function renderIndexTimeSeries(
    data: IndexTimeSeriesPoint[],
    elementId: string,
    metricKey: 'mispricing' | 'residualMispricing'
) {
    const el = document.getElementById(elementId);
    if (!el) return;

    if (metricKey === 'residualMispricing') {
        // Warning: Python backend might not provide residual timeseries yet.
        // If data doesn't have it, we can't plot.
        if (data.length > 0 && data[0].residualMispricing === undefined) {
            el.innerHTML = '<div class="text-center text-gray-500 pt-10">Residual Time Series not available</div>';
            return;
        }
    }

    // Group by index
    const statsByIndex = new Map<string, { x: string[], y: number[] }>();
    data.forEach(d => {
        if (!statsByIndex.has(d.index)) {
            statsByIndex.set(d.index, { x: [], y: [] });
        }
        const val = metricKey === 'residualMispricing' ? (d.residualMispricing || 0) : d.mispricing;
        statsByIndex.get(d.index)!.x.push(d.date);
        statsByIndex.get(d.index)!.y.push(val);
    });

    const traces: any[] = [];
    statsByIndex.forEach((series, indexName) => {
        traces.push({
            name: indexName,
            x: series.x,
            y: series.y,
            type: 'scatter',
            mode: 'lines+markers',
            line: { width: 2 }
        });
    });

    const layout = {
        margin: { t: 20, r: 20, b: 40, l: 40 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {
            tickfont: { color: '#8b919a' },
            gridcolor: '#2a2e38'
        },
        yaxis: {
            tickformat: '.0%',
            tickfont: { color: '#8b919a' },
            gridcolor: '#2a2e38'
        },
        legend: {
            orientation: 'h',
            y: 1.1,
            font: { color: '#8b919a' }
        },
        hoverlabel: { bgcolor: '#1a1d24', bordercolor: '#2a2e38', font: { color: '#e8eaed' } }
    };

    const config = { responsive: true, displayModeBar: false };
    Plotly.newPlot(elementId, traces, layout, config);
}
