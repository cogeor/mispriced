import Plotly from 'plotly.js-dist-min';
import type { ScatterPoint } from '../types';

export function renderValuationMap(
    data: ScatterPoint[],
    elementId: string,
    colorBy: 'sector' | 'mispricing',
    metricKey: 'mispricing' | 'residualMispricing'
) {
    const el = document.getElementById(elementId);
    if (!el) return;

    const x = data.map(d => d.actual); // Billions
    const y = data.map(d => d.predicted); // Billions
    const text = data.map(d => {
        const mVal = d[metricKey];
        const mPct = (mVal * 100).toFixed(1) + '%';
        return `${d.company} (${d.ticker})<br>Sector: ${d.sector}<br>Actual: $${d.actual.toFixed(2)}B<br>Predicted: $${d.predicted.toFixed(2)}B<br>Mispricing: ${mPct}`;
    });

    const customdata = data.map(d => [d.ticker, d.sector, d.company, d[metricKey]]);

    let markerColor: any;
    let showScale = false;

    if (colorBy === 'sector') {
        markerColor = data.map(d => d.sectorColor);
    } else {
        const mValues = data.map(d => d[metricKey]); // -1 to 1 usually
        markerColor = mValues;
        showScale = true;
    }

    const trace: any = {
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
            line: { width: 1, color: '#1a1d24' }
        }
    };

    if (showScale) {
        trace.marker.colorscale = 'RdYlGn';
        trace.marker.reversescale = true;
        trace.marker.cmin = -0.5;
        trace.marker.cmax = 0.5;
        trace.marker.colorbar = {
            title: 'Mispricing',
            thickness: 10,
            tickformat: '.0%',
            tickfont: { color: '#8b919a' }
        };
    }

    // Diagonal Line
    const minVal = Math.min(...x, ...y) * 0.8;
    const maxVal = Math.max(...x, ...y) * 1.2;

    const layout = {
        margin: { t: 20, r: 20, b: 40, l: 60 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {
            title: 'Actual Market Cap ($B)',
            type: 'log',
            gridcolor: '#2a2e38',
            zerolinecolor: '#2a2e38',
            tickfont: { color: '#8b919a' },
            titlefont: { color: '#8b919a' }
        },
        yaxis: {
            title: 'Predicted Market Cap ($B)',
            type: 'log',
            gridcolor: '#2a2e38',
            zerolinecolor: '#2a2e38',
            tickfont: { color: '#8b919a' },
            titlefont: { color: '#8b919a' }
        },
        shapes: [
            {
                type: 'line',
                x0: minVal, y0: minVal,
                x1: maxVal, y1: maxVal,
                line: { color: '#2a2e38', width: 2, dash: 'dot' }
            }
        ],
        hoverlabel: { bgcolor: '#1a1d24', bordercolor: '#2a2e38', font: { color: '#e8eaed' } }
    };

    const config = { responsive: true, displayModeBar: false };

    Plotly.newPlot(elementId, [trace], layout, config);

    // Click event
    (el as any).removeAllListeners?.('plotly_click'); // Remove if generic Element doesn't have it, but Plotly adds it
    (el as any).on('plotly_click', (data: any) => {
        const pt = data.points[0];
        const ticker = pt.customdata[0];
        window.open(`https://www.google.com/finance/quote/${ticker}:NYSE`, '_blank');
    });
}
