import './style.css';
import type { DashboardData, ScatterPoint, BacktestItem, MispricingMode, ColorMode, MetricKey, QuarterValuationData, CoreData, ScatterData, BacktestPayload } from './types';
import { aggregateToSummary, formatQuarter, formatMcap } from './utils';
import { renderValuationMap } from './charts/valuationMap';
import { renderIndexChart, renderIndexTimeSeriesMulti, renderSectorTimeSeriesMulti } from './charts/indexCharts';
import { buildICHeatmap, renderSignalDecay, type HorizonDecayItem } from './charts/signalQuality';
import { renderSectorChart, renderUncertaintyChart, renderSizePremiumChart } from './charts/miscCharts';

// Dashboard data assembled from split files
let dashboardData: DashboardData | null = null;
let coreDataLoaded = false;
let scatterDataLoaded = false;
let backtestDataLoaded = false;
let mispricingMode: MispricingMode = 'raw';
const colorBy: ColorMode = 'mispricing'; // Always use mispricing color
let selectedQuarter: string | null = null;
// Default enabled indices/sectors for time series charts (set after data loads)
let enabledIndices: Set<string> = new Set();
let enabledSectors: Set<string> = new Set(['Global']);

// Cache for lazily loaded quarter data
const quarterDataCache: Map<string, QuarterValuationData> = new Map();

// Cache current scatter data for fast mode switching
let currentScatterData: ScatterPoint[] = [];

// Stock table state
let stockSortColumn: string = 'actual';
let stockSortAscending: boolean = false;
let stockSearchQuery: string = '';
let allStockData: ScatterPoint[] = [];
let stockPage: number = 0;
const STOCKS_PER_PAGE = 100;
let filteredStockData: ScatterPoint[] = [];

// Loading helpers
function hideLoadingOverlay(): void {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.classList.add('fade-out');
        setTimeout(() => overlay.remove(), 300);
    }
}

function nextFrame(): Promise<void> {
    return new Promise(resolve => requestAnimationFrame(() => resolve()));
}

function showChartSpinner(chartId: string): void {
    const el = document.getElementById(chartId);
    if (el) {
        el.innerHTML = '<div class="chart-loading"><div class="spinner"></div></div>';
    }
}

function clearChartSpinner(chartId: string, message: string = ''): void {
    const el = document.getElementById(chartId);
    if (el) {
        el.innerHTML = message ? `<div class="chart-loading" style="color: #6b7280; font-size: 14px;">${message}</div>` : '';
    }
}

function updateLoadingText(text: string): void {
    const el = document.querySelector('#loadingOverlay .loading-text');
    if (el) el.textContent = text;
}

async function init(): Promise<void> {
    try {
        // Initialize empty dashboard data structure
        dashboardData = {
            generated_at: '',
            stats: {} as DashboardData['stats'],
            scatter_data: [],
            index_chart_data: [],
            sector_breakdown: [],
            backtest_data: { sector_ts: [], index_ts: [], sector_summary: [], index_summary: [], horizon: [] },
            index_timeseries: [],
            sector_timeseries: [],
            size_premium_curve: [],
            size_coefficients: [],
            available_quarters: [],
        };

        // Setup event listeners early
        setupEventListeners();

        // PHASE 1: Load core data (~35KB) - fast, enables header + basic charts
        updateLoadingText('Loading core data...');
        const coreResponse = await fetch('/core.json');
        if (!coreResponse.ok) throw new Error(`HTTP ${coreResponse.status} loading core data`);
        const coreData: CoreData = await coreResponse.json();

        // Merge core data into dashboardData
        dashboardData.generated_at = coreData.generated_at;
        dashboardData.stats = coreData.stats;
        dashboardData.index_chart_data = coreData.index_chart_data;
        dashboardData.sector_breakdown = coreData.sector_breakdown;
        dashboardData.index_timeseries = coreData.index_timeseries;
        dashboardData.sector_timeseries = coreData.sector_timeseries;
        dashboardData.size_premium_curve = coreData.size_premium_curve;
        dashboardData.size_coefficients = coreData.size_coefficients;
        dashboardData.available_quarters = coreData.available_quarters;
        dashboardData.size_correction_model = coreData.size_correction_model;
        coreDataLoaded = true;

        // Hide overlay and render core charts immediately
        hideLoadingOverlay();
        populateQuarterDropdown();
        updateStats();
        await renderCoreCharts();

        // PHASE 2: Load scatter data (~1.1MB) in background
        showChartSpinner('valuationChart');
        showChartSpinner('sectorChart');
        showChartSpinner('uncertaintyChart');
        const scatterPromise = fetch('/scatter.json')
            .then(r => r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)))
            .then(async (data: ScatterData) => {
                dashboardData!.scatter_data = data.scatter_data;
                scatterDataLoaded = true;
                currentScatterData = data.scatter_data;
                applySizeCorrection(currentScatterData);
                await renderScatterCharts();
            })
            .catch(e => {
                console.error('Failed to load scatter data:', e);
                // Clear spinners on error
                clearChartSpinner('valuationChart', 'Failed to load');
                clearChartSpinner('sectorChart', 'Failed to load');
                clearChartSpinner('uncertaintyChart', 'Failed to load');
            });

        // PHASE 3: Load backtest data (~250KB) in background
        showChartSpinner('icSectorChart');
        showChartSpinner('icIndexChart');
        showChartSpinner('icSectorDecayChart');
        showChartSpinner('icIndexDecayChart');
        const backtestPromise = fetch('/backtest.json')
            .then(r => r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)))
            .then(async (data: BacktestPayload) => {
                dashboardData!.backtest_data = data.backtest_data;
                backtestDataLoaded = true;
                await renderBacktestCharts();
            })
            .catch(e => {
                console.error('Failed to load backtest data:', e);
                // Clear spinners on error
                clearChartSpinner('icSectorChart', 'Failed to load');
                clearChartSpinner('icIndexChart', 'Failed to load');
                clearChartSpinner('icSectorDecayChart', 'Failed to load');
                clearChartSpinner('icIndexDecayChart', 'Failed to load');
            });

        // Wait for all data to load
        await Promise.all([scatterPromise, backtestPromise]);

    } catch (e) {
        console.error(e);
        hideLoadingOverlay();
        const app = document.getElementById('app');
        if (app) {
            app.innerHTML = `
                <div style="display: flex; justify-content: center; align-items: center; height: 100vh;">
                    <div style="text-align: center; padding: 2rem; background: rgba(127, 29, 29, 0.2); border: 1px solid rgba(248, 113, 113, 0.5); border-radius: 0.5rem; max-width: 32rem;">
                        <h2 style="font-size: 1.5rem; font-weight: 700; color: #ef4444; margin-bottom: 0.5rem;">Error Loading Data</h2>
                        <p style="color: #d1d5db; margin-bottom: 1rem;">${e instanceof Error ? e.message : String(e)}</p>
                        <p style="font-size: 0.875rem; color: #6b7280;">Ensure data files exist in the 'public' folder.</p>
                    </div>
                </div>`;
        }
    }
}

async function renderCoreCharts(): Promise<void> {
    if (!dashboardData || !coreDataLoaded) return;
    const metricKey: MetricKey = mispricingMode === 'sizeNeutral' ? 'residualMispricing' : 'mispricing';

    // Initialize default enabled indices
    if (enabledIndices.size === 0 && dashboardData.index_timeseries.length > 0) {
        const indexCounts: Record<string, number> = {};
        dashboardData.index_timeseries.forEach(d => {
            indexCounts[d.index] = (indexCounts[d.index] || 0) + d.count;
        });
        const sortedIndices = Object.entries(indexCounts)
            .sort((a, b) => b[1] - a[1])
            .map(([idx]) => idx);
        if (sortedIndices.length > 0) {
            enabledIndices.add(sortedIndices[0]);
        }
    }

    await nextFrame();
    // Index bar chart (uses index_chart_data, works without scatter)
    renderIndexChart(dashboardData.index_chart_data, [], 'indexChart', metricKey);

    await nextFrame();
    // Time series charts
    renderIndexTimeSeriesMulti(dashboardData.index_timeseries, 'indexTimeSeriesChart', metricKey, enabledIndices);
    renderSectorTimeSeriesMulti(dashboardData.sector_timeseries, 'sectorTimeSeriesChart', metricKey, enabledSectors);

    await nextFrame();
    // Size premium chart
    renderSizePremiumChart(dashboardData.size_coefficients, 'sizePremiumChart');
}

async function renderScatterCharts(): Promise<void> {
    if (!dashboardData || !scatterDataLoaded) return;
    const metricKey: MetricKey = mispricingMode === 'sizeNeutral' ? 'residualMispricing' : 'mispricing';

    // Update stats now that scatter data is loaded
    updateStats();

    await nextFrame();
    renderValuationMap(currentScatterData, 'valuationChart', colorBy, metricKey);

    await nextFrame();
    renderSectorChart(currentScatterData, 'sectorChart', metricKey);
    // Re-render index chart with scatter data for richer tooltips
    renderIndexChart(dashboardData.index_chart_data, currentScatterData, 'indexChart', metricKey);

    await nextFrame();
    renderUncertaintyChart(currentScatterData, 'uncertaintyChart', metricKey);

    await nextFrame();
    allStockData = [...currentScatterData];
    renderStockTable(metricKey);
}

async function renderBacktestCharts(): Promise<void> {
    if (!dashboardData || !backtestDataLoaded) return;
    const backtestMetricKey = mispricingMode === 'sizeNeutral' ? 'residual' : 'raw';

    // Compute sector mcap for ordering
    const sectorMcap = new Map<string, number>();
    if (scatterDataLoaded) {
        dashboardData.scatter_data.forEach(d => {
            sectorMcap.set(d.sector, (sectorMcap.get(d.sector) || 0) + d.actual);
        });
        const totalMcap = dashboardData.scatter_data.reduce((sum, d) => sum + d.actual, 0);
        sectorMcap.set('Global', totalMcap);
    }

    // Compute index mcap for ordering
    const indexMcap = new Map<string, number>();
    dashboardData.index_chart_data.forEach(d => {
        indexMcap.set(d.index, d.count);
    });

    await nextFrame();
    const sectorResult = aggregateToSummary(dashboardData.backtest_data.sector_ts, selectedQuarter);
    buildICHeatmap(sectorResult.data, 'icSectorChart', backtestMetricKey, sectorResult.quarter, sectorMcap, true);
    const sectorDecayData = recomputeDecay(dashboardData.backtest_data.sector_ts, backtestMetricKey, sectorResult.quarter);
    renderSignalDecay(sectorDecayData, 'icSectorDecayChart');

    await nextFrame();
    const indexResult = aggregateToSummary(dashboardData.backtest_data.index_ts, selectedQuarter);
    buildICHeatmap(indexResult.data, 'icIndexChart', backtestMetricKey, indexResult.quarter, indexMcap, false);
    const indexDecayData = recomputeDecay(dashboardData.backtest_data.index_ts, backtestMetricKey, indexResult.quarter);
    renderSignalDecay(indexDecayData, 'icIndexDecayChart');

    // Show note if backtest quarter differs from selected quarter
    const backtestNote = document.getElementById('backtestQuarterNote');
    if (backtestNote) {
        const usedQuarter = sectorResult.quarter || indexResult.quarter;
        if (selectedQuarter && usedQuarter && selectedQuarter !== usedQuarter) {
            backtestNote.textContent = `⚠ Showing ${formatQuarter(usedQuarter)} data (${formatQuarter(selectedQuarter)} not yet available)`;
            backtestNote.style.display = 'inline';
        } else {
            backtestNote.style.display = 'none';
        }
    }
}

function updateStats(): void {
    if (!dashboardData) return;

    const s = dashboardData.stats;
    setText('generatedAt', new Date(dashboardData.generated_at).toLocaleString());

    // Format quarter as Q1-Q4
    const quarter = s.quarter_date ? formatQuarter(s.quarter_date) : '';
    setText('quarterDate', quarter);

    // Always show LRAE as the metric
    setText('metricDesc', 'LRAE');

    setText('indicesTracked', s.indices_tracked.toLocaleString());

    // Use currentScatterData for dynamic stats (updates when quarter changes)
    const scatterData = currentScatterData.length > 0 ? currentScatterData : dashboardData.scatter_data;

    // If no scatter data yet, show loading state
    if (scatterData.length === 0) {
        setText('totalTickers', '...');
        setText('totalMcap', '...');
        setText('sectorsTracked', '...');
        setText('medianMispricing', '...');
        setText('avgMispricing', '...');
        return;
    }

    // Compute stats from filtered data (not static stats)
    setText('totalTickers', scatterData.length.toLocaleString());
    const totalMcapT = scatterData.reduce((sum, d) => sum + d.actual, 0) / 1000; // B to T
    setText('totalMcap', totalMcapT.toFixed(1));

    // Calculate and display sectors tracked
    if (scatterData) {
        const uniqueSectors = new Set(scatterData.map(d => d.sector));
        setText('sectorsTracked', uniqueSectors.size.toString());
    }

    // Compute dynamic stats based on mode
    const key: MetricKey = mispricingMode === 'sizeNeutral' ? 'residualMispricing' : 'mispricing';

    // Filter valid data points
    const validData = scatterData.filter(d =>
        d[key] !== undefined && d[key] !== null && !isNaN(d[key]) && d.actual > 0
    );
    const n = validData.length;
    if (n === 0) {
        setText('medianMispricing', 'N/A');
        setText('avgMispricing', 'N/A');
        return;
    }

    // Simple median (equal weight per stock)
    const values = validData.map(d => d[key]).sort((a, b) => a - b);
    const median = n % 2 === 0
        ? (values[n / 2 - 1] + values[n / 2]) / 2
        : values[Math.floor(n / 2)];

    // Market-cap weighted average: Sum(mispricing × mcap) / Sum(mcap)
    const totalMcap = validData.reduce((sum, d) => sum + d.actual, 0);
    const weightedSum = validData.reduce((sum, d) => sum + d[key] * d.actual, 0);
    const weightedAvg = weightedSum / totalMcap;

    // Negate so positive = overvalued
    const negMedian = -median;
    const negAvg = -weightedAvg;
    setText('medianMispricing', (negMedian > 0 ? '+' : '') + (negMedian * 100).toFixed(1) + '%');
    // Note: HTML already has % after this span
    setText('avgMispricing', (negAvg > 0 ? '+' : '') + (negAvg * 100).toFixed(1));
}

/**
 * Progressive rendering of all charts with current data.
 * Uses requestAnimationFrame to avoid blocking the main thread.
 * Called by updateCharts() after data fetch and by setMode() for fast switching.
 */
async function renderAllCharts(metricKey: MetricKey, progressive: boolean = true): Promise<void> {
    if (!dashboardData) return;

    // Helper to render with optional frame break
    const render = async (fn: () => void) => {
        if (progressive) await nextFrame();
        fn();
    };

    // Main valuation map first (most important, above fold)
    await render(() => renderValuationMap(currentScatterData, 'valuationChart', colorBy, metricKey));

    // Sector and index charts (visible on scroll)
    await render(() => {
        renderSectorChart(currentScatterData, 'sectorChart', metricKey);
        renderIndexChart(dashboardData!.index_chart_data, currentScatterData, 'indexChart', metricKey);
    });

    // Initialize default enabled indices (highest mcap) if not set
    if (enabledIndices.size === 0 && dashboardData.index_timeseries.length > 0) {
        const indexCounts: Record<string, number> = {};
        dashboardData.index_timeseries.forEach(d => {
            indexCounts[d.index] = (indexCounts[d.index] || 0) + d.count;
        });
        const sortedIndices = Object.entries(indexCounts)
            .sort((a, b) => b[1] - a[1])
            .map(([idx]) => idx);
        if (sortedIndices.length > 0) {
            enabledIndices.add(sortedIndices[0]); // Add highest mcap index
        }
    }

    // Time series charts
    await render(() => {
        renderIndexTimeSeriesMulti(dashboardData!.index_timeseries, 'indexTimeSeriesChart', metricKey, enabledIndices);
        renderSectorTimeSeriesMulti(dashboardData!.sector_timeseries, 'sectorTimeSeriesChart', metricKey, enabledSectors);
    });

    const backtestMetricKey = mispricingMode === 'sizeNeutral' ? 'residual' : 'raw';

    // Compute sector mcap for ordering
    const sectorMcap = new Map<string, number>();
    dashboardData.scatter_data.forEach(d => {
        sectorMcap.set(d.sector, (sectorMcap.get(d.sector) || 0) + d.actual);
    });
    // Add Global with total mcap
    const totalMcap = dashboardData.scatter_data.reduce((sum, d) => sum + d.actual, 0);
    sectorMcap.set('Global', totalMcap);

    // Compute index mcap for ordering (from index_chart_data)
    const indexMcap = new Map<string, number>();
    dashboardData.index_chart_data.forEach(d => {
        // Use count as proxy since actual mcap not directly available
        indexMcap.set(d.index, d.count);
    });

    // Backtest charts (lower priority, below fold)
    await render(() => {
        const sectorResult = aggregateToSummary(dashboardData!.backtest_data.sector_ts, selectedQuarter);
        buildICHeatmap(sectorResult.data, 'icSectorChart', backtestMetricKey, sectorResult.quarter, sectorMcap, true);
        const sectorDecayData = recomputeDecay(dashboardData!.backtest_data.sector_ts, backtestMetricKey, sectorResult.quarter);
        renderSignalDecay(sectorDecayData, 'icSectorDecayChart');

        const indexResult = aggregateToSummary(dashboardData!.backtest_data.index_ts, selectedQuarter);
        buildICHeatmap(indexResult.data, 'icIndexChart', backtestMetricKey, indexResult.quarter, indexMcap, false);
        const indexDecayData = recomputeDecay(dashboardData!.backtest_data.index_ts, backtestMetricKey, indexResult.quarter);
        renderSignalDecay(indexDecayData, 'icIndexDecayChart');

        // Show note if backtest quarter differs from selected quarter
        const backtestNote = document.getElementById('backtestQuarterNote');
        if (backtestNote) {
            const usedQuarter = sectorResult.quarter || indexResult.quarter;
            if (selectedQuarter && usedQuarter && selectedQuarter !== usedQuarter) {
                backtestNote.textContent = `⚠ Showing ${formatQuarter(usedQuarter)} data (${formatQuarter(selectedQuarter)} not yet available)`;
                backtestNote.style.display = 'inline';
            } else {
                backtestNote.style.display = 'none';
            }
        }
    });

    // Uncertainty and size premium charts
    await render(() => {
        renderUncertaintyChart(currentScatterData, 'uncertaintyChart', metricKey);
        renderSizePremiumChart(dashboardData!.size_coefficients, 'sizePremiumChart');
    });

    // Stock table last (can be deferred)
    await render(() => {
        allStockData = [...currentScatterData];
        renderStockTable(metricKey);
    });
}

/**
 * Fetch data for selected quarter and render all charts.
 * Called on quarter change.
 */
async function updateCharts(): Promise<void> {
    if (!dashboardData) return;

    // Get scatter data for selected quarter
    currentScatterData = dashboardData.scatter_data;
    const latestQuarter = dashboardData.available_quarters?.[0];

    // If a non-latest quarter is selected, fetch its data lazily
    if (selectedQuarter && selectedQuarter !== latestQuarter) {
        showChartSpinner('valuationChart');
        const quarterData = await fetchQuarterData(selectedQuarter);
        if (quarterData) {
            currentScatterData = quarterData.scatter_data;
        }
    }

    // Render all charts with fetched data
    applySizeCorrection(currentScatterData);

    // Use progressive rendering
    await renderScatterCharts();
    await renderCoreCharts();
    if (backtestDataLoaded) await renderBacktestCharts();
}

function applySizeCorrection(data: ScatterPoint[]): void {
    if (!dashboardData?.size_correction_model) return;

    // The model in dashboardData applies to the latest quarter.
    const latestQuarter = dashboardData.available_quarters?.[0];
    if (selectedQuarter && latestQuarter && selectedQuarter !== latestQuarter) {
        // For historical quarters, we rely on the pre-calculated residuals in the JSON
        return;
    }

    const model = dashboardData.size_correction_model; // { coefficients: [a, b, c], ... }
    const [a, b, c] = model.coefficients;

    // Apply model: log(1 + mispricing) = a * log(mcap)^2 + b * log(mcap) + c
    let count = 0;
    data.forEach(d => {
        // d.actual is in Billions. Model fits on log(Raw Dollars).
        const mcapRaw = d.actual * 1e9;
        if (mcapRaw <= 0) return;

        const logMcap = Math.log(mcapRaw);

        // Expected log(1+m)
        const expectedLog = a * logMcap * logMcap + b * logMcap + c;

        // Actual log(1+m)
        // d.mispricing is the raw relative error
        const actualLog = Math.log(1 + d.mispricing);

        // Residual in log space
        const residualLog = actualLog - expectedLog;

        // Convert back to percentage
        // residual = exp(residual_log) - 1
        const residual = Math.exp(residualLog) - 1;

        d.residualMispricing = residual;
        d.residualMispricingPct = (residual * 100).toFixed(1) + '%';
        count++;
    });

    if (count > 0) {
        console.log(`Applied client-side size correction (log-quadratic) to ${count} points.`);
    }
}

function recomputeDecay(ts: BacktestItem[], metric: string, quarter: string | null = null): HorizonDecayItem[] {
    let filtered = ts.filter(d => (d.metric || 'raw') === metric);
    if (quarter) {
        filtered = filtered.filter(d => d.quarter === quarter);
    }
    const byHorizon = new Map<number, number[]>();
    filtered.forEach(d => {
        if (!byHorizon.has(d.horizon)) byHorizon.set(d.horizon, []);
        byHorizon.get(d.horizon)!.push(d.ic);
    });
    return Array.from(byHorizon.entries()).map(([h, ics]) => {
        const avg = ics.reduce((a, b) => a + b, 0) / ics.length;
        const variance = ics.reduce((sum, ic) => sum + Math.pow(ic - avg, 2), 0) / ics.length;
        const std = Math.sqrt(variance);
        const se = std / Math.sqrt(ics.length);
        return {
            horizon: h,
            avg_ic: avg,
            std_error: se
        };
    });
}

/**
 * Fetch quarter data lazily from separate JSON file.
 * Returns cached data if already fetched.
 */
async function fetchQuarterData(quarter: string): Promise<QuarterValuationData | null> {
    // Check cache first
    if (quarterDataCache.has(quarter)) {
        return quarterDataCache.get(quarter)!;
    }

    try {
        const response = await fetch(`/quarters/${quarter}.json`);
        if (!response.ok) {
            console.warn(`Failed to load quarter data for ${quarter}: HTTP ${response.status}`);
            return null;
        }
        const data: QuarterValuationData = await response.json();
        quarterDataCache.set(quarter, data);
        return data;
    } catch (e) {
        console.error(`Error fetching quarter data for ${quarter}:`, e);
        return null;
    }
}

function populateQuarterDropdown(): void {
    const select = document.getElementById('quarterSelect') as HTMLSelectElement | null;
    if (!select || select.dataset.populated || !dashboardData) return;

    // Use available_quarters from dashboard data (already sorted desc)
    const quarters = dashboardData.available_quarters || [];

    if (quarters.length === 0) return;

    // Default to latest (first) quarter
    if (!selectedQuarter) selectedQuarter = quarters[0];

    select.innerHTML = quarters.map(q =>
        `<option value="${q}" ${q === selectedQuarter ? 'selected' : ''}>${formatQuarter(q)}</option>`
    ).join('');

    select.dataset.populated = 'true';
}

function renderStockTable(metricKey: MetricKey, resetPage: boolean = true): void {
    const el = document.getElementById('stockTable');
    if (!el) return;

    if (resetPage) stockPage = 0;

    // Filter by search
    let filtered = allStockData;
    if (stockSearchQuery) {
        const q = stockSearchQuery.toLowerCase();
        filtered = filtered.filter(d =>
            d.ticker.toLowerCase().includes(q) ||
            (d.company && d.company.toLowerCase().includes(q)) ||
            d.sector.toLowerCase().includes(q)
        );
    }

    // Sort
    filteredStockData = [...filtered].sort((a, b) => {
        let valA: number | string = 0;
        let valB: number | string = 0;

        switch (stockSortColumn) {
            case 'ticker':
                valA = a.ticker;
                valB = b.ticker;
                break;
            case 'company':
                valA = a.company || a.ticker;
                valB = b.company || b.ticker;
                break;
            case 'sector':
                valA = a.sector;
                valB = b.sector;
                break;
            case 'actual':
                valA = a.actual;
                valB = b.actual;
                break;
            case 'predicted':
                // Sort by effective predicted (actual * (1 + mispricing))
                valA = a.actual * (1 + (a[metricKey] || 0));
                valB = b.actual * (1 + (b[metricKey] || 0));
                break;
            case 'mispricing':
                valA = a[metricKey] || 0;
                valB = b[metricKey] || 0;
                break;
            case 'uncertainty':
                valA = a.relStd;
                valB = b.relStd;
                break;
        }

        if (typeof valA === 'string' && typeof valB === 'string') {
            return stockSortAscending ? valA.localeCompare(valB) : valB.localeCompare(valA);
        }
        return stockSortAscending ? (valA as number) - (valB as number) : (valB as number) - (valA as number);
    });

    // Paginate - only render current page
    const startIdx = stockPage * STOCKS_PER_PAGE;
    const pageData = filteredStockData.slice(startIdx, startIdx + STOCKS_PER_PAGE);

    el.innerHTML = pageData.map(r => {
        // Negate so positive = overvalued
        const mispricingVal = -(r[metricKey] || 0);
        // positive (overvalued) = green, negative (undervalued) = red
        const mispricingClass = mispricingVal > 0 ? 'text-success' : 'text-danger';
        // Calculate predicted based on mode: predicted = actual * (1 + mispricing)
        // For size-neutral mode, use residualMispricing to get size-adjusted prediction
        const effectivePredicted = r.actual * (1 + (r[metricKey] || 0));
        return `
            <tr class="hover:bg-gray-800/50 cursor-pointer" onclick="window.open('https://www.google.com/finance/quote/${r.ticker}', '_blank')">
                <td class="py-2 px-4 font-bold text-white">${r.ticker}</td>
                <td class="py-2 px-2 truncate text-gray-400" style="max-width: 150px;" title="${r.company || r.ticker}">${r.company || r.ticker}</td>
                <td class="py-2 px-2 truncate text-gray-500" style="max-width: 100px;" title="${r.sector}">${r.sector}</td>
                <td class="py-2 px-2 text-right text-gray-400">$${formatMcap(r.actual)}</td>
                <td class="py-2 px-2 text-right text-gray-400">$${formatMcap(effectivePredicted)}</td>
                <td class="py-2 px-2 text-right font-bold ${mispricingClass}">${mispricingVal > 0 ? '+' : ''}${(mispricingVal * 100).toFixed(1)}%</td>
                <td class="py-2 px-4 text-right font-mono text-gray-500">${(r.relStd * 100).toFixed(1)}%</td>
            </tr>
        `;
    }).join('');

    // Update sort indicators
    updateSortIndicators();

    // Update pagination info
    updatePaginationControls();
}

function updatePaginationControls(): void {
    const totalPages = Math.ceil(filteredStockData.length / STOCKS_PER_PAGE);
    const startIdx = stockPage * STOCKS_PER_PAGE + 1;
    const endIdx = Math.min((stockPage + 1) * STOCKS_PER_PAGE, filteredStockData.length);

    const paginationEl = document.getElementById('paginationInfo');
    if (paginationEl) {
        paginationEl.textContent = `${startIdx}-${endIdx} of ${filteredStockData.length}`;
    }

    const prevBtn = document.getElementById('prevPage') as HTMLButtonElement | null;
    const nextBtn = document.getElementById('nextPage') as HTMLButtonElement | null;

    if (prevBtn) {
        prevBtn.disabled = stockPage === 0;
        prevBtn.classList.toggle('opacity-50', stockPage === 0);
    }
    if (nextBtn) {
        nextBtn.disabled = stockPage >= totalPages - 1;
        nextBtn.classList.toggle('opacity-50', stockPage >= totalPages - 1);
    }
}

function updateSortIndicators(): void {
    document.querySelectorAll('.sortable-header').forEach(header => {
        const col = header.getAttribute('data-sort');
        const indicator = header.querySelector('.sort-indicator');
        if (indicator) {
            if (col === stockSortColumn) {
                indicator.classList.add('active');
                indicator.textContent = stockSortAscending ? '↑' : '↓';
            } else {
                indicator.classList.remove('active');
                indicator.textContent = '↕';
            }
        }
    });
}

function exportTableToCsv(): void {
    if (!dashboardData) return;

    const metricKey: MetricKey = mispricingMode === 'sizeNeutral' ? 'residualMispricing' : 'mispricing';

    // Export top 100 by market cap (or current filtered view if smaller)
    const dataToExport = [...filteredStockData]
        .sort((a, b) => b.actual - a.actual)
        .slice(0, 100);

    const headers = ['Ticker', 'Company', 'Sector', 'Actual ($B)', 'Predicted ($B)', 'Mispricing (%)', 'Uncertainty (%)'];
    const rows = dataToExport.map(r => {
        // Calculate effective predicted based on mode: actual * (1 + mispricing)
        const effectivePredicted = r.actual * (1 + (r[metricKey] || 0));
        return [
            r.ticker,
            r.company || r.ticker,
            r.sector,
            r.actual.toFixed(2),
            effectivePredicted.toFixed(2),
            // Negate so positive = overvalued
            (-(r[metricKey] || 0) * 100).toFixed(2),
            (r.relStd * 100).toFixed(2)
        ];
    });

    const csv = [headers.join(','), ...rows.map(row => row.map(cell => `"${cell}"`).join(','))].join('\n');

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `mispriced_stocks_${new Date().toISOString().split('T')[0]}.csv`;
    link.click();
    URL.revokeObjectURL(url);
}

function setText(id: string, val: string): void {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

function setupEventListeners(): void {
    // Mode toggle
    const modeRaw = document.getElementById('modeRaw');
    const modeSizeNeutral = document.getElementById('modeSizeNeutral');

    if (modeRaw) modeRaw.addEventListener('click', () => setMode('raw'));
    if (modeSizeNeutral) modeSizeNeutral.addEventListener('click', () => setMode('sizeNeutral'));

    // Sticky mode toggle - use right positioning relative to parent card
    const modeToggle = document.getElementById('modeToggle');
    if (modeToggle) {
        const parentCard = modeToggle.closest('.stat-card');

        const getFixedRight = (): number => {
            // Get parent card's right edge, then add the 16px (right-4) offset
            if (parentCard) {
                const parentRect = parentCard.getBoundingClientRect();
                const viewportWidth = document.documentElement.clientWidth;
                // Parent's right edge distance from viewport + original 16px padding
                return viewportWidth - parentRect.right + 16;
            }
            return 16;
        };

        const onScroll = () => {
            if (parentCard) {
                const parentRect = parentCard.getBoundingClientRect();
                // Become sticky when parent card's bottom edge scrolls past 80px from viewport top
                if (parentRect.bottom < 80) {
                    if (!modeToggle.classList.contains('sticky')) {
                        modeToggle.style.right = `${getFixedRight()}px`;
                        modeToggle.classList.add('sticky');
                    }
                } else {
                    if (modeToggle.classList.contains('sticky')) {
                        modeToggle.style.right = '';
                        modeToggle.classList.remove('sticky');
                    }
                }
            }
        };

        // Recalculate on resize
        const onResize = () => {
            if (modeToggle.classList.contains('sticky')) {
                modeToggle.style.right = `${getFixedRight()}px`;
            }
        };

        window.addEventListener('scroll', onScroll, { passive: true });
        window.addEventListener('resize', onResize, { passive: true });
    }

    // Quarter select - fetch quarter data lazily
    const quarterSelect = document.getElementById('quarterSelect') as HTMLSelectElement | null;
    if (quarterSelect) {
        quarterSelect.addEventListener('change', async (e) => {
            const newQuarter = (e.target as HTMLSelectElement).value || null;
            if (newQuarter === selectedQuarter) return;

            selectedQuarter = newQuarter;

            // Show loading state
            quarterSelect.disabled = true;
            quarterSelect.style.opacity = '0.5';

            try {
                await updateCharts();
            } finally {
                quarterSelect.disabled = false;
                quarterSelect.style.opacity = '1';
            }
        });
    }

    // Stock search
    const stockSearch = document.getElementById('stockSearch') as HTMLInputElement | null;
    if (stockSearch) {
        stockSearch.addEventListener('input', (e) => {
            stockSearchQuery = (e.target as HTMLInputElement).value;
            const metricKey: MetricKey = mispricingMode === 'sizeNeutral' ? 'residualMispricing' : 'mispricing';
            renderStockTable(metricKey);
        });
    }

    // Export CSV
    const exportBtn = document.getElementById('exportCsv');
    if (exportBtn) {
        exportBtn.addEventListener('click', exportTableToCsv);
    }

    // Sortable headers
    document.querySelectorAll('.sortable-header').forEach(header => {
        header.addEventListener('click', () => {
            const col = header.getAttribute('data-sort');
            if (col) {
                if (stockSortColumn === col) {
                    stockSortAscending = !stockSortAscending;
                } else {
                    stockSortColumn = col;
                    stockSortAscending = col === 'mispricing' ? true : true; // Default ascending, except mispricing starts with undervalued
                }
                const metricKey: MetricKey = mispricingMode === 'sizeNeutral' ? 'residualMispricing' : 'mispricing';
                renderStockTable(metricKey);
            }
        });
    });

    // Pagination
    const prevBtn = document.getElementById('prevPage');
    const nextBtn = document.getElementById('nextPage');

    if (prevBtn) {
        prevBtn.addEventListener('click', () => {
            if (stockPage > 0) {
                stockPage--;
                const metricKey: MetricKey = mispricingMode === 'sizeNeutral' ? 'residualMispricing' : 'mispricing';
                renderStockTable(metricKey, false);
            }
        });
    }

    if (nextBtn) {
        nextBtn.addEventListener('click', () => {
            const totalPages = Math.ceil(filteredStockData.length / STOCKS_PER_PAGE);
            if (stockPage < totalPages - 1) {
                stockPage++;
                const metricKey: MetricKey = mispricingMode === 'sizeNeutral' ? 'residualMispricing' : 'mispricing';
                renderStockTable(metricKey, false);
            }
        });
    }
}

function setMode(mode: MispricingMode): void {
    mispricingMode = mode;

    const rawBtn = document.getElementById('modeRaw');
    const sizeNeutralBtn = document.getElementById('modeSizeNeutral');

    if (mode === 'raw') {
        rawBtn?.classList.add('active');
        sizeNeutralBtn?.classList.remove('active');
    } else {
        sizeNeutralBtn?.classList.add('active');
        rawBtn?.classList.remove('active');
    }

    if (dashboardData) {
        updateStats();
        // Use cached data for instant mode switching (no progressive rendering)
        const metricKey: MetricKey = mode === 'sizeNeutral' ? 'residualMispricing' : 'mispricing';
        renderAllCharts(metricKey, false);  // false = no progressive, instant switch
    }
}


// Start
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
