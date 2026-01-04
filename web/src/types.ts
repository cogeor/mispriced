export interface DashboardStats {
    total_tickers: number;
    total_actual_mcap_t: number;
    total_predicted_mcap_t: number;
    undervalued_count: number;
    overvalued_count: number;
    avg_mispricing_pct: number;
    median_mispricing_pct: number;
    indices_tracked: number;
    quarter_date: string;
    metric_name: string;
    metric_formula: string;
    metric_description: string;
    size_premium?: unknown;
}

export interface ScatterPoint {
    ticker: string;
    actual: number;
    predicted: number;
    mispricing: number;
    residualMispricing: number;
    relStd: number;
    mispricingPct: string;
    residualMispricingPct: string;
    sector: string;
    company: string;
    industry: string;
    sectorColor: string;
    actual_mcap: number;
}

export interface IndexChartItem {
    index: string;
    mispricing: number;
    residualMispricing: number;
    mispricingPct: string;
    residualMispricingPct: string;
    color: string;
    status: string;
    totalActual: string;
    totalPredicted: string;
    count: number;
    officialCount: number;
}

export interface SectorBreakdownItem {
    sector: string;
    count: number;
    avgMispricing: number;
    totalMcap: number;
    color: string;
}

export interface BacktestItem {
    name: string;
    horizon: number;
    ic: number;
    pval: number;
    n_obs: number;
    spread: number;
    hit_rate: number;
    metric: string;
    quarter?: string;
    log_pval?: number;
}

export interface BacktestData {
    sector_ts: BacktestItem[];
    index_ts: BacktestItem[];
    sector_summary: BacktestItem[];
    index_summary: BacktestItem[];
    horizon: {
        horizon: number;
        avg_ic: number;
        avg_spread: number;
    }[];
}

export interface IndexTimeSeriesPoint {
    index: string;
    date: string;
    mispricing: number;
    residualMispricing?: number;
    count: number;
}

export interface SectorTimeSeriesPoint {
    sector: string;
    date: string;
    mispricing: number;
    residualMispricing?: number;
    count: number;
}

export interface SizePremiumPoint {
    logMcap: number;
    mcapB: number;
    expectedMispricing: number;
}

export interface SizeCoefficient {
    quarter: string;
    slope: number;
    se?: number;
    slopeSE?: number;
    slopeTstat?: number;
    slopePval?: number;
    beta?: number;
    intercept: number;
    rSquared: number;
    nObs: number;
}

export interface QuarterValuationData {
    scatter_data: ScatterPoint[];
    sector_breakdown: SectorBreakdownItem[];
}

export interface SizeCorrectionModel {
    method: string;
    coefficients: number[];
    equation: string;
}

export interface DashboardData {
    generated_at: string;
    stats: DashboardStats;
    scatter_data: ScatterPoint[];
    index_chart_data: IndexChartItem[];
    sector_breakdown: SectorBreakdownItem[];
    backtest_data: BacktestData;
    index_timeseries: IndexTimeSeriesPoint[];
    sector_timeseries: SectorTimeSeriesPoint[];
    size_premium_curve: SizePremiumPoint[];
    size_coefficients: SizeCoefficient[];
    available_quarters: string[];  // List of available quarters for lazy loading
    size_correction_model?: SizeCorrectionModel; // Quadratic model coefficients
}

export type MispricingMode = 'raw' | 'sizeNeutral';
export type ColorMode = 'sector' | 'mispricing';
export type MetricKey = 'mispricing' | 'residualMispricing';
