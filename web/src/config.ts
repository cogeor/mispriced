/**
 * Frontend configuration constants.
 *
 * IC_INTERPRETATION controls only the *narrative copy* used in chart subtitles,
 * tooltips, and methodology paragraphs. It does NOT change the displayed sign
 * of the IC: both framings show the same raw Spearman IC and differ only in
 * which pole (positive or negative) the copy treats as confirmation.
 *
 * Signal convention (consistent across the repo):
 *   signal_raw = (predicted_mcap - actual_mcap) / actual_mcap
 *   signal_raw > 0  =>  undervalued (predicted price higher than actual)
 *   signal_raw < 0  =>  overvalued
 *
 * Under that convention:
 *   - 'reversion' framing: positive IC = book-residual predicted mean reversion
 *     (i.e. the value strategy worked over the horizon).
 *   - 'intangibles' framing: negative IC = market continued to price the
 *     non-book premium (intangibles/growth dominated book value).
 */

export type ICInterpretation = 'reversion' | 'intangibles';

export const IC_INTERPRETATION: ICInterpretation = 'reversion';

export interface ICInterpretationCopy {
    subtitle: string;
    positivePole: string;
    negativePole: string;
    tooltipPhrase: string;
    methodologyParagraph: string;
}

export const IC_INTERPRETATION_COPY: Record<ICInterpretation, ICInterpretationCopy> = {
    reversion: {
        subtitle:
            'Positive IC = book residual predicted mean reversion (value worked).',
        positivePole: 'value reverted',
        negativePole: 'intangibles/growth dominated',
        tooltipPhrase:
            'Spearman IC of signal vs forward return. Positive = book-based value predicted mean reversion.',
        methodologyParagraph:
            'signal_raw = (predicted_mcap - actual_mcap) / actual_mcap, so signal_raw > 0 means the model thinks the stock is undervalued. IC is the raw Spearman rank correlation of the signal against the forward return &mdash; no sign flip is applied. Under the reversion framing, a positive IC means the book-residual predicted mean reversion (the value strategy worked over the horizon); a negative IC means the market continued to price the non-book premium.',
    },
    intangibles: {
        subtitle:
            'Negative IC = market continued to price the non-book premium (intangibles dominated).',
        positivePole: 'value reverted',
        negativePole: 'intangibles/growth dominated',
        tooltipPhrase:
            'Spearman IC of signal vs forward return. Negative = market kept rewarding the non-book premium.',
        methodologyParagraph:
            'signal_raw = (predicted_mcap - actual_mcap) / actual_mcap, so signal_raw > 0 means the model thinks the stock is undervalued. IC is the raw Spearman rank correlation of the signal against the forward return &mdash; no sign flip is applied. Under the intangibles framing, a negative IC is the expected headline: the intangibles/growth premium continued to be paid and undervalued stocks kept underperforming; a positive IC would indicate the book residual reverted.',
    },
};

export const IC_COPY: ICInterpretationCopy = IC_INTERPRETATION_COPY[IC_INTERPRETATION];
