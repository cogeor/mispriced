/**
 * Chart info tooltip system.
 *
 * Adds "?" icon buttons next to chart titles. Hovering (desktop) or tapping
 * (mobile) shows a fixed-position tooltip with Data, Axes, and Insights.
 */

import { CHART_META, type ChartMeta } from './chartMeta';

let activeTooltip: string | null = null;
let tooltipEl: HTMLElement | null = null;
let hideTimeout: ReturnType<typeof setTimeout> | null = null;

function getTooltipEl(): HTMLElement {
    if (!tooltipEl) {
        tooltipEl = document.getElementById('chartInfoTooltip');
    }
    return tooltipEl!;
}

function buildTooltipHTML(meta: ChartMeta): string {
    return `
        <div class="cit-section">
            <div class="cit-label">Data</div>
            <div class="cit-text">${meta.data}</div>
        </div>
        <div class="cit-section">
            <div class="cit-label">Axes</div>
            <div class="cit-text">${meta.axes}</div>
        </div>
        <div class="cit-section">
            <div class="cit-label">Insights</div>
            <div class="cit-text">${meta.insights}</div>
        </div>
    `;
}

function positionTooltip(btn: HTMLElement, tip: HTMLElement): void {
    const rect = btn.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    // Mobile: show as bottom sheet
    if (vw < 768) {
        tip.style.left = '8px';
        tip.style.right = '8px';
        tip.style.top = '';
        tip.style.bottom = '80px'; // above mobile control bar
        tip.style.maxWidth = '';
        return;
    }

    // Desktop: position near button
    const tipWidth = 360;
    const tipHeight = tip.offsetHeight || 280;

    let left = rect.right + 8;
    let top = rect.top;

    // Clamp to viewport
    if (left + tipWidth > vw - 16) {
        left = rect.left - tipWidth - 8;
    }
    if (left < 8) left = 8;
    if (top + tipHeight > vh - 16) {
        top = vh - tipHeight - 16;
    }
    if (top < 8) top = 8;

    tip.style.left = `${left}px`;
    tip.style.top = `${top}px`;
    tip.style.right = '';
    tip.style.bottom = '';
    tip.style.maxWidth = `${tipWidth}px`;
}

function showTooltip(chartId: string, btn: HTMLElement): void {
    const meta = CHART_META[chartId];
    if (!meta) return;

    if (hideTimeout) {
        clearTimeout(hideTimeout);
        hideTimeout = null;
    }

    const tip = getTooltipEl();
    tip.innerHTML = buildTooltipHTML(meta);
    tip.classList.add('visible');
    activeTooltip = chartId;

    // Position after content is rendered
    requestAnimationFrame(() => positionTooltip(btn, tip));
}

function hideTooltip(): void {
    const tip = getTooltipEl();
    tip.classList.remove('visible');
    activeTooltip = null;
}

function scheduleHide(): void {
    hideTimeout = setTimeout(hideTooltip, 200);
}

function cancelHide(): void {
    if (hideTimeout) {
        clearTimeout(hideTimeout);
        hideTimeout = null;
    }
}

export function setupChartInfoTooltips(): void {
    const tip = getTooltipEl();
    if (!tip) return;

    const isMobile = window.innerWidth < 768;

    // Tooltip hover: keep visible when hovering the tooltip itself
    tip.addEventListener('mouseenter', cancelHide);
    tip.addEventListener('mouseleave', scheduleHide);

    document.querySelectorAll<HTMLButtonElement>('.chart-info-btn').forEach(btn => {
        const chartId = btn.dataset.chart;
        if (!chartId || !CHART_META[chartId]) return;

        if (isMobile) {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                if (activeTooltip === chartId) {
                    hideTooltip();
                } else {
                    showTooltip(chartId, btn);
                }
            });
        } else {
            btn.addEventListener('mouseenter', () => showTooltip(chartId, btn));
            btn.addEventListener('mouseleave', scheduleHide);
        }
    });

    // Dismiss on scroll or outside click
    window.addEventListener('scroll', hideTooltip, { passive: true });
    document.addEventListener('click', (e) => {
        if (activeTooltip && tip && !tip.contains(e.target as Node)) {
            const isBtn = (e.target as HTMLElement).closest('.chart-info-btn');
            if (!isBtn) hideTooltip();
        }
    });

    // Dismiss on Escape
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && activeTooltip) hideTooltip();
    });
}
