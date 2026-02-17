# Mobile Layout Fixes for Dashboard

## Problem Statement

The dashboard has several layout issues on mobile devices:

1. **Overlapping buttons**: The quarter select dropdown and mode toggle buttons (Raw/Size-Neutral) are well aligned on desktop but overlap on mobile
2. **Sticky behavior missing on mobile**: The buttons don't follow on scroll on mobile like they should
3. **Overflowing vignettes**: The Market Coverage and Mispricing stat cards at the top take too much horizontal space and overflow on the right side
4. **Single screen width**: All elements should fit in a single screen width without horizontal scrolling

## Current Implementation

- `quarterSelectWrapper`: Fixed position at `top: 24px; right: 30%`
- `modeToggle`: Absolute positioned inside mispricing card, becomes sticky on scroll via JS
- KPI cards: Two-column grid on desktop (`md:grid-cols-2`), single column on mobile
- Stat values displayed in horizontal flex rows with `•` separators

## Requirements

1. Replace the two separate buttons (quarter select + mode toggle) with a mobile-friendly interface
2. Ensure all stat vignettes fit within a single screen width on mobile
3. Maintain current desktop behavior
4. Clean, professional solution that works across all breakpoints

## Files to Modify

- `web/index.html` - HTML structure and inline styles
- `web/src/style.css` - CSS styles (uses Tailwind)
- `web/src/main.ts` - JavaScript event handlers for mode toggle and sticky behavior

## Technical Constraints

- Uses Tailwind CSS via CDN
- Uses Plotly for charts
- Must maintain existing functionality (quarter switching, mode toggle)
