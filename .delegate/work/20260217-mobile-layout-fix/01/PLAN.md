# Loop 01: Mobile Layout Fixes

## Objective
Fix mobile layout issues: overlapping controls, missing sticky behavior, and overflowing stat cards.

## Implementation Plan

### 1. Create Mobile Control Bar
- Add a new unified control bar at the top that appears only on mobile
- Contains both quarter selector and mode toggle in a single row
- Hidden on desktop (md:hidden), visible on mobile
- Fixed position at bottom of screen on mobile for better thumb accessibility

### 2. Hide Desktop Controls on Mobile
- Quarter selector wrapper: Add `hidden md:block` to hide on mobile
- Mode toggle in stat card: Add `hidden md:flex` to hide on mobile

### 3. Fix Stat Card Overflow
- Market Coverage card: Change horizontal flex to grid/wrap on mobile
- Use `flex-wrap` and adjust gap/sizing
- Stack values vertically on very small screens

### 4. Mobile Sticky Behavior
- Mobile control bar is always fixed at bottom
- Remove complex JS sticky logic for mobile (not needed with fixed bottom bar)

## Files to Modify

1. **web/index.html**
   - Add mobile control bar HTML
   - Add responsive classes to existing controls
   - Fix stat card layout classes

2. **web/src/style.css**
   - Add mobile control bar styles
   - Add responsive stat card styles

3. **web/src/main.ts**
   - Ensure mode toggle event listeners work with new mobile elements

## Implementation Details

### Mobile Control Bar (new HTML)
```html
<!-- Mobile Control Bar - fixed at bottom -->
<div id="mobileControlBar" class="fixed bottom-0 left-0 right-0 z-50 md:hidden bg-gray-900/95 border-t border-gray-700 backdrop-blur-sm p-3">
  <div class="flex justify-between items-center gap-3">
    <!-- Quarter Select (clone) -->
    <select id="quarterSelectMobile" class="...">
    </select>
    <!-- Mode Toggle (clone) -->
    <div class="flex gap-2">
      <button id="modeRawMobile" class="...">Raw</button>
      <button id="modeSizeNeutralMobile" class="...">Size-Neutral</button>
    </div>
  </div>
</div>
```

### Stat Card Fix
Change from:
```html
<div class="flex items-center gap-3">
```
To:
```html
<div class="flex flex-wrap items-center gap-x-3 gap-y-1">
```

And hide separators on mobile when wrapping occurs.
