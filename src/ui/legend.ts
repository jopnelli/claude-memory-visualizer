// Legend component for time/category coloring

import * as THREE from 'three';
import { state } from '../state';
import { TIME_GRADIENT, CATEGORY_COLORS } from '../three/colors';
import { hasCategoryData } from '../three/points';

// Update legend display
export function updateLegend(legendEl: HTMLElement): void {
  // Check for category-based coloring
  if (hasCategoryData()) {
    // Collect unique categories from data
    const categories = new Set<string>();
    for (const doc of state.data!.documents) {
      const cat = doc.metadata?.category as string;
      if (cat) categories.add(cat);
    }

    const categoryItems = Array.from(categories)
      .map((cat) => {
        const color = CATEGORY_COLORS[cat] || new THREE.Color(0x666666);
        return `
        <div class="legend-item">
          <div class="legend-dot" style="background: #${color.getHexString()};"></div>
          <span>${cat}</span>
        </div>
      `;
      })
      .join('');

    legendEl.innerHTML = `
      <div style="font-size: 11px; color: #888; margin-bottom: 8px;">
        <strong>Categories</strong>
      </div>
      ${categoryItems}
    `;
    return;
  }

  if (!state.timeRange) {
    legendEl.innerHTML =
      '<div style="color: #666; font-size: 11px;">No timestamp data</div>';
    return;
  }

  const formatDate = (d: Date) =>
    d.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });

  // Create gradient bar
  const gradientColors = TIME_GRADIENT.map((c) => '#' + c.getHexString()).join(
    ', '
  );

  legendEl.innerHTML = `
    <div style="font-size: 11px; color: #888; margin-bottom: 8px;">
      <strong>Time Range</strong>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
      <span style="font-size: 10px; color: #666;">${formatDate(state.timeRange.min)}</span>
      <div style="flex: 1; height: 8px; border-radius: 4px; background: linear-gradient(to right, ${gradientColors});"></div>
      <span style="font-size: 10px; color: #666;">${formatDate(state.timeRange.max)}</span>
    </div>
    <div style="font-size: 10px; color: #555; margin-top: 4px; text-align: center;">
      older → newer
    </div>
  `;
}
