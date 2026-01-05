// UI utility functions

// Escape HTML to prevent XSS
export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// Add spinner CSS (call once on init)
export function initSpinnerStyles(): void {
  const spinnerStyle = document.createElement('style');
  spinnerStyle.textContent = `
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    .spinner {
      width: 14px;
      height: 14px;
      border: 2px solid #333;
      border-top-color: #3b82f6;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
  `;
  document.head.appendChild(spinnerStyle);
}
