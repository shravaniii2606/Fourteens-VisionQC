/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    colors: {
      navy: '#1E3A5F', ink: '#25364A', canvas: '#F6F8FB', white: '#FFFFFF', line: '#E5E9F0',
      pass: '#22A06B', fail: '#E5484D', recapture: '#F5A524', accent: '#3B82B8', slate: '#64748B',
    },
    borderRadius: { card: '12px' },
    boxShadow: { soft: '0 2px 10px rgba(24, 50, 76, 0.045)' },
    extend: {
      spacing: { page: '34px', panel: '24px' },
      fontFamily: { sans: ['DM Sans', 'Inter', 'system-ui', 'sans-serif'], display: ['Manrope', 'Inter', 'sans-serif'] },
    },
  },
  plugins: [],
};
