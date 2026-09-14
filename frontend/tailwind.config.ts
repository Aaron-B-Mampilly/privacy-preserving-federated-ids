import type { Config } from 'tailwindcss';

/**
 * Design tokens are ported from the project's published results artifact
 * so the dashboard and the thesis write-up read as one system: the same
 * teal accent, the same Fraunces/IBM Plex pairing, and the same
 * finding / caveat / mechanism / resolved callout vocabulary.
 *
 * Colours are declared as CSS variables in styles/tokens.css and only
 * referenced here, so light and dark are a single source of truth.
 */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: 'rgb(var(--bg) / <alpha-value>)',
        surface: 'rgb(var(--surface) / <alpha-value>)',
        'surface-alt': 'rgb(var(--surface-alt) / <alpha-value>)',
        ink: 'rgb(var(--ink) / <alpha-value>)',
        'ink-muted': 'rgb(var(--ink-muted) / <alpha-value>)',
        border: 'rgb(var(--border) / <alpha-value>)',
        accent: 'rgb(var(--accent) / <alpha-value>)',
        'accent-strong': 'rgb(var(--accent-strong) / <alpha-value>)',
        'accent-soft': 'rgb(var(--accent-soft) / <alpha-value>)',
        finding: 'rgb(var(--finding) / <alpha-value>)',
        'finding-soft': 'rgb(var(--finding-soft) / <alpha-value>)',
        caveat: 'rgb(var(--caveat) / <alpha-value>)',
        'caveat-soft': 'rgb(var(--caveat-soft) / <alpha-value>)',
        good: 'rgb(var(--good) / <alpha-value>)',
        'good-soft': 'rgb(var(--good-soft) / <alpha-value>)',
        danger: 'rgb(var(--danger) / <alpha-value>)',
        'danger-soft': 'rgb(var(--danger-soft) / <alpha-value>)',
      },
      fontFamily: {
        display: ['Fraunces', 'Georgia', 'serif'],
        sans: ['"IBM Plex Sans"', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.6875rem', { lineHeight: '1rem' }],
      },
      borderRadius: {
        card: '10px',
      },
      boxShadow: {
        card: 'var(--shadow-card)',
        lift: 'var(--shadow-lift)',
      },
      keyframes: {
        'fade-in': {
          from: { opacity: '0', transform: 'translateY(4px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          '100%': { transform: 'translateX(100%)' },
        },
      },
      animation: {
        // Deliberately short and subtle: this is a research dashboard,
        // not a marketing page.
        'fade-in': 'fade-in 180ms ease-out',
        shimmer: 'shimmer 1.4s infinite',
      },
    },
  },
  plugins: [],
} satisfies Config;
