/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: "var(--bg-canvas)",
        surface: "var(--bg-surface)",
        subtle: "var(--bg-subtle)",
        hairline: "var(--border-hairline)",
        primary: "var(--text-primary)",
        muted: "var(--text-muted)",
        critical: "var(--color-risk-critical)",
        highrisk: "var(--color-risk-high)",
        caution: "var(--color-caution)",
        protective: "var(--color-protective)",
        accent: "var(--color-accent)",
      },
      fontFamily: {
        serif: ['"Fraunces"', 'Georgia', 'serif'],
        sans: ['"Inter Tight"', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'Menlo', 'Consolas', 'monospace'],
      },
      boxShadow: {
        hairline: "0 0 0 1px var(--border-hairline)",
        card: "0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 0 0 1px var(--border-hairline)",
        modal: "0 20px 25px -5px rgba(0, 0, 0, 0.15), 0 0 0 1px var(--border-hairline)",
      },
      animation: {
        "pulse-subtle": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "fade-in": "fadeIn 250ms cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "slide-down": "slideDown 300ms cubic-bezier(0.16, 1, 0.3, 1) forwards",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0", transform: "translateY(4px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        slideDown: {
          "0%": { opacity: "0", transform: "translateY(-10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [],
}
