/** @type {import('tailwindcss').Config} */
export default {
    content: ['./src/**/*.{html,js,svelte,ts}'],
    theme: {
        extend: {
            colors: {
                profit: { DEFAULT: '#22c55e', light: '#86efac', dark: '#15803d' },
                loss: { DEFAULT: '#ef4444', light: '#fca5a5', dark: '#b91c1c' },
                neutral: { DEFAULT: '#6b7280', light: '#9ca3af', dark: '#4b5563' },
                surface: {
                    DEFAULT: '#1e293b',
                    light: '#334155',
                    dark: '#0f172a',
                },
            },
        },
    },
    plugins: [],
};
