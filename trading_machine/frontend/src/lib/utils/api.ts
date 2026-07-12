const API_BASE = '/api';

async function request<T>(url: string, options?: RequestInit): Promise<T> {
    const res = await fetch(`${API_BASE}${url}`, {
        headers: { 'Content-Type': 'application/json' },
        ...options,
    });
    if (!res.ok) {
        const error = await res.text();
        throw new Error(`HTTP ${res.status}: ${error}`);
    }
    return res.json();
}

export const api = {
    tickers: {
        list: () => request<string[]>('/tickers/'),
        price: (ticker: string) => request<any>(`/tickers/${ticker}/price`),
        historical: (ticker: string, start: string, end: string) =>
            request<any>(`/tickers/${ticker}/historical?start_date=${start}&end_date=${end}`),
        batchPrices: () => request<any>('/tickers/prices/batch'),
    },
    signals: {
        live: () => request<any>('/signals/live'),
        ticker: (ticker: string) => request<any>(`/signals/live/${ticker}`),
        positions: () => request<any>('/signals/positions'),
    },
    backtest: {
        summary: (start: string, end: string, ticker?: string) => {
            const p = new URLSearchParams({ start_date: start, end_date: end });
            if (ticker) p.set('ticker', ticker);
            return request<any>(`/backtest/summary?${p}`);
        },
        results: (start: string, end: string, ticker?: string) => {
            const p = new URLSearchParams({ start_date: start, end_date: end });
            if (ticker) p.set('ticker', ticker);
            return request<any>(`/backtest/results?${p}`);
        },
        trades: (start: string, end: string, ticker?: string, page = 1, pageSize = 100) => {
            const p = new URLSearchParams({ start_date: start, end_date: end, page: String(page), page_size: String(pageSize) });
            if (ticker) p.set('ticker', ticker);
            return request<any>(`/backtest/trades?${p}`);
        },
        equityCurve: (start: string, end: string, ticker?: string) => {
            const p = new URLSearchParams({ start_date: start, end_date: end });
            if (ticker) p.set('ticker', ticker);
            return request<any>(`/backtest/equity-curve?${p}`);
        },
        export: (start: string, end: string, format: 'csv' | 'excel', ticker?: string) => {
            const p = new URLSearchParams({ start_date: start, end_date: end, format });
            if (ticker) p.set('ticker', ticker);
            return `${API_BASE}/backtest/export?${p}`;
        },
    },
    learning: {
        log: (ticker?: string, limit = 50) => {
            const p = new URLSearchParams({ limit: String(limit) });
            if (ticker) p.set('ticker', ticker);
            return request<any>(`/learning/log?${p}`);
        },
        regimes: (ticker?: string) => {
            const p = new URLSearchParams();
            if (ticker) p.set('ticker', ticker);
            return request<any>(`/learning/regimes?${p}`);
        },
        modelVersions: () => request<any>('/learning/model-versions'),
    },
    system: {
        status: () => request<any>('/system/status'),
        startTraining: () => request<any>('/system/start-training', { method: 'POST' }),
    },
    health: () => request<any>('/health'),
};
