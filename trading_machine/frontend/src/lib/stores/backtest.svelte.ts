import type { BacktestSummary, BacktestFullResult, TickerBreakdown, EquityPoint, DrawdownPoint } from '$types/backtest';

class BacktestStore {
    summary = $state<BacktestSummary | null>(null);
    equityCurve = $state<EquityPoint[]>([]);
    drawdownCurve = $state<DrawdownPoint[]>([]);
    tickerBreakdown = $state<TickerBreakdown[]>([]);
    loading = $state(false);
    error = $state<string | null>(null);

    async fetch(startDate: string, endDate: string, ticker?: string) {
        this.loading = true;
        this.error = null;
        try {
            const params = new URLSearchParams({ start_date: startDate, end_date: endDate });
            if (ticker) params.set('ticker', ticker);
            const res = await fetch(`/api/backtest/results?${params}`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data: BacktestFullResult = await res.json();
            this.summary = data.summary;
            this.equityCurve = data.equity_curve || [];
            this.drawdownCurve = data.drawdown_curve || [];
            this.tickerBreakdown = data.ticker_breakdown || [];
        } catch (e) {
            this.error = e instanceof Error ? e.message : 'Failed to fetch backtest';
        } finally {
            this.loading = false;
        }
    }

    get hasData(): boolean {
        return this.summary !== null && this.summary.total_trades > 0;
    }
}

export const backtestStore = new BacktestStore();
