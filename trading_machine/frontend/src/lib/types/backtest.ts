export interface BacktestSummary {
    ticker: string | null;
    start_date: string;
    end_date: string;
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    win_rate: number;
    loss_rate: number;
    total_wins: number;
    total_losses: number;
    net_profit: number;
    profit_factor: number;
    avg_win: number;
    avg_loss: number;
    largest_win: number;
    largest_loss: number;
    max_drawdown_pct: number;
    sharpe_ratio: number;
    win_loss_ratio: number;
    model_version: string;
}

export interface EquityPoint {
    date: string;
    equity: number;
}

export interface DrawdownPoint {
    date: string;
    drawdown_pct: number;
}

export interface TickerBreakdown {
    ticker: string;
    net_profit: number;
    win_rate: number;
    trades: number;
}

export interface BacktestFullResult {
    summary: BacktestSummary;
    equity_curve: EquityPoint[];
    drawdown_curve: DrawdownPoint[];
    ticker_breakdown: TickerBreakdown[];
}
