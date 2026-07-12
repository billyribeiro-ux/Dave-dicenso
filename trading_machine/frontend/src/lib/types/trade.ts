export interface Trade {
    id: number;
    ticker: string;
    direction: 'LONG' | 'SHORT';
    entry_time: string;
    entry_price: number;
    exit_time: string;
    exit_price: number;
    pnl: number;
    pnl_pct: number;
    exit_reason: string;
    loss_classification: string | null;
    model_version: string;
    created_at: string;
}

export interface TradeListResponse {
    trades: Trade[];
    total: number;
    page: number;
    page_size: number;
}
