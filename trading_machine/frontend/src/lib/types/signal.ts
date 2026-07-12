export interface Signal {
    ticker: string;
    signal: 'BUY' | 'SELL' | 'NEUTRAL';
    entry_price: number | null;
    target_price: number | null;
    stop_price: number | null;
    confidence: number;
    timestamp: string;
    regime: string | null;
    latent_state_version: string;
}

export interface Position {
    ticker: string;
    direction: 'LONG' | 'SHORT';
    entry_price: number;
    current_price: number;
    unrealized_pnl: number;
    unrealized_pnl_pct: number;
    target_price: number | null;
    stop_price: number | null;
    entry_time: string;
    duration_minutes: number;
}

export interface SignalsResponse {
    signals: Signal[];
    timestamp: string;
}

export interface PositionsResponse {
    positions: Position[];
    count: number;
    total_unrealized_pnl: number;
}
