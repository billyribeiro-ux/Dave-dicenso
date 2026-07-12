export interface SystemStatus {
    status: 'READY' | 'TRAINING' | 'BACKTESTING' | 'LIVE' | 'ERROR';
    uptime_seconds: number;
    active_tickers: number;
    trained_models: number;
    current_model_versions: Record<string, string>;
    last_backtest: string | null;
    last_training: string | null;
    active_positions: number;
    api_calls_today: number;
    errors_today: number;
}

export interface ModelVersion {
    ticker: string;
    version: string;
    trained: boolean;
}

export interface LearningLogEntry {
    id: number;
    ticker: string;
    date: string;
    classification: string;
    description: string;
    action_taken: string;
    result: string | null;
    created_at: string;
}
