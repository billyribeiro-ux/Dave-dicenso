export interface ApiResponse<T> {
    data: T;
    error?: string;
    timestamp: string;
}

export interface PaginatedResponse<T> {
    items: T[];
    total: number;
    page: number;
    page_size: number;
}

export interface DateRange {
    start_date: string;
    end_date: string;
    ticker?: string;
}

export interface ExportRequest {
    start_date: string;
    end_date: string;
    ticker?: string;
    format: 'csv' | 'excel';
}

export interface HealthCheck {
    status: string;
    timestamp: string;
    ws_connections: number;
}
