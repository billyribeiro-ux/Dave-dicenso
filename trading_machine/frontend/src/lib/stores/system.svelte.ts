import type { SystemStatus } from '$types/system';

class SystemStore {
    status = $state<SystemStatus | null>(null);
    loading = $state(false);
    error = $state<string | null>(null);

    async fetch() {
        this.loading = true;
        this.error = null;
        try {
            const res = await fetch('/api/system/status');
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            this.status = await res.json();
        } catch (e) {
            this.error = e instanceof Error ? e.message : 'Failed to fetch system status';
        } finally {
            this.loading = false;
        }
    }

    get isLive(): boolean {
        return this.status?.status === 'LIVE';
    }

    get isReady(): boolean {
        return this.status?.status === 'READY' || this.status?.status === 'LIVE';
    }

    get trainedCount(): number {
        return this.status?.trained_models ?? 0;
    }

    get totalTickers(): number {
        return this.status?.active_tickers ?? 0;
    }

    get modelVersions(): Record<string, string> {
        return this.status?.current_model_versions ?? {};
    }
}

export const systemStore = new SystemStore();
