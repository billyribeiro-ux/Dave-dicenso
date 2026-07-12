import type { Position } from '$types/signal';

class PositionsStore {
    positions = $state<Position[]>([]);
    loading = $state(false);
    error = $state<string | null>(null);
    totalUnrealizedPnl = $derived(
        this.positions.reduce((sum, p) => sum + p.unrealized_pnl, 0)
    );
    openCount = $derived(this.positions.length);

    async fetch() {
        this.loading = true;
        this.error = null;
        try {
            const res = await fetch('/api/signals/positions');
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            this.positions = data.positions || [];
        } catch (e) {
            this.error = e instanceof Error ? e.message : 'Failed to fetch positions';
        } finally {
            this.loading = false;
        }
    }

    getByTicker(ticker: string): Position | undefined {
        return this.positions.find(p => p.ticker === ticker);
    }

    getLongPositions(): Position[] {
        return this.positions.filter(p => p.direction === 'LONG');
    }

    getShortPositions(): Position[] {
        return this.positions.filter(p => p.direction === 'SHORT');
    }
}

export const positionsStore = new PositionsStore();
