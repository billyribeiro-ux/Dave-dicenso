import type { Signal } from '$types/signal';

class SignalsStore {
    signals = $state<Signal[]>([]);
    loading = $state(false);
    error = $state<string | null>(null);
    lastUpdate = $state<string>('');

    async fetch() {
        this.loading = true;
        this.error = null;
        try {
            const res = await fetch('/api/signals/live');
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            this.signals = data.signals || [];
            this.lastUpdate = data.timestamp || new Date().toISOString();
        } catch (e) {
            this.error = e instanceof Error ? e.message : 'Failed to fetch signals';
        } finally {
            this.loading = false;
        }
    }

    getBuySignals(): Signal[] {
        return this.signals.filter(s => s.signal === 'BUY').sort((a, b) => b.confidence - a.confidence);
    }

    getSellSignals(): Signal[] {
        return this.signals.filter(s => s.signal === 'SELL').sort((a, b) => b.confidence - a.confidence);
    }

    getByTicker(ticker: string): Signal | undefined {
        return this.signals.find(s => s.ticker === ticker);
    }
}

export const signalsStore = new SignalsStore();
