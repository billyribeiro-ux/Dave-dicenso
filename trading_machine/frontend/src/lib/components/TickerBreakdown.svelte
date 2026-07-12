<script lang="ts">
    import type { TickerBreakdown } from '$types/backtest';
    import { formatCurrency } from '$utils/formatters';
    let { breakdown = [] as TickerBreakdown[] } = $props();
</script>
<div class="card">
    <h3 class="text-sm font-semibold text-slate-300 mb-3">Ticker Breakdown</h3>
    <div class="space-y-2">
        {#each breakdown as item}
            <div class="flex items-center justify-between py-1.5 border-b border-slate-700/50 last:border-0">
                <span class="text-white font-medium">{item.ticker}</span>
                <div class="flex items-center gap-4 text-sm">
                    <span class="text-slate-400">{item.trades} trades</span>
                    <span class="text-slate-400">{(item.win_rate * 100).toFixed(1)}% win</span>
                    <span class="font-mono {item.net_profit >= 0 ? 'metric-up' : 'metric-down'}">
                        {formatCurrency(item.net_profit)}
                    </span>
                </div>
            </div>
        {/each}
    </div>
</div>
