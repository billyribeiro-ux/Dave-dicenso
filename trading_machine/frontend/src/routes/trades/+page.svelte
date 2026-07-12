<script lang="ts">
    import TradeRow from '$components/TradeRow.svelte';
    import MetricCard from '$components/MetricCard.svelte';
    import DateRangeFilter from '$components/DateRangeFilter.svelte';
    import TickerSelector from '$components/TickerSelector.svelte';
    import ExportButtons from '$components/ExportButtons.svelte';
    import EmptyState from '$components/EmptyState.svelte';
    import type { Trade } from '$types/trade';
    import { DEFAULT_START_DATE, DEFAULT_END_DATE } from '$utils/constants';
    import { formatCurrency, formatPercent } from '$utils/formatters';
    import { onMount } from 'svelte';

    let trades = $state<Trade[]>([]);
    let total = $state(0);
    let loading = $state(false);
    let startDate = $state(DEFAULT_START_DATE);
    let endDate = $state(DEFAULT_END_DATE);
    let ticker = $state('');

    async function fetchTrades() {
        loading = true;
        try {
            const p = new URLSearchParams({ start_date: startDate, end_date: endDate, page: '1', page_size: '100' });
            if (ticker) p.set('ticker', ticker);
            const res = await fetch(`/api/backtest/trades?${p}`);
            const data = await res.json();
            trades = data.trades || [];
            total = data.total || 0;
        } catch (e) { console.error(e); } finally { loading = false; }
    }

    onMount(() => { fetchTrades(); });

    const totalPnl = $derived(trades.reduce((s, t) => s + t.pnl, 0));
    const wins = $derived(trades.filter(t => t.pnl > 0).length);
    const winRate = $derived(trades.length > 0 ? wins / trades.length : 0);

    function doExport(format: 'csv' | 'excel') {
        const p = new URLSearchParams({ start_date: startDate, end_date: endDate, format });
        if (ticker) p.set('ticker', ticker);
        window.open(`/api/backtest/export?${p}`, '_blank');
    }
</script>

<svelte:head><title>Trades — Trading Machine</title></svelte:head>

<div class="space-y-6">
    <div class="flex items-center justify-between">
        <h2 class="text-xl font-bold text-white">💹 Trade History</h2>
        <div class="flex items-center gap-3">
            <TickerSelector bind:selected={ticker} onSelect={fetchTrades} />
            <DateRangeFilter bind:startDate bind:endDate onApply={fetchTrades} />
            <ExportButtons {startDate} {endDate} {ticker} onExport={doExport} />
        </div>
    </div>

    <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard label="Total Trades" value={total.toString()} trend="neutral" />
        <MetricCard label="Net P&L" value={formatCurrency(totalPnl)} trend={totalPnl >= 0 ? 'up' : 'down'} />
        <MetricCard label="Win Rate" value={formatPercent(winRate * 100)} trend={winRate >= 0.5 ? 'up' : 'down'} sublabel={`${wins}/${trades.length}`} />
        <MetricCard label="Avg P&L" value={formatCurrency(trades.length > 0 ? totalPnl / trades.length : 0)} trend={totalPnl >= 0 ? 'up' : 'down'} />
    </div>

    {#if trades.length > 0}
        <div class="card overflow-x-auto">
            <table class="w-full">
                <thead><tr class="text-slate-400 text-xs uppercase"><th class="text-left py-2 px-3">Ticker</th><th class="text-left py-2 px-3">Dir</th><th class="text-left py-2 px-3">Entry</th><th class="text-left py-2 px-3">Exit</th><th class="text-right py-2 px-3">P&L</th><th class="text-left py-2 px-3">Reason</th></tr></thead>
                <tbody>{#each trades as trade}<TradeRow {trade} />{/each}</tbody>
            </table>
        </div>
    {:else if loading}
        <p class="text-slate-400">Loading trades...</p>
    {:else}
        <EmptyState message="No trades in this period" />
    {/if}
</div>
