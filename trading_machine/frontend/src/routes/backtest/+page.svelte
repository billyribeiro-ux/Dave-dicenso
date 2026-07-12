<script lang="ts">
    import MetricCard from '$components/MetricCard.svelte';
    import EquityCurve from '$components/EquityCurve.svelte';
    import DrawdownCurve from '$components/DrawdownCurve.svelte';
    import TickerBreakdown from '$components/TickerBreakdown.svelte';
    import DateRangeFilter from '$components/DateRangeFilter.svelte';
    import TickerSelector from '$components/TickerSelector.svelte';
    import { backtestStore } from '$stores/backtest.svelte.ts';
    import { DEFAULT_START_DATE, DEFAULT_END_DATE } from '$utils/constants';
    import { formatCurrency, formatPercent } from '$utils/formatters';
    import { onMount } from 'svelte';

    let startDate = $state(DEFAULT_START_DATE);
    let endDate = $state(DEFAULT_END_DATE);
    let ticker = $state('');

    onMount(() => { backtestStore.fetch(startDate, endDate, ticker || undefined); });

    function applyFilter() { backtestStore.fetch(startDate, endDate, ticker || undefined); }
    const s = $derived(backtestStore.summary);
</script>

<svelte:head><title>Backtest — Trading Machine</title></svelte:head>

<div class="space-y-6">
    <div class="flex items-center justify-between">
        <h2 class="text-xl font-bold text-white">📈 Backtest</h2>
        <div class="flex items-center gap-3">
            <TickerSelector bind:selected={ticker} onSelect={() => {}} />
            <DateRangeFilter bind:startDate bind:endDate onApply={applyFilter} />
        </div>
    </div>

    {#if s?.total_trades}
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard label="Net P&L" value={formatCurrency(s.net_profit)} trend={s.net_profit >= 0 ? 'up' : 'down'} />
            <MetricCard label="Win Rate" value={formatPercent(s.win_rate * 100)} trend={s.win_rate >= 0.5 ? 'up' : 'down'} sublabel={`${s.total_trades} trades`} />
            <MetricCard label="Sharpe" value={s.sharpe_ratio.toFixed(2)} trend={s.sharpe_ratio >= 1 ? 'up' : 'neutral'} />
            <MetricCard label="Max DD" value={formatPercent(s.max_drawdown_pct * 100)} trend="down" />
        </div>
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <EquityCurve data={backtestStore.equityCurve} />
            <DrawdownCurve data={backtestStore.drawdownCurve} />
        </div>
        <TickerBreakdown breakdown={backtestStore.tickerBreakdown} />
    {:else if backtestStore.loading}
        <p class="text-slate-400">Loading...</p>
    {:else}
        <p class="text-slate-400">No backtest results for this period.</p>
    {/if}
</div>
