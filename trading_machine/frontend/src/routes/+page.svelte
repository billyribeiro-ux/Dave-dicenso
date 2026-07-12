<script lang="ts">
    import MetricCard from '$components/MetricCard.svelte';
    import EquityCurve from '$components/EquityCurve.svelte';
    import DrawdownCurve from '$components/DrawdownCurve.svelte';
    import { backtestStore } from '$stores/backtest.svelte.ts';
    import { signalsStore } from '$stores/signals.svelte.ts';
    import { positionsStore } from '$stores/positions.svelte.ts';
    import { systemStore } from '$stores/system.svelte.ts';
    import { DEFAULT_START_DATE, DEFAULT_END_DATE } from '$utils/constants';
    import { formatCurrency, formatPercent } from '$utils/formatters';
    import { onMount } from 'svelte';

    onMount(() => {
        backtestStore.fetch(DEFAULT_START_DATE, DEFAULT_END_DATE);
        signalsStore.fetch();
        positionsStore.fetch();
        systemStore.fetch();
    });

    const s = $derived(backtestStore.summary);
</script>

<svelte:head><title>Dashboard — Trading Machine</title></svelte:head>

<div class="space-y-6">
    <h2 class="text-xl font-bold text-white">📊 Dashboard</h2>

    {#if s}
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard label="Net P&L" value={formatCurrency(s.net_profit)} trend={s.net_profit >= 0 ? 'up' : 'down'} />
            <MetricCard label="Win Rate" value={formatPercent(s.win_rate * 100)} trend={s.win_rate >= 0.5 ? 'up' : 'down'} sublabel={`${s.total_trades} trades`} />
            <MetricCard label="Sharpe Ratio" value={s.sharpe_ratio.toFixed(2)} trend={s.sharpe_ratio >= 1 ? 'up' : s.sharpe_ratio >= 0 ? 'neutral' : 'down'} />
            <MetricCard label="Max Drawdown" value={formatPercent(s.max_drawdown_pct * 100)} trend="down" sublabel={`Profit Factor: ${s.profit_factor.toFixed(2)}`} />
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <EquityCurve data={backtestStore.equityCurve} />
            <DrawdownCurve data={backtestStore.drawdownCurve} />
        </div>

        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard label="Avg Win" value={formatCurrency(s.avg_win)} trend="up" />
            <MetricCard label="Avg Loss" value={formatCurrency(s.avg_loss)} trend="down" />
            <MetricCard label="Largest Win" value={formatCurrency(s.largest_win)} trend="up" />
            <MetricCard label="Largest Loss" value={formatCurrency(s.largest_loss)} trend="down" />
        </div>
    {:else}
        <p class="text-slate-400">Loading dashboard data...</p>
    {/if}
</div>
