<script lang="ts">
    import { signalsStore } from '$stores/signals.svelte.ts';
    import { positionsStore } from '$stores/positions.svelte.ts';
    import SignalCard from '$components/SignalCard.svelte';
    import PositionRow from '$components/PositionRow.svelte';
    import EmptyState from '$components/EmptyState.svelte';
    import { onMount } from 'svelte';

    onMount(() => { signalsStore.fetch(); positionsStore.fetch(); });
</script>

<svelte:head><title>Signals — Trading Machine</title></svelte:head>

<div class="space-y-6">
    <h2 class="text-xl font-bold text-white">📡 Live Signals</h2>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div class="space-y-3">
            <h3 class="text-sm font-semibold text-slate-400 uppercase">Active Signals</h3>
            {#if signalsStore.signals.length > 0}
                {#each signalsStore.signals as signal}
                    <SignalCard {signal} />
                {/each}
            {:else}
                <EmptyState message="No active signals" />
            {/if}
        </div>
        <div class="space-y-3">
            <h3 class="text-sm font-semibold text-slate-400 uppercase">Open Positions</h3>
            {#if positionsStore.positions.length > 0}
                <div class="card overflow-x-auto">
                    <table class="w-full text-sm">
                        <thead>
                            <tr class="text-slate-400 text-xs uppercase">
                                <th class="text-left py-2 px-4">Ticker</th><th class="text-left py-2 px-4">Dir</th>
                                <th class="text-right py-2 px-4">Entry</th><th class="text-right py-2 px-4">Current</th>
                                <th class="text-right py-2 px-4">P&L</th><th class="text-right py-2 px-4">Time</th>
                            </tr>
                        </thead>
                        <tbody>
                            {#each positionsStore.positions as pos}
                                <PositionRow position={pos} />
                            {/each}
                        </tbody>
                    </table>
                </div>
            {:else}
                <EmptyState message="No open positions" />
            {/if}
        </div>
    </div>
</div>
