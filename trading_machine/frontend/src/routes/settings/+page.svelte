<script lang="ts">
    import { systemStore } from '$stores/system.svelte.ts';
    import { formatDuration } from '$utils/formatters';
    import { onMount } from 'svelte';

    onMount(() => { systemStore.fetch(); });
    const s = $derived(systemStore.status);
</script>

<svelte:head><title>Settings — Trading Machine</title></svelte:head>

<div class="space-y-6 max-w-2xl">
    <h2 class="text-xl font-bold text-white">⚙️ System Status</h2>

    {#if s}
        <div class="card space-y-4">
            <div class="flex justify-between"><span class="text-slate-400">Status</span><span class="badge-profit">{s.status}</span></div>
            <div class="flex justify-between"><span class="text-slate-400">Uptime</span><span class="text-white">{formatDuration(s.uptime_seconds)}</span></div>
            <div class="flex justify-between"><span class="text-slate-400">Trained Models</span><span class="text-white">{s.trained_models}/{s.active_tickers}</span></div>
            <div class="flex justify-between"><span class="text-slate-400">Active Positions</span><span class="text-white">{s.active_positions}</span></div>
            <div class="flex justify-between"><span class="text-slate-400">API Calls Today</span><span class="text-white">{s.api_calls_today}</span></div>
        </div>

        <div class="card">
            <h3 class="text-sm font-semibold text-slate-300 mb-3">Model Versions</h3>
            {#each Object.entries(s.current_model_versions) as [t, v]}
                <div class="flex justify-between text-sm py-1 border-b border-slate-700/50 last:border-0">
                    <span class="text-white">{t}</span>
                    <span class="text-slate-400">{v === '0.0.0' ? 'Not trained' : v}</span>
                </div>
            {/each}
        </div>
    {:else}
        <p class="text-slate-400">Loading system status...</p>
    {/if}
</div>
