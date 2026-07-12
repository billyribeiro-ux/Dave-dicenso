<script lang="ts">
    import LearningTimeline from '$components/LearningTimeline.svelte';
    import EmptyState from '$components/EmptyState.svelte';
    import type { LearningLogEntry, ModelVersion } from '$types/system';
    import { onMount } from 'svelte';

    let entries = $state<LearningLogEntry[]>([]);
    let versions = $state<Record<string, string>>({});
    let loading = $state(false);

    onMount(async () => {
        loading = true;
        try {
            const [logRes, verRes] = await Promise.all([
                fetch('/api/learning/log?limit=50'),
                fetch('/api/learning/model-versions'),
            ]);
            const log = await logRes.json();
            const ver = await verRes.json();
            entries = log.entries || [];
            versions = ver.versions || {};
        } catch (e) { console.error(e); } finally { loading = false; }
    });
</script>

<svelte:head><title>Learning — Trading Machine</title></svelte:head>

<div class="space-y-6">
    <h2 class="text-xl font-bold text-white">🧠 Learning Log</h2>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div class="lg:col-span-2">
            {#if entries.length > 0}
                <LearningTimeline {entries} />
            {:else}
                <EmptyState message="No learning events yet" />
            {/if}
        </div>
        <div class="card">
            <h3 class="text-sm font-semibold text-slate-300 mb-3">Model Versions</h3>
            <div class="space-y-2">
                {#each Object.entries(versions) as [ticker, version]}
                    <div class="flex justify-between text-sm py-1 border-b border-slate-700/50 last:border-0">
                        <span class="text-white font-medium">{ticker}</span>
                        <span class="text-slate-400">{version === '0.0.0' ? 'Not trained' : `v${version}`}</span>
                    </div>
                {/each}
            </div>
        </div>
    </div>
</div>
