<script lang="ts">
    import type { LearningLogEntry } from '$types/system';
    let { entries = [] as LearningLogEntry[] } = $props();
</script>
<div class="card">
    <h3 class="text-sm font-semibold text-slate-300 mb-3">Learning Timeline</h3>
    <div class="space-y-3">
        {#each entries as entry}
            <div class="flex gap-3 text-sm">
                <div class="w-2 h-2 mt-1.5 rounded-full shrink-0 {entry.classification === 'TYPE_A' ? 'bg-yellow-500' : entry.classification === 'TYPE_B' ? 'bg-red-500' : entry.classification === 'TYPE_C' ? 'bg-blue-500' : 'bg-slate-500'}"></div>
                <div class="flex-1 min-w-0">
                    <p class="text-white">{entry.description}</p>
                    <p class="text-slate-500 text-xs mt-0.5">{entry.ticker} · {entry.action_taken}</p>
                </div>
            </div>
        {/each}
    </div>
</div>
