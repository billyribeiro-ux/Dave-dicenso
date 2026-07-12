<script lang="ts">
    let { label = '', value = '', trend = '' as 'up' | 'down' | 'neutral', sublabel = '' } = $props();
</script>

<div class="card flex flex-col gap-1">
    <span class="text-xs text-slate-400 uppercase tracking-wider">{label}</span>
    <span class="text-2xl font-bold {trend === 'up' ? 'metric-up' : trend === 'down' ? 'metric-down' : 'text-white'}">
        {value}
    </span>
    {#if sublabel}
        <span class="text-xs text-slate-500">{sublabel}</span>
    {/if}
</div>
