<script lang="ts">
    import { TICKERS } from '$utils/constants';
    let { selected = $bindable(''), onSelect = (_t: string) => {} } = $props();
</script>
<select bind:value={selected} onchange={() => onSelect(selected)} class="bg-surface-light border border-slate-600 rounded-lg px-3 py-1.5 text-sm text-white">
    <option value="">All Tickers</option>
    {#each TICKERS as t}
        <option value={t}>{t}</option>
    {/each}
</select>
