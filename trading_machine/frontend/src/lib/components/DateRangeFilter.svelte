<script lang="ts">
    import { DEFAULT_START_DATE, DEFAULT_END_DATE } from '$utils/constants';

    let {
        startDate = $bindable(DEFAULT_START_DATE),
        endDate = $bindable(DEFAULT_END_DATE),
        onApply = () => {},
    } = $props();
</script>

<div class="flex items-center gap-3 flex-wrap">
    <div class="flex items-center gap-2">
        <label class="text-sm text-slate-400">From</label>
        <input type="date" bind:value={startDate} class="bg-surface-light border border-slate-600 rounded-lg px-3 py-1.5 text-sm text-white" />
    </div>
    <div class="flex items-center gap-2">
        <label class="text-sm text-slate-400">To</label>
        <input type="date" bind:value={endDate} class="bg-surface-light border border-slate-600 rounded-lg px-3 py-1.5 text-sm text-white" />
    </div>
    <button class="btn-primary text-sm" onclick={onApply}>Apply</button>
</div>
