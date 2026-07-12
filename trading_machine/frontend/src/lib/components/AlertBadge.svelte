<script lang="ts">
    import { formatCurrency } from '$utils/formatters';
    let { totalPnl = 0, count = 0 } = $props();
</script>
<div class="flex items-center gap-2 text-sm">
    <span class={totalPnl >= 0 ? 'text-profit' : 'text-loss'}>
        {formatCurrency(totalPnl)}
    </span>
    <span class="text-slate-500">({count} open)</span>
</div>
