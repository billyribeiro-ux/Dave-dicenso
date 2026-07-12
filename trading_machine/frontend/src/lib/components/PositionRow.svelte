<script lang="ts">
    import type { Position } from '$types/signal';
    import { formatCurrency, formatPercent } from '$utils/formatters';
    let { position } = $props<{ position: Position }>();
</script>

<tr class="border-b border-slate-700/50 hover:bg-slate-700/30 transition-colors">
    <td class="py-3 px-4 font-medium text-white">{position.ticker}</td>
    <td class="py-3 px-4">
        <span class={position.direction === 'LONG' ? 'badge-profit' : 'badge-loss'}>{position.direction}</span>
    </td>
    <td class="py-3 px-4 font-mono text-slate-300">${position.entry_price.toFixed(2)}</td>
    <td class="py-3 px-4 font-mono text-white">${position.current_price.toFixed(2)}</td>
    <td class="py-3 px-4 font-mono {position.unrealized_pnl >= 0 ? 'metric-up' : 'metric-down'}">
        {formatCurrency(position.unrealized_pnl)} ({formatPercent(position.unrealized_pnl_pct)})
    </td>
    <td class="py-3 px-4 text-slate-400">{position.duration_minutes.toFixed(0)}m</td>
</tr>
