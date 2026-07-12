<script lang="ts">
    import type { Trade } from '$types/trade';
    import { formatCurrency, formatPercent, formatDateTime } from '$utils/formatters';
    let { trade } = $props<{ trade: Trade }>();
</script>

<tr class="border-b border-slate-700/50 hover:bg-slate-700/30 transition-colors text-sm">
    <td class="py-2 px-3 font-medium text-white">{trade.ticker}</td>
    <td class="py-2 px-3">
        <span class={trade.direction === 'LONG' ? 'badge-profit' : 'badge-loss'}>{trade.direction}</span>
    </td>
    <td class="py-2 px-3 text-slate-400">{formatDateTime(trade.entry_time)}</td>
    <td class="py-2 px-3 text-slate-400">{trade.exit_time ? formatDateTime(trade.exit_time) : '—'}</td>
    <td class="py-2 px-3 font-mono {trade.pnl >= 0 ? 'metric-up' : 'metric-down'}">
        {formatCurrency(trade.pnl)}
    </td>
    <td class="py-2 px-3 text-slate-400 text-xs max-w-[120px] truncate" title={trade.exit_reason}>
        {trade.exit_reason}
    </td>
</tr>
