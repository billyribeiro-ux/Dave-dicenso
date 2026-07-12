<script lang="ts">
    import type { Signal } from '$types/signal';
    let { signal } = $props<{ signal: Signal }>();

    const signalColors: Record<string, string> = {
        BUY: 'border-l-profit bg-profit/5',
        SELL: 'border-l-loss bg-loss/5',
        NEUTRAL: 'border-l-neutral bg-neutral/5',
    };
    const signalBadges: Record<string, string> = {
        BUY: 'badge-profit',
        SELL: 'badge-loss',
        NEUTRAL: 'badge-neutral',
    };
</script>

<div class="card border-l-4 {signalColors[signal.signal] || 'border-l-slate-600'} flex items-center justify-between">
    <div class="flex items-center gap-4">
        <span class="text-lg font-bold text-white">{signal.ticker}</span>
        <span class={signalBadges[signal.signal] || 'badge-neutral'}>{signal.signal}</span>
    </div>
    <div class="flex items-center gap-6 text-sm">
        {#if signal.entry_price}
            <div class="text-right">
                <span class="text-slate-400 text-xs">Entry</span>
                <p class="text-white font-mono">${signal.entry_price.toFixed(2)}</p>
            </div>
        {/if}
        <div class="text-right">
            <span class="text-slate-400 text-xs">Confidence</span>
            <p class="text-white font-mono">{(signal.confidence * 100).toFixed(1)}%</p>
        </div>
        {#if signal.regime}
            <div class="text-right">
                <span class="text-slate-400 text-xs">Regime</span>
                <p class="text-slate-300">{signal.regime}</p>
            </div>
        {/if}
    </div>
</div>
