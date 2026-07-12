<script lang="ts">
    import { websocketStore } from '$stores/websocket.svelte.ts';
    import { systemStore } from '$stores/system.svelte.ts';
    import StatusIndicator from './StatusIndicator.svelte';
</script>

<header class="h-14 bg-surface border-b border-slate-700 flex items-center justify-between px-6 shrink-0">
    <div class="flex items-center gap-4">
        <StatusIndicator connected={websocketStore.connected} />
        <span class="text-sm text-slate-400">
            {websocketStore.connected ? 'Live data connected' : 'Connecting...'}
        </span>
    </div>
    <div class="flex items-center gap-4 text-sm text-slate-400">
        <span>Models: {systemStore.trainedCount}/{systemStore.totalTickers}</span>
        <span>Status: {systemStore.status?.status ?? 'Unknown'}</span>
    </div>
</header>
