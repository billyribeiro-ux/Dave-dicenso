<script lang="ts">
    import { page } from '$app/stores';
    import { systemStore } from '$stores/system.svelte.ts';

    const navItems = [
        { href: '/', label: 'Dashboard', icon: '📊' },
        { href: '/signals', label: 'Signals', icon: '📡' },
        { href: '/backtest', label: 'Backtest', icon: '📈' },
        { href: '/trades', label: 'Trades', icon: '💹' },
        { href: '/learning', label: 'Learning', icon: '🧠' },
        { href: '/settings', label: 'Settings', icon: '⚙️' },
    ];
</script>

<aside class="w-64 bg-surface border-r border-slate-700 flex flex-col shrink-0">
    <div class="p-4 border-b border-slate-700">
        <h1 class="text-lg font-bold text-blue-400">🤖 Trading Machine</h1>
        <p class="text-xs text-slate-400 mt-1">Autonomous Self-Learning</p>
    </div>
    <nav class="flex-1 p-2 space-y-1">
        {#each navItems as item}
            {@const isActive = $page.url.pathname === item.href}
            <a
                href={item.href}
                class="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors {isActive
                    ? 'bg-blue-600/20 text-blue-400 font-medium'
                    : 'text-slate-300 hover:bg-slate-700/50 hover:text-white'}"
            >
                <span>{item.icon}</span>
                <span>{item.label}</span>
            </a>
        {/each}
    </nav>
    <div class="p-4 border-t border-slate-700">
        <div class="flex items-center gap-2 text-xs text-slate-400">
            <span class="inline-block w-2 h-2 rounded-full {systemStore.isLive ? 'bg-green-500' : 'bg-yellow-500'}"></span>
            <span>{systemStore.isLive ? 'Live' : 'Ready'}</span>
        </div>
        <p class="text-xs text-slate-500 mt-1">{systemStore.trainedCount}/{systemStore.totalTickers} trained</p>
    </div>
</aside>
