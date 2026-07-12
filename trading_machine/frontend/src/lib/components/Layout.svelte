<script lang="ts">
    import Sidebar from './Sidebar.svelte';
    import Header from './Header.svelte';
    import { websocketStore } from '$stores/websocket.svelte.ts';
    import { onMount } from 'svelte';

    let { children } = $props();

    onMount(() => {
        websocketStore.connect();
        return () => websocketStore.disconnect();
    });
</script>

<div class="flex h-screen overflow-hidden">
    <Sidebar />
    <div class="flex-1 flex flex-col overflow-hidden">
        <Header />
        <main class="flex-1 overflow-y-auto p-6">
            {@render children?.()}
        </main>
    </div>
</div>
