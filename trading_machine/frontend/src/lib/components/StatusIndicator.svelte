<script lang="ts">
    let { connected = false } = $props();
</script>
<span class="inline-block w-2.5 h-2.5 rounded-full {connected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}"></span>
