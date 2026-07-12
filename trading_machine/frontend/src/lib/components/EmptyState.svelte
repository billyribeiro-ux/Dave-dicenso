<script lang="ts">
    let { message = 'No data available yet.' } = $props();
</script>
<div class="card flex flex-col items-center justify-center py-12 text-center">
    <span class="text-4xl mb-4">📭</span>
    <p class="text-slate-400 text-lg">{message}</p>
</div>
