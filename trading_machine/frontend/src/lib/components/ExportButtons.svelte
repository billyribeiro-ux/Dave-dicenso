<script lang="ts">
    let { startDate = '2023-01-01', endDate = '2026-07-10', ticker = '', onExport = (_f: 'csv'|'excel') => {} } = $props();
</script>
<div class="flex items-center gap-2">
    <button class="btn-secondary text-sm" onclick={() => onExport('csv')}>📥 Export CSV</button>
    <button class="btn-secondary text-sm" onclick={() => onExport('excel')}>📊 Export Excel</button>
</div>
