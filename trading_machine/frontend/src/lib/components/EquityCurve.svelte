<script lang="ts">
    import type { EquityPoint } from '$types/backtest';
    import { formatCurrency } from '$utils/formatters';
    let { data = [] as EquityPoint[] } = $props();

    const chartId = 'equity-' + Math.random().toString(36).slice(2, 8);
    let canvas: HTMLCanvasElement;
    let chart: any = null;

    $effect(() => {
        if (!canvas || data.length === 0) return;
        const initChart = async () => {
            const { Chart, LineController, LineElement, PointElement, LinearScale, CategoryScale, Filler, Tooltip } = await import('chart.js');
            Chart.register(LineController, LineElement, PointElement, LinearScale, CategoryScale, Filler, Tooltip);
            if (chart) chart.destroy();
            chart = new Chart(canvas, {
                type: 'line',
                data: {
                    labels: data.map(d => d.date.slice(0, 10)),
                    datasets: [{
                        label: 'Equity',
                        data: data.map(d => d.equity),
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59,130,246,0.1)',
                        fill: true,
                        tension: 0.2,
                        pointRadius: 0,
                    }],
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { ticks: { color: '#94a3b8', maxTicksLimit: 10 } }, y: { ticks: { color: '#94a3b8', callback: (v: any) => formatCurrency(v) } } } },
            });
        };
        initChart();
        return () => { if (chart) chart.destroy(); };
    });
</script>

<div class="card">
    <h3 class="text-sm font-semibold text-slate-300 mb-3">Equity Curve</h3>
    <div class="h-64">
        <canvas bind:this={canvas} id={chartId}></canvas>
    </div>
</div>
