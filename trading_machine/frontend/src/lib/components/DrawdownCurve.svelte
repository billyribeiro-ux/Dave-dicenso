<script lang="ts">
    import type { DrawdownPoint } from '$types/backtest';
    let { data = [] as DrawdownPoint[] } = $props();

    const chartId = 'dd-' + Math.random().toString(36).slice(2, 8);
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
                        label: 'Drawdown %',
                        data: data.map(d => d.drawdown_pct * 100),
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239,68,68,0.1)',
                        fill: true,
                        tension: 0.2,
                        pointRadius: 0,
                    }],
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { ticks: { color: '#94a3b8', maxTicksLimit: 10 } }, y: { ticks: { color: '#94a3b8', callback: (v: any) => `${v}%` }, reverse: true } } },
            });
        };
        initChart();
        return () => { if (chart) chart.destroy(); };
    });
</script>

<div class="card">
    <h3 class="text-sm font-semibold text-slate-300 mb-3">Drawdown</h3>
    <div class="h-48">
        <canvas bind:this={canvas} id={chartId}></canvas>
    </div>
</div>
