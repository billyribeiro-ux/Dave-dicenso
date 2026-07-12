import { signalsStore } from './signals.svelte.ts';
import { positionsStore } from './positions.svelte.ts';
import { systemStore } from './system.svelte.ts';

class WebSocketStore {
    connected = $state(false);
    private ws: WebSocket | null = null;
    private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    private reconnectDelay = 1000;

    connect() {
        if (this.ws) return;
        try {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            this.ws = new WebSocket(`${protocol}//${location.host}/ws/live`);
            this.ws.onopen = () => {
                this.connected = true;
                this.reconnectDelay = 1000;
                console.log('[WS] Connected');
            };
            this.ws.onmessage = (event) => {
                try {
                    const msg = JSON.parse(event.data);
                    this.handleMessage(msg);
                } catch (e) {
                    console.error('[WS] Parse error:', e);
                }
            };
            this.ws.onclose = () => {
                this.connected = false;
                this.ws = null;
                this.scheduleReconnect();
            };
            this.ws.onerror = () => {
                this.ws?.close();
            };
        } catch (e) {
            this.scheduleReconnect();
        }
    }

    disconnect() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        this.ws?.close();
        this.ws = null;
        this.connected = false;
    }

    private scheduleReconnect() {
        if (this.reconnectTimer) return;
        this.reconnectTimer = setTimeout(() => {
            this.reconnectTimer = null;
            this.reconnectDelay = Math.min(this.reconnectDelay * 2, 30000);
            this.connect();
        }, this.reconnectDelay);
    }

    private handleMessage(msg: { type: string; data: any }) {
        switch (msg.type) {
            case 'signals':
                if (msg.data?.signals) {
                    signalsStore.signals = msg.data.signals;
                    signalsStore.lastUpdate = msg.data.timestamp || '';
                }
                break;
            case 'positions':
                if (msg.data?.positions) {
                    positionsStore.positions = msg.data.positions;
                }
                break;
            case 'system_status':
                if (msg.data) {
                    systemStore.status = msg.data;
                }
                break;
        }
    }
}

export const websocketStore = new WebSocketStore();
