"""
backend/api/websocket.py — WebSocket manager for real-time data streaming.
"""

from fastapi import WebSocket
from typing import Set, Dict, Any
import asyncio
import json
from datetime import datetime

from utils.logger import get_logger

logger = get_logger()


class WebSocketManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message_type: str, data: Dict[str, Any]):
        message = json.dumps({
            "type": message_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }, default=str)
        dead: Set[WebSocket] = set()
        async with self._lock:
            connections = self.active_connections.copy()
        for connection in connections:
            try:
                await connection.send_text(message)
            except Exception:
                dead.add(connection)
        if dead:
            async with self._lock:
                self.active_connections -= dead

    async def broadcast_signals(self, signals: list):
        await self.broadcast("signals", {"signals": signals})

    async def broadcast_positions(self, positions: list):
        await self.broadcast("positions", {"positions": positions})

    async def broadcast_system_status(self, status: dict):
        await self.broadcast("system_status", status)

    @property
    def connection_count(self) -> int:
        return len(self.active_connections)


ws_manager = WebSocketManager()
