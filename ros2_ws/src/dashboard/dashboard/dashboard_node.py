"""
ROS2 + FastAPI in one process.
- rclpy runs in a background thread, populates self.state under self.state_lock.
- FastAPI runs uvicorn on the main thread, reads state on each WS tick.
"""
import asyncio
import json
import threading
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from custom_msgs.msg import MLClassification


# Shared state — accessed by both ROS thread and FastAPI thread.
state = {
    'vision_label': 'unknown',
    'vision_confidence': 0.0,
    'vision_inference_ms': 0.0,
}
state_lock = threading.Lock()


class DashboardNode(Node):
    def __init__(self):
        super().__init__('dashboard')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub_vision = self.create_subscription(
            MLClassification, '/ml/vision', self.on_vision, qos
        )
        self.get_logger().info('dashboard ROS node ready')

    def on_vision(self, msg):
        with state_lock:
            state['vision_label'] = msg.label
            state['vision_confidence'] = msg.confidence
            state['vision_inference_ms'] = msg.inference_ms


# ── FastAPI app ──────────────────────────────────────────────────────
app = FastAPI()

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Roboception</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif;
         background: #111; color: #eee;
         display: flex; align-items: center; justify-content: center;
         height: 100vh; margin: 0; }
  .card { text-align: center; padding: 40px 80px;
          border: 1px solid #333; border-radius: 12px; }
  .label { font-size: 4rem; font-weight: 600; margin: 0; }
  .conf  { font-size: 1.2rem; color: #888; margin-top: 10px; }
  .meta  { font-size: 0.9rem; color: #555; margin-top: 30px; }
</style>
</head>
<body>
<div class="card">
  <p class="label" id="label">—</p>
  <p class="conf"  id="conf">—</p>
  <p class="meta"  id="meta">connecting...</p>
</div>
<script>
const ws = new WebSocket(`ws://${location.host}/ws`);
ws.onmessage = (e) => {
  const s = JSON.parse(e.data);
  document.getElementById('label').textContent = s.vision_label;
  document.getElementById('conf').textContent =
    `confidence ${(s.vision_confidence * 100).toFixed(0)}%`;
  document.getElementById('meta').textContent =
    `inference ${s.vision_inference_ms.toFixed(1)} ms`;
};
ws.onclose = () => {
  document.getElementById('meta').textContent = 'disconnected';
};
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return INDEX_HTML


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            with state_lock:
                snapshot = dict(state)
            await ws.send_text(json.dumps(snapshot))
            await asyncio.sleep(0.5)  # 2 Hz updates
    except WebSocketDisconnect:
        pass


# ── Glue ─────────────────────────────────────────────────────────────
def ros_thread_main():
    rclpy.init()
    node = DashboardNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    ros_thread = threading.Thread(target=ros_thread_main, daemon=True)
    ros_thread.start()

    # uvicorn blocks the main thread, which is what we want.
    uvicorn.run(app, host='0.0.0.0', port=8765, log_level='warning')


if __name__ == '__main__':
    main()