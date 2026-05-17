"""
ROS2 + FastAPI in one process. Renders fused state banner, three modality
cards, and a bandwidth telemetry panel.
"""
import asyncio
import json
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from custom_msgs.msg import MLClassification, EnvironmentState


state = {
    'env_state': 'CALM',
    'fusion_reason': '',

    'vision_label': 'unknown',
    'vision_confidence': 0.0,
    'vision_inference_ms': 0.0,

    'audio_label': 'unknown',
    'audio_confidence': 0.0,
    'audio_inference_ms': 0.0,

    'imu_label': 'unknown',
    'imu_confidence': 0.0,
    'imu_inference_ms': 0.0,

    'bw_raw_kbps': 0.0,
    'bw_actual_kbps': 0.0,
    'bw_saved_pct': 0.0,
    'frames_received': 0,
    'frames_processed': 0,
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

        self.create_subscription(MLClassification, '/ml/vision', self.on_vision, qos)
        self.create_subscription(MLClassification, '/ml/audio',  self.on_audio,  qos)
        self.create_subscription(MLClassification, '/ml/imu',    self.on_imu,    qos)
        self.create_subscription(EnvironmentState, '/environment/state',
                                 self.on_env, qos)
        self.get_logger().info('dashboard ROS node ready')

    def on_vision(self, msg):
        with state_lock:
            state['vision_label'] = msg.label
            state['vision_confidence'] = msg.confidence
            state['vision_inference_ms'] = msg.inference_ms

    def on_audio(self, msg):
        with state_lock:
            state['audio_label'] = msg.label
            state['audio_confidence'] = msg.confidence
            state['audio_inference_ms'] = msg.inference_ms

    def on_imu(self, msg):
        with state_lock:
            state['imu_label'] = msg.label
            state['imu_confidence'] = msg.confidence
            state['imu_inference_ms'] = msg.inference_ms

    def on_env(self, msg):
        with state_lock:
            state['env_state'] = msg.env_state
            state['fusion_reason'] = msg.fusion_reason
            state['bw_raw_kbps'] = msg.bandwidth_raw_kbps
            state['bw_actual_kbps'] = msg.bandwidth_actual_kbps
            state['bw_saved_pct'] = msg.bandwidth_saved_pct
            state['frames_received'] = msg.frames_received
            state['frames_processed'] = msg.frames_processed


app = FastAPI()

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Roboception</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, system-ui, sans-serif;
         background: #0e0e10; color: #eee;
         display: flex; align-items: center; justify-content: center;
         min-height: 100vh; margin: 0; padding: 20px; }
  .wrap { display: flex; flex-direction: column; align-items: center;
          width: 100%; max-width: 1000px; }
  .banner { margin-bottom: 28px; text-align: center;
            padding: 22px 40px; border-radius: 14px;
            background: #18181b; border: 1px solid #2a2a2e;
            min-width: 600px; transition: background 0.4s, border-color 0.4s; }
  .banner-state { font-size: 1.8rem; font-weight: 700; letter-spacing: 0.1em; }
  .banner-reason { font-size: 0.85rem; color: #888; margin-top: 6px;
                   min-height: 1.1rem; }
  .banner.calm  { background: #18181b; border-color: #2a2a2e; }
  .banner.event { background: #1e3a5f; border-color: #3b82f6; }
  .banner.alert { background: #5f1e1e; border-color: #ef4444; }
  .banner.warn  { background: #5f4a1e; border-color: #f59e0b; }
  .deck { display: flex; gap: 24px; flex-wrap: wrap;
          justify-content: center; margin-bottom: 28px; }
  .card { background: #18181b; border: 1px solid #2a2a2e;
          border-radius: 14px; padding: 28px 32px;
          width: 280px; text-align: center; }
  .modality-name { font-size: 0.75rem; color: #888;
                   letter-spacing: 0.25em; margin-bottom: 18px; }
  .label { font-size: 2.6rem; font-weight: 600; margin: 0;
           min-height: 3.4rem; line-height: 1.2; word-break: break-word; }
  .conf  { font-size: 1.1rem; color: #b8b8c0; margin-top: 14px; }
  .meta  { font-size: 0.75rem; color: #555; margin-top: 16px;
           letter-spacing: 0.05em; }
  .bw    { background: #18181b; border: 1px solid #2a2a2e;
           border-radius: 14px; padding: 22px 36px;
           min-width: 600px; }
  .bw-title { font-size: 0.75rem; color: #888;
              letter-spacing: 0.25em; margin-bottom: 14px; text-align: center; }
  .bw-row { display: flex; justify-content: space-around;
            align-items: baseline; gap: 20px; }
  .bw-cell { text-align: center; flex: 1; }
  .bw-num   { font-size: 1.6rem; font-weight: 600; }
  .bw-num.savings { color: #4ade80; }
  .bw-lbl   { font-size: 0.7rem; color: #666; letter-spacing: 0.15em;
              margin-top: 4px; }
  .conn  { position: fixed; top: 16px; right: 20px;
           font-size: 0.7rem; color: #555; letter-spacing: 0.1em; }
  .live  { color: #4ade80; }
</style>
</head>
<body>
<div class="conn" id="conn">CONNECTING</div>
<div class="wrap">
  <div class="banner calm" id="banner">
    <div class="banner-state" id="env_state">—</div>
    <div class="banner-reason" id="fusion_reason">—</div>
  </div>
  <div class="deck">
    <div class="card">
      <div class="modality-name">VISION</div>
      <p class="label" id="vision_label">—</p>
      <p class="conf"  id="vision_conf">—</p>
      <p class="meta"  id="vision_meta">—</p>
    </div>
    <div class="card">
      <div class="modality-name">AUDIO</div>
      <p class="label" id="audio_label">—</p>
      <p class="conf"  id="audio_conf">—</p>
      <p class="meta"  id="audio_meta">—</p>
    </div>
    <div class="card">
      <div class="modality-name">IMU</div>
      <p class="label" id="imu_label">—</p>
      <p class="conf"  id="imu_conf">—</p>
      <p class="meta"  id="imu_meta">—</p>
    </div>
  </div>
  <div class="bw">
    <div class="bw-title">BANDWIDTH</div>
    <div class="bw-row">
      <div class="bw-cell">
        <div class="bw-num" id="bw_raw">—</div>
        <div class="bw-lbl">RAW Kbps</div>
      </div>
      <div class="bw-cell">
        <div class="bw-num" id="bw_actual">—</div>
        <div class="bw-lbl">ACTUAL Kbps</div>
      </div>
      <div class="bw-cell">
        <div class="bw-num savings" id="bw_saved">—</div>
        <div class="bw-lbl">SAVED</div>
      </div>
      <div class="bw-cell">
        <div class="bw-num" id="bw_frames">—</div>
        <div class="bw-lbl">FRAMES / S</div>
      </div>
    </div>
  </div>
</div>
<script>
const STATE_CLASS = {
  'AUDIO_TRIGGERED': 'alert',
  'IMU_DISTURBANCE': 'warn',
  'VISUAL_EVENT': 'event',
  'MULTIMODAL_DISAGREEMENT': 'warn',
  'CALM': 'calm',
};

const ws = new WebSocket(`ws://${location.host}/ws`);
const setText = (id, v) => { document.getElementById(id).textContent = v; };

ws.onopen = () => {
  const c = document.getElementById('conn');
  c.textContent = 'LIVE'; c.classList.add('live');
};
ws.onclose = () => {
  const c = document.getElementById('conn');
  c.textContent = 'DISCONNECTED'; c.classList.remove('live');
};
ws.onmessage = (e) => {
  const s = JSON.parse(e.data);

  for (const m of ['vision', 'audio', 'imu']) {
    setText(`${m}_label`, s[`${m}_label`]);
    setText(`${m}_conf`,  `${(s[`${m}_confidence`] * 100).toFixed(0)}% confidence`);
    setText(`${m}_meta`,  `inference ${s[`${m}_inference_ms`].toFixed(1)} ms`);
  }

  setText('env_state', s.env_state);
  setText('fusion_reason', s.fusion_reason || 'no events');
  document.getElementById('banner').className =
    'banner ' + (STATE_CLASS[s.env_state] || 'calm');

  setText('bw_raw',    s.bw_raw_kbps.toFixed(1));
  setText('bw_actual', s.bw_actual_kbps.toFixed(1));
  setText('bw_saved',  `${s.bw_saved_pct.toFixed(0)}%`);
  setText('bw_frames', `${s.frames_processed} / ${s.frames_received}`);
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
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass


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
    uvicorn.run(app, host='0.0.0.0', port=8765, log_level='warning')


if __name__ == '__main__':
    main()