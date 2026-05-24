"""
ROS2 + FastAPI dashboard.

Panels:
  - Live camera feed with YOLO bounding boxes  (/camera/annotated → MJPEG)
  - Vision / Audio / IMU classification cards
  - Audio start/stop button + auto-save clips on AUDIO_TRIGGERED
  - Bandwidth telemetry
"""
import asyncio
import collections
import json
import threading
import time
import wave
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from cv_bridge import CvBridge
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
import uvicorn

from sensor_msgs.msg import Image
from custom_msgs.msg import MLClassification, EnvironmentState, AudioChunk


CLIPS_DIR = Path('/tmp/roboception_clips')
AUDIO_BUF_LEN = 12       # chunks to keep (~12 s at 1 Hz)
CLIP_COOLDOWN = 10.0     # minimum seconds between auto-saves


# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    'env_state': 'CALM',
    'fusion_reason': '',
    'vision_label': 'unknown',  'vision_confidence': 0.0,  'vision_inference_ms': 0.0,
    'audio_label': 'unknown',   'audio_confidence': 0.0,   'audio_inference_ms': 0.0,
    'imu_label': 'unknown',     'imu_confidence': 0.0,     'imu_inference_ms': 0.0,
    'bw_raw_kbps': 0.0, 'bw_actual_kbps': 0.0, 'bw_saved_pct': 0.0,
    'frames_received': 0, 'frames_processed': 0,
    'audio_active': True,
    'audio_clips': [],
}
state_lock = threading.Lock()

_frame_jpg: bytes = b''
_frame_lock = threading.Lock()

_audio_buf: collections.deque = collections.deque(maxlen=AUDIO_BUF_LEN)
_audio_lock = threading.Lock()

_bridge = CvBridge()


# ── ROS node ──────────────────────────────────────────────────────────────────
class DashboardNode(Node):
    def __init__(self):
        super().__init__('dashboard')
        CLIPS_DIR.mkdir(parents=True, exist_ok=True)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(MLClassification, '/ml/vision',  self.on_vision, qos)
        self.create_subscription(MLClassification, '/ml/audio',   self.on_audio,  qos)
        self.create_subscription(MLClassification, '/ml/imu',     self.on_imu,    qos)
        self.create_subscription(EnvironmentState, '/environment/state', self.on_env, qos)
        self.create_subscription(Image,       '/camera/annotated',  self.on_annotated,   qos)
        self.create_subscription(AudioChunk,  '/audio/chunk',       self.on_audio_chunk, qos)

        self._last_save = 0.0
        self.get_logger().info('dashboard node ready')

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
            prev = state['env_state']
            state['env_state'] = msg.env_state
            state['fusion_reason'] = msg.fusion_reason
            state['bw_raw_kbps'] = msg.bandwidth_raw_kbps
            state['bw_actual_kbps'] = msg.bandwidth_actual_kbps
            state['bw_saved_pct'] = msg.bandwidth_saved_pct
            state['frames_received'] = msg.frames_received
            state['frames_processed'] = msg.frames_processed
            active = state['audio_active']
            audio_label = state['audio_label']

        # Save clip on rising edge of AUDIO_TRIGGERED
        now = time.time()
        if (msg.env_state == 'AUDIO_TRIGGERED' and
                prev != 'AUDIO_TRIGGERED' and
                active and
                now - self._last_save > CLIP_COOLDOWN):
            self._last_save = now
            self._save_clip(audio_label)

    def on_annotated(self, msg):
        global _frame_jpg
        try:
            frame = _bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                with _frame_lock:
                    _frame_jpg = buf.tobytes()
        except Exception as e:
            self.get_logger().warn(f'annotated frame: {e}', throttle_duration_sec=2.0)

    def on_audio_chunk(self, msg):
        with _audio_lock:
            _audio_buf.append(msg)

    def _save_clip(self, label: str):
        with _audio_lock:
            chunks = list(_audio_buf)
        if not chunks:
            return
        ts = time.strftime('%H%M%S')
        safe = label.replace(' ', '_')[:20]
        filename = f'{ts}_{safe}.wav'
        try:
            sr = chunks[0].sample_rate
            combined = np.concatenate([
                (np.array(c.samples) * 32767).astype(np.int16) for c in chunks
            ])
            with wave.open(str(CLIPS_DIR / filename), 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(combined.tobytes())
            with state_lock:
                state['audio_clips'] = [filename] + state['audio_clips'][:9]
            self.get_logger().info(f'saved clip: {filename}')
        except Exception as e:
            self.get_logger().warn(f'clip save: {e}')


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI()


async def _mjpeg_gen():
    """Yield latest annotated frame as MJPEG at ~15 fps."""
    while True:
        with _frame_lock:
            jpg = _frame_jpg
        if jpg:
            yield b'--f\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n'
        await asyncio.sleep(1 / 15)


@app.get('/video_feed')
async def video_feed():
    return StreamingResponse(
        _mjpeg_gen(),
        media_type='multipart/x-mixed-replace; boundary=f',
    )


@app.websocket('/ws')
async def ws_state(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            with state_lock:
                snap = dict(state)
            await ws.send_text(json.dumps(snap))
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass


@app.post('/audio/toggle')
async def audio_toggle():
    with state_lock:
        state['audio_active'] = not state['audio_active']
        return {'audio_active': state['audio_active']}


@app.get('/clips/{name}')
async def get_clip(name: str):
    path = CLIPS_DIR / Path(name).name   # strip any path components
    if not path.exists():
        raise HTTPException(status_code=404)
    return FileResponse(str(path), media_type='audio/wav', filename=name)


INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Roboception</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,system-ui,sans-serif;background:#0e0e10;color:#eee;min-height:100vh;padding:20px}
.wrap{max-width:1260px;margin:0 auto;display:flex;flex-direction:column;gap:18px}

.banner{padding:18px 32px;border-radius:14px;background:#18181b;border:1px solid #2a2a2e;text-align:center;transition:background .4s,border-color .4s}
.banner.calm{background:#18181b;border-color:#2a2a2e}
.banner.event{background:#1e3a5f;border-color:#3b82f6}
.banner.alert{background:#5f1e1e;border-color:#ef4444}
.banner.warn{background:#5f4a1e;border-color:#f59e0b}
.banner-state{font-size:1.6rem;font-weight:700;letter-spacing:.1em}
.banner-reason{font-size:.85rem;color:#888;margin-top:6px}

.panel{background:#18181b;border:1px solid #2a2a2e;border-radius:14px;overflow:hidden}
.panel-title{font-size:.7rem;color:#888;letter-spacing:.25em;padding:10px 14px 8px;border-bottom:1px solid #2a2a2e}
#cam-img{max-height:600px;width:auto;margin:0 auto;display:block;aspect-ratio:16/10;object-fit:cover;background:#000}
.no-feed{aspect-ratio:16/10;display:flex;align-items:center;justify-content:center;color:#444;font-size:.8rem;letter-spacing:.2em}

.deck{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}
.card{background:#18181b;border:1px solid #2a2a2e;border-radius:14px;padding:22px;text-align:center}
.card-title{font-size:.7rem;color:#888;letter-spacing:.25em;margin-bottom:12px}
.card-label{font-size:2rem;font-weight:600;min-height:2.6rem;line-height:1.2;word-break:break-word}
.card-conf{font-size:.95rem;color:#b8b8c0;margin-top:10px}
.card-meta{font-size:.7rem;color:#555;margin-top:8px}

.audio-btn{margin-top:14px;padding:7px 18px;border-radius:8px;font-size:.72rem;font-weight:600;letter-spacing:.1em;cursor:pointer;border:1px solid;transition:all .2s}
.audio-btn.running{background:#1c3a2e;border-color:#4ade80;color:#4ade80}
.audio-btn.stopped{background:#3a1c1c;border-color:#ef4444;color:#ef4444}

.bw{background:#18181b;border:1px solid #2a2a2e;border-radius:14px;padding:18px 28px}
.bw-title{font-size:.7rem;color:#888;letter-spacing:.25em;margin-bottom:12px;text-align:center}
.bw-row{display:flex;justify-content:space-around;align-items:baseline}
.bw-cell{text-align:center;flex:1}
.bw-num{font-size:1.4rem;font-weight:600}
.bw-num.savings{color:#4ade80}
.bw-lbl{font-size:.65rem;color:#666;letter-spacing:.15em;margin-top:4px}

.clips{background:#18181b;border:1px solid #2a2a2e;border-radius:14px;padding:16px 22px}
.clips-title{font-size:.7rem;color:#888;letter-spacing:.25em;margin-bottom:10px}
.clip-list{display:flex;flex-wrap:wrap;gap:8px}
.clip-a{font-size:.72rem;color:#60a5fa;text-decoration:none;background:#1e2a3a;border:1px solid #3b82f6;border-radius:6px;padding:4px 10px}
.clip-a:hover{background:#2a3a5a}
.clip-empty{font-size:.72rem;color:#555}

.conn{position:fixed;top:14px;right:18px;font-size:.7rem;color:#555;letter-spacing:.1em}
.conn.live{color:#4ade80}
</style>
</head>
<body>
<div class="conn" id="conn">CONNECTING</div>
<div class="wrap">

  <div class="banner calm" id="banner">
    <div class="banner-state" id="env_state">—</div>
    <div class="banner-reason" id="fusion_reason">—</div>
  </div>

  <div class="panel">
    <div class="panel-title">CAMERA · YOLO INFERENCE</div>
    <img id="cam-img" src="/video_feed" alt=""
         onerror="this.style.display='none';document.getElementById('no-feed').style.display='flex'">
    <div class="no-feed" id="no-feed" style="display:none">NO FEED</div>
  </div>

  <div class="deck">
    <div class="card">
      <div class="card-title">VISION</div>
      <p class="card-label" id="vision_label">—</p>
      <p class="card-conf"  id="vision_conf">—</p>
      <p class="card-meta"  id="vision_meta">—</p>
    </div>
    <div class="card">
      <div class="card-title">AUDIO</div>
      <p class="card-label" id="audio_label">—</p>
      <p class="card-conf"  id="audio_conf">—</p>
      <p class="card-meta"  id="audio_meta">—</p>
      <button class="audio-btn running" id="audio-btn">■ STOP</button>
    </div>
    <div class="card">
      <div class="card-title">IMU</div>
      <p class="card-label" id="imu_label">—</p>
      <p class="card-conf"  id="imu_conf">—</p>
      <p class="card-meta"  id="imu_meta">—</p>
    </div>
  </div>

  <div class="bw">
    <div class="bw-title">BANDWIDTH</div>
    <div class="bw-row">
      <div class="bw-cell"><div class="bw-num" id="bw_raw">—</div><div class="bw-lbl">RAW Kbps</div></div>
      <div class="bw-cell"><div class="bw-num" id="bw_actual">—</div><div class="bw-lbl">ACTUAL Kbps</div></div>
      <div class="bw-cell"><div class="bw-num savings" id="bw_saved">—</div><div class="bw-lbl">SAVED</div></div>
      <div class="bw-cell"><div class="bw-num" id="bw_frames">—</div><div class="bw-lbl">FRAMES / S</div></div>
    </div>
  </div>

  <div class="clips">
    <div class="clips-title">AUDIO CLIPS</div>
    <div class="clip-list" id="clip-list"><span class="clip-empty">no clips yet</span></div>
  </div>

</div>

<script>
const STATE_CLASS = {
  AUDIO_TRIGGERED: 'alert', IMU_DISTURBANCE: 'warn',
  VISUAL_EVENT: 'event', MULTIMODAL_DISAGREEMENT: 'warn', CALM: 'calm',
};
const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };

const ws = new WebSocket(`ws://${location.host}/ws`);
ws.onopen  = () => { const c = document.getElementById('conn'); c.textContent = 'LIVE'; c.classList.add('live'); };
ws.onclose = () => { const c = document.getElementById('conn'); c.textContent = 'DISCONNECTED'; c.classList.remove('live'); };
ws.onmessage = (e) => {
  const s = JSON.parse(e.data);

  set('env_state',    s.env_state);
  set('fusion_reason', s.fusion_reason || 'no events');
  document.getElementById('banner').className = 'banner ' + (STATE_CLASS[s.env_state] || 'calm');

  for (const m of ['vision', 'imu']) {
    set(`${m}_label`, s[`${m}_label`]);
    set(`${m}_conf`,  `${(s[`${m}_confidence`] * 100).toFixed(0)}%`);
    set(`${m}_meta`,  `${s[`${m}_inference_ms`].toFixed(1)} ms`);
  }

  const btn = document.getElementById('audio-btn');
  if (!s.audio_active) {
    set('audio_label', 'PAUSED'); set('audio_conf', '—'); set('audio_meta', '—');
    btn.textContent = '▶ START'; btn.className = 'audio-btn stopped';
  } else {
    set('audio_label', s.audio_label);
    set('audio_conf',  `${(s.audio_confidence * 100).toFixed(0)}%`);
    set('audio_meta',  `${s.audio_inference_ms.toFixed(1)} ms`);
    btn.textContent = '■ STOP'; btn.className = 'audio-btn running';
  }

  set('bw_raw',    s.bw_raw_kbps.toFixed(1));
  set('bw_actual', s.bw_actual_kbps.toFixed(1));
  set('bw_saved',  `${s.bw_saved_pct.toFixed(0)}%`);
  set('bw_frames', `${s.frames_processed} / ${s.frames_received}`);

  const list = document.getElementById('clip-list');
  if (s.audio_clips && s.audio_clips.length) {
    list.innerHTML = s.audio_clips.map(n =>
      `<a class="clip-a" href="/clips/${encodeURIComponent(n)}" download="${n}">${n}</a>`
    ).join('');
  }
};

document.getElementById('audio-btn').addEventListener('click', () => {
  fetch('/audio/toggle', { method: 'POST' });
});
</script>
</body>
</html>"""


@app.get('/', response_class=HTMLResponse)
async def index():
    return INDEX_HTML


def _ros_main():
    rclpy.init()
    node = DashboardNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    threading.Thread(target=_ros_main, daemon=True).start()
    uvicorn.run(app, host='0.0.0.0', port=8765, log_level='warning')


if __name__ == '__main__':
    main()
