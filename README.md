# Roboception

**Multimodal environment perception using a smartphone as a sensor suite, ROS 2 as middleware, and on-device ML for real-time scene understanding.**

Roboception turns an Android phone into a robot's eyes, ears, and inertial sensor using the IP Webcam app. Three bridge nodes stream camera, microphone, and IMU data into a ROS 2 graph. Three ML inference nodes classify each modality in real time using ONNX models. A fusion aggregator merges the per-modality labels into a single environment state, and a live web dashboard renders the result over WebSocket.

---

## Architecture

```
┌─────────────────┐     HTTP / MJPEG / WAV / JSON
│  Android Phone  │ ──────────────────────────────────┐
│  (IP Webcam)    │                                   │
└─────────────────┘                                   ▼
                                             ┌────────────────┐
                                             │  phone_bridge  │
                                             │  ├ camera      │──▶ /camera/image_raw (15 Hz)
                                             │  │             │──▶ /camera/camera_info
                                             │  │             │──▶ /camera/telemetry (1 Hz)
                                             │  ├ audio       │──▶ /audio/chunk (1 Hz)
                                             │  └ imu         │──▶ /imu/data (50 Hz)
                                             └────────────────┘
                                                      │
                              ┌───────────────────────┼────────────────────────┐
                              ▼                       ▼                        ▼
                       ┌────────────┐         ┌─────────────┐         ┌──────────────┐
                       │  YOLOv8n   │         │  Audio CNN  │         │  IMU 1D-CNN  │
                       │  (ONNX)    │         │  (ONNX)     │         │  (ONNX)      │
                       └─────┬──────┘         └──────┬──────┘         └──────┬───────┘
                             │                       │                        │
                             ▼                       ▼                        ▼
                       /ml/vision              /ml/audio                 /ml/imu
                       /camera/annotated
                             │                       │                        │
                             └───────────────────────┼────────────────────────┘
                                                     ▼
                                            ┌────────────────┐
                                            │    fusion /    │──▶ /environment/state
                                            │   aggregator   │──▶ /camera/boost_rate
                                            └───────┬────────┘
                                                    │
                                                    ▼
                                            ┌────────────────┐
                                            │   dashboard    │──▶ http://localhost:8765
                                            │ (FastAPI + WS) │
                                            └────────────────┘
```

---

## Packages

| Package | Build type | Description |
|---|---|---|
| `custom_msgs` | `ament_cmake` | `AudioChunk`, `MLClassification`, `EnvironmentState` message definitions |
| `phone_bridge` | `ament_python` | Camera, audio, and IMU bridge nodes (HTTP → ROS 2) |
| `ml_inference` | `ament_python` | Vision (YOLOv8n), audio (mel-CNN), and IMU (1D-CNN) classifiers |
| `fusion` | `ament_python` | Rule-based multimodal aggregator — produces five environment states |
| `dashboard` | `ament_python` | FastAPI + WebSocket live dashboard on port 8765 |
| `roboception_bringup` | `ament_python` | Single launch file (`full_system.launch.py`) for the whole graph |

---

## Topic Graph

| Topic | Type | Hz | Publisher | Subscriber(s) |
|---|---|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` (bgr8, 640×400) | 15–30 | camera_bridge | vision_detector |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | 15–30 | camera_bridge | vision_detector |
| `/camera/annotated` | `sensor_msgs/Image` (bgr8, 640×400) | 15–30 | vision_detector | dashboard |
| `/camera/telemetry` | `std_msgs/Float32MultiArray` | 1 | camera_bridge | aggregator |
| `/camera/boost_rate` | `std_msgs/Bool` | event | aggregator | camera_bridge |
| `/audio/chunk` | `custom_msgs/AudioChunk` | 1 | audio_bridge | audio_classifier, dashboard |
| `/imu/data` | `sensor_msgs/Imu` | 50 | imu_bridge | imu_classifier |
| `/ml/vision` | `custom_msgs/MLClassification` | 15–30 | vision_detector | aggregator, dashboard |
| `/ml/audio` | `custom_msgs/MLClassification` | 1 | audio_classifier | aggregator, dashboard |
| `/ml/imu` | `custom_msgs/MLClassification` | 2 | imu_classifier | aggregator, dashboard |
| `/environment/state` | `custom_msgs/EnvironmentState` | 2 | aggregator | dashboard |

All topics use **BEST_EFFORT / KEEP_LAST** QoS for low-latency delivery.

---

## Nodes

### phone_bridge

**`camera_bridge`** — pulls the MJPEG stream from IP Webcam and republishes as ROS Images.

| Parameter | Default | Notes |
|---|---|---|
| `phone_url` | `http://192.168.0.102:8080` | Base URL of the IP Webcam server |

- Base rate **15 Hz**; boosts to **30 Hz** for 5 seconds when `/camera/boost_rate` receives `True`.
- Publishes `/camera/telemetry` every second: `[raw_kbps, actual_kbps, saved_pct, frames_received, frames_published]`.
- Approximate intrinsics for 640×400 at ~60° FOV are baked in (`fx=fy=554`). Replace after running `camera_calibration`.

**`audio_bridge`** — streams WAV audio from IP Webcam and publishes 1-second chunks.

| Parameter | Default |
|---|---|
| `phone_url` | `http://192.168.0.102:8080` |
| `chunk_duration_sec` | `1.0` |
| `reconnect_delay_sec` | `2.0` |

- Multi-channel audio is mixed to mono.
- Samples are normalised to `float32` in `[-1, 1]`.

**`imu_bridge`** — polls the JSON sensors endpoint and publishes IMU messages at 50 Hz.

| Parameter | Default |
|---|---|
| `phone_url` | `http://192.168.0.102:8080` |
| `poll_rate_hz` | `50.0` |
| `http_timeout_sec` | `1.0` |

- Publishes `linear_acceleration` (m/s²) and `angular_velocity` (rad/s).
- `orientation_covariance[0] = -1` — orientation is **not** estimated.

---

### ml_inference

**`vision_detector`** — runs YOLOv8n on every RGB frame; publishes the top detection and an annotated image.

| Parameter | Default |
|---|---|
| `model_path` | `~/roboception/models/yolov8n.onnx` |
| `labels_path` | `~/roboception/models/labels/coco_labels.txt` |
| `conf_threshold` | `0.35` |
| `nms_threshold` | `0.45` |
| `input_size` | `640` |

- Preprocessing: letterbox pad to 640×640, RGB normalised `[0, 1]`.
- Postprocessing: NMS, top-5 detections returned; box coordinates unproject from letterbox space back to original frame.
- Publishes bounding-box annotated frames on `/camera/annotated`.

**`audio_classifier`** — converts each 1-second audio chunk into a mel-spectrogram and classifies it.

| Parameter | Default |
|---|---|
| `model_path` | `~/roboception/models/audio_model.onnx` |
| `labels_path` | `~/roboception/models/labels/esc50_macro_labels.txt` |

- Preprocessing: resample to 22 050 Hz → 64-mel spectrogram (FFT 1024, hop 512, 44 frames) → z-normalise per channel.
- Input shape: `(1, 1, 64, 44)`. Output: 4 classes — `silence`, `human`, `appliance`, `alert`.
- Preprocessing parameters **must** match training exactly.

**`imu_classifier`** — classifies motion activity from a sliding 128-sample window of accelerometer + gyroscope data.

| Parameter | Default |
|---|---|
| `model_path` | `~/roboception/models/imu_model.onnx` |
| `labels_path` | `~/roboception/models/labels/uci_har_labels.txt` |
| `inference_rate_hz` | `2.0` |

- Accumulates 128 samples (≈2.56 s at 50 Hz) in a ring buffer; infers at 2 Hz on a timer.
- Per-window mean subtraction approximates gravity removal.
- Input shape: `(1, 128, 6)` → `[accel_xyz, gyro_xyz]`. Output: 3 classes — `still`, `walking`, `disturbance`.

---

### fusion

**`aggregator`** — reads all three `/ml/*` topics and `/camera/telemetry`, applies priority rules, and publishes `/environment/state` at 2 Hz.

**Environment states (first matching rule wins):**

| State | Rule |
|---|---|
| `AUDIO_TRIGGERED` | Audio label ∈ {human, appliance, alert} AND confidence ≥ 0.7 — **also triggers 30 Hz camera boost for 5 s** |
| `IMU_DISTURBANCE` | IMU = disturbance AND vision idle (empty / unknown) |
| `MULTIMODAL_DISAGREEMENT` | Vision = person (> 60%) AND audio = silence (> 60%) AND IMU = still (> 60%) |
| `VISUAL_EVENT` | Vision not idle AND confidence > 0.4 |
| `CALM` | No events |

- Modality data older than **2 seconds** is treated as stale and falls back to `unknown`.

---

### dashboard

**`dashboard`** — subscribes to all output topics, serves a FastAPI web application on port **8765**.

**Web endpoints:**

| Endpoint | Description |
|---|---|
| `GET /` | HTML5 SPA dashboard |
| `GET /video_feed` | MJPEG stream of `/camera/annotated` at ~15 fps |
| `WS /ws` | JSON state updates every 0.5 s (env state, labels, confidence, bandwidth) |
| `POST /audio/toggle` | Pause / resume audio recording |
| `GET /clips/{name}` | Download a saved WAV clip |

**Audio clip auto-save:**
- Triggers on the rising edge of `AUDIO_TRIGGERED`.
- Keeps a 12-second rolling buffer of audio chunks.
- 10-second cooldown between saves; clips written to `/tmp/roboception_clips/`.
- Filename format: `HHMMSS_{label}.wav` (16-bit PCM, mono).

---

## ML Models

| Modality | File | Input shape | Training data | Classes |
|---|---|---|---|---|
| Vision | `yolov8n.onnx` (12 MB) | `(1, 3, 640, 640)` | COCO (pre-trained) | 80 COCO classes |
| Audio | `audio_model.onnx` (22 KB + 941 KB data) | `(1, 1, 64, 44)` | ESC-50 → 4 macro-classes | silence, human, appliance, alert |
| IMU | `imu_model.onnx` (19 KB + 142 KB data) | `(1, 128, 6)` | UCI-HAR → 3 macro-classes | still, walking, disturbance |

All inference runs on **CPU** via ONNX Runtime (`CPUExecutionProvider`).

Model files live under `~/roboception/models/`. Label files are in `~/roboception/models/labels/`.

---

## Custom Messages

**`AudioChunk.msg`**
```
std_msgs/Header header
float32[]        samples      # PCM normalised to [-1, 1]
uint32           sample_rate
uint8            channels
```

**`MLClassification.msg`**
```
std_msgs/Header header
string           modality        # 'vision' | 'audio' | 'imu'
string           label           # top prediction
float32          confidence
string[]         all_labels      # top-5 for vision, all classes otherwise
float32[]        all_confidences
float32          inference_ms
```

**`EnvironmentState.msg`**
```
std_msgs/Header header
string           env_state
string           vision_label
float32          vision_confidence
string           audio_label
float32          audio_confidence
string           imu_label
float32          imu_confidence
string           fusion_reason
float32          bandwidth_raw_kbps
float32          bandwidth_actual_kbps
float32          bandwidth_saved_pct
uint32           frames_received
uint32           frames_processed
```

---

## Prerequisites

- **Ubuntu 24.04** (or WSL2 on Windows)
- **ROS 2 Jazzy** desktop install
- **Python 3.12**
- **Android phone** with [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) installed, exposing `/video`, `/audio.wav`, and `/sensors.json`

---

## Installation

### 1. Install ROS 2 Jazzy

```bash
sudo apt update && sudo apt install -y software-properties-common curl
sudo add-apt-repository universe -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) \
  signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install -y ros-jazzy-desktop python3-colcon-common-extensions
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 2. Clone and Build

```bash
git clone https://github.com/ramessri/roboception.git
cd roboception/ros2_ws
colcon build
source install/setup.bash
```

### 3. Install Python Dependencies

```bash
pip install opencv-python numpy requests onnxruntime
pip install torch torchaudio
pip install fastapi uvicorn websockets
```

### 4. Prepare Models

**Vision — download YOLOv8n and export to ONNX:**
```bash
pip install ultralytics
cd ~/roboception/models
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=640)"
```

**Audio classifier — train on ESC-50:**
```bash
python ~/roboception/training/audio_train.py
# Downloads ESC-50, trains 4-class CNN, exports audio_model.onnx
```

**IMU classifier — train on UCI-HAR:**
```bash
python ~/roboception/training/imu_train.py
# Downloads UCI-HAR, trains 1D-CNN, exports imu_model.onnx
```

---

## Usage

### Quick Start

Start IP Webcam on your phone, then:

```bash
cd ~/roboception/ros2_ws
source install/setup.bash

ros2 launch roboception_bringup full_system.launch.py \
  phone_url:=http://<PHONE_IP>:8080
```

Open **http://localhost:8765** in a browser.

### Launch Arguments

| Argument | Default | Description |
|---|---|---|
| `phone_url` | `http://192.168.0.102:8080` | Base URL of the IP Webcam server |
| `vision_conf` | `0.35` | YOLOv8 confidence threshold |
| `log_level` | `info` | ROS log level (`debug` / `info` / `warn` / `error`) |

### Manual Node Launch

```bash
# Bridges
ros2 run phone_bridge camera_bridge --ros-args -p phone_url:=http://<PHONE_IP>:8080
ros2 run phone_bridge audio_bridge  --ros-args -p phone_url:=http://<PHONE_IP>:8080
ros2 run phone_bridge imu_bridge    --ros-args -p phone_url:=http://<PHONE_IP>:8080

# Classifiers
ros2 run ml_inference vision_detector
ros2 run ml_inference audio_classifier
ros2 run ml_inference imu_classifier

# Fusion
ros2 run fusion aggregator

# Dashboard
ros2 run dashboard dashboard
```

### Verify Phone Connectivity

```bash
curl -I http://<PHONE_IP>:8080/video
curl -I http://<PHONE_IP>:8080/audio.wav
curl -I http://<PHONE_IP>:8080/sensors.json
```

---

## Project Structure

```
roboception/
├── models/
│   ├── yolov8n.onnx                     # Vision — COCO pre-trained
│   ├── audio_model.onnx                 # Audio CNN — ESC-50 4-class
│   ├── audio_model.onnx.data
│   ├── imu_model.onnx                   # IMU 1D-CNN — UCI-HAR 3-class
│   ├── imu_model.onnx.data
│   └── labels/
│       ├── coco_labels.txt              # 80 COCO classes
│       ├── esc50_macro_labels.txt       # silence, human, appliance, alert
│       └── uci_har_labels.txt           # still, walking, disturbance
├── training/
│   ├── audio_train.py                   # ESC-50 → 4-class CNN → ONNX
│   └── imu_train.py                     # UCI-HAR → 3-class 1D-CNN → ONNX
├── ros2_ws/src/
│   ├── custom_msgs/                     # ROS 2 message definitions
│   │   └── msg/
│   │       ├── AudioChunk.msg
│   │       ├── MLClassification.msg
│   │       └── EnvironmentState.msg
│   ├── phone_bridge/
│   │   └── phone_bridge/
│   │       ├── camera_bridge_node.py
│   │       ├── audio_bridge_node.py
│   │       └── imu_bridge_node.py
│   ├── ml_inference/
│   │   └── ml_inference/
│   │       ├── vision_detector_node.py
│   │       ├── audio_classifier_node.py
│   │       └── imu_classifier_node.py
│   ├── fusion/
│   │   └── fusion/
│   │       └── aggregator_node.py
│   ├── dashboard/
│   │   └── dashboard/
│   │       └── dashboard_node.py
│   └── roboception_bringup/
│       └── launch/
│           └── full_system.launch.py
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## License

[MIT](LICENSE)
