# Roboception

**Multimodal environment perception using a smartphone as a sensor suite, ROS 2 as middleware, and on-device ML for real-time scene understanding.**

Roboception turns an Android phone into a robot's eyes, ears, and inertial sensor using IP Webcam . Three bridge nodes stream camera, microphone, and IMU data into a ROS 2 graph. Three ML inference nodes classify each modality in real time using ONNX models. A fusion aggregator merges the per-modality labels into a single environment state, and a live web dashboard renders the result over WebSocket.

---

## Architecture

```
┌──────────────┐       HTTP / MJPEG / WAV / JSON
│  Android     │ ─────────────────────────────────────┐
│  IP Webcam   │                                      │
└──────────────┘                                      ▼
                                              ┌──────────────┐
                                              │ phone_bridge  │
                                              │  ├ camera     │──▶ /camera/image_raw
                                              │  ├ audio      │──▶ /audio/chunk
                                              │  └ imu        │──▶ /imu/data
                                              └──────────────┘
                                                      │
                                     ┌────────────────┼────────────────┐
                                     ▼                ▼                ▼
                              ┌────────────┐  ┌─────────────┐  ┌────────────┐
                              │ YOLOv8n    │  │ Audio CNN   │  │ IMU 1D-CNN │
                              │ (ONNX)     │  │ (ONNX)      │  │ (ONNX)     │
                              └─────┬──────┘  └──────┬──────┘  └─────┬──────┘
                                    │                │               │
                                    ▼                ▼               ▼
                              /ml/vision        /ml/audio       /ml/imu
                                    │                │               │
                                    └────────┬───────┘───────────────┘
                                             ▼
                                     ┌──────────────┐
                                     │   fusion      │
                                     │  aggregator   │──▶ /environment/state
                                     └──────┬───────┘
                                            │
                                            ▼
                                     ┌──────────────┐
                                     │  dashboard    │
                                     │  (FastAPI +   │──▶ http://localhost:8765
                                     │   WebSocket)  │
                                     └──────────────┘
```

---

## ROS 2 Packages

| Package | Type | Description |
|---|---|---|
| **custom_msgs** | `ament_cmake` | `AudioChunk`, `MLClassification`, `EnvironmentState` message definitions |
| **phone_bridge** | `ament_python` | Camera, audio, and IMU bridge nodes — HTTP to ROS 2 |
| **ml_inference** | `ament_python` | Vision (YOLOv8n), audio (mel-spectrogram CNN), and IMU (1D-CNN) classifiers |
| **fusion** | `ament_python` | Rule-based multimodal aggregator producing five environment states |
| **dashboard** | `ament_python` | FastAPI + WebSocket live dashboard on port 8765 |
| **roboception_bringup** | `ament_python` | Single launch file to start the full 8-node graph |

---

## ROS 2 Topic Graph

| Topic | Message Type | Publisher | Subscriber(s) |
|---|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | camera_bridge | vision_detector |
| `/camera/telemetry` | `std_msgs/Float32MultiArray` | camera_bridge | aggregator |
| `/camera/boost_rate` | `std_msgs/Bool` | aggregator | camera_bridge |
| `/audio/chunk` | `custom_msgs/AudioChunk` | audio_bridge | audio_classifier |
| `/imu/data` | `sensor_msgs/Imu` | imu_bridge | imu_classifier |
| `/ml/vision` | `custom_msgs/MLClassification` | vision_detector | aggregator, dashboard |
| `/ml/audio` | `custom_msgs/MLClassification` | audio_classifier | aggregator, dashboard |
| `/ml/imu` | `custom_msgs/MLClassification` | imu_classifier | aggregator, dashboard |
| `/environment/state` | `custom_msgs/EnvironmentState` | aggregator | dashboard |

---

## Fusion States

The aggregator node produces one of five environment states (first matching rule wins):

| State | Trigger |
|---|---|
| `AUDIO_TRIGGERED` | Audio label ∈ {human, appliance, alert} with confidence ≥ 0.7 |
| `IMU_DISTURBANCE` | IMU = disturbance while vision is idle |
| `MULTIMODAL_DISAGREEMENT` | Vision sees a person but audio = silence and IMU = still |
| `VISUAL_EVENT` | Non-idle vision label with confidence > 0.4 |
| `CALM` | No events detected |

On `AUDIO_TRIGGERED`, the aggregator publishes a boost signal to the camera bridge, temporarily raising the frame rate from 5 Hz to 15 Hz for 5 seconds.

---

## ML Models

| Modality | Model | Input Shape | Training Data | Labels |
|---|---|---|---|---|
| Vision | YOLOv8n (Ultralytics → ONNX) | (1, 3, 640, 640) | COCO (pre-trained) | 80 COCO classes |
| Audio | 4-layer CNN | (1, 1, 64, 44) mel-spectrogram | ESC-50 → 4 macro-classes | silence, human, appliance, alert |
| IMU | 3-layer 1D-CNN | (1, 128, 6) | UCI-HAR → 3 macro-classes | still, walking, disturbance |

All inference runs on CPU via ONNX Runtime.

---

## Prerequisites

- **Ubuntu 24.04** (or WSL2)
- **ROS 2 Jazzy** desktop install
- **Python 3.11+**
- **Android phone** with [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) installed (or similar app exposing `/video`, `/audio.wav`, `/sensors.json` endpoints)

---

## Installation

### 1. Install ROS 2 Jazzy

```bash
sudo apt update
sudo apt install -y software-properties-common curl
sudo add-apt-repository universe -y

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
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

Disclaimer: Experimented with uv in the intial stages of project, learned not the best case for WSL ROS2

```bash
pip install opencv-python numpy requests onnxruntime
pip install torch torchaudio soundfile
pip install fastapi uvicorn websockets
```

### 4. Download / Train Models

**Vision (YOLOv8n):**
```bash
cd models
python download_yolov8.py
```

**Audio classifier (train from scratch):**
```bash
python training/audio_train.py    # downloads ESC-50, trains CNN, exports ONNX
```

**IMU classifier (train from scratch):**
```bash
python training/imu_train.py      # downloads UCI-HAR, trains 1D-CNN, exports ONNX
```

---

## Usage

### Quick Start — Single Launch File

Start your phone's IP Webcam server, then:

```bash
cd ~/roboception/ros2_ws
source install/setup.bash

ros2 launch roboception_bringup full_system.launch.py \
  phone_url:=http://<PHONE_IP>:8080
```

Open **http://localhost:8765** in a browser to view the live dashboard.

### Manual Launch (Individual Nodes)

```bash
# Terminal 1-3: Bridges
ros2 run phone_bridge camera_bridge --ros-args -p phone_url:=http://<PHONE_IP>:8080
ros2 run phone_bridge audio_bridge  --ros-args -p phone_url:=http://<PHONE_IP>:8080
ros2 run phone_bridge imu_bridge    --ros-args -p phone_url:=http://<PHONE_IP>:8080

# Terminal 4-6: Classifiers
ros2 run ml_inference vision_detector
ros2 run ml_inference audio_classifier
ros2 run ml_inference imu_classifier

# Terminal 7: Fusion
ros2 run fusion aggregator

# Terminal 8: Dashboard
ros2 run dashboard dashboard
```

### Verify Phone Connectivity

```bash
curl -I http://<PHONE_IP>:8080/video
curl -I http://<PHONE_IP>:8080/audio.wav
curl -I http://<PHONE_IP>:8080/sensor.json
```

---

## Project Structure

```
roboception/
├── models/
│   ├── yolov8n.onnx                    # Vision model (COCO pre-trained)
│   ├── audio_model.onnx                # Audio CNN (trained on ESC-50)
│   ├── imu_model.onnx                  # IMU 1D-CNN (trained on UCI-HAR)
│   └── labels/
│       ├── coco_labels.txt             # 80 COCO class names
│       ├── esc50_macro_labels.txt      # silence, human, appliance, alert
│       └── uci_har_labels.txt          # still, walking, disturbance
├── training/
│   ├── audio_train.py                  # ESC-50 → 4-class CNN → ONNX
│   └── imu_train.py                    # UCI-HAR → 3-class 1D-CNN → ONNX
├── ros2_ws/src/
│   ├── custom_msgs/                    # ROS 2 message definitions
│   │   └── msg/
│   │       ├── AudioChunk.msg
│   │       ├── MLClassification.msg
│   │       └── EnvironmentState.msg
│   ├── phone_bridge/                   # Sensor bridge nodes
│   │   └── phone_bridge/
│   │       ├── camera_bridge_node.py
│   │       ├── audio_bridge_node.py
│   │       └── imu_bridge_node.py
│   ├── ml_inference/                   # ML classifier nodes
│   │   └── ml_inference/
│   │       ├── vision_detector_node.py
│   │       ├── audio_classifier_node.py
│   │       └── imu_classifier_node.py
│   ├── fusion/                         # Multimodal fusion
│   │   └── fusion/
│   │       └── aggregator_node.py
│   ├── dashboard/                      # Live web dashboard
│   │   └── dashboard/
│   │       └── dashboard_node.py
│   └── roboception_bringup/           # Launch configuration
│       └── launch/
│           └── full_system.launch.py
├── main.py                             # Standalone camera test script
├── pyproject.toml
├── LICENSE                             # MIT
└── README.md
```

---

## Custom Message Definitions

**AudioChunk.msg**
```
std_msgs/Header header
float32[] samples
uint32 sample_rate
uint8 channels
```

**MLClassification.msg**
```
std_msgs/Header header
string modality
string label
float32 confidence
string[] all_labels
float32[] all_confidences
float32 inference_ms
```

**EnvironmentState.msg**
```
std_msgs/Header header
string env_state
string vision_label      /  float32 vision_confidence
string audio_label       /  float32 audio_confidence
string imu_label         /  float32 imu_confidence
string fusion_reason
float32 bandwidth_raw_kbps / bandwidth_actual_kbps / bandwidth_saved_pct
uint32 frames_received   /  uint32 frames_processed
```

---

## License

[MIT](LICENSE)
