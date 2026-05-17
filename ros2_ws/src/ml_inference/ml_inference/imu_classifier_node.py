"""
Subscribes to /imu/data (sensor_msgs/Imu @ 50 Hz), maintains a sliding
128-sample window (~2.56 sec at 50 Hz), runs imu_model.onnx at 2 Hz,
publishes /ml/imu as custom_msgs/MLClassification.

Window shape: (1, 128, 6) — body_acc xyz + gyro xyz, matches training.

Note on units / normalization: UCI-HAR body_acc is gravity-removed and
normalized in 'g' units (~[-1, 1]). Phone reports linear_acceleration
in m/s² with gravity included (~9.81 on one axis at rest). At inference
we subtract per-window mean to approximate gravity removal — works well
for a 3-class problem where variance dominates.
"""
import time
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Imu
from custom_msgs.msg import MLClassification


WINDOW_LEN = 128
N_CHANNELS = 6


class ImuClassifierNode(Node):
    def __init__(self):
        super().__init__('imu_classifier')

        self.declare_parameter(
            'model_path',
            str(Path.home() / 'roboception' / 'models' / 'imu_model.onnx')
        )
        self.declare_parameter(
            'labels_path',
            str(Path.home() / 'roboception' / 'models' / 'labels' / 'uci_har_labels.txt')
        )
        self.declare_parameter('inference_rate_hz', 2.0)

        model_path = self.get_parameter('model_path').value
        labels_path = self.get_parameter('labels_path').value
        infer_rate = self.get_parameter('inference_rate_hz').value

        self.labels = Path(labels_path).read_text().strip().splitlines()
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.get_logger().info(f'loaded {model_path}')

        # Ring buffer of recent samples; each sample is [ax,ay,az,gx,gy,gz].
        self.buffer = deque(maxlen=WINDOW_LEN)

        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        qos_pub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )

        self.sub = self.create_subscription(Imu, '/imu/data', self.on_imu, qos_sub)
        self.pub = self.create_publisher(MLClassification, '/ml/imu', qos_pub)
        self.timer = self.create_timer(1.0 / infer_rate, self.run_inference)

        self.get_logger().info(
            f'imu_classifier ready  (window={WINDOW_LEN}, infer @ {infer_rate} Hz)'
        )

    def on_imu(self, msg: Imu):
        self.buffer.append([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ])

    def softmax(self, logits):
        e = np.exp(logits - np.max(logits))
        return e / e.sum()

    def run_inference(self):
        if len(self.buffer) < WINDOW_LEN:
            return  # not enough samples yet

        t0 = time.time()
        window = np.array(self.buffer, dtype=np.float32)            # (128, 6)
        # Per-channel zero-mean normalization (approximates gravity removal).
        window = window - window.mean(axis=0, keepdims=True)
        tensor = np.expand_dims(window, axis=0)                     # (1, 128, 6)

        logits = self.session.run(None, {self.input_name: tensor})[0][0]
        probs = self.softmax(logits)
        top_idx = int(np.argmax(probs))
        inference_ms = (time.time() - t0) * 1000.0

        out = MLClassification()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'phone_imu'
        out.modality = 'imu'
        out.label = self.labels[top_idx]
        out.confidence = float(probs[top_idx])
        out.all_labels = list(self.labels)
        out.all_confidences = [float(p) for p in probs]
        out.inference_ms = float(inference_ms)
        self.pub.publish(out)


def main():
    rclpy.init()
    node = ImuClassifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()