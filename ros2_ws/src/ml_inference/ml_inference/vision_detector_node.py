"""
Subscribes to /camera/image_raw, runs YOLOv8n via ONNX Runtime,
publishes top detection on /ml/vision as custom_msgs/MLClassification.

Maps the 80 COCO classes to a single scene-level label by picking the
highest-confidence detection in the frame. If nothing crosses the
confidence threshold, label is 'empty'.
"""
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from custom_msgs.msg import MLClassification


class VisionDetectorNode(Node):
    def __init__(self):
        super().__init__('vision_detector')

        # ── Parameters ────────────────────────────────────────────
        self.declare_parameter(
            'model_path',
            str(Path.home() / 'roboception' / 'models' / 'yolov8n.onnx')
        )
        self.declare_parameter(
            'labels_path',
            str(Path.home() / 'roboception' / 'models' / 'labels' / 'coco_labels.txt')
        )
        self.declare_parameter('conf_threshold', 0.35)
        self.declare_parameter('nms_threshold', 0.45)
        self.declare_parameter('input_size', 640)

        model_path = self.get_parameter('model_path').value
        labels_path = self.get_parameter('labels_path').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.nms_threshold = self.get_parameter('nms_threshold').value
        self.input_size = self.get_parameter('input_size').value

        # ── Load labels ───────────────────────────────────────────
        self.labels = Path(labels_path).read_text().strip().splitlines()
        assert len(self.labels) == 80, f'expected 80 COCO labels, got {len(self.labels)}'

        # ── Load ONNX model ───────────────────────────────────────
        # CPUExecutionProvider only — WSL2 doesn't have CUDA passthrough by
        # default. On a real robot you'd add 'CUDAExecutionProvider' first
        # and let onnxruntime pick the best available.
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider'],
        )
        self.input_name = self.session.get_inputs()[0].name  # 'images'
        self.get_logger().info(f'loaded {model_path}')

        # ── QoS: BEST_EFFORT to match camera_bridge ───────────────
        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_pub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.bridge = CvBridge()
        self.pub = self.create_publisher(MLClassification, '/ml/vision', qos_pub)
        self.sub = self.create_subscription(
            Image, '/camera/image_raw', self.on_image, qos_sub
        )

        self.get_logger().info('vision_detector ready')

    # ── Preprocessing ────────────────────────────────────────────
    def preprocess(self, frame_bgr):
        """
        BGR (H, W, 3) uint8 → (1, 3, 640, 640) float32 in [0,1], RGB.
        Letterbox: scale preserving aspect ratio, pad with gray.
        Returns (tensor, scale, pad_x, pad_y) — scale/pad needed to map
        boxes back to original image coords later (unused in this demo
        because we don't draw boxes, but kept for clarity).
        """
        h, w = frame_bgr.shape[:2]
        scale = self.input_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame_bgr, (new_w, new_h))

        # Pad to square 640x640
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # BGR → RGB, HWC → CHW, uint8 → float32, [0,1]
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)           # CHW
        tensor = np.expand_dims(tensor, 0)            # NCHW
        return tensor, scale, pad_x, pad_y

    # ── Postprocessing ───────────────────────────────────────────
    def postprocess(self, output):
        """
        YOLOv8 ONNX output is (1, 84, 8400):
          - axis 0: batch (1)
          - axis 1: 4 box coords (cx, cy, w, h) + 80 class scores
          - axis 2: 8400 candidate boxes

        Returns: list of (class_id, confidence) for surviving boxes after NMS.
        """
        # Transpose to (8400, 84) — one row per candidate.
        preds = output[0].T

        # Split: first 4 cols are bbox, remaining 80 are class scores.
        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]

        # Top class per row + its score.
        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        # Filter by confidence threshold.
        mask = confidences >= self.conf_threshold
        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # cv2.dnn.NMSBoxes wants xywh as [x, y, w, h] with x,y = top-left.
        # YOLOv8 gives center-xywh, so convert.
        xy_tl = boxes_xywh[:, :2] - boxes_xywh[:, 2:] / 2
        boxes_for_nms = np.concatenate([xy_tl, boxes_xywh[:, 2:]], axis=1)

        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms.tolist(),
            confidences.tolist(),
            self.conf_threshold,
            self.nms_threshold,
        )

        if len(indices) == 0:
            return []

        # OpenCV returns indices as np.ndarray in newer versions, flat list in older.
        indices = np.array(indices).flatten()

        results = [(int(class_ids[i]), float(confidences[i])) for i in indices]
        # Sort by confidence descending.
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ── Callback ─────────────────────────────────────────────────
    def on_image(self, msg):
        t0 = time.time()
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge: {e}', throttle_duration_sec=2.0)
            return

        tensor, _, _, _ = self.preprocess(frame)
        output = self.session.run(None, {self.input_name: tensor})[0]
        detections = self.postprocess(output)
        inference_ms = (time.time() - t0) * 1000.0

        # Build MLClassification message.
        out = MLClassification()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = msg.header.frame_id
        out.modality = 'vision'
        out.inference_ms = float(inference_ms)

        if not detections:
            out.label = 'empty'
            out.confidence = 1.0
            out.all_labels = []
            out.all_confidences = []
        else:
            top_class_id, top_conf = detections[0]
            out.label = self.labels[top_class_id]
            out.confidence = top_conf
            # Send up to top 5 for the dashboard.
            top5 = detections[:5]
            out.all_labels = [self.labels[c] for c, _ in top5]
            out.all_confidences = [float(c) for _, c in top5]

        self.pub.publish(out)


def main():
    rclpy.init()
    node = VisionDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()