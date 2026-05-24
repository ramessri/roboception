"""
Subscribes to /camera/image_raw, runs YOLOv8n via ONNX Runtime,
publishes top detection on /ml/vision as custom_msgs/MLClassification,
and publishes an annotated image with bounding boxes on /camera/annotated.
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

# Colour palette — one colour per class, cycling modulo 80.
_PALETTE = [
    (0, 255, 80), (255, 80, 0), (0, 180, 255), (255, 220, 0),
    (180, 0, 255), (0, 255, 200), (255, 100, 100), (100, 255, 100),
]


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
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider'],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.get_logger().info(f'loaded {model_path}')

        # ── QoS ───────────────────────────────────────────────────
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
        self.pub_annotated = self.create_publisher(Image, '/camera/annotated', qos_pub)
        self.sub = self.create_subscription(
            Image, '/camera/image_raw', self.on_image, qos_sub
        )

        self.get_logger().info('vision_detector ready')

    # ── Preprocessing ────────────────────────────────────────────
    def preprocess(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        scale = self.input_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame_bgr, (new_w, new_h))

        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)
        tensor = np.expand_dims(tensor, 0)
        return tensor, scale, pad_x, pad_y

    # ── Postprocessing ───────────────────────────────────────────
    def postprocess(self, output):
        """
        Returns list of (class_id, confidence, x1, y1, x2, y2) in letterbox space.
        Box coords are in [0, input_size] range — caller undoes padding/scale.
        """
        preds = output[0].T
        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        mask = confidences >= self.conf_threshold
        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

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

        indices = np.array(indices).flatten()
        results = []
        for i in indices:
            x1 = float(boxes_for_nms[i][0])
            y1 = float(boxes_for_nms[i][1])
            x2 = float(x1 + boxes_for_nms[i][2])
            y2 = float(y1 + boxes_for_nms[i][3])
            results.append((int(class_ids[i]), float(confidences[i]), x1, y1, x2, y2))
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

        tensor, scale, pad_x, pad_y = self.preprocess(frame)
        output = self.session.run(None, {self.input_name: tensor})[0]
        detections = self.postprocess(output)
        inference_ms = (time.time() - t0) * 1000.0

        # ── Draw bounding boxes ───────────────────────────────────
        annotated = frame.copy()
        h_orig, w_orig = frame.shape[:2]
        for class_id, conf, bx1, by1, bx2, by2 in detections:
            ox1 = max(0, int((bx1 - pad_x) / scale))
            oy1 = max(0, int((by1 - pad_y) / scale))
            ox2 = min(w_orig - 1, int((bx2 - pad_x) / scale))
            oy2 = min(h_orig - 1, int((by2 - pad_y) / scale))
            colour = _PALETTE[class_id % len(_PALETTE)]
            cv2.rectangle(annotated, (ox1, oy1), (ox2, oy2), colour, 2)
            text = f'{self.labels[class_id]} {conf:.0%}'
            cv2.putText(annotated, text, (ox1, max(oy1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

        ann_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        ann_msg.header = msg.header
        self.pub_annotated.publish(ann_msg)

        # ── MLClassification ──────────────────────────────────────
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
            out.label = self.labels[detections[0][0]]
            out.confidence = detections[0][1]
            top5 = detections[:5]
            out.all_labels = [self.labels[d[0]] for d in top5]
            out.all_confidences = [float(d[1]) for d in top5]

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
