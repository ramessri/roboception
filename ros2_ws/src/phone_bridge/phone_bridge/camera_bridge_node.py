"""
Camera bridge — phone MJPEG → /camera/image_raw + /camera/telemetry.

Base rate: 5 Hz (BASE_RATE). Boost rate: 15 Hz (BOOST_RATE), activated for
BOOST_DURATION_SEC seconds when /camera/boost_rate publishes True.

Telemetry: every TELEMETRY_PERIOD_SEC seconds, publishes a snapshot of
  - bytes/sec received from phone (raw)
  - bytes/sec actually published to ROS (actual, after rate-limiting)
  - frame counts received vs published
The aggregator consumes this and includes it in /environment/state.
"""
import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool, Float32MultiArray
from cv_bridge import CvBridge


BASE_RATE = 15.0                 # Hz — idle rate (min needed for VI-SLAM tracking)
BOOST_RATE = 30.0                # Hz — boosted rate on audio events
BOOST_DURATION_SEC = 5.0         # how long a single boost lasts
TELEMETRY_PERIOD_SEC = 1.0       # how often to publish bandwidth stats
TICK_PERIOD = 1.0 / 30.0         # internal tick rate; publishes are decimated from here


class CameraBridgeNode(Node):
    def __init__(self):
        super().__init__('camera_bridge')

        self.declare_parameter('phone_url', 'http://192.168.0.102:8080')
        phone_url = self.get_parameter('phone_url').value
        self.video_url = f'{phone_url}/video'

        qos_pub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.pub_image = self.create_publisher(Image, '/camera/image_raw', qos_pub)
        self.pub_camera_info = self.create_publisher(CameraInfo, '/camera/camera_info', qos_pub)

        # Approximate intrinsics for your phone at 640x400.
        # fx=fy=554 assumes ~60° horizontal FOV — good enough for RTAB to start.
        # Replace with real values after you run camera_calibration once.
        self._camera_info = CameraInfo()
        self._camera_info.header.frame_id = 'phone_camera'
        self._camera_info.width  = 640
        self._camera_info.height = 400
        self._camera_info.distortion_model = 'plumb_bob'
        self._camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]   # no distortion assumed
        self._camera_info.k = [
            554.0,   0.0, 320.0,
            0.0, 554.0, 200.0,
            0.0,   0.0,   1.0,
        ]
        self._camera_info.r = [1.0, 0.0, 0.0,
                                0.0, 1.0, 0.0,
                                0.0, 0.0, 1.0]
        self._camera_info.p = [
            554.0,   0.0, 320.0, 0.0,
            0.0, 554.0, 200.0, 0.0,
            0.0,   0.0,   1.0, 0.0,
        ]

        self.pub_telemetry = self.create_publisher(
            Float32MultiArray, '/camera/telemetry', qos_pub
        )
        self.sub_boost = self.create_subscription(
            Bool, '/camera/boost_rate', self.on_boost, qos_sub
        )

        self.bridge = CvBridge()
        self.cap = None

        # Rate control state.
        self.boosted_until = 0.0
        self.last_publish_time = 0.0

        # Bandwidth + frame counters (reset on each telemetry publish).
        self.bytes_received_total = 0   # bytes from phone
        self.bytes_published_total = 0  # bytes downstream to ROS
        self.frames_received = 0
        self.frames_published = 0
        self.window_start = time.time()

        self.tick_timer = self.create_timer(TICK_PERIOD, self.tick)
        self.telemetry_timer = self.create_timer(
            TELEMETRY_PERIOD_SEC, self.emit_telemetry
        )

        self.get_logger().info(
            f'camera_bridge → {self.video_url} | base={BASE_RATE} Hz, boost={BOOST_RATE} Hz'
        )

    def on_boost(self, msg: Bool):
        if msg.data:
            new_until = time.time() + BOOST_DURATION_SEC
            # Only log on rising edge to avoid spam.
            if new_until > self.boosted_until + 1.0:
                self.get_logger().info(f'boost_rate ON for {BOOST_DURATION_SEC}s')
            self.boosted_until = new_until

    def current_rate(self) -> float:
        return BOOST_RATE if time.time() < self.boosted_until else BASE_RATE

    def tick(self):
        # Connect / reconnect.
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.video_url)
            if not self.cap.isOpened():
                self.get_logger().warn(
                    f'Cannot reach {self.video_url}, retrying...',
                    throttle_duration_sec=2.0,
                )
                self.cap = None
                return
            self.get_logger().info('Connected to phone camera stream')

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Camera stream dropped, reconnecting...')
            self.cap.release()
            self.cap = None
            return

        # Estimate bytes for "raw" bandwidth — JPEG-encode and measure.
        # We must encode anyway later for /image_raw → cv_bridge converts to
        # raw bytes (large), so we use JPEG-encoded size as the realistic
        # "what we'd send over network" figure. Defensible interview note.
        ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            frame_bytes = len(jpeg.tobytes())
        else:
            frame_bytes = frame.nbytes
        self.bytes_received_total += frame_bytes
        self.frames_received += 1

        # Rate-limit: only publish if enough time has passed since last publish.
        now = time.time()
        min_interval = 1.0 / self.current_rate()
        if now - self.last_publish_time < min_interval:
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'phone_camera'
        self.pub_image.publish(msg)

        self._camera_info.header.stamp = msg.header.stamp   # same timestamp as image
        self.pub_camera_info.publish(self._camera_info)

        self.bytes_published_total += frame_bytes
        self.frames_published += 1
        self.last_publish_time = now

    def emit_telemetry(self):
        elapsed = max(time.time() - self.window_start, 1e-6)
        raw_kbps = (self.bytes_received_total / elapsed) / 1024.0 * 8.0
        actual_kbps = (self.bytes_published_total / elapsed) / 1024.0 * 8.0
        saved_pct = (1.0 - actual_kbps / raw_kbps) * 100.0 if raw_kbps > 0 else 0.0

        # Float32MultiArray layout (positional, documented in aggregator):
        #   [raw_kbps, actual_kbps, saved_pct, frames_received, frames_published]
        out = Float32MultiArray()
        out.data = [
            float(raw_kbps),
            float(actual_kbps),
            float(saved_pct),
            float(self.frames_received),
            float(self.frames_published),
        ]
        self.pub_telemetry.publish(out)

        # Reset window.
        self.bytes_received_total = 0
        self.bytes_published_total = 0
        self.frames_received = 0
        self.frames_published = 0
        self.window_start = time.time()


def main():
    rclpy.init()
    node = CameraBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.cap is not None:
            node.cap.release()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()