# ros2_ws/src/phone_bridge/phone_bridge/camera_bridge_node.py
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraBridgeNode(Node):
    def __init__(self):
        super().__init__('camera_bridge')

        # Parameters — declare so they can be overridden via launch file / CLI
        self.declare_parameter('phone_url', 'http://192.168.1.42:8080')
        self.declare_parameter('publish_rate_hz', 15.0)

        phone_url = self.get_parameter('phone_url').value
        rate = self.get_parameter('publish_rate_hz').value
        self.video_url = f'{phone_url}/video'

        # QoS: BEST_EFFORT for sensor streams. If you use RELIABLE (the
        # default), subscribers that fall behind will cause the publisher
        # to block. For a 15 Hz camera you want drop-on-overflow.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub = self.create_publisher(Image, '/camera/image_raw', qos)

        self.bridge = CvBridge()
        self.cap = None  # TODO: open cv2.VideoCapture(self.video_url) — but
                        # consider: what if the phone isn't reachable yet at
                        # startup? Should the node die, or retry?

        self.timer = self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(f'camera_bridge → {self.video_url} @ {rate} Hz')

    def tick(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.video_url)

            if not self.cap.isOpened():
                self.get_logger().warn(
                    f'Cannot reach {self.video_url}, retrying...',
                    throttle_duration_sec=2.0
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

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'phone_camera'
        self.pub.publish(msg)

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