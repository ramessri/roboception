# ros2_ws/src/phone_bridge/phone_bridge/imu_bridge_node.py
import os
import requests
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu


class ImuBridgeNode(Node):
    def __init__(self):
        super().__init__('imu_bridge')

        _ip = os.environ.get('PHONE_IP', '')
        _default_url = f'http://{_ip}:8080' if _ip else 'http://[set PHONE_IP]'
        self.declare_parameter('phone_url', _default_url)
        self.declare_parameter('poll_rate_hz', 50.0)
        self.declare_parameter('http_timeout_sec', 1.0)

        phone_url = self.get_parameter('phone_url').value
        rate = self.get_parameter('poll_rate_hz').value
        self.http_timeout = self.get_parameter('http_timeout_sec').value
        self.sensors_url = f'{phone_url}/sensors.json'

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.pub = self.create_publisher(Imu, '/imu/data', qos)

        # requests.Session reuses the underlying TCP connection — significantly
        # faster than creating a new one each poll. At 50 Hz this matters.
        self.session = requests.Session()

        self.timer = self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(f'imu_bridge → {self.sensors_url} @ {rate} Hz')

    def tick(self):
        try:
            response = self.session.get(
                self.sensors_url,
                timeout=self.http_timeout
            )
            response.raise_for_status()
            data = response.json()

        except requests.RequestException as e:
            self.get_logger().warn(
                f'IMU HTTP error: {e}',
                throttle_duration_sec=2.0
            )
            return

        except ValueError as e:
            self.get_logger().warn(
                f'IMU JSON parse error: {e}',
                throttle_duration_sec=2.0
            )
            return

        accel_samples = data.get('accel', {}).get('data', [])
        gyro_samples = data.get('gyro', {}).get('data', [])

        if not accel_samples or not gyro_samples:
            self.get_logger().warn(
                'missing accel or gyro samples',
                throttle_duration_sec=2.0
            )
            return

        try:
            _, accel = accel_samples[-1]
            _, gyro = gyro_samples[-1]

            ax, ay, az = accel
            gx, gy, gz = gyro

        except (ValueError, TypeError) as e:
            self.get_logger().warn(
                f'bad IMU sample format: {e}',
                throttle_duration_sec=2.0
            )
            return

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'phone_imu'

        msg.linear_acceleration.x = float(ax)
        msg.linear_acceleration.y = float(ay)
        msg.linear_acceleration.z = float(az)

        msg.angular_velocity.x = float(gx)
        msg.angular_velocity.y = float(gy)
        msg.angular_velocity.z = float(gz)

        # Orientation is not estimated in this bridge.
        msg.orientation_covariance[0] = -1.0

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = ImuBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()