"""
Multimodal fusion: combines /ml/vision, /ml/audio, /ml/imu into one
EnvironmentState at 2 Hz. Also relays /camera/telemetry → bandwidth fields
in EnvironmentState, and publishes /camera/boost_rate on audio events.

Rule priority (first match wins):
  1. AUDIO_TRIGGERED         — loud sound, attention-demanding
  2. IMU_DISTURBANCE         — phone bumped/shaken
  3. MULTIMODAL_DISAGREEMENT — vision sees a person but audio+IMU say no
  4. VISUAL_EVENT            — something interesting in view
  5. CALM                    — default
"""
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Bool, Float32MultiArray
from custom_msgs.msg import MLClassification, EnvironmentState


STALENESS_SEC = 2.0
BOOST_DURATION_SEC = 5.0

VISION_IDLE = {'empty', 'unknown'}
AUDIO_EVENT_LABELS = {'human', 'appliance', 'alert'}
AUDIO_EVENT_THRESHOLD = 0.7


class AggregatorNode(Node):
    def __init__(self):
        super().__init__('aggregator')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.last = {'vision': None, 'audio': None, 'imu': None}
        self.last_time = {'vision': 0.0, 'audio': 0.0, 'imu': 0.0}

        # Latest telemetry from camera_bridge.
        self.bandwidth = {
            'raw_kbps': 0.0,
            'actual_kbps': 0.0,
            'saved_pct': 0.0,
            'frames_received': 0,
            'frames_published': 0,
        }

        self.boost_until = 0.0

        self.create_subscription(MLClassification, '/ml/vision',
                                 lambda m: self.on_ml('vision', m), qos)
        self.create_subscription(MLClassification, '/ml/audio',
                                 lambda m: self.on_ml('audio', m), qos)
        self.create_subscription(MLClassification, '/ml/imu',
                                 lambda m: self.on_ml('imu', m), qos)
        self.create_subscription(Float32MultiArray, '/camera/telemetry',
                                 self.on_telemetry, qos)

        self.pub_state = self.create_publisher(
            EnvironmentState, '/environment/state', qos
        )
        self.pub_boost = self.create_publisher(Bool, '/camera/boost_rate', qos)

        self.timer = self.create_timer(0.5, self.aggregate)
        self.get_logger().info('aggregator ready')

    def on_ml(self, modality, msg):
        self.last[modality] = msg
        self.last_time[modality] = time.time()

    def on_telemetry(self, msg: Float32MultiArray):
        if len(msg.data) >= 5:
            self.bandwidth['raw_kbps'] = msg.data[0]
            self.bandwidth['actual_kbps'] = msg.data[1]
            self.bandwidth['saved_pct'] = msg.data[2]
            self.bandwidth['frames_received'] = int(msg.data[3])
            self.bandwidth['frames_published'] = int(msg.data[4])

    def fresh(self, modality):
        return (time.time() - self.last_time[modality]) < STALENESS_SEC

    def get(self, modality):
        if not self.fresh(modality) or self.last[modality] is None:
            return 'unknown', 0.0
        m = self.last[modality]
        return m.label, m.confidence

    def decide(self, v, vc, a, ac, i, ic):
        if a in AUDIO_EVENT_LABELS and ac >= AUDIO_EVENT_THRESHOLD:
            return 'AUDIO_TRIGGERED', f'audio={a} ({ac:.2f})'

        if i == 'disturbance' and v in VISION_IDLE:
            return 'IMU_DISTURBANCE', f'imu=disturbance, vision={v}'

        if (v == 'person' and vc > 0.6 and
                a == 'silence' and ac > 0.6 and
                i == 'still' and ic > 0.6):
            return ('MULTIMODAL_DISAGREEMENT',
                    'vision=person but audio=silent, imu=still')

        if v not in VISION_IDLE and vc > 0.4:
            return 'VISUAL_EVENT', f'vision={v} ({vc:.2f})'

        return 'CALM', 'no events'

    def aggregate(self):
        v, vc = self.get('vision')
        a, ac = self.get('audio')
        i, ic = self.get('imu')

        env_state, reason = self.decide(v, vc, a, ac, i, ic)

        now = time.time()
        if env_state == 'AUDIO_TRIGGERED':
            self.boost_until = now + BOOST_DURATION_SEC
        boost = now < self.boost_until
        self.pub_boost.publish(Bool(data=boost))

        out = EnvironmentState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'aggregator'
        out.env_state = env_state
        out.vision_label = v
        out.vision_confidence = float(vc)
        out.audio_label = a
        out.audio_confidence = float(ac)
        out.imu_label = i
        out.imu_confidence = float(ic)
        out.fusion_reason = reason
        out.bandwidth_raw_kbps = float(self.bandwidth['raw_kbps'])
        out.bandwidth_actual_kbps = float(self.bandwidth['actual_kbps'])
        out.bandwidth_saved_pct = float(self.bandwidth['saved_pct'])
        out.frames_received = int(self.bandwidth['frames_received'])
        out.frames_processed = int(self.bandwidth['frames_published'])
        self.pub_state.publish(out)


def main():
    rclpy.init()
    node = AggregatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()