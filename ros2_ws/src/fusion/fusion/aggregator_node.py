"""
Multimodal fusion: combines /ml/vision, /ml/audio, /ml/imu into one
high-level environment state at 2 Hz, publishes /environment/state.

Also emits /camera/boost_rate (Bool) — True when an audio event warrants
higher camera frame rate for visual confirmation.

Rule priority (first match wins):
  1. AUDIO_TRIGGERED         — loud sound, attention-demanding
  2. IMU_DISTURBANCE         — phone bumped/shaken
  3. MULTIMODAL_DISAGREEMENT — vision sees a person but audio+IMU say no one's here
  4. VISUAL_EVENT            — something interesting in view
  5. CALM                    — default
"""
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Bool
from custom_msgs.msg import MLClassification, EnvironmentState


STALENESS_SEC = 2.0           # ignore a modality's last reading if older than this
BOOST_DURATION_SEC = 5.0      # how long to hold boost_rate=True after audio event

VISION_ACTIVE = {'empty', 'unknown'}  # these mean "nothing of interest"
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

        # Latest classification per modality + arrival time (for staleness).
        self.last = {
            'vision': None, 'audio': None, 'imu': None,
        }
        self.last_time = {
            'vision': 0.0, 'audio': 0.0, 'imu': 0.0,
        }

        # Boost state.
        self.boost_until = 0.0

        # Subscribers — one per modality.
        self.create_subscription(MLClassification, '/ml/vision',
                                 lambda m: self.on_ml('vision', m), qos)
        self.create_subscription(MLClassification, '/ml/audio',
                                 lambda m: self.on_ml('audio', m), qos)
        self.create_subscription(MLClassification, '/ml/imu',
                                 lambda m: self.on_ml('imu', m), qos)

        # Publishers.
        self.pub_state = self.create_publisher(EnvironmentState, '/environment/state', qos)
        self.pub_boost = self.create_publisher(Bool, '/camera/boost_rate', qos)

        # 2 Hz aggregation timer.
        self.timer = self.create_timer(0.5, self.aggregate)
        self.get_logger().info('aggregator ready')

    def on_ml(self, modality: str, msg: MLClassification):
        self.last[modality] = msg
        self.last_time[modality] = time.time()

    def fresh(self, modality: str) -> bool:
        return (time.time() - self.last_time[modality]) < STALENESS_SEC

    def get(self, modality: str):
        """Returns (label, confidence) or ('unknown', 0.0) if stale/missing."""
        if not self.fresh(modality) or self.last[modality] is None:
            return 'unknown', 0.0
        m = self.last[modality]
        return m.label, m.confidence

    def decide(self, v_label, v_conf, a_label, a_conf, i_label, i_conf):
        """Returns (env_state, reason). Priority-ordered rule evaluation."""

        # Rule 1: AUDIO_TRIGGERED
        if a_label in AUDIO_EVENT_LABELS and a_conf >= AUDIO_EVENT_THRESHOLD:
            return 'AUDIO_TRIGGERED', f'audio={a_label} ({a_conf:.2f})'

        # Rule 2: IMU_DISTURBANCE
        if i_label == 'disturbance' and v_label in VISION_ACTIVE:
            return 'IMU_DISTURBANCE', f'imu=disturbance, vision={v_label}'

        # Rule 3: MULTIMODAL_DISAGREEMENT
        # Vision says person, but audio and IMU both indicate empty space.
        if (v_label == 'person' and v_conf > 0.6 and
                a_label == 'silence' and a_conf > 0.6 and
                i_label == 'still' and i_conf > 0.6):
            return 'MULTIMODAL_DISAGREEMENT', (
                f'vision=person but audio=silent, imu=still'
            )

        # Rule 4: VISUAL_EVENT
        if v_label not in VISION_ACTIVE and v_conf > 0.4:
            return 'VISUAL_EVENT', f'vision={v_label} ({v_conf:.2f})'

        # Rule 5: CALM
        return 'CALM', 'no events'

    def aggregate(self):
        v_label, v_conf = self.get('vision')
        a_label, a_conf = self.get('audio')
        i_label, i_conf = self.get('imu')

        env_state, reason = self.decide(
            v_label, v_conf, a_label, a_conf, i_label, i_conf
        )

        # Boost rate handling.
        now = time.time()
        if env_state == 'AUDIO_TRIGGERED':
            self.boost_until = now + BOOST_DURATION_SEC

        boost = now < self.boost_until
        self.pub_boost.publish(Bool(data=boost))

        # Build EnvironmentState message.
        out = EnvironmentState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'aggregator'
        out.env_state = env_state
        out.vision_label = v_label
        out.vision_confidence = float(v_conf)
        out.audio_label = a_label
        out.audio_confidence = float(a_conf)
        out.imu_label = i_label
        out.imu_confidence = float(i_conf)
        out.fusion_reason = reason
        # Bandwidth fields left at default for now — Step 7 populates them.
        out.bandwidth_raw"""
Multimodal fusion: combines /ml/vision, /ml/audio, /ml/imu into one
high-level environment state at 2 Hz, publishes /environment/state.

Also emits /camera/boost_rate (Bool) — True when an audio event warrants
higher camera frame rate for visual confirmation.

Rule priority (first match wins):
  1. AUDIO_TRIGGERED         — loud sound, attention-demanding
  2. IMU_DISTURBANCE         — phone bumped/shaken
  3. MULTIMODAL_DISAGREEMENT — vision sees a person but audio+IMU say no one's here
  4. VISUAL_EVENT            — something interesting in view
  5. CALM                    — default
"""
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Bool
from custom_msgs.msg import MLClassification, EnvironmentState


STALENESS_SEC = 2.0           # ignore a modality's last reading if older than this
BOOST_DURATION_SEC = 5.0      # how long to hold boost_rate=True after audio event

VISION_ACTIVE = {'empty', 'unknown'}  # these mean "nothing of interest"
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

        # Latest classification per modality + arrival time (for staleness).
        self.last = {
            'vision': None, 'audio': None, 'imu': None,
        }
        self.last_time = {
            'vision': 0.0, 'audio': 0.0, 'imu': 0.0,
        }

        # Boost state.
        self.boost_until = 0.0

        # Subscribers — one per modality.
        self.create_subscription(MLClassification, '/ml/vision',
                                 lambda m: self.on_ml('vision', m), qos)
        self.create_subscription(MLClassification, '/ml/audio',
                                 lambda m: self.on_ml('audio', m), qos)
        self.create_subscription(MLClassification, '/ml/imu',
                                 lambda m: self.on_ml('imu', m), qos)

        # Publishers.
        self.pub_state = self.create_publisher(EnvironmentState, '/environment/state', qos)
        self.pub_boost = self.create_publisher(Bool, '/camera/boost_rate', qos)

        # 2 Hz aggregation timer.
        self.timer = self.create_timer(0.5, self.aggregate)
        self.get_logger().info('aggregator ready')

    def on_ml(self, modality: str, msg: MLClassification):
        self.last[modality] = msg
        self.last_time[modality] = time.time()

    def fresh(self, modality: str) -> bool:
        return (time.time() - self.last_time[modality]) < STALENESS_SEC

    def get(self, modality: str):
        """Returns (label, confidence) or ('unknown', 0.0) if stale/missing."""
        if not self.fresh(modality) or self.last[modality] is None:
            return 'unknown', 0.0
        m = self.last[modality]
        return m.label, m.confidence

    def decide(self, v_label, v_conf, a_label, a_conf, i_label, i_conf):
        """Returns (env_state, reason). Priority-ordered rule evaluation."""

        # Rule 1: AUDIO_TRIGGERED
        if a_label in AUDIO_EVENT_LABELS and a_conf >= AUDIO_EVENT_THRESHOLD:
            return 'AUDIO_TRIGGERED', f'audio={a_label} ({a_conf:.2f})'

        # Rule 2: IMU_DISTURBANCE
        if i_label == 'disturbance' and v_label in VISION_ACTIVE:
            return 'IMU_DISTURBANCE', f'imu=disturbance, vision={v_label}'

        # Rule 3: MULTIMODAL_DISAGREEMENT
        # Vision says person, but audio and IMU both indicate empty space.
        if (v_label == 'person' and v_conf > 0.6 and
                a_label == 'silence' and a_conf > 0.6 and
                i_label == 'still' and i_conf > 0.6):
            return 'MULTIMODAL_DISAGREEMENT', (
                f'vision=person but audio=silent, imu=still'
            )

        # Rule 4: VISUAL_EVENT
        if v_label not in VISION_ACTIVE and v_conf > 0.4:
            return 'VISUAL_EVENT', f'vision={v_label} ({v_conf:.2f})'

        # Rule 5: CALM
        return 'CALM', 'no events'

    def aggregate(self):
        v_label, v_conf = self.get('vision')
        a_label, a_conf = self.get('audio')
        i_label, i_conf = self.get('imu')

        env_state, reason = self.decide(
            v_label, v_conf, a_label, a_conf, i_label, i_conf
        )

        # Boost rate handling.
        now = time.time()
        if env_state == 'AUDIO_TRIGGERED':
            self.boost_until = now + BOOST_DURATION_SEC

        boost = now < self.boost_until
        self.pub_boost.publish(Bool(data=boost))

        # Build EnvironmentState message.
        out = EnvironmentState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'aggregator'
        out.env_state = env_state
        out.vision_label = v_label
        out.vision_confidence = float(v_conf)
        out.audio_label = a_label
        out.audio_confidence = float(a_conf)
        out.imu_label = i_label
        out.imu_confidence = float(i_conf)
        out.fusion_reason = reason
        # Bandwidth fields left at default for now — Step 7 populates them.
        out.bandwidth_raw_kbps = 0.0
        out.bandwidth_actual_kbps = 0.0
        out.bandwidth_saved_pct = 0.0
        out.frames_received = 0
        out.frames_processed = 0
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