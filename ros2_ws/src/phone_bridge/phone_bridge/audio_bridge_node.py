# ros2_ws/src/phone_bridge/phone_bridge/audio_bridge_node.py
import struct
import threading
import requests
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from custom_msgs.msg import AudioChunk


WAV_HEADER_SIZE = 44


class AudioBridgeNode(Node):
    def __init__(self):
        super().__init__('audio_bridge')

        self.declare_parameter('phone_url', 'http://192.168.0.102:8080')
        self.declare_parameter('chunk_duration_sec', 1.0)
        self.declare_parameter('reconnect_delay_sec', 2.0)

        phone_url = self.get_parameter('phone_url').value
        self.chunk_duration = self.get_parameter('chunk_duration_sec').value
        self.reconnect_delay = self.get_parameter('reconnect_delay_sec').value
        self.audio_url = f'{phone_url}/audio.wav'

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )
        self.pub = self.create_publisher(AudioChunk, '/audio/chunk', qos)

        # Stream parameters — set by parse_wav_header after first connect.
        self.sample_rate = None
        self.channels = None
        self.bits_per_sample = None

        # Why a thread? requests.iter_content blocks. If we ran it in the
        # ROS2 executor's main thread, rclpy.spin() couldn't process callbacks
        # (no callbacks here, but parameters, services, lifecycle hooks all
        # break). Cleaner: thread does I/O, ROS publishing is thread-safe.
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.stream_loop, daemon=True)
        self.thread.start()

        self.get_logger().info(f'audio_bridge → {self.audio_url}')

    def parse_wav_header(self, header_bytes):
        """Pulls out (sample_rate, channels, bits_per_sample) from a 44-byte
        standard PCM WAV header. Logs and returns None if malformed."""
        if len(header_bytes) < WAV_HEADER_SIZE:
            return None
        if header_bytes[0:4] != b'RIFF' or header_bytes[8:12] != b'WAVE':
            self.get_logger().error('not a WAV stream (missing RIFF/WAVE magic)')
            return None

        # struct format: '<' = little-endian, 'H' = uint16, 'I' = uint32
        channels = struct.unpack('<H', header_bytes[22:24])[0]
        sample_rate = struct.unpack('<I', header_bytes[24:28])[0]
        bits_per_sample = struct.unpack('<H', header_bytes[34:36])[0]
        return sample_rate, channels, bits_per_sample

    def stream_loop(self):
        """Long-running reconnection loop. One iteration = one HTTP connection."""
        while not self.stop_event.is_set():
            try:
                self.consume_stream()
            except Exception as e:
                self.get_logger().warn(f'stream error: {e}, reconnecting in {self.reconnect_delay}s')
                self.stop_event.wait(self.reconnect_delay)

    def consume_stream(self):
        response = requests.get(self.audio_url, stream=True, timeout=10)
        response.raise_for_status()
        raw = response.raw

        header = raw.read(WAV_HEADER_SIZE)
        parsed = self.parse_wav_header(header)
        if parsed is None:
            raise RuntimeError('bad WAV header')

        self.sample_rate, self.channels, self.bits_per_sample = parsed
        self.get_logger().info(
            f'audio stream: {self.sample_rate} Hz, '
            f'{self.channels} ch, {self.bits_per_sample}-bit'
        )

        bytes_per_sample = self.bits_per_sample // 8
        samples_per_chunk = int(self.sample_rate * self.chunk_duration)
        bytes_per_chunk = samples_per_chunk * self.channels * bytes_per_sample

        while not self.stop_event.is_set():
            # Read a full chunk, looping over short reads.
            chunk_bytes = b''
            while len(chunk_bytes) < bytes_per_chunk:
                piece = raw.read(bytes_per_chunk - len(chunk_bytes))
                if not piece:
                    raise ConnectionError('audio stream EOF')
                chunk_bytes += piece

            pcm = np.frombuffer(chunk_bytes, dtype=np.int16)
            if self.channels == 2:
                pcm = pcm.reshape(-1, 2).mean(axis=1).astype(np.int16)
            samples = pcm.astype(np.float32) / 32768.0

            msg = AudioChunk()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'phone_mic'
            msg.samples = samples.tolist()
            msg.sample_rate = self.sample_rate
            msg.channels = 1
            self.pub.publish(msg)

    def destroy_node(self):
        self.stop_event.set()
        super().destroy_node()


def main():
    rclpy.init()
    node = AudioBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()