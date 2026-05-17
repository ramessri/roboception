"""
Subscribes to /audio/chunk (custom_msgs/AudioChunk at ~1 Hz, 44.1 kHz mono),
preprocesses to (1, 1, 64, 44) mel-spectrogram, runs audio_model.onnx,
publishes /ml/audio as custom_msgs/MLClassification.

Preprocessing must EXACTLY MATCH training (audio_train.py). If you change
SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, TIME_FRAMES in one, change it in
both — otherwise the model sees out-of-distribution input and predicts
garbage at inference time even with a well-trained checkpoint.
"""
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
import torchaudio.transforms as taT
import torch.nn as nn

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from custom_msgs.msg import AudioChunk, MLClassification


# These MUST match training/audio_train.py — see docstring above.
SAMPLE_RATE = 22050
WINDOW_SAMPLES = SAMPLE_RATE
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64
TIME_FRAMES = 44


class AudioClassifierNode(Node):
    def __init__(self):
        super().__init__('audio_classifier')

        self.declare_parameter(
            'model_path',
            str(Path.home() / 'roboception' / 'models' / 'audio_model.onnx')
        )
        self.declare_parameter(
            'labels_path',
            str(Path.home() / 'roboception' / 'models' / 'labels' / 'esc50_macro_labels.txt')
        )
        model_path = self.get_parameter('model_path').value
        labels_path = self.get_parameter('labels_path').value

        self.labels = Path(labels_path).read_text().strip().splitlines()
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.get_logger().info(f'loaded {model_path}')

        # Preprocess pipeline mirrors training.
        self.mel = nn.Sequential(
            taT.MelSpectrogram(
                sample_rate=SAMPLE_RATE, n_fft=N_FFT,
                hop_length=HOP_LENGTH, n_mels=N_MELS,
            ),
            taT.AmplitudeToDB(),
        )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )
        self.pub = self.create_publisher(MLClassification, '/ml/audio', qos)
        self.sub = self.create_subscription(AudioChunk, '/audio/chunk', self.on_audio, qos)
        self.get_logger().info('audio_classifier ready')

    def preprocess(self, msg: AudioChunk) -> np.ndarray:
        """AudioChunk → (1, 1, 64, 44) float32."""
        samples = torch.tensor(msg.samples, dtype=torch.float32).unsqueeze(0)  # (1, N)

        if msg.sample_rate != SAMPLE_RATE:
            samples = torchaudio.functional.resample(samples, msg.sample_rate, SAMPLE_RATE)

        n = samples.shape[1]
        if n > WINDOW_SAMPLES:
            start = (n - WINDOW_SAMPLES) // 2
            samples = samples[:, start:start + WINDOW_SAMPLES]
        elif n < WINDOW_SAMPLES:
            samples = torch.nn.functional.pad(samples, (0, WINDOW_SAMPLES - n))

        mel = self.mel(samples)                                 # (1, 64, T)
        if mel.shape[2] > TIME_FRAMES:
            mel = mel[:, :, :TIME_FRAMES]
        elif mel.shape[2] < TIME_FRAMES:
            mel = torch.nn.functional.pad(mel, (0, TIME_FRAMES - mel.shape[2]))

        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        return mel.unsqueeze(0).numpy().astype(np.float32)      # (1, 1, 64, 44)

    def softmax(self, logits):
        e = np.exp(logits - np.max(logits))
        return e / e.sum()

    def on_audio(self, msg: AudioChunk):
        if not msg.samples:
            return
        t0 = time.time()
        tensor = self.preprocess(msg)
        logits = self.session.run(None, {self.input_name: tensor})[0][0]
        probs = self.softmax(logits)
        top_idx = int(np.argmax(probs))
        inference_ms = (time.time() - t0) * 1000.0

        out = MLClassification()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = msg.header.frame_id
        out.modality = 'audio'
        out.label = self.labels[top_idx]
        out.confidence = float(probs[top_idx])
        out.all_labels = list(self.labels)
        out.all_confidences = [float(p) for p in probs]
        out.inference_ms = float(inference_ms)
        self.pub.publish(out)


def main():
    rclpy.init()
    node = AudioClassifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()