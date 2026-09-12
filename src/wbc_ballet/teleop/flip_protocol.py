"""Flip-task UDP protocol (G1-29DoF), wire-compatible with gamepad v2.

Sibling of :mod:`wbc_ballet.teleop.protocol` (the ballet task's UDP
protocol). Flip needs an extra ``flip`` (upright/inverted) field, so it gets
its own packet shape and receiver instead of reusing the ballet one.

Packet layout (little-endian float32): ``[targets(29), masks(29), vx, vy,
wz, flip]`` = 62 floats.
"""

from dataclasses import dataclass
import socket
import weakref
import numpy as np

PACKET_FLOATS = 62
PACKET_BYTES = PACKET_FLOATS * 4
DEFAULT_PORT = 55002


@dataclass(frozen=True)
class FlipPacket:
    targets: np.ndarray
    mask: np.ndarray
    velocity: np.ndarray
    flip: int = 0

    @classmethod
    def neutral(cls):
        return cls(np.zeros(29, dtype=np.float32), np.zeros(29, dtype=np.float32),
                   np.zeros(3, dtype=np.float32), 0)

    @classmethod
    def from_bytes(cls, payload):
        if len(payload) != PACKET_BYTES:
            raise ValueError(f'expected {PACKET_BYTES} bytes, got {len(payload)}')
        values = np.frombuffer(payload, dtype='<f4').copy()
        if not np.isfinite(values).all():
            raise ValueError('packet contains NaN/Inf')
        if values[61] not in (0., 1.):
            raise ValueError('flip must be exactly 0 or 1')
        mask = (values[29:58] >= .5).astype(np.float32)
        targets = np.where(mask > 0, np.clip(values[:29], -1., 1.), 0.)
        return cls(targets, mask, values[58:61], int(values[61]))


class FlipReceiver:
    def __init__(self, host='127.0.0.1', port=DEFAULT_PORT):
        self.packet = FlipPacket.neutral()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((host, port))
        self.socket.setblocking(False)
        self._finalizer = weakref.finalize(self, self.socket.close)

    def poll(self):
        # Bound work even if the sender floods the socket.
        for _ in range(256):
            try:
                payload, _ = self.socket.recvfrom(65535)
            except BlockingIOError:
                break
            try:
                self.packet = FlipPacket.from_bytes(payload)
            except ValueError:
                continue
        return self.packet

    def close(self):
        self._finalizer()


def receiver_for(env, host, port):
    if not hasattr(env, '_flip_udp_receiver'):
        env._flip_udp_receiver = FlipReceiver(host, port)
    return env._flip_udp_receiver
