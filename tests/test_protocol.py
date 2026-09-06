import socket

import numpy as np
import pytest

from wbc_ballet.teleop.protocol import NUM_JOINTS, PACKET_BYTES, BalletCommand, UdpCommandReceiver


def test_packet_round_trip() -> None:
    expected = BalletCommand(
        np.linspace(-1, 1, NUM_JOINTS),
        np.arange(NUM_JOINTS) % 2,
        np.array([0.4, -0.2, 0.7]),
    )
    actual = BalletCommand.from_bytes(expected.to_bytes())
    np.testing.assert_allclose(actual.targets, expected.targets)
    np.testing.assert_array_equal(actual.mask, expected.mask)
    np.testing.assert_allclose(actual.velocity, expected.velocity)


def test_rejects_invalid_packet() -> None:
    with pytest.raises(ValueError, match="expected"):
        BalletCommand.from_bytes(bytes(PACKET_BYTES - 4))


def test_receiver_keeps_latest_valid_packet() -> None:
    receiver = UdpCommandReceiver(port=0)
    port = receiver.socket.getsockname()[1]
    expected = BalletCommand.neutral()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        sender.sendto(b"invalid", ("127.0.0.1", port))
        sender.sendto(expected.to_bytes(), ("127.0.0.1", port))
    actual = receiver.poll()
    receiver.close()
    np.testing.assert_array_equal(actual.targets, expected.targets)
