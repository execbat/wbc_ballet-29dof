"""Small terminal sender for the Orbit-compatible UDP command protocol."""

from __future__ import annotations

import argparse
import socket

import numpy as np

from .protocol import NUM_JOINTS, BalletCommand


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=55001)
    parser.add_argument("--joint", type=int, choices=range(NUM_JOINTS))
    parser.add_argument("--target", type=float, default=0.0)
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    args = parser.parse_args()
    targets = np.zeros(NUM_JOINTS, dtype=np.float32)
    mask = np.zeros(NUM_JOINTS, dtype=np.float32)
    if args.joint is not None:
        targets[args.joint] = np.clip(args.target, -1.0, 1.0)
        mask[args.joint] = 1.0
    cmd = BalletCommand(targets, mask, np.array([args.vx, args.vy, args.yaw]))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(cmd.to_bytes(), (args.host, args.port))
