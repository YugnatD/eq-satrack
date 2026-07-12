"""Connection CLI plumbing shared by characterize.py and track_pass.py."""

from __future__ import annotations

import argparse

from .mock_mount import MockConfig, MockMount
from .transport import SerialTransport, TCPTransport, Transport


def add_connection_args(parser: argparse.ArgumentParser) -> None:
    conn = parser.add_mutually_exclusive_group(required=True)
    conn.add_argument("--mock", action="store_true", help="use the in-process mock mount")
    conn.add_argument("--serial", metavar="PORT", help="serial device, e.g. /dev/ttyACM0")
    conn.add_argument("--tcp", metavar="HOST[:PORT]", help="WiFi endpoint, e.g. 192.168.4.1:4030")
    parser.add_argument("--mock-seed", type=int, default=None, help="mock only: RNG seed for reproducible runs")


def build_transport(args: argparse.Namespace, mock_config: MockConfig | None = None) -> Transport:
    if args.mock:
        return MockMount(mock_config or MockConfig(), seed=args.mock_seed)
    if args.serial:
        return SerialTransport(args.serial, baudrate=9600)
    if args.tcp:
        host, _, port_str = args.tcp.partition(":")
        port = int(port_str) if port_str else 4030
        return TCPTransport(host, port)
    raise SystemExit("one of --mock / --serial / --tcp is required")
