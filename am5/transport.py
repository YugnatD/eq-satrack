"""Byte-level transports to the mount: real serial/TCP, or the in-process mock.

Every transport exposes the same tiny interface (write / read_until_hash /
read_exact / close) so `characterize.py` and `Mount` never need to know
whether they are talking to hardware or to `MockMount`.
"""

from __future__ import annotations

import socket
import time
from abc import ABC, abstractmethod


class Transport(ABC):
    @abstractmethod
    def write(self, data: bytes) -> None: ...

    @abstractmethod
    def read_until_hash(self, timeout: float) -> bytes:
        """Read bytes up to and including the next '#', or whatever arrived
        before `timeout` elapses (possibly empty)."""

    @abstractmethod
    def read_exact(self, n: int, timeout: float) -> bytes:
        """Read exactly `n` bytes, or fewer if `timeout` elapses first. For
        the handful of LX200 replies (:Sr#/:Sd#) that are a single raw
        character with no '#' terminator — read_until_hash would otherwise
        block for the full timeout on every one of them."""

    @abstractmethod
    def close(self) -> None: ...


class SerialTransport(Transport):
    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 2.0):
        import serial  # local import: not needed in --mock runs

        self._ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        # Discard whatever the OS-level tty buffer accumulated before this
        # process opened the port (e.g. an unread reply from a previous
        # session) — otherwise the first read here can splice onto stale
        # bytes and silently return a corrupted value.
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()

    def write(self, data: bytes) -> None:
        self._ser.write(data)

    def read_until_hash(self, timeout: float) -> bytes:
        self._ser.timeout = timeout
        return self._ser.read_until(b"#")

    def read_exact(self, n: int, timeout: float) -> bytes:
        self._ser.timeout = timeout
        return self._ser.read(n)

    def close(self) -> None:
        self._ser.close()


class TCPTransport(Transport):
    def __init__(self, host: str, port: int = 4030, timeout: float = 2.0):
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._buf = b""

    def write(self, data: bytes) -> None:
        self._sock.sendall(data)

    def read_until_hash(self, timeout: float) -> bytes:
        deadline = time.monotonic() + timeout
        while b"#" not in self._buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._sock.settimeout(remaining)
            try:
                chunk = self._sock.recv(256)
            except (socket.timeout, OSError):
                break
            if not chunk:
                break
            self._buf += chunk
        if b"#" in self._buf:
            idx = self._buf.index(b"#") + 1
            out, self._buf = self._buf[:idx], self._buf[idx:]
            return out
        out, self._buf = self._buf, b""
        return out

    def read_exact(self, n: int, timeout: float) -> bytes:
        deadline = time.monotonic() + timeout
        while len(self._buf) < n:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._sock.settimeout(remaining)
            try:
                chunk = self._sock.recv(256)
            except (socket.timeout, OSError):
                break
            if not chunk:
                break
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def close(self) -> None:
        self._sock.close()
