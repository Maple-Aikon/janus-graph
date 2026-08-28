"""Phase 5: Cross-architecture binary detection tests."""
from __future__ import annotations

from unittest import mock

from janus_graph.migrate import detect_artifact


def test_detect_linux_arm64() -> None:
    """Linux ARM64 (aarch64) normalizes to arm64."""
    with mock.patch("platform.system", return_value="Linux"), \
         mock.patch("platform.machine", return_value="aarch64"):
        info = detect_artifact()
    assert info["system"] == "linux"
    assert info["normalized_arch"] == "arm64"
    assert info["artifact"] == "falkordb-linux-arm64"
    assert info["supported"] is True


def test_detect_linux_x86_64() -> None:
    """Linux x86_64 (amd64) normalizes to x86_64."""
    with mock.patch("platform.system", return_value="Linux"), \
         mock.patch("platform.machine", return_value="x86_64"):
        info = detect_artifact()
    assert info["normalized_arch"] == "x86_64"
    assert info["artifact"] == "falkordb-linux-x86_64"
    assert info["supported"] is True


def test_detect_macos_arm64() -> None:
    """macOS Apple Silicon normalizes to arm64."""
    with mock.patch("platform.system", return_value="Darwin"), \
         mock.patch("platform.machine", return_value="arm64"):
        info = detect_artifact()
    assert info["system"] == "darwin"
    assert info["normalized_arch"] == "arm64"
    assert info["supported"] is True


def test_detect_unsupported_arch() -> None:
    """Unknown architecture reports supported=False."""
    with mock.patch("platform.system", return_value="Linux"), \
         mock.patch("platform.machine", return_value="riscv64"):
        info = detect_artifact()
    assert info["normalized_arch"] == "riscv64"
    assert info["supported"] is False


def test_detect_real_platform() -> None:
    """Detect on the current host (no mocks) returns a valid dict."""
    info = detect_artifact()
    assert info["system"] in ("linux", "darwin", "windows", "freebsd")
    assert info["machine"] is not None
    assert info["normalized_arch"] in ("arm64", "x86_64")
    assert isinstance(info["supported"], bool)
