"""FalkorDB embedded server lifecycle manager."""

from __future__ import annotations

import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Optional
from ..config import EngineConfig


class FalkorDBServerManager:
    """Manages the FalkorDB (Redis + module) server process."""

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.bin_dir = Path(self.config.bin_dir).resolve()
        self.pid_file = Path(self.config.pid_file).resolve()
        self.log_file = Path(self.config.log_file).resolve()
        self.data_dir = Path(self.config.data_dir).resolve()

    def resolve_binary_paths(self) -> tuple[Path, Path]:
        """Resolve platform-specific binary paths for redis-server and falkordb.so."""
        arch = platform.machine()
        
        # Check standard names first
        redis_bin = self.bin_dir / "redis-server-8"
        if not redis_bin.exists():
            redis_bin = self.bin_dir / "redis-server"

        falkordb_module = self.bin_dir / "falkordb.so"
        if not falkordb_module.exists():
            falkordb_module = self.bin_dir / f"falkordb.{arch}.so"

        return redis_bin, falkordb_module

    def is_running(self) -> bool:
        """Check if FalkorDB process is currently running."""
        if not self.pid_file.exists():
            return False
        try:
            pid = int(self.pid_file.read_text().strip())
            os.kill(pid, 0)
            return True
        except (ValueError, OSError):
            return False

    def start(self) -> bool:
        """Start the FalkorDB server if not already running."""
        if self.is_running():
            return True

        redis_bin, falkordb_module = self.resolve_binary_paths()
        if not redis_bin.exists() or not falkordb_module.exists():
            raise FileNotFoundError(
                f"Missing engine binaries: redis={redis_bin.exists()}, falkordb={falkordb_module.exists()}"
            )

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(redis_bin),
            "--port", str(self.config.port),
            "--bind", self.config.host,
            "--dir", str(self.data_dir),
            "--loadmodule", str(falkordb_module),
            "--daemonize", "yes",
            "--pidfile", str(self.pid_file),
            "--logfile", str(self.log_file),
        ]

        subprocess.run(cmd, check=True)
        time.sleep(0.5)
        return self.is_running()

    def stop(self) -> bool:
        """Stop the FalkorDB server."""
        if not self.is_running():
            return True
        try:
            pid = int(self.pid_file.read_text().strip())
            os.kill(pid, 15)  # SIGTERM
            time.sleep(0.5)
            if self.pid_file.exists():
                self.pid_file.unlink()
            return True
        except (ValueError, OSError):
            return False
