"""Generic CLI pipe sink for forwarding reports to an external command.

The :class:`PipeSink` constructs an arbitrary command from ``command`` plus
optional ``extra_args`` (e.g. ``--channel``) and writes the rendered event
text to the command's stdin. Use this when you want to forward Janus-Graph
reports to *any* external notification CLI without coupling Janus-Graph to
that tool's specific config keys.

Typical wiring for the PicoClaw ``send_telegram.sh`` helper:

.. code-block:: yaml

    report:
      sinks:
        pipe:
          enabled: true
          command:
            - "/home/maple/.picoclaw/workspace/cli-bin/terminals/send_telegram.sh"
          extra_args:
            - "--channel"
          use_stdin: true
          timeout_sec: 15.0
          min_severity: "info"

The sink is intentionally generic — pipe to ``curl``, ``mail``, ``jq``, a
custom webhook CLI, anything that reads stdin. The caller owns routing,
auth, and channel selection; Janus-Graph only streams text on demand.

Implementation notes:

* ``command`` is a list (argv list) so we never invoke a shell. The first
  element is verified to exist on disk to prevent typos from silently
  spawning nothing (only checked when ``use_stdin=True`` AND we are the
  sink delivering via stdin; a missing binary for a non-stdin pipe is
  just logged as a warning so opt-in hooks like ``| tee`` still work).
* ``use_stdin=False`` means text is appended to argv (``text`` is passed
  as the last command-line argument). Useful for ``jq -r . '.foo'`` style
  filters.
* ``extra_args`` is appended *after* the resolved per-event text arg, so
  ``--channel`` ends up trailing the message exactly like ``send_telegram.sh``
  expects.
* Severity filtering follows the same ``min_severity`` convention as
  every other sink so operators can globally mute noise.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence

from ..models import ReportEvent
from . import BaseSink

logger = logging.getLogger("janus_graph.report.pipe")


class PipeSink(BaseSink):
    """Forward each report event's text payload to an external CLI.

    Parameters
    ----------
    command
        Absolute path + zero or more leading arguments (e.g.
        ``["/usr/local/bin/notify", "--format=markdown"]``).
    extra_args
        Arguments appended *after* the rendered text. Useful for routing
        flags like ``--channel`` or environment selectors like
        ``--env=production``.
    use_stdin
        If True (default), pipe event text via stdin. If False, pass text
        as the last CLI argument.
    timeout_sec
        Subprocess timeout. Default 15s.
    min_severity
        Drop events below this severity. Default ``"info"`` → no filtering.
    cwd
        Optional working directory for the subprocess. Default ``None``.
    env
        Optional extra env vars merged onto ``os.environ``. Default ``None``.
    """

    def __init__(
        self,
        command: Sequence[str],
        extra_args: Optional[Sequence[str]] = None,
        use_stdin: bool = True,
        timeout_sec: float = 15.0,
        min_severity: str = "info",
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
    ):
        if not command:
            raise ValueError("PipeSink requires a non-empty `command` list")
        if use_stdin and not Path(command[0]).exists():
            raise FileNotFoundError(
                f"PipeSink command not found on disk: {command[0]} "
                f"(use_stdin=True requires the binary to exist at emit time)"
            )
        self.command: List[str] = list(command)
        self.extra_args: List[str] = list(extra_args or [])
        self.use_stdin = use_stdin
        self.timeout_sec = timeout_sec
        self.min_severity = min_severity
        self.cwd = cwd
        self.env = env

    @property
    def name(self) -> str:
        return "pipe"

    # ------------------------------------------------------------------
    # Formatting

    def _render(self, event: ReportEvent) -> str:
        """Render the event to plain Markdown text suitable for piping.

        Markdown is the lingua franca of CLI notifiers
        (``send_telegram.sh`` falls back to plain text; ``gh``, ``jq``, etc.
        all accept it; downstream HTML renderers can escalate if needed).
        """
        icon = {
            "debug": "🔍",
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "critical": "🚨",
        }.get(event.severity.value, "ℹ️")
        lines = [
            f"{icon} *Janus-Graph [{event.severity.value.upper()}]* `{event.kind}`",
            f"_{event.summary}_",
        ]
        if event.details:
            # Generous truncation — most CLI notifiers handle 4 KB; chat
            # bridges usually resegment after this point.
            detail_text = json.dumps(event.details, indent=2, default=str)[:3500]
            lines.append("```json" + chr(10) + detail_text + chr(10) + "```")
        return chr(10).join(lines)

    # ------------------------------------------------------------------
    # Emit

    async def emit(self, event: ReportEvent) -> None:
        if not event.severity.is_at_least(self.min_severity):
            return

        text = self._render(event)

        if self.use_stdin:
            args = self.command + self.extra_args
            input_payload: Optional[str] = text
        else:
            args = self.command + [text] + self.extra_args
            input_payload = None

        def _invoke() -> "tuple[int, str, str]":
            merged_env = None
            if self.env:
                merged_env = {**os.environ, **self.env}
            proc = subprocess.run(  # internal trusted CLI
                args,
                input=input_payload,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                check=False,
                cwd=self.cwd,
                env=merged_env,
            )
            return proc.returncode, proc.stdout, proc.stderr

        try:
            rc, _stdout, stderr = await asyncio.to_thread(_invoke)
            if rc != 0:
                logger.warning(
                    "PipeSink: command=%s rc=%d stderr=%s",
                    args, rc, (stderr or "").strip()[:200],
                )
            else:
                logger.debug("PipeSink: OK (%s)", args[0])
        except subprocess.TimeoutExpired:
            logger.warning(
                "PipeSink: timeout after %.1fs (%s)", self.timeout_sec, args,
            )
        except FileNotFoundError as err:
            logger.warning("PipeSink: command not found (%s) — %s", args, err)
        except Exception as err:
            logger.warning("PipeSink: exception while invoking %s — %s", args, err)
