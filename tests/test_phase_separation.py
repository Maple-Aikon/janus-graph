"""Tests for Phase 2/3 clean architectural boundary separation (per R22.4)."""

from pathlib import Path


def test_phase2_does_not_import_phase3():
    """Phase 2 (heuristics) must not directly import Phase 3 (report)."""
    heuristics_dir = Path(__file__).resolve().parent.parent / "janus_graph" / "heuristics"
    for py_file in heuristics_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert "from janus_graph.report" not in content, (
            f"Phase 2 file {py_file.name} illegally imports from janus_graph.report"
        )
        assert "import janus_graph.report" not in content, (
            f"Phase 2 file {py_file.name} illegally imports janus_graph.report"
        )


def test_phase3_does_not_import_phase2():
    """Phase 3 (report) must not directly import Phase 2 (heuristics)."""
    report_dir = Path(__file__).resolve().parent.parent / "janus_graph" / "report"
    for py_file in report_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert "from janus_graph.heuristics" not in content, (
            f"Phase 3 file {py_file.name} illegally imports from janus_graph.heuristics"
        )
        assert "import janus_graph.heuristics" not in content, (
            f"Phase 3 file {py_file.name} illegally imports janus_graph.heuristics"
        )
