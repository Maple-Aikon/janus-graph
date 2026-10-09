"""`contracts.DreamSettings` is a shadow, not the live Dream Mode setting.

Why this file exists: two classes describe "Dream Mode settings" —
`config.DreamConfig` (pydantic, what the daemon actually loads) and
`contracts.DreamSettings` (a frozen dataclass with no readers). Their
`dedup_threshold` values already disagree: 0.80 live, 0.85 here. Editing the
wrong one is silent — it changes nothing and the tests still pass.

These tests pin the divergence so it stays a documented, deliberate difference
rather than an accident nobody notices until a threshold "doesn't take effect".

A note on the import guard at the bottom: the mutating test edits a MODULE
GLOBAL and restores it in a `finally`. That is deliberate — patching is not
used because the property under test is "the shadow dataclass disagrees with
the live config", which a mock would erase by definition.
"""

import dataclasses

from dataclasses import FrozenInstanceError

import pytest

from janus_graph.config import DreamConfig, JanusSettings, load_config
from janus_graph.core.contracts import DreamSettings, Settings


def test_dream_settings_is_a_frozen_dataclass():
    """The Phase-0.5 contract guarantees immutability; dream must keep it."""
    shadow = DreamSettings()
    assert dataclasses.is_dataclass(shadow)
    with pytest.raises(FrozenInstanceError):
        shadow.dedup_threshold = 0.5


def test_contracts_settings_carries_the_shadow_not_the_live_model():
    """A bare contracts.Settings yields DreamSettings — NOT DreamConfig.

    This is the exact shape that made the drift hazardous: the flag added on
    2026-10-08 (`dedup_apply`) exists only on DreamConfig, so anything built
    from contracts.Settings silently loses it.
    """
    settings = Settings()
    assert isinstance(settings.pipeline.dream, DreamSettings)
    assert not isinstance(settings.pipeline.dream, DreamConfig)
    assert not hasattr(settings.pipeline.dream, "dedup_apply")


def test_shadow_threshold_differs_from_the_live_one():
    """The divergence is intentional and measured — assert it is still there.

    If this ever stops holding, the two classes were unified and this whole
    file should be deleted rather than kept as folklore.
    """
    assert DreamSettings().dedup_threshold == 0.85
    assert DreamConfig().dedup_threshold == 0.80
    assert DreamSettings().dedup_threshold != DreamConfig().dedup_threshold


def test_live_config_resolves_dedup_apply_but_shadow_cannot():
    """Production reads DreamConfig; the shadow is unreachable for this flag."""
    cfg = load_config()
    assert isinstance(cfg, JanusSettings)
    assert isinstance(cfg.pipeline.dream, DreamConfig)
    assert cfg.pipeline.dream.dedup_apply is False  # repo default, not the live file

    # Same getattr chain dream.py uses.
    flag = bool(
        getattr(
            getattr(getattr(cfg, "pipeline", None), "dream", None),
            "dedup_apply",
            False,
        )
    )
    assert flag is False

    # The shadow has no such attribute at all, so the chain falls back to False
    # on a contracts.Settings — the failure is a silent dry-run, not an error.
    bare = Settings()
    assert (
        bool(getattr(getattr(getattr(bare, "pipeline", None), "dream", None),
                     "dedup_apply", False))
        is False
    )


def test_dream_shadow_is_not_the_live_setting():
    """Guard the guard: mutating the shadow must not reach the live model.

    Flipping the shadow's dedup_threshold is the mistake this file exists to
    catch. Assert it cannot propagate: the two classes hold independent values.

    The value is read off ``__dataclass_fields__``, not off an instance. A
    dataclass bakes its defaults into the generated ``__init__`` at class-creation
    time, so ``DreamSettings.dedup_threshold = 0.10`` rebinds only the class
    attribute while ``DreamSettings()`` still returns 0.85 — the first version of
    this test asserted on the instance and failed for that reason.
    """
    original = DreamSettings.__dataclass_fields__["dedup_threshold"].default
    try:
        DreamSettings.__dataclass_fields__["dedup_threshold"].default = 0.10
        assert DreamSettings.__dataclass_fields__["dedup_threshold"].default == 0.10
        assert DreamConfig().dedup_threshold == 0.80
    finally:
        DreamSettings.__dataclass_fields__["dedup_threshold"].default = original
    assert DreamSettings.__dataclass_fields__["dedup_threshold"].default == original
    assert DreamSettings().dedup_threshold == original
