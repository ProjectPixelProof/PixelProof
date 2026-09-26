"""Regression tests for the protected clean-container verifier handoff."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "harbor/agents/offline_verifier.py"


def _load_offline_verifier():
    names = ("harbor", "harbor.environments", "harbor.environments.base")
    saved = {name: sys.modules.get(name) for name in names}
    for name in names:
        sys.modules[name] = types.ModuleType(name)
    sys.modules["harbor.environments.base"].BaseEnvironment = object
    try:
        spec = importlib.util.spec_from_file_location(
            "offline_verifier_under_test",
            MODULE_PATH,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, prior in saved.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


def test_recycle_finishes_before_propagating_cancellation(monkeypatch) -> None:
    offline_verifier = _load_offline_verifier()
    completed = False

    async def fake_recycle(environment, *, required_artifact: str) -> None:
        del environment, required_artifact
        nonlocal completed
        await asyncio.sleep(0.02)
        completed = True

    monkeypatch.setattr(
        offline_verifier,
        "_recycle_for_offline_verifier",
        fake_recycle,
    )

    async def scenario() -> None:
        task = asyncio.create_task(
            offline_verifier.recycle_for_offline_verifier(
                object(),
                required_artifact="process/runtime.json",
            )
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert completed is True
