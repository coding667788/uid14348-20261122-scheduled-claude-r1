"""Shared helpers for the unittest suite."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")


def load(name: str) -> dict[str, Any]:
    with open(os.path.join(EXAMPLES, name), "r", encoding="utf-8") as handle:
        return json.load(handle)


def clone(case: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(case)


def run_cli(command: str, case: dict[str, Any] | str) -> tuple[int, dict[str, Any]]:
    """Run ``python -m solution <command> CASE.json`` in a subprocess."""

    if isinstance(case, dict):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        try:
            json.dump(case, handle, ensure_ascii=False)
            handle.close()
            payload = run_cli_path(command, handle.name)
        finally:
            os.unlink(handle.name)
        return payload
    return run_cli_path(command, case)


def run_cli_path(command: str, path: str) -> tuple[int, dict[str, Any]]:
    proc = subprocess.run(
        [sys.executable, "-m", "solution", command, path],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    text = proc.stdout.decode("utf-8")
    return proc.returncode, json.loads(text)


def simulation(command: str, case: dict[str, Any]) -> dict[str, Any]:
    code, payload = run_cli(command, case)
    if code != 0 and payload.get("ok") is not False:
        raise AssertionError(f"unexpected exit {code}: {payload}")
    return payload


class BaseCase(unittest.TestCase):
    """Small helpers for asserting on the JSON shape."""

    def assertOk(self, payload: dict[str, Any]) -> None:
        self.assertTrue(payload.get("ok"), msg=json.dumps(payload, ensure_ascii=False)[:800])

    def codes(self, payload: dict[str, Any], field: str = "violations") -> list[str]:
        return [item["code"] for item in payload[field]]

    def steps(self, payload: dict[str, Any], field: str = "violations") -> list[int]:
        return [item["step"] for item in payload[field]]
