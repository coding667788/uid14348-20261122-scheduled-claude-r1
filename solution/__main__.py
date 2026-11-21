"""``python -m solution <command> CASE.json``.

Commands
--------
``replay``  execute the supplied timeline and report barrier witnesses, marker
            fates, and the earliest violation prefix.
``plan``    search for the fixed release table.
``check``   validate the file structure and echo a normalised summary.

stdout always receives one deterministic UTF-8 JSON document.  Structural
problems are reported as ``{"ok": false, "error": {"code", "field", ...}}``
with a non-zero exit status; valid documents exit 0 even when they report an
infeasible plan or a replay violation, because those are model results, not
program errors.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from .model import compile_case
from .plan import run_plan
from .replay import run_replay
from .schema import CaseError, validate_case

USAGE = "usage: python -m solution {replay|plan|check} CASE.json"


def _dump(payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)
    sys.stdout.buffer.write(text.encode("utf-8"))
    sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.flush()


def _read(path: str) -> Any:
    with open(path, "rb") as handle:
        data = handle.read()
    return json.loads(data.decode("utf-8"))


def summary(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "command": "check",
        "case": case.get("name", "case"),
        "case_version": case["case_version"],
        "counts": {
            "nodes": len(case["nodes"]),
            "links": len(case["links"]),
            "identities": len(case["identities"]),
            "services": len(case["services"]),
            "requests": len(case["requests"]),
            "actions": len(case.get("actions", [])),
            "inbound_rules": len(case["policy"]["inbound"]),
            "egress_rules": len(case["policy"]["egress"]),
            "timeline_events": len(case.get("timeline", [])),
        },
        "ttl": case["ttl"],
        "horizon": case["horizon"],
        "limits": {
            "proxies": 14,
            "actions": 16,
            "slots": 10,
            "ttl": 6,
        },
        "scenarios": {
            "latency": [s["id"] for s in (case.get("scenarios") or {}).get("latency", [])],
            "fault": (case.get("scenarios") or {}).get("fault"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] not in {"replay", "plan", "check"}:
        _dump(
            {
                "ok": False,
                "error": {"code": "E_USAGE", "field": "$", "message": USAGE},
            }
        )
        return 2
    command, path = args
    try:
        raw = _read(path)
    except FileNotFoundError:
        _dump(
            {
                "ok": False,
                "command": command,
                "error": {
                    "code": "E_IO",
                    "field": "$",
                    "message": f"case file not found: {path}",
                },
            }
        )
        return 2
    except UnicodeDecodeError:
        _dump(
            {
                "ok": False,
                "command": command,
                "error": {
                    "code": "E_ENCODING",
                    "field": "$",
                    "message": "case file is not valid UTF-8",
                },
            }
        )
        return 2
    except json.JSONDecodeError as exc:
        _dump(
            {
                "ok": False,
                "command": command,
                "error": {
                    "code": "E_JSON",
                    "field": f"offset {exc.pos} (line {exc.lineno}, column {exc.colno})",
                    "message": "case file is not valid JSON",
                },
            }
        )
        return 2

    try:
        case = validate_case(raw)
    except CaseError as exc:
        payload = exc.to_dict()
        payload["command"] = command
        _dump(payload)
        return 2

    compiled = compile_case(case)
    if command == "check":
        _dump(summary(case))
        return 0
    if command == "replay":
        result = run_replay(compiled)
    else:
        result = run_plan(compiled)
    _dump(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
