"""``replay`` -- execute a supplied timeline and explain every outcome."""

from __future__ import annotations

from typing import Any

from .model import (
    CompiledCase,
    Trace,
    horizon_for,
    make_fault,
    make_latency,
    simulate,
    violation_key,
)


def scenario_list(case: CompiledCase) -> list[dict[str, Any]]:
    """Every permitted scenario, as ``{id, latency, fault}``.

    The plan has to be safe for all of them; replay uses exactly the one the
    case selects.
    """

    out: list[dict[str, Any]] = []
    variants: list[tuple[str, dict[str, dict[str, int]] | None]] = [("baseline", None)]
    for scen in case.latency_scenarios:
        variants.append((scen["id"], scen.get("links") or {}))
    for name, override in variants:
        out.append(
            {
                "id": name,
                "latency": override,
                "fault": bool(case.fault),
                "fault_window": case.fault,
            }
        )
    return out


def timeline_steps(case: CompiledCase) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Read the supplied ordering and turn it into ``action -> step``.

    An event without ``step`` takes its index in the timeline, so a bare order
    is also a valid timeline.  Events for unknown action ids are reported by
    the caller rather than silently ignored.
    """

    steps: dict[str, int] = {}
    resolved: list[dict[str, Any]] = []
    for index, event in enumerate(case.raw.get("timeline") or []):
        aid = event["id"]
        step = int(event.get("step", index))
        steps[aid] = step
        resolved.append({"id": aid, "step": step})
    return steps, resolved


def select_scenario(case: CompiledCase) -> tuple[str, dict[str, dict[str, int]] | None, bool]:
    """Pick the replay scenario named by ``case["replay"]``."""

    spec = case.raw.get("replay") or {}
    name = spec.get("latency") or "baseline"
    if name == "baseline":
        override = None
    else:
        match = next((s for s in case.latency_scenarios if s["id"] == name), None)
        if match is None:
            raise KeyError(name)
        override = match.get("links") or {}
    fault = bool(spec.get("fault", case.fault is not None))
    return name, override, fault


def prefix_of(trace: Trace, violation: dict[str, Any]) -> list[dict[str, Any]]:
    """Every event up to and including the violation step."""

    limit = int(violation["step"])
    return [e for e in trace.events if int(e["step"]) <= limit]


def run_replay(case: CompiledCase) -> dict[str, Any]:
    steps, resolved = timeline_steps(case)
    unknown = [e["id"] for e in resolved if e["id"] not in case.actions]
    if unknown:
        return {
            "ok": False,
            "command": "replay",
            "error": {
                "code": "E_REFERENCE",
                "field": "$.timeline",
                "message": "timeline references unknown actions: " + ", ".join(sorted(set(unknown))),
            },
        }
    for record in case.raw.get("actions", []):
        if "step" in record:
            steps[record["id"]] = int(record["step"])
    name, override, use_fault = select_scenario(case)
    latency = make_latency(case, override)
    fault = make_fault(case, use_fault)
    until = int((case.raw.get("replay") or {}).get("until", 0) or 0) or horizon_for(case, steps)
    trace = simulate(case, steps, latency, fault, until)

    earliest = trace.earliest()
    isolation = [v for v in trace.violations if v["code"] == "tenant_isolation"]
    reach = [v for v in trace.violations if v["code"].startswith("unreachable")]

    return {
        "ok": True,
        "command": "replay",
        "case": case.raw.get("name", "case"),
        "scenario": {
            "latency": name,
            "fault": bool(fault),
            "fault_window": (
                {
                    "domain": fault.domain,
                    "fail_at": fault.fail_at,
                    "recover_at": fault.recover_at,
                    "recover_stretch": fault.stretch,
                }
                if fault is not None
                else None
            ),
            "ttl": case.ttl,
            "horizon": until,
        },
        "timeline": sorted(resolved, key=lambda e: (e["step"], e["id"])),
        "actions_applied": trace.actions_applied,
        "barriers": trace.barriers,
        "deliveries": trace.deliveries,
        "receipts": {
            request: {branch: dict(rec) for branch, rec in sorted(branches.items())}
            for request, branches in sorted(trace.receipts.items())
        },
        "drops": trace.drops,
        "rejections": trace.rejections,
        "late_deliveries": trace.late,
        "duplicate_deliveries": trace.duplicates,
        "generation_mismatches": trace.generation_mismatches,
        "in_flight": trace.in_flight,
        "marker_fates": {k: trace.marker_fates[k] for k in sorted(trace.marker_fates)},
        "violations": sorted(trace.violations, key=violation_key),
        "earliest_violation": earliest,
        "violation_prefix": prefix_of(trace, earliest) if earliest is not None else [],
        "isolated": not isolation,
        "reachable": not reach,
        "compression": {
            "steps_visited": len(trace.visited_steps),
            "steps_skipped": trace.compressed_steps,
            "visited": trace.visited_steps,
            "reachability_end": until,
        },
    }
