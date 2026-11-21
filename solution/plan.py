"""``plan`` -- choose a fixed release table that is safe under every scenario.

A *table* maps every action id to a slot in ``[0, slots)``.  Several actions may
share a slot and apply in id order; an ordering constraint ``before -> after``
requires ``slot(before) < slot(after)``, so simultaneous actions never satisfy a
dependency.

Constraints
-----------
* ``after`` dependencies given by the input;
* the implicit protocol dependency: a retire must follow a barrier at the same
  proxy whose generation covers it;
* per-action ``slots`` whitelists and fixed ``step`` values;
* at most ``plan.slots`` distinct slots;
* a barrier may pick a wait from ``wait_slots``; the drain is checked at the
  end of the wait, never in the middle.

Objective (lexicographic, smaller is better)
--------------------------------------------
1. total paused-proxy steps, summed over proxies on a fixed cost window;
2. number of distinct slots carrying a release action
   (``install``/``switch``/``retire``) -- barriers are drains, not releases;
3. the action-id sequence in slot order.

Safety is checked by simulating the table for every permitted latency scenario
under both the fault-free and the single-fault variant.  One table must work for
all of them: a plan that is only safe because one scenario is expected at run
time is rejected.

The search is a depth-first enumeration of slot contents with dominance
pruning.  Its size is bounded by ``plan.search_budget`` (default 200 000
nodes); when the bound is hit the answer is still the best table found but is
reported with ``"complete": false`` instead of being claimed optimal.

When no table exists, the planner reports, for the input's ``baseline_order``,
either the earliest scheduling-constraint conflict or -- if the baseline
satisfies the scheduling constraints -- the shortest isolation / reachability
failure prefix, ties broken by the event-id sequence.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Iterator

from .model import (
    CompiledCase,
    Trace,
    horizon_for,
    make_fault,
    make_latency,
    simulate,
    violation_key,
)

RELEASE_KINDS = {"install_inbound", "switch_egress", "retire_inbound", "retire_egress"}
DEFAULT_NODE_BUDGET = 200_000


# ---------------------------------------------------------------------------
# Constraint graph
# ---------------------------------------------------------------------------


@dataclass
class ConstraintGraph:
    ids: list[str]
    edges: dict[str, set[str]]  # action -> must run strictly after these
    fixed: dict[str, int]
    windows: dict[str, tuple[int, int]]
    waits: dict[str, tuple[int, ...]]
    slots: int

    def closure(self) -> dict[str, set[str]]:
        memo: dict[str, set[str]] = {}

        def visit(aid: str, stack: tuple[str, ...]) -> set[str]:
            if aid in memo:
                return memo[aid]
            if aid in stack:
                return set()
            out: set[str] = set()
            for dep in self.edges.get(aid, ()):
                out.add(dep)
                out |= visit(dep, stack + (aid,))
            memo[aid] = out
            return out

        for aid in self.ids:
            visit(aid, ())
        return memo

    def cycle(self) -> list[str] | None:
        colour: dict[str, int] = {}
        path: list[str] = []

        def dfs(node: str) -> list[str] | None:
            colour[node] = 1
            path.append(node)
            for dep in sorted(self.edges.get(node, ())):
                state = colour.get(dep, 0)
                if state == 1:
                    return path[path.index(dep) :] + [dep]
                if state == 0:
                    found = dfs(dep)
                    if found:
                        return found
            path.pop()
            colour[node] = 2
            return None

        for aid in self.ids:
            if colour.get(aid, 0) == 0:
                found = dfs(aid)
                if found:
                    return found
        return None


def build_graph(case: CompiledCase) -> ConstraintGraph:
    ids = list(case.action_order)
    edges: dict[str, set[str]] = {aid: set() for aid in ids}
    for aid in ids:
        for dep in case.actions[aid].after:
            if dep in case.actions:
                edges[aid].add(dep)

    barriers = [aid for aid in ids if case.actions[aid].kind == "barrier"]
    for aid in ids:
        act = case.actions[aid]
        if act.kind in ("retire_inbound", "retire_egress"):
            gen = int(act.generation or 0)
            for barrier in barriers:
                other = case.actions[barrier]
                if other.proxy == act.proxy and int(other.generation or 0) >= gen:
                    edges[aid].add(barrier)

    # A resume of the same proxy must come strictly after its pause: a table
    # cannot pause and resume in one slot, because that is not an outage.
    for aid in ids:
        act = case.actions[aid]
        if act.kind != "resume":
            continue
        for other in ids:
            if case.actions[other].kind == "pause" and case.actions[other].proxy == act.proxy:
                edges[aid].add(other)

    raw_by_id = {a["id"]: a for a in case.raw.get("actions", [])}
    fixed: dict[str, int] = {}
    windows: dict[str, tuple[int, int]] = {}
    waits: dict[str, tuple[int, ...]] = {}
    for aid in ids:
        act = case.actions[aid]
        raw = raw_by_id.get(aid, {})
        if act.slots:
            windows[aid] = (min(act.slots), max(act.slots))
        if "step" in raw:
            fixed[aid] = int(raw["step"])
        if act.kind == "barrier":
            raw_waits = raw.get("wait_slots")
            if raw_waits:
                waits[aid] = tuple(sorted({int(w) for w in raw_waits}))
            else:
                waits[aid] = (int(raw.get("wait", 0) or 0),)
    return ConstraintGraph(
        ids=ids,
        edges=edges,
        fixed=fixed,
        windows=windows,
        waits=waits,
        slots=case.policy_slots,
    )


# ---------------------------------------------------------------------------
# Scenario beds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bed:
    name: str
    latency: dict[str, dict[str, int]] | None
    fault: bool


def scenarios(case: CompiledCase) -> list[Bed]:
    beds = [Bed("baseline", None, False)]
    for scen in case.latency_scenarios:
        beds.append(Bed(scen["id"], scen.get("links") or {}, False))
    if case.fault is not None:
        beds.append(Bed("baseline+fault", None, True))
        for scen in case.latency_scenarios:
            beds.append(Bed(f"{scen['id']}+fault", scen.get("links") or {}, True))
    seen: set[str] = set()
    unique: list[Bed] = []
    for bed in beds:
        key = stable_key(bed.latency) + ("+fault" if bed.fault else "")
        if key in seen:
            continue
        seen.add(key)
        unique.append(bed)
    return unique


def stable_key(value: Any) -> str:
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{stable_key(v)}" for k, v in sorted(value.items())) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(stable_key(v) for v in value) + "]"
    return str(value)


# ---------------------------------------------------------------------------
# Costs and evaluation
# ---------------------------------------------------------------------------


def cost_horizon(case: CompiledCase, graph: ConstraintGraph) -> int:
    """A fixed comparison window shared by every candidate table.

    Using one window keeps pause durations comparable even when some tables
    place their last action earlier than others.
    """

    worst = {aid: max(graph.slots - 1, 0) for aid in graph.ids}
    span = horizon_for(case, worst)
    return max(span, case.horizon, 1)


def paused_steps(
    case: CompiledCase,
    assignment: dict[str, int],
    waits: dict[str, int],
    window: int,
) -> int:
    """Sum of paused-proxy durations implied by the table alone.

    Pausing is independent of latency and faults: the proxy is paused from the
    pause slot until the matching resume slot (or the end of the window).
    """

    total = 0
    by_proxy: dict[str, list[tuple[int, str]]] = {}
    for aid, slot in assignment.items():
        act = case.actions[aid]
        if act.kind in ("pause", "resume"):
            by_proxy.setdefault(act.proxy, []).append((slot, act.kind))
    for proxy in sorted(case.proxies):
        events = sorted(by_proxy.get(proxy, []), key=lambda e: (e[0], e[1]))
        start: int | None = None
        for slot, kind in events:
            if kind == "pause" and start is None:
                start = slot
            elif kind == "resume" and start is not None:
                total += max(0, min(slot, window) - start)
                start = None
        if start is not None:
            total += max(0, window - start)
    return total


def release_slot_count(case: CompiledCase, assignment: dict[str, int]) -> int:
    return len(
        {assignment[aid] for aid in assignment if case.actions[aid].kind in RELEASE_KINDS}
    )


@dataclass
class Outcome:
    trace: Trace


def evaluate(
    case: CompiledCase,
    assignment: dict[str, int],
    waits: dict[str, int],
    bed: Bed,
    until: int,
) -> Outcome:
    latency = make_latency(case, bed.latency)
    fault = make_fault(case, bed.fault)
    trace = simulate(case, assignment, latency, fault, until, waits=waits)
    return Outcome(trace=trace)


def is_safe(outcome: Outcome) -> bool:
    return not outcome.trace.violations


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    key: tuple[Any, ...]
    assignment: dict[str, int]
    waits: dict[str, int]
    outcomes: list[Outcome] = field(default_factory=list)


class Search:
    """Depth-first slot enumeration with sound lower-bound pruning."""

    def __init__(
        self,
        case: CompiledCase,
        graph: ConstraintGraph,
        beds: list[Bed],
        budget: int,
    ) -> None:
        self.case = case
        self.graph = graph
        self.beds = beds
        self.budget = budget
        self.closure = graph.closure()
        self.window = cost_horizon(case, graph)
        self.best: Candidate | None = None
        self.leaves = 0
        self.safe_leaves = 0
        self.nodes = 0
        self.complete = True

    # -- driver -----------------------------------------------------------

    def run(self) -> Candidate | None:
        self._dfs(
            slot=0,
            scheduled=frozenset(),
            assignment={},
            waits={},
            open_pauses={},
            closed_pause=0,
        )
        return self.best

    def _release_slots_used(self, assignment: dict[str, int]) -> int:
        return len(
            {
                slot
                for aid, slot in assignment.items()
                if self.case.actions[aid].kind in RELEASE_KINDS
            }
        )

    def _min_extra_release_slots(self, remaining: frozenset[str]) -> int:
        """Longest chain of remaining release actions.

        A chain must occupy strictly increasing slots, so its length is a sound
        lower bound on the number of *new* release slots still required.  The
        longest chain is computed as the height of the release sub-poset.
        """

        release = [a for a in remaining if self.case.actions[a].kind in RELEASE_KINDS]
        if not release:
            return 0
        height: dict[str, int] = {}

        def depth(aid: str) -> int:
            if aid in height:
                return height[aid]
            deps = [d for d in self.closure[aid] if d in release and d != aid]
            height[aid] = 1 + max((depth(d) for d in deps), default=0)
            return height[aid]

        return max(depth(a) for a in release)

    def _fire(self, aid: str, slot: int, waits: dict[str, int]) -> int:
        """The step at which an action takes effect from a table."""

        act = self.case.actions[aid]
        if act.kind == "barrier":
            wait = waits.get(aid, self.graph.waits.get(aid, (act.wait,))[0])
            return slot + wait
        return slot

    def _dfs(
        self,
        slot: int,
        scheduled: frozenset[str],
        assignment: dict[str, int],
        waits: dict[str, int],
        open_pauses: dict[str, int],
        closed_pause: int,
    ) -> None:
        if not self.complete:
            return
        self.nodes += 1
        if self.nodes > self.budget:
            self.complete = False
            return
        graph = self.graph
        ids = graph.ids
        if len(scheduled) == len(ids):
            self._evaluate(assignment, waits)
            return
        if slot >= graph.slots:
            return
        if self.best is not None:
            if closed_pause > self.best.key[0]:
                return
            remaining = frozenset(a for a in ids if a not in scheduled)
            used = self._release_slots_used(assignment)
            floor = used + self._min_extra_release_slots(remaining)
            if floor > self.best.key[1]:
                return

        ready: list[str] = []
        forced: list[str] = []
        for aid in ids:
            if aid in scheduled:
                continue
            if not self.closure[aid].issubset(scheduled):
                continue
            if aid in graph.fixed:
                if graph.fixed[aid] < slot:
                    return  # the fixed slot has passed
                if graph.fixed[aid] == slot:
                    forced.append(aid)
                continue
            window = graph.windows.get(aid)
            if window is not None:
                if window[1] < slot:
                    return  # the whitelist no longer permits it
                if window[0] > slot:
                    continue  # cannot run yet
            ready.append(aid)

        if forced:
            missing = [a for a in forced if a not in ready]
            if missing:
                return
            base = tuple(sorted(forced))
            optional = [a for a in ready if a not in forced]
        else:
            base = ()
            optional = ready

        # Enumerate eagerly: try a slot that carries as much work as possible
        # first, so a good incumbent appears early and prunes the rest.
        for chosen in self._subsets(base, optional):
            new_open = dict(open_pauses)
            new_closed = closed_pause
            for aid in chosen:
                act = self.case.actions[aid]
                if act.kind == "pause":
                    new_open.setdefault(act.proxy, slot)
                elif act.kind == "resume" and act.proxy in new_open:
                    new_closed += slot - new_open.pop(act.proxy)
            if self.best is not None and new_closed > self.best.key[0]:
                continue
            for aid in chosen:
                assignment[aid] = slot
            for combo in self._wait_combos(chosen):
                waits.update(combo)
                self._dfs(
                    slot + 1,
                    scheduled | set(chosen),
                    assignment,
                    waits,
                    new_open,
                    new_closed,
                )
                for key in combo:
                    waits.pop(key, None)
                if not self.complete:
                    break
            for aid in chosen:
                del assignment[aid]
            if not self.complete:
                return

    def _subsets(self, base: tuple[str, ...], optional: list[str]) -> Iterator[tuple[str, ...]]:
        """Largest subsets first, then smaller ones; deterministic within a size."""

        for size in range(len(optional), -1, -1):
            for extra in itertools.combinations(optional, size):
                yield tuple(sorted(base + extra))

    def _wait_combos(self, chosen: tuple[str, ...]) -> Iterator[dict[str, int]]:
        barrier_ids = [a for a in chosen if self.case.actions[a].kind == "barrier"]
        if not barrier_ids:
            yield {}
            return
        combos: list[dict[str, int]] = [{}]
        for aid in barrier_ids:
            options = self.graph.waits.get(aid, (0,))
            combos = [{**combo, aid: w} for combo in combos for w in options]
        for combo in combos:
            yield combo

    # -- leaves -----------------------------------------------------------

    def _fire_order_ok(
        self, assignment: dict[str, int], waits: dict[str, int]
    ) -> tuple[bool, dict[str, Any] | None]:
        """Dependencies must be satisfied by effect time, not merely by slot.

        A barrier waits before it is checked, so its slot order can differ from
        the order in which its proof becomes available.  A table whose barrier
        is *scheduled* later but *fires* earlier does not satisfy the protocol.
        """

        fire = {
            aid: self._fire(aid, slot, waits) for aid, slot in assignment.items()
        }
        for aid in self.graph.ids:
            if aid not in fire:
                continue
            for dep in sorted(self.graph.edges.get(aid, ())):
                if dep not in fire:
                    continue
                if fire[dep] >= fire[aid]:
                    return False, {
                        "step": fire[aid],
                        "code": "schedule_order",
                        "before": dep,
                        "after": aid,
                        "before_fire_step": fire[dep],
                        "after_fire_step": fire[aid],
                        "detail": "a dependency must take effect strictly earlier",
                    }
        for aid in self.graph.ids:
            step = self.graph.fixed.get(aid)
            if step is not None and fire.get(aid) != step:
                return False, {
                    "step": fire.get(aid, step),
                    "code": "fixed_step",
                    "after": aid,
                    "expected": step,
                    "detail": "the table must fire the action at its fixed step",
                }
        return True, None

    def _evaluate(self, assignment: dict[str, int], waits: dict[str, int]) -> None:
        self.leaves += 1
        if self.leaves > self.budget:
            self.complete = False
            return
        order_ok, order_problem = self._fire_order_ok(assignment, waits)
        if not order_ok:
            return
        paused = paused_steps(self.case, assignment, waits, self.window)
        release_slots = release_slot_count(self.case, assignment)
        if self.best is not None and (paused, release_slots) > self.best.key[:2]:
            return
        outcomes: list[Outcome] = []
        for bed in self.beds:
            outcome = evaluate(self.case, assignment, waits, bed, self.window)
            if not is_safe(outcome):
                return
            outcomes.append(outcome)
        self.safe_leaves += 1
        sequence = tuple(
            aid for _, aid in sorted((slot, aid) for aid, slot in assignment.items())
        )
        key = (paused, release_slots, sequence)
        if self.best is None or key < self.best.key:
            self.best = Candidate(
                key=key,
                assignment=dict(assignment),
                waits=dict(waits),
                outcomes=outcomes,
            )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_plan(
    case: CompiledCase, *, budget: int = DEFAULT_NODE_BUDGET
) -> dict[str, Any]:
    graph = build_graph(case)
    cycle = graph.cycle()
    if cycle is not None:
        return {
            "ok": False,
            "command": "plan",
            "error": {
                "code": "E_SCHEDULE",
                "field": "$.actions",
                "message": "dependency cycle: " + " -> ".join(cycle),
                "cycle": cycle,
            },
        }
    for aid, step in graph.fixed.items():
        if step >= graph.slots:
            return {
                "ok": False,
                "command": "plan",
                "error": {
                    "code": "E_SCHEDULE",
                    "field": "$.actions",
                    "message": f"fixed step of {aid!r} does not fit in {graph.slots} slots",
                },
            }
    beds = scenarios(case)
    if not graph.ids:
        return _empty_plan(case, beds, graph)

    search = Search(case, graph, beds, budget)
    best = search.run()
    if best is None:
        return _unsatisfiable(case, graph, beds, search)
    return _success(case, graph, best, beds, search)


def _empty_plan(case: CompiledCase, beds: list[Bed], graph: ConstraintGraph) -> dict[str, Any]:
    window = cost_horizon(case, graph)
    outcomes = [evaluate(case, {}, {}, bed, window) for bed in beds]
    if any(not is_safe(o) for o in outcomes):
        return {
            "ok": False,
            "command": "plan",
            "error": {
                "code": "E_UNSATISFIABLE",
                "field": "$",
                "message": "no fixed release table is safe for every permitted scenario",
                "reason": "the input has no actions and the static state already fails",
                "failures": sorted(
                    (v for o in outcomes for v in o.trace.violations), key=violation_key
                ),
            },
        }
    return {
        "ok": True,
        "command": "plan",
        "case": case.raw.get("name", "case"),
        "table": [],
        "sequence": [],
        "objective": {"paused_proxy_steps": 0, "release_steps": 0, "action_id_sequence": []},
        "scenarios": [
            {
                "scenario": bed.name,
                "fault": bed.fault,
                "isolated": True,
                "reachable": True,
                "failures": [],
                "receipts": {
                    req: sorted(branches)
                    for req, branches in sorted(o.trace.receipts.items())
                },
                "barriers": [],
            }
            for bed, o in zip(beds, outcomes)
        ],
        "constraints": [],
        "search": {"leaves": 1, "safe_leaves": 1, "complete": True, "slots": graph.slots},
    }


def _success(
    case: CompiledCase,
    graph: ConstraintGraph,
    best: Candidate,
    beds: list[Bed],
    search: Search,
) -> dict[str, Any]:
    assignment = best.assignment
    waits = best.waits
    table = []
    for slot, aid in sorted((slot, aid) for aid, slot in assignment.items()):
        act = case.actions[aid]
        wait = waits.get(aid, graph.waits.get(aid, (act.wait,))[0])
        table.append(
            {
                "slot": slot,
                "id": aid,
                "kind": act.kind,
                "proxy": act.proxy,
                "version": act.version,
                "generation": act.generation,
                "wait": wait,
                "fire_step": slot + (wait if act.kind == "barrier" else 0),
            }
        )
    per_bed = []
    for bed, outcome in zip(beds, best.outcomes):
        trace = outcome.trace
        per_bed.append(
            {
                "scenario": bed.name,
                "fault": bed.fault,
                "isolated": not any(v["code"] == "tenant_isolation" for v in trace.violations),
                "reachable": not any(
                    v["code"].startswith("unreachable") for v in trace.violations
                ),
                "failures": sorted(trace.violations, key=violation_key),
                "receipts": {
                    req: sorted(branches) for req, branches in sorted(trace.receipts.items())
                },
                "barriers": [
                    {
                        "step": b["step"],
                        "event": b["event"],
                        "proxy": b["proxy"],
                        "generation": b["generation"],
                        "ok": b["ok"],
                        "blocking": [m["marker"] for m in b["blocking"]],
                    }
                    for b in trace.barriers
                ],
            }
        )
    sequence = [entry["id"] for entry in table]
    return {
        "ok": True,
        "command": "plan",
        "case": case.raw.get("name", "case"),
        "table": table,
        "sequence": sequence,
        "objective": {
            "paused_proxy_steps": best.key[0],
            "release_steps": best.key[1],
            "action_id_sequence": sequence,
            "lexicographic": [best.key[0], best.key[1], sequence],
        },
        "scenarios": per_bed,
        "constraints": ordering_constraints(case, graph),
        "search": {
            "leaves": search.leaves,
            "safe_leaves": search.safe_leaves,
            "nodes": search.nodes,
            "complete": search.complete,
            "slots": graph.slots,
            "cost_window": search.window,
        },
    }


def ordering_constraints(case: CompiledCase, graph: ConstraintGraph) -> list[dict[str, Any]]:
    closure = graph.closure()
    rows: list[dict[str, Any]] = []
    for aid in graph.ids:
        act = case.actions[aid]
        for dep in sorted(closure[aid]):
            if dep in act.after:
                reason = "explicit_after"
            elif case.actions[dep].kind == "barrier" and act.kind.startswith("retire"):
                reason = "retire_requires_barrier"
            else:
                reason = "implicit_protocol"
            rows.append({"before": dep, "after": aid, "reason": reason, "proxy": act.proxy})
    rows.sort(key=lambda r: (r["before"], r["after"], r["reason"]))
    return rows


# ---------------------------------------------------------------------------
# Unsatisfiable: baseline diagnostics
# ---------------------------------------------------------------------------


def _unsatisfiable(
    case: CompiledCase,
    graph: ConstraintGraph,
    beds: list[Bed],
    search: Search,
) -> dict[str, Any]:
    baseline = baseline_positions(case)
    conflict = earliest_schedule_conflict(case, graph, baseline)
    base = {
        "ok": False,
        "command": "plan",
        "error": {
            "code": "E_UNSATISFIABLE",
            "field": "$",
            "message": "no fixed release table is safe for every permitted scenario",
        },
        "search": {
            "leaves": search.leaves,
            "safe_leaves": search.safe_leaves,
            "nodes": search.nodes,
            "complete": search.complete,
        },
        "scenarios": [bed.name for bed in beds],
    }
    if conflict is not None:
        base["error"]["reason"] = "baseline_order violates the scheduling constraints"
        base["error"]["baseline_conflict"] = conflict
        return base
    base["error"]["reason"] = "baseline_order is schedulable but not safe"
    base["error"]["failure_prefix"] = shortest_failure_prefix(case, graph, beds, search.window)
    return base


def baseline_positions(case: CompiledCase) -> dict[str, int]:
    order = list(case.baseline)
    seen = set(order)
    for aid in case.action_order:
        if aid not in seen:
            order.append(aid)
            seen.add(aid)
    return {aid: index for index, aid in enumerate(order)}


def earliest_schedule_conflict(
    case: CompiledCase, graph: ConstraintGraph, position: dict[str, int]
) -> dict[str, Any] | None:
    conflicts: list[dict[str, Any]] = []
    for aid in graph.ids:
        act = case.actions[aid]
        for dep in sorted(act.after):
            if position[dep] >= position[aid]:
                conflicts.append(
                    {
                        "step": position[aid],
                        "code": "explicit_after",
                        "before": dep,
                        "after": aid,
                        "detail": f"{dep!r} must be scheduled strictly before {aid!r}",
                    }
                )
    for aid in graph.ids:
        act = case.actions[aid]
        if act.kind not in ("retire_inbound", "retire_egress"):
            continue
        covering = [
            b
            for b in graph.ids
            if case.actions[b].kind == "barrier"
            and case.actions[b].proxy == act.proxy
            and int(case.actions[b].generation or 0) >= int(act.generation or 0)
        ]
        if covering and not any(position[b] < position[aid] for b in covering):
            conflicts.append(
                {
                    "step": position[aid],
                    "code": "retire_requires_barrier",
                    "after": aid,
                    "proxy": act.proxy,
                    "barrier_candidates": sorted(covering),
                    "detail": "retire has no earlier covering barrier",
                }
            )
    for aid in graph.ids:
        if aid in graph.windows:
            lo, hi = graph.windows[aid]
            if not (lo <= position[aid] <= hi):
                conflicts.append(
                    {
                        "step": position[aid],
                        "code": "slot_window",
                        "after": aid,
                        "window": [lo, hi],
                        "detail": "baseline position is outside the permitted slots",
                    }
                )
        if aid in graph.fixed and position[aid] != graph.fixed[aid]:
            conflicts.append(
                {
                    "step": position[aid],
                    "code": "fixed_step",
                    "after": aid,
                    "expected": graph.fixed[aid],
                    "detail": "baseline position differs from the fixed step",
                }
            )
    if not conflicts:
        return None
    conflicts.sort(key=lambda c: (c["step"], c["code"], c.get("before", ""), c.get("after", "")))
    first = conflicts[0]
    ordered = [aid for aid, _ in sorted(position.items(), key=lambda kv: (kv[1], kv[0]))]
    first["prefix"] = [aid for aid in ordered if position[aid] <= first["step"]]
    return first


def shortest_failure_prefix(
    case: CompiledCase,
    graph: ConstraintGraph,
    beds: list[Bed],
    window: int,
) -> dict[str, Any]:
    """Shortest baseline prefix in which isolation or reachability fails.

    The prefix is measured in baseline events: the actions scheduled at or
    before the violating instant.  Ties are broken by the event-id sequence,
    then by the scenario name, so the answer is stable.
    """

    position = baseline_positions(case)
    assignment = dict(position)
    waits = {
        aid: graph.waits[aid][0] for aid in graph.ids if case.actions[aid].kind == "barrier"
    }
    candidates: list[dict[str, Any]] = []
    for bed in beds:
        outcome = evaluate(case, assignment, waits, bed, window)
        trace = outcome.trace
        if not trace.violations:
            continue
        earliest = min(trace.violations, key=violation_key)
        limit = int(earliest["step"])
        prefix = [
            aid
            for aid, idx in sorted(position.items(), key=lambda kv: (kv[1], kv[0]))
            if idx <= limit
        ]
        candidates.append(
            {
                "scenario": bed.name,
                "step": int(earliest["step"]),
                "violation": earliest,
                "prefix": prefix,
                "prefix_ids": sorted(prefix, key=lambda a: (position[a], a)),
            }
        )
    if not candidates:
        return {}
    candidates.sort(key=lambda c: (len(c["prefix"]), c["step"], c["scenario"], c["prefix_ids"]))
    return candidates[0]
