"""Compiled case model and the discrete-step marker simulator.

Model summary
-------------
A *marker* is a payload-free abstract identity token.  It carries the original
tenant, the calling identity, the target service, a label, a generation, and an
absolute expiry step.  The original tenant never changes; only the label may be
rewritten by an approved egress mapping.

At a proxy a marker is processed in this order:

1. inbound check -- an active inbound rule for this proxy must match
   ``(tenant, identity, service, label, generation)`` exactly;
2. egress mapping -- an active egress rule must match
   ``(tenant, identity, service, label_in, generation)`` and name one of this
   proxy's links; the marker leaves with ``label_out``;
3. transit -- ``forward`` steps at the proxy plus ``delivery`` steps on the
   link; the marker's absolute expiry is unchanged, so a long loop can expire.

A marker arriving at a service node is a *delivery*.  It is legal only when
``(original tenant, label)`` is listed in that service's ``accepts`` set.

Lifetime
--------
``expires_at = injection_step + ttl - 1``.  A marker is alive while
``step <= expires_at`` and is discarded at the first step after that, so it can
make at most ``ttl`` hops/steps including its injection step.  ``ttl_left`` is
``expires_at - step + 1``: the number of steps it can still be used.

Steps are discrete integers.  The simulator is event driven: it visits only
steps at which something can change (actions, request starts, fault window
edges, arrivals, expiries) and skips every step that cannot affect the future
because no marker is in flight and no policy change is pending.  This is the
timeline compression described in the README.

Everything is a pure function of the case, the scenario and the event
timeline, so replay is bit-for-bit reproducible.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

DROP_TTL = "ttl_expired"
DROP_FAULT = "fault_dropped"
DROP_REJECTED = "inbound_rejected"
DROP_NO_EGRESS = "no_egress_mapping"
DROP_CLOSED_GENERATION = "closed_generation"
DROP_STALE_INJECTION = "stale_injection"
DROP_PAUSED_EXPIRED = "expired_while_paused"
DROP_ZERO_DELAY = "zero_delay_cycle"


@dataclass
class Action:
    id: str
    kind: str
    proxy: str
    version: str | None = None
    generation: int | None = None
    wait: int = 0
    after: tuple[str, ...] = ()
    slots: tuple[int, ...] | None = None

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "Action":
        slots = raw.get("slots")
        return cls(
            id=raw["id"],
            kind=raw["kind"],
            proxy=raw["proxy"],
            version=raw.get("version"),
            generation=raw.get("generation"),
            wait=int(raw.get("wait", 0) or 0),
            after=tuple(sorted(raw.get("after", []) or [])),
            slots=tuple(sorted(slots)) if slots else None,
        )


@dataclass(frozen=True)
class Thread:
    """One independent injection branch of a request."""

    request: str
    branch: str
    forced_link: str | None


@dataclass
class CompiledCase:
    raw: dict[str, Any]
    ttl: int
    horizon: int
    node_kind: dict[str, str]
    node_service: dict[str, str]
    services: dict[str, set[tuple[str, str]]]
    links: dict[str, dict[str, Any]]
    out_links: dict[str, list[str]]
    identities: dict[str, dict[str, str]]
    tenants: tuple[str, ...]
    inbound: list[dict[str, Any]]
    egress: list[dict[str, Any]]
    requests: list[dict[str, Any]]
    actions: dict[str, Action]
    action_order: list[str]
    baseline: list[str]
    latency_scenarios: list[dict[str, Any]]
    fault: dict[str, Any] | None
    policy_slots: int
    proxies: tuple[str, ...] = ()
    service_nodes: tuple[str, ...] = ()
    reach_to: dict[str, set[str]] = field(default_factory=dict)
    reach_from: dict[str, set[str]] = field(default_factory=dict)
    initial_active: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    egress_groups: dict[tuple[str, str], list[dict[str, Any]]] = field(default_factory=dict)
    rule_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)

    def links_from(self, proxy: str) -> list[str]:
        return self.out_links.get(proxy, [])

    def request_start(self, req: dict[str, Any]) -> int:
        return int(req.get("start", 0) or 0)

    def request_starts(self, req: dict[str, Any]) -> list[int]:
        """Every step at which the request is (re-)injected.

        ``start`` is the initial injection.  ``retries`` lists additional
        steps; each retry is an independent copy with its own marker id but the
        *same* request and branch, which is how duplicate deliveries and
        duplicate receipts are exercised.
        """

        starts = [self.request_start(req)]
        starts.extend(int(x) for x in (req.get("retries") or []))
        return sorted(starts)

    def request_generation(self, req: dict[str, Any]) -> int:
        return int(req.get("generation", 0) or 0)

    def threads(self, req: dict[str, Any]) -> list[Thread]:
        """Independent injection threads for one request.

        A fan-out request injects one thread per egress link of its entry
        proxy, so two branches can never share a receipt and can never
        overwrite each other.  A plain request has a single thread named by
        ``branch`` (default ``main``).
        """

        if req.get("fanout"):
            return [
                Thread(req["id"], f"{req['id']}#{lid}", lid)
                for lid in self.links_from(req["entry"])
            ]
        return [Thread(req["id"], req.get("branch") or "main", None)]


def compile_case(case: dict[str, Any]) -> CompiledCase:
    services: dict[str, set[tuple[str, str]]] = {}
    for svc in case["services"]:
        services[svc["id"]] = {(a["tenant"], a["label"]) for a in svc["accepts"]}

    node_kind: dict[str, str] = {}
    node_service: dict[str, str] = {}
    for node in case["nodes"]:
        node_kind[node["id"]] = node["kind"]
        if node["kind"] == "service":
            node_service[node["id"]] = node["service"]

    links = {link["id"]: link for link in case["links"]}
    proxies = tuple(sorted(n for n, k in node_kind.items() if k == "proxy"))
    service_nodes = tuple(sorted(n for n, k in node_kind.items() if k == "service"))
    out_links: dict[str, list[str]] = {p: [] for p in proxies}
    for link in case["links"]:
        out_links[link["from"]].append(link["id"])
    for val in out_links.values():
        val.sort()

    actions = {a["id"]: Action.from_raw(a) for a in case.get("actions", [])}
    action_order = [a["id"] for a in case.get("actions", [])]

    reach_to: dict[str, set[str]] = {}
    for origin in list(node_kind):
        seen: set[str] = {origin}
        stack = [origin]
        while stack:
            cur = stack.pop()
            for lid in out_links.get(cur, []):
                nxt = links[lid]["to"]
                if nxt in seen:
                    continue
                seen.add(nxt)
                stack.append(nxt)
        reach_to[origin] = seen
    reach_from: dict[str, set[str]] = {n: set() for n in node_kind}
    for src, targets in reach_to.items():
        for dst in targets:
            reach_from[dst].add(src)

    # Initial policy state: a version introduced by an install/switch action
    # starts inactive; every other version starts active.  ``policy.initial``
    # overrides this per proxy when supplied.
    introduced_in: dict[str, set[str]] = {p: set() for p in proxies}
    introduced_eg: dict[str, set[str]] = {p: set() for p in proxies}
    for act in actions.values():
        if act.kind == "install_inbound" and act.version:
            introduced_in[act.proxy].add(act.version)
        elif act.kind == "switch_egress" and act.version:
            introduced_eg[act.proxy].add(act.version)

    initial_active: dict[str, dict[str, set[str]]] = {}
    explicit = (case.get("policy") or {}).get("initial") or {}
    for proxy in proxies:
        if proxy in explicit:
            spec = explicit[proxy]
            initial_active[proxy] = {
                "inbound": set(spec.get("inbound", [])),
                "egress": set(spec.get("egress", [])),
            }
            continue
        initial_active[proxy] = {
            "inbound": {
                r["version"]
                for r in case["policy"]["inbound"]
                if r["proxy"] == proxy and r["version"] not in introduced_in[proxy]
            },
            "egress": {
                r["version"]
                for r in case["policy"]["egress"]
                if r["proxy"] == proxy and r["version"] not in introduced_eg[proxy]
            },
        }

    # An egress *switch group* is the set of mutually exclusive mappings: when
    # the group's version is switched to, every other version in the group is
    # deactivated in the same step.  Rules without a group switch on their own.
    egress_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rule in case["policy"]["egress"]:
        group = rule.get("switch_group") or rule["version"]
        egress_groups.setdefault((rule["proxy"], group), []).append(rule)
    for val in egress_groups.values():
        val.sort(key=lambda r: r["id"])

    scenarios = case.get("scenarios") or {}
    rule_by_id: dict[str, dict[str, Any]] = {}
    for rule in case["policy"]["inbound"] + case["policy"]["egress"]:
        rule_by_id[rule["id"]] = rule

    return CompiledCase(
        raw=case,
        ttl=int(case["ttl"]),
        horizon=int(case["horizon"]),
        node_kind=node_kind,
        node_service=node_service,
        services=services,
        links=links,
        out_links=out_links,
        identities={i["id"]: i for i in case["identities"]},
        tenants=tuple(sorted(case["tenants"])),
        inbound=list(case["policy"]["inbound"]),
        egress=list(case["policy"]["egress"]),
        requests=list(case["requests"]),
        actions=actions,
        action_order=action_order,
        baseline=list(case.get("baseline_order") or []),
        latency_scenarios=list(scenarios.get("latency") or []),
        fault=scenarios.get("fault"),
        policy_slots=int((case.get("plan") or {}).get("slots", 10)),
        proxies=proxies,
        service_nodes=service_nodes,
        reach_to=reach_to,
        reach_from=reach_from,
        initial_active=initial_active,
        egress_groups=egress_groups,
        rule_by_id=rule_by_id,
    )


# ---------------------------------------------------------------------------
# Latency / fault
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaultWindow:
    domain: str
    fail_at: int
    recover_at: int
    stretch: int = 0

    def down(self, step: int) -> bool:
        return self.fail_at <= step < self.recover_at

    def recovered(self, step: int) -> bool:
        return step >= self.recover_at


class Latency:
    """Resolved per-link delays for one scenario, with recovery stretching."""

    def __init__(
        self,
        base: dict[str, dict[str, Any]],
        override: dict[str, dict[str, int]] | None,
        stretch: int,
    ) -> None:
        self._base = base
        self._override = override or {}
        self.stretch = stretch

    def domain(self, link_id: str) -> str:
        return self._base[link_id]["domain"]

    def delays(self, link_id: str, recovered: bool = False) -> tuple[int, int]:
        link = self._base[link_id]
        patch = self._override.get(link_id, {})
        delivery = int(patch.get("delivery", link["delivery"]))
        forward = int(patch.get("forward", link["forward"]))
        if recovered:
            delivery += self.stretch
        return forward, delivery


# ---------------------------------------------------------------------------
# Markers and trace containers
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Marker:
    mid: str
    request: str
    branch: str
    identity: str
    tenant: str
    service: str
    label: str
    generation: int
    injected_at: int
    expires_at: int
    node: str | None = None
    link: str | None = None
    arrive_at: int | None = None
    depart_at: int | None = None
    queued_since: int | None = None
    hops: int = 0
    trail: list[str] = field(default_factory=list)

    def ttl_left(self, step: int) -> int:
        return max(0, self.expires_at - step + 1)

    def as_dict(self, step: int) -> dict[str, Any]:
        out: dict[str, Any] = {
            "marker": self.mid,
            "request": self.request,
            "branch": self.branch,
            "tenant": self.tenant,
            "identity": self.identity,
            "service": self.service,
            "label": self.label,
            "generation": self.generation,
            "ttl_left": self.ttl_left(step),
            "trail": list(self.trail),
        }
        if self.node is not None:
            out["at"] = self.node
        if self.link is not None:
            out["link"] = self.link
            out["arrive_at"] = self.arrive_at
        if self.queued_since is not None:
            out["queued_since"] = self.queued_since
        return out


@dataclass
class Trace:
    barriers: list[dict[str, Any]] = field(default_factory=list)
    deliveries: list[dict[str, Any]] = field(default_factory=list)
    drops: list[dict[str, Any]] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    violations: list[dict[str, Any]] = field(default_factory=list)
    receipts: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    late: list[dict[str, Any]] = field(default_factory=list)
    duplicates: list[dict[str, Any]] = field(default_factory=list)
    generation_mismatches: list[dict[str, Any]] = field(default_factory=list)
    actions_applied: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    compressed_steps: int = 0
    visited_steps: list[int] = field(default_factory=list)
    last_step: int = 0
    marker_fates: dict[str, dict[str, Any]] = field(default_factory=dict)
    in_flight: list[dict[str, Any]] = field(default_factory=list)

    def earliest(self) -> dict[str, Any] | None:
        if not self.violations:
            return None
        return min(self.violations, key=violation_key)


def violation_key(v: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(v.get("step", 0)),
        str(v.get("code", "")),
        str(v.get("event", "")),
        str(v.get("request", "")),
        str(v.get("marker", "")),
        stable(v),
    )


def stable(value: Any) -> str:
    """Stable short rendering used only for deterministic tie-breaking."""

    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{stable(v)}" for k, v in sorted(value.items())) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(stable(v) for v in value) + "]"
    return str(value)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class Simulator:
    """Executes one event timeline under one latency/fault scenario."""

    def __init__(
        self,
        case: CompiledCase,
        latency: Latency,
        fault: FaultWindow | None,
        event_steps: dict[str, int],
        waits: dict[str, int] | None = None,
    ) -> None:
        self.case = case
        self.latency = latency
        self.fault = fault
        self.waits = dict(waits or {})
        self.trace = Trace()
        self.step = 0
        self.markers: list[Marker] = []
        self.next_index = 0
        self.inbound_active = {p: set(v["inbound"]) for p, v in case.initial_active.items()}
        self.egress_active = {p: set(v["egress"]) for p, v in case.initial_active.items()}
        self.retired: set[tuple[str, str, str]] = set()  # (proxy, kind, version)
        self.closed: dict[str, set[int]] = {p: set() for p in case.proxies}
        self.paused: set[str] = set()
        self.proof: dict[str, int] = {}
        self._times: list[int] = []
        self._seen: set[int] = set()
        self.actions_at = self._schedule(event_steps)
        self.requests_at: dict[int, list[dict[str, Any]]] = {}
        for req in case.requests:
            for start in case.request_starts(req):
                self.requests_at.setdefault(start, []).append(req)

    # -- time management --------------------------------------------------

    def wait_of(self, action: "Action") -> int:
        if action.id in self.waits:
            return int(self.waits[action.id])
        return action.wait if action.kind == "barrier" else 0

    def _schedule(self, event_steps: dict[str, int]) -> dict[int, list[str]]:
        table: dict[int, list[str]] = {}
        for aid, step in event_steps.items():
            act = self.case.actions.get(aid)
            if act is None:
                continue
            table.setdefault(step + self.wait_of(act), []).append(aid)
        for val in table.values():
            val.sort()
        return table

    def _push(self, step: int) -> None:
        if step not in self._seen:
            self._seen.add(step)
            heapq.heappush(self._times, step)

    def _seed_times(self) -> None:
        for step in self.actions_at:
            self._push(step)
        for step in self.requests_at:
            self._push(step)
        if self.fault is not None:
            self._push(self.fault.fail_at)
            self._push(self.fault.recover_at)

    def _after_step(self) -> None:
        """Schedule every time that this step's state can still influence."""

        for marker in self.markers:
            if marker.arrive_at is not None and marker.arrive_at > self.step:
                self._push(marker.arrive_at)
            # Expiry is still relevant: a marker cannot outlive it, and the
            # drop has to be recorded at the exact step it happens.
            if marker.expires_at + 1 > self.step:
                self._push(marker.expires_at + 1)
        if self.fault is not None and self.step < self.fault.recover_at:
            # In-flight markers crossing the window are dropped when it opens
            # (handled by drop_faulted), so no per-step scan is needed.
            pass

    # -- rule lookups -----------------------------------------------------

    def _rule_active(self, rule: dict[str, Any], kind: str) -> bool:
        if (rule["proxy"], kind, rule["version"]) in self.retired:
            return False
        active = self.inbound_active if kind == "inbound" else self.egress_active
        if rule["version"] not in active[rule["proxy"]]:
            return False
        if self._closed(rule["proxy"], int(rule.get("generation", 0))):
            return False
        return True

    def _closed(self, proxy: str, generation: int) -> bool:
        return any(generation < g for g in self.closed.get(proxy, ()))

    def _match_inbound(self, proxy: str, marker: Marker) -> dict[str, Any] | None:
        for rule in self.case.inbound:
            if rule["proxy"] != proxy or rule["tenant"] != marker.tenant:
                continue
            if rule["service"] != marker.service or rule["identity"] != marker.identity:
                continue
            if rule["label"] != marker.label or int(rule.get("generation", 0)) != marker.generation:
                continue
            if self._rule_active(rule, "inbound"):
                return rule
        return None

    def _match_egress(
        self, proxy: str, marker: Marker, forced_link: str | None, branch: str
    ) -> dict[str, Any] | None:
        candidates = []
        for rule in self.case.egress:
            if rule["proxy"] != proxy or rule["tenant"] != marker.tenant:
                continue
            if rule["service"] != marker.service or rule["identity"] != marker.identity:
                continue
            if rule["label_in"] != marker.label or int(rule.get("generation", 0)) != marker.generation:
                continue
            if forced_link is not None and rule["link"] != forced_link:
                continue
            if not self._rule_active(rule, "egress"):
                continue
            candidates.append(rule)
        if not candidates:
            return None
        pinned = [r for r in candidates if r.get("branch") == branch]
        chosen = pinned if pinned else candidates
        return sorted(chosen, key=lambda r: r["id"])[0]

    # -- per-step phases (the order is part of the model contract) --------

    def at_step(self, step: int) -> None:
        self.step = step
        self.trace.last_step = step
        self.trace.visited_steps.append(step)
        # Order matters and is part of the contract:
        #  1. a link that is down at this step destroys in-flight markers;
        #  2. policy actions take effect at the *start* of the step;
        #  3. expired markers are discarded;
        #  4. new requests are injected at the (possibly just changed) entry;
        #  5. arrivals are processed, including zero-delay cascades;
        #  6. barriers are checked at the *end* of the step, so an arrival at
        #     this very step is counted as still in reach.
        self.drop_faulted(step)
        self.apply_actions(step, phase="start")
        self.drop_expired(step)
        self.inject(step)
        self.settle(step)
        self.apply_actions(step, phase="end")

    def settle(self, step: int) -> None:
        """Process arrivals at ``step`` until zero-delay hops are exhausted."""

        budget = 4 * (len(self.case.node_kind) + 4)
        for marker in self.markers:
            marker.hops = 0
        while True:
            pending = [m for m in self.markers if m.arrive_at == step]
            if not pending:
                break
            for marker in pending:
                if marker not in self.markers:
                    continue
                marker.hops += 1
                if marker.hops > budget:
                    self.trace.rejections.append(
                        {
                            "step": step,
                            "code": DROP_ZERO_DELAY,
                            "marker": marker.mid,
                            "reason": "marker did not advance within one step",
                        }
                    )
                    self._drop(marker, step, DROP_ZERO_DELAY)
                    continue
                self._arrive_one(marker, step)

    def drop_faulted(self, step: int) -> None:
        if self.fault is None or not self.fault.down(step):
            return
        for marker in list(self.markers):
            if marker.link is None:
                continue
            if self.latency.domain(marker.link) != self.fault.domain:
                continue
            self._drop(marker, step, DROP_FAULT)

    def drop_expired(self, step: int) -> None:
        for marker in list(self.markers):
            if marker.expires_at >= step:
                continue
            reason = (
                DROP_PAUSED_EXPIRED
                if (marker.node in self.paused and marker.link is None)
                else DROP_TTL
            )
            self._drop(marker, step, reason)

    def inject(self, step: int) -> None:
        for req in self.requests_at.get(step, ()):
            gen = self.case.request_generation(req)
            entry = req["entry"]
            if self._closed(entry, gen):
                self.trace.rejections.append(
                    {
                        "step": step,
                        "code": DROP_STALE_INJECTION,
                        "request": req["id"],
                        "proxy": entry,
                        "generation": gen,
                        "reason": "generation already closed at the entry proxy",
                    }
                )
                continue
            if entry in self.paused:
                self.trace.rejections.append(
                    {
                        "step": step,
                        "code": "entry_paused",
                        "request": req["id"],
                        "proxy": entry,
                        "generation": gen,
                        "reason": "entry proxy is paused; new markers are not accepted",
                    }
                )
                continue
            for thread in self.case.threads(req):
                self.next_index += 1
                marker = Marker(
                    mid=f"{req['id']}/{thread.branch}/{self.next_index:03d}",
                    request=req["id"],
                    branch=thread.branch,
                    identity=req["identity"],
                    tenant=req["tenant"],
                    service=req["service"],
                    label=req.get("label", ""),
                    generation=gen,
                    injected_at=step,
                    expires_at=step + self.case.ttl - 1,
                    node=entry,
                    trail=[entry],
                )
                self.markers.append(marker)
                self.trace.events.append(
                    {"step": step, "event": "inject", "marker": marker.mid, "proxy": entry}
                )
                # A marker created at a proxy is checked exactly like one that
                # arrived: inbound rule, then egress mapping.
                self._at_proxy(marker, step, entry, thread.forced_link)

    # -- arrivals ---------------------------------------------------------

    def _arrive_one(self, marker: Marker, step: int) -> None:
        link = marker.link
        marker.arrive_at = None
        marker.link = None
        node = marker.node or ""
        self.trace.events.append(
            {"step": step, "event": "arrive", "marker": marker.mid, "node": node, "link": link}
        )
        if self.case.node_kind.get(node) == "service":
            self._deliver(marker, step, node)
            return
        self._at_proxy(marker, step, node, None)

    def _at_proxy(self, marker: Marker, step: int, proxy: str, forced_link: str | None) -> None:
        if proxy in self.paused:
            marker.node = proxy
            marker.link = None
            marker.arrive_at = None
            marker.queued_since = step
            self.trace.events.append(
                {"step": step, "event": "queue", "marker": marker.mid, "proxy": proxy}
            )
            return
        if self._closed(proxy, marker.generation):
            self.trace.generation_mismatches.append(
                {
                    "step": step,
                    "code": "generation_closed_at_proxy",
                    "marker": marker.mid,
                    "proxy": proxy,
                    "generation": marker.generation,
                    "closed": sorted(self.closed[proxy]),
                    "reason": "marker generation is closed at this proxy",
                }
            )
            self._drop(marker, step, DROP_CLOSED_GENERATION)
            return
        rule = self._match_inbound(proxy, marker)
        if rule is None:
            self.trace.rejections.append(
                {
                    "step": step,
                    "code": DROP_REJECTED,
                    "marker": marker.mid,
                    "proxy": proxy,
                    "label": marker.label,
                    "generation": marker.generation,
                    "reason": "no active inbound rule matches identity, service, label and generation",
                }
            )
            self._drop(marker, step, DROP_REJECTED)
            return
        marker.node = proxy
        marker.link = None
        marker.queued_since = None
        self.trace.events.append(
            {
                "step": step,
                "event": "accept",
                "marker": marker.mid,
                "proxy": proxy,
                "rule": rule["id"],
                "version": rule["version"],
            }
        )
        self._forward(marker, step, proxy, forced_link)

    def _forward(self, marker: Marker, step: int, proxy: str, forced_link: str | None) -> None:
        rule = self._match_egress(proxy, marker, forced_link, marker.branch)
        if rule is None:
            self.trace.rejections.append(
                {
                    "step": step,
                    "code": DROP_NO_EGRESS,
                    "marker": marker.mid,
                    "proxy": proxy,
                    "label": marker.label,
                    "generation": marker.generation,
                    "reason": "no active egress mapping for the current label",
                }
            )
            self._drop(marker, step, DROP_NO_EGRESS)
            return
        link_id = rule["link"]
        link = self.case.links[link_id]
        in_domain = self.fault is not None and self.latency.domain(link_id) == self.fault.domain
        if in_domain and self.fault.down(step):
            self._drop(marker, step, DROP_FAULT)
            return
        forward_d, delivery_d = self.latency.delays(
            link_id, recovered=bool(in_domain and self.fault.recovered(step))
        )
        arrive = step + forward_d + delivery_d
        if in_domain:
            # A hop that is in flight when the window opens is lost; the
            # marker is not revived by the later recovery, and a marker that
            # already expired during the outage stays expired.
            if any(self.fault.down(t) for t in range(step, arrive + 1)):
                self._drop(marker, step, DROP_FAULT)
                return
        marker.node = link["to"]
        marker.link = link_id
        marker.depart_at = step
        marker.arrive_at = arrive
        marker.label = rule["label_out"]
        marker.trail.append(f"{link_id}>{rule['label_out']}")
        self.trace.events.append(
            {
                "step": step,
                "event": "forward",
                "marker": marker.mid,
                "proxy": proxy,
                "rule": rule["id"],
                "version": rule["version"],
                "link": link_id,
                "label_out": rule["label_out"],
                "arrive_at": arrive,
            }
        )
        if arrive > step:
            # Same-step hops are handled by ``settle``; scheduling the current
            # step again would re-run its action and injection phases.
            self._push(arrive)

    def _deliver(self, marker: Marker, step: int, node: str) -> None:
        service = self.case.node_service[node]
        req = next((r for r in self.case.requests if r["id"] == marker.request), None)
        accepted = (marker.tenant, marker.label) in self.case.services.get(service, set())
        deadline = int(req["deadline"]) if req is not None else 0
        late = step > deadline
        self.markers.remove(marker)
        self.trace.deliveries.append(
            {
                "step": step,
                "marker": marker.mid,
                "request": marker.request,
                "branch": marker.branch,
                "service": service,
                "node": node,
                "tenant": marker.tenant,
                "label": marker.label,
                "generation": marker.generation,
                "accepted": accepted,
                "late": late,
                "trail": list(marker.trail),
            }
        )
        self.trace.events.append(
            {"step": step, "event": "deliver", "marker": marker.mid, "node": node, "service": service}
        )
        self.trace.marker_fates[marker.mid] = {
            "outcome": "delivered",
            "step": step,
            "service": service,
            "label": marker.label,
            "late": late,
            "accepted": accepted,
        }
        if not accepted:
            self.trace.violations.append(
                {
                    "step": step,
                    "code": "tenant_isolation",
                    "marker": marker.mid,
                    "request": marker.request,
                    "branch": marker.branch,
                    "service": service,
                    "tenant": marker.tenant,
                    "label": marker.label,
                    "detail": "service does not accept this (tenant, label) pair",
                }
            )
            return

        slot = self.trace.receipts.setdefault(marker.request, {})
        branch = marker.branch
        if branch in slot:
            self.trace.duplicates.append(
                {
                    "step": step,
                    "marker": marker.mid,
                    "request": marker.request,
                    "branch": branch,
                    "kept_marker": slot[branch]["marker"],
                    "kept_step": slot[branch]["step"],
                    "reason": "branch already has a receipt; the earliest receipt is kept",
                }
            )
            self.trace.events.append(
                {"step": step, "event": "duplicate_receipt", "marker": marker.mid, "branch": branch}
            )
            return
        slot[branch] = {
            "step": step,
            "marker": marker.mid,
            "service": service,
            "tenant": marker.tenant,
            "label": marker.label,
            "generation": marker.generation,
            "late": late,
        }
        if late:
            self.trace.late.append(
                {
                    "step": step,
                    "marker": marker.mid,
                    "request": marker.request,
                    "branch": branch,
                    "deadline": deadline,
                    "reason": "delivery arrived after the request deadline",
                }
            )
        expected = self.case.request_generation(req) if req is not None else marker.generation
        if marker.generation != expected:
            self.trace.generation_mismatches.append(
                {
                    "step": step,
                    "marker": marker.mid,
                    "request": marker.request,
                    "expected": expected,
                    "actual": marker.generation,
                    "reason": "receipt generation differs from the request generation",
                }
            )

    # -- drops ------------------------------------------------------------

    def _drop(self, marker: Marker, step: int, reason: str) -> None:
        if marker not in self.markers:
            return
        self.markers.remove(marker)
        self.trace.drops.append(
            {
                "step": step,
                "code": reason,
                "marker": marker.mid,
                "request": marker.request,
                "branch": marker.branch,
                "label": marker.label,
                "generation": marker.generation,
                "ttl_left": marker.ttl_left(step),
                "expires_at": marker.expires_at,
                "trail": list(marker.trail),
            }
        )
        self.trace.events.append(
            {"step": step, "event": "drop", "marker": marker.mid, "code": reason}
        )
        self.trace.marker_fates[marker.mid] = {"outcome": "dropped", "code": reason, "step": step}

    # -- actions ----------------------------------------------------------

    def apply_actions(self, step: int, phase: str = "start") -> None:
        """Apply actions scheduled at ``step``.

        Barrier drains are evaluated at the end of the step (``phase="end"``);
        every other action takes effect at the start.  This keeps the
        semantics "wait a slot, then check at the end of that slot" exact: a
        marker delivered during the same slot is still in flight at the check
        and blocks the barrier.
        """

        for aid in self.actions_at.get(step, ()):
            act = self.case.actions[aid]
            if (act.kind == "barrier") != (phase == "end"):
                continue
            record: dict[str, Any] = {"step": step, "id": aid, "kind": act.kind, "proxy": act.proxy}
            if act.kind == "install_inbound":
                self.inbound_active[act.proxy].add(act.version)
                record["version"] = act.version
                record["note"] = "install only adds inbound acceptance; egress is untouched"
            elif act.kind == "switch_egress":
                self._switch_egress(act, record)
            elif act.kind in ("retire_inbound", "retire_egress"):
                self._retire(act, record)
            elif act.kind == "pause":
                self.paused.add(act.proxy)
            elif act.kind == "resume":
                self.paused.discard(act.proxy)
                record["resumed_markers"] = self._resume(act.proxy, step)
            elif act.kind == "barrier":
                record.update(self._barrier(act, step))
            self.trace.actions_applied.append(record)

    def _switch_egress(self, act: Action, record: dict[str, Any]) -> None:
        """Switch the egress mapping to ``version`` atomically.

        Every version that shares an egress group with ``version`` at this
        proxy is deactivated in the same step; the new version becomes active.
        """

        self.egress_active[act.proxy].add(act.version)
        group_key = None
        for (proxy, group), rules in self.case.egress_groups.items():
            if proxy != act.proxy:
                continue
            if any(r["version"] == act.version for r in rules):
                group_key = (proxy, group)
                break
        deactivated: list[str] = []
        if group_key is not None:
            versions = {r["version"] for r in self.case.egress_groups[group_key]}
            for old in sorted(versions - {act.version}):
                self.egress_active[act.proxy].discard(old)
                deactivated.append(old)
        record["version"] = act.version
        record["deactivated"] = deactivated

    def _retire(self, act: Action, record: dict[str, Any]) -> None:
        """Retire a rule version after a real drain proof.

        The proof must cover every generation the retired rules can still be
        matched with, not merely the generation the action names.
        """

        kind = "inbound" if act.kind == "retire_inbound" else "egress"
        rules = self.case.inbound if kind == "inbound" else self.case.egress
        affected = [
            int(rule.get("generation", 0))
            for rule in rules
            if rule["proxy"] == act.proxy and rule["version"] == act.version
        ]
        required = max(affected + [int(act.generation or 0)])
        proven = self.proof.get(act.proxy, -1)
        if proven < required:
            self.trace.violations.append(
                {
                    "step": self.step,
                    "code": "retire_without_drain",
                    "event": act.id,
                    "proxy": act.proxy,
                    "version": act.version,
                    "required_generation": required,
                    "proven_generation": proven,
                    "detail": "no successful barrier proves the affected generations are drained",
                }
            )
            record["retired"] = False
            return
        self.retired.add((act.proxy, kind, act.version))
        record["version"] = act.version
        record["retired"] = True
        record["required_generation"] = required

    def _resume(self, proxy: str, step: int) -> int:
        resumed = 0
        for marker in list(self.markers):
            if marker.node != proxy or marker.arrive_at is not None:
                continue
            if marker.queued_since is None:
                continue
            marker.queued_since = None
            if marker.expires_at < step:
                self._drop(marker, step, DROP_PAUSED_EXPIRED)
                continue
            if self._closed(proxy, marker.generation):
                self._drop(marker, step, DROP_CLOSED_GENERATION)
                continue
            rule = self._match_inbound(proxy, marker)
            if rule is None:
                self.trace.rejections.append(
                    {
                        "step": step,
                        "code": DROP_REJECTED,
                        "marker": marker.mid,
                        "proxy": proxy,
                        "label": marker.label,
                        "generation": marker.generation,
                        "reason": "queued marker no longer matches an active inbound rule",
                    }
                )
                self._drop(marker, step, DROP_REJECTED)
                continue
            self._forward(marker, step, proxy, None)
            resumed += 1
        return resumed

    def _reachable_to(self, marker: Marker, proxy: str) -> bool:
        target = marker.node or ""
        if marker.link is not None and marker.node is not None:
            target = marker.node
        if target == proxy:
            return True
        return proxy in self.case.reach_to.get(target, set())

    def _barrier(self, act: Action, step: int) -> dict[str, Any]:
        generation = int(act.generation or 0)
        checked: list[dict[str, Any]] = []
        blocking: list[dict[str, Any]] = []
        for marker in self.markers:
            if marker.generation >= generation:
                continue
            if not self._reachable_to(marker, act.proxy):
                continue
            snapshot = marker.as_dict(step)
            checked.append(snapshot)
            blocking.append(snapshot)
        checked.sort(key=lambda m: m["marker"])
        blocking.sort(key=lambda m: m["marker"])
        witness: dict[str, Any] = {
            "step": step,
            "event": act.id,
            "proxy": act.proxy,
            "generation": generation,
            "ok": not blocking,
            "old_markers_within_reach": len(checked),
            "blocking": blocking,
            "closed_generations": [[], sorted(self.closed[act.proxy])],
            "paused_proxies": sorted(self.paused),
        }
        if blocking:
            self.trace.violations.append(
                {
                    "step": step,
                    "code": "barrier_unproven",
                    "event": act.id,
                    "proxy": act.proxy,
                    "generation": generation,
                    "blocking": [m["marker"] for m in blocking],
                    "detail": "old-generation markers can still reach the proxy",
                }
            )
            self.trace.barriers.append(witness)
            return {"barrier_ok": False}

        # A successful barrier closes every generation below its own at this
        # proxy.  New markers of a closed generation can never be accepted
        # here again; retiring the rules themselves is a separate action.
        previously_closed = sorted(self.closed[act.proxy])
        self.closed[act.proxy].add(generation)
        self.proof[act.proxy] = max(self.proof.get(act.proxy, -1), generation)
        witness["previously_closed"] = previously_closed
        witness["closed_generations"] = sorted(self.closed[act.proxy])
        self.trace.barriers.append(witness)
        self.trace.events.append(
            {"step": step, "event": "barrier", "id": act.id, "proxy": act.proxy, "ok": True}
        )
        return {"barrier_ok": True}

    # -- reachability -----------------------------------------------------

    def finalize(self, until: int) -> None:
        for req in self.case.requests:
            threads = [t.branch for t in self.case.threads(req)]
            receipts = self.trace.receipts.get(req["id"], {})
            eligible = set(req.get("eligible") or [req["service"]])
            legal = [
                b
                for b in threads
                if b in receipts and not receipts[b]["late"] and receipts[b]["service"] in eligible
            ]
            off_list = [
                b
                for b in threads
                if b in receipts and receipts[b]["service"] not in eligible
            ]
            if off_list:
                self.trace.violations.append(
                    {
                        "step": min(
                            until,
                            min(int(receipts[b]["step"]) for b in off_list),
                        ),
                        "code": "delivery_outside_eligible",
                        "request": req["id"],
                        "deadline": req["deadline"],
                        "eligible": sorted(eligible),
                        "branches": sorted(off_list),
                        "detail": "a delivery reached a service outside the request's eligible list",
                    }
                )
            if not legal:
                self.trace.violations.append(
                    {
                        "step": min(until, int(req["deadline"])),
                        "code": "unreachable",
                        "request": req["id"],
                        "deadline": req["deadline"],
                        "eligible": sorted(eligible),
                        "branches": threads,
                        "detail": "no legal delivery satisfied the reachability list by the deadline",
                    }
                )
            elif req.get("require_all_branches") and len(legal) < len(threads):
                self.trace.violations.append(
                    {
                        "step": min(until, int(req["deadline"])),
                        "code": "unreachable_branch",
                        "request": req["id"],
                        "deadline": req["deadline"],
                        "missing": sorted(set(threads) - set(legal)),
                        "detail": "not every injection branch delivered in time",
                    }
                )

    def snapshot_in_flight(self, step: int) -> list[dict[str, Any]]:
        return sorted(
            (m.as_dict(step) for m in self.markers),
            key=lambda m: (m["marker"],),
        )

    # -- driver -----------------------------------------------------------

    def run(self, until: int) -> Trace:
        self._seed_times()
        self._push(until)
        while self._times:
            step = heapq.heappop(self._times)
            self._seen.discard(step)
            if step > until:
                continue
            self.at_step(step)
            self._after_step()
        self.trace.visited_steps.sort()
        self.trace.compressed_steps = max(0, (until + 1) - len(set(self.trace.visited_steps)))
        self.finalize(until)
        self.trace.in_flight = self.snapshot_in_flight(until)
        self.trace.last_step = until
        return self.trace


def make_latency(case: CompiledCase, override: dict[str, dict[str, int]] | None) -> Latency:
    stretch = int((case.fault or {}).get("recover_stretch", 0) or 0)
    return Latency(case.links, override or {}, stretch)


def make_fault(case: CompiledCase, active: bool) -> FaultWindow | None:
    if not active or case.fault is None:
        return None
    f = case.fault
    return FaultWindow(
        domain=f["domain"],
        fail_at=int(f["fail_at"]),
        recover_at=int(f["recover_at"]),
        stretch=int(f.get("recover_stretch", 0) or 0),
    )


def simulate(
    case: CompiledCase,
    event_steps: dict[str, int],
    latency: Latency,
    fault: FaultWindow | None,
    until: int,
    waits: dict[str, int] | None = None,
) -> Trace:
    return Simulator(case, latency, fault, event_steps, waits).run(until)


def horizon_for(case: CompiledCase, event_steps: dict[str, int]) -> int:
    last_event = max(
        [*event_steps.values(), *(s for r in case.requests for s in case.request_starts(r))],
        default=0,
    )
    until = max(case.horizon, last_event + case.ttl + 2)
    if case.fault is not None:
        until = max(until, int(case.fault["recover_at"]) + case.ttl + 2)
    return until
