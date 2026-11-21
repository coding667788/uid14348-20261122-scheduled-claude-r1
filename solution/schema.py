"""Structural validation for case files.

Every validation failure carries a machine readable ``code`` and a JSON-path
``field`` so that callers can point at the exact structural position.

Codes
-----
E_VERSION     unsupported ``case_version``
E_MISSING     required member absent
E_TYPE        wrong JSON type
E_ENUM        value outside the permitted set
E_RANGE       numeric value outside its permitted range
E_DUPLICATE   duplicated identifier
E_REFERENCE   identifier does not resolve
E_GRAPH       link/nodes disagree about the topology
E_ACTION      action definition is inconsistent
E_SCENARIO    scenario definition is inconsistent
E_TIMELINE    replay timeline is inconsistent
"""

from __future__ import annotations

from typing import Any

_MAX_ERRORS = 25

_ACTION_KINDS = {
    "install_inbound",
    "switch_egress",
    "retire_inbound",
    "retire_egress",
    "barrier",
    "pause",
    "resume",
}

_NODE_KINDS = {"proxy", "service"}


class CaseError(Exception):
    """Raised when a case file is structurally invalid."""

    def __init__(self, errors: list[dict[str, str]]):
        self.errors = errors
        first = errors[0]
        super().__init__(f"{first['code']} at {first['field']}: {first['message']}")

    def to_dict(self) -> dict[str, Any]:
        return {"ok": False, "error": self.errors[0], "errors": self.errors}


class _V:
    """Small accumulator that records the first N structural problems."""

    def __init__(self) -> None:
        self.errors: list[dict[str, str]] = []

    @property
    def full(self) -> bool:
        return len(self.errors) >= _MAX_ERRORS

    def add(self, code: str, field: str, message: str) -> None:
        if not self.full:
            self.errors.append({"code": code, "field": field, "message": message})

    # -- primitive helpers -------------------------------------------------

    def obj(self, value: Any, field: str, *, quiet: bool = False) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            if not quiet:
                self.add("E_TYPE", field, "expected a JSON object")
            return None
        return value

    def arr(self, value: Any, field: str, *, quiet: bool = False) -> list[Any] | None:
        if not isinstance(value, list):
            if not quiet:
                self.add("E_TYPE", field, "expected a JSON array")
            return None
        return value

    def req(self, obj: dict[str, Any], key: str, field: str) -> Any:
        if key not in obj:
            self.add("E_MISSING", f"{field}.{key}", "required member is absent")
            return None
        return obj[key]

    def str_(self, obj: dict[str, Any], key: str, field: str, *, required: bool = True) -> str | None:
        if key not in obj:
            if required:
                self.add("E_MISSING", f"{field}.{key}", "required string member is absent")
            return None
        value = obj[key]
        if not isinstance(value, str) or value == "":
            self.add("E_TYPE", f"{field}.{key}", "expected a non-empty string")
            return None
        return value

    def int_(
        self,
        obj: dict[str, Any],
        key: str,
        field: str,
        *,
        required: bool = True,
        lo: int | None = None,
        hi: int | None = None,
    ) -> int | None:
        if key not in obj:
            if required:
                self.add("E_MISSING", f"{field}.{key}", "required integer member is absent")
            return None
        value = obj[key]
        if isinstance(value, bool) or not isinstance(value, int):
            self.add("E_TYPE", f"{field}.{key}", "expected an integer")
            return None
        if lo is not None and value < lo:
            self.add("E_RANGE", f"{field}.{key}", f"must be >= {lo}")
            return None
        if hi is not None and value > hi:
            self.add("E_RANGE", f"{field}.{key}", f"must be <= {hi}")
            return None
        return value

    def bool_(
        self, obj: dict[str, Any], key: str, field: str, *, default: bool = False
    ) -> bool:
        if key not in obj:
            return default
        value = obj[key]
        if not isinstance(value, bool):
            self.add("E_TYPE", f"{field}.{key}", "expected a boolean")
            return default
        return value

    def enum(self, obj: dict[str, Any], key: str, field: str, allowed: set[str]) -> str | None:
        value = self.str_(obj, key, field)
        if value is None:
            return None
        if value not in allowed:
            self.add(
                "E_ENUM",
                f"{field}.{key}",
                "expected one of: " + ", ".join(sorted(allowed)),
            )
            return None
        return value

    def str_list(
        self, obj: dict[str, Any], key: str, field: str, *, required: bool = True
    ) -> list[str] | None:
        if key not in obj:
            if required:
                self.add("E_MISSING", f"{field}.{key}", "required array member is absent")
            return None
        raw = self.arr(obj[key], f"{field}.{key}")
        if raw is None:
            return None
        out: list[str] = []
        for idx, item in enumerate(raw):
            if not isinstance(item, str) or item == "":
                self.add("E_TYPE", f"{field}.{key}[{idx}]", "expected a non-empty string")
                continue
            out.append(item)
        return out

    def int_list(
        self,
        obj: dict[str, Any],
        key: str,
        field: str,
        *,
        required: bool = True,
        lo: int = 0,
        hi: int = 10_000,
    ) -> list[int] | None:
        if key not in obj:
            if required:
                self.add("E_MISSING", f"{field}.{key}", "required array member is absent")
            return None
        raw = self.arr(obj[key], f"{field}.{key}")
        if raw is None:
            return None
        out: list[int] = []
        for idx, item in enumerate(raw):
            if isinstance(item, bool) or not isinstance(item, int):
                self.add("E_TYPE", f"{field}.{key}[{idx}]", "expected an integer")
                continue
            if item < lo or item > hi:
                self.add("E_RANGE", f"{field}.{key}[{idx}]", f"must be within [{lo}, {hi}]")
                continue
            out.append(item)
        return out

    def unique(self, seen: set[str], ident: str, field: str) -> bool:
        if ident in seen:
            self.add("E_DUPLICATE", field, f"duplicate identifier {ident!r}")
            return False
        seen.add(ident)
        return True


def validate_case(raw: Any) -> dict[str, Any]:
    """Validate ``raw`` and return it unchanged when structurally sound."""

    v = _V()
    case = v.obj(raw, "$")
    if case is None:
        raise CaseError(v.errors)

    version = v.req(case, "case_version", "$")
    if version is not None and (isinstance(version, bool) or version != 1):
        v.add("E_VERSION", "$.case_version", "only case_version 1 is supported")

    name = case.get("name", "case")
    if not isinstance(name, str) or name == "":
        v.add("E_TYPE", "$.name", "expected a non-empty string")
        name = "case"

    ttl = v.int_(case, "ttl", "$", lo=1, hi=6)
    horizon = v.int_(case, "horizon", "$", lo=1, hi=256)

    tenants = v.str_list(case, "tenants", "$")
    tenant_set: set[str] = set()
    for i, t in enumerate(tenants or []):
        v.unique(tenant_set, t, f"$.tenants[{i}]")

    # -- services ---------------------------------------------------------
    services_present = "services" in case
    services_raw = v.arr(v.req(case, "services", "$"), "$.services", quiet=True) or []
    service_ids: set[str] = set()
    service_accepts: dict[str, set[tuple[str, str]]] = {}
    for i, item in enumerate(services_raw):
        fld = f"$.services[{i}]"
        svc = v.obj(item, fld)
        if svc is None:
            continue
        sid = v.str_(svc, "id", fld)
        if sid is not None:
            v.unique(service_ids, sid, f"{fld}.id")
        accepts = v.arr(v.req(svc, "accepts", fld), f"{fld}.accepts") or []
        pairs: set[tuple[str, str]] = set()
        for j, acc in enumerate(accepts):
            afld = f"{fld}.accepts[{j}]"
            a = v.obj(acc, afld)
            if a is None:
                continue
            tenant = v.str_(a, "tenant", afld)
            label = v.str_(a, "label", afld)
            if tenant is not None and tenant not in tenant_set:
                v.add("E_REFERENCE", f"{afld}.tenant", f"unknown tenant {tenant!r}")
            if tenant is not None and label is not None:
                pairs.add((tenant, label))
        if sid is not None:
            service_accepts[sid] = pairs
    if services_present and not services_raw:
        v.add("E_RANGE", "$.services", "at least one service is required")

    # -- nodes ------------------------------------------------------------
    nodes_raw = v.arr(v.req(case, "nodes", "$"), "$.nodes", quiet=True) or []
    node_kind: dict[str, str] = {}
    proxy_ids: list[str] = []
    service_nodes: list[str] = []
    for i, item in enumerate(nodes_raw):
        fld = f"$.nodes[{i}]"
        node = v.obj(item, fld)
        if node is None:
            continue
        nid = v.str_(node, "id", fld)
        kind = v.enum(node, "kind", fld, _NODE_KINDS)
        v.str_(node, "region", fld)
        if nid is not None:
            v.unique(set(node_kind), nid, f"{fld}.id")
        if kind == "service":
            svc = v.str_(node, "service", fld)
            if svc is not None and svc not in service_ids:
                v.add("E_REFERENCE", f"{fld}.service", f"unknown service {svc!r}")
            if nid is not None:
                service_nodes.append(nid)
        elif kind == "proxy" and nid is not None:
            proxy_ids.append(nid)
        if nid is not None and kind is not None:
            node_kind[nid] = kind
    if "nodes" in case and not proxy_ids:
        v.add("E_RANGE", "$.nodes", "at least one proxy node is required")

    # -- links ------------------------------------------------------------
    links_raw = v.arr(v.req(case, "links", "$"), "$.links", quiet=True) or []
    link_ids: set[str] = set()
    link_src: dict[str, str] = {}
    domains: set[str] = set()
    for i, item in enumerate(links_raw):
        fld = f"$.links[{i}]"
        link = v.obj(item, fld)
        if link is None:
            continue
        lid = v.str_(link, "id", fld)
        src = v.str_(link, "from", fld)
        dst = v.str_(link, "to", fld)
        v.int_(link, "delivery", fld, lo=0, hi=64)
        v.int_(link, "forward", fld, lo=0, hi=64)
        dom = v.str_(link, "domain", fld)
        if dom is not None:
            domains.add(dom)
        if lid is not None:
            v.unique(link_ids, lid, f"{fld}.id")
        if src is not None and src not in proxy_ids:
            v.add("E_GRAPH", f"{fld}.from", f"{src!r} is not a proxy node; only proxies originate links")
        if dst is not None and dst not in node_kind:
            v.add("E_REFERENCE", f"{fld}.to", f"unknown node {dst!r}")
        if lid is not None and src is not None:
            link_src[lid] = src

    # -- identities -------------------------------------------------------
    idents_raw = v.arr(v.req(case, "identities", "$"), "$.identities", quiet=True) or []
    identity_ids: set[str] = set()
    identity_tenant: dict[str, str] = {}
    for i, item in enumerate(idents_raw):
        fld = f"$.identities[{i}]"
        ident = v.obj(item, fld)
        if ident is None:
            continue
        iid = v.str_(ident, "id", fld)
        tenant = v.str_(ident, "tenant", fld)
        svc = v.str_(ident, "service", fld)
        if iid is not None:
            v.unique(identity_ids, iid, f"{fld}.id")
        if tenant is not None and tenant not in tenant_set:
            v.add("E_REFERENCE", f"{fld}.tenant", f"unknown tenant {tenant!r}")
        if svc is not None and svc not in service_ids:
            v.add("E_REFERENCE", f"{fld}.service", f"unknown service {svc!r}")
        if iid is not None and tenant is not None:
            identity_tenant[iid] = tenant

    # -- policy -----------------------------------------------------------
    policy = v.obj(v.req(case, "policy", "$"), "$.policy", quiet=True) or {}
    inbound_raw = v.arr(v.req(policy, "inbound", "$.policy"), "$.policy.inbound", quiet=True) or []
    egress_raw = v.arr(v.req(policy, "egress", "$.policy"), "$.policy.egress", quiet=True) or []

    def _policy_common(rule: dict[str, Any], fld: str) -> None:
        v.str_(rule, "version", fld)
        v.str_(rule, "proxy", fld)
        v.str_(rule, "tenant", fld)
        v.str_(rule, "service", fld)
        v.str_(rule, "identity", fld)
        v.int_(rule, "generation", fld, lo=0, hi=64)
        proxy = rule.get("proxy")
        if isinstance(proxy, str) and proxy not in node_kind:
            v.add("E_REFERENCE", f"{fld}.proxy", f"unknown node {proxy!r}")
        elif isinstance(proxy, str) and node_kind.get(proxy) != "proxy":
            v.add("E_GRAPH", f"{fld}.proxy", "policy must attach to a proxy node")
        for key in ("tenant", "service", "identity"):
            val = rule.get(key)
            if not isinstance(val, str):
                continue
            target = tenant_set if key == "tenant" else service_ids if key == "service" else identity_ids
            if val not in target:
                v.add("E_REFERENCE", f"{fld}.{key}", f"unknown {key} {val!r}")

    inbound_ids: set[str] = set()
    for i, item in enumerate(inbound_raw):
        fld = f"$.policy.inbound[{i}]"
        rule = v.obj(item, fld)
        if rule is None:
            continue
        rid = v.str_(rule, "id", fld)
        if rid is not None:
            v.unique(inbound_ids, rid, f"{fld}.id")
        _policy_common(rule, fld)
        v.str_(rule, "label", fld)

    egress_ids: set[str] = set()
    for i, item in enumerate(egress_raw):
        fld = f"$.policy.egress[{i}]"
        rule = v.obj(item, fld)
        if rule is None:
            continue
        rid = v.str_(rule, "id", fld)
        if rid is not None:
            v.unique(egress_ids, rid, f"{fld}.id")
        _policy_common(rule, fld)
        v.str_(rule, "label_in", fld)
        v.str_(rule, "label_out", fld)
        link = v.str_(rule, "link", fld)
        if link is not None and link not in link_ids:
            v.add("E_REFERENCE", f"{fld}.link", f"unknown link {link!r}")
        elif link is not None:
            proxy = rule.get("proxy")
            if isinstance(proxy, str) and link in link_src and link_src[link] != proxy:
                v.add("E_GRAPH", f"{fld}.link", "egress link does not start at the policy proxy")
        if "branch" in rule:
            v.str_(rule, "branch", fld)
        if "switch_group" in rule:
            v.str_(rule, "switch_group", fld)

    # -- requests ---------------------------------------------------------
    requests_raw = v.arr(v.req(case, "requests", "$"), "$.requests", quiet=True) or []
    request_ids: set[str] = set()
    for i, item in enumerate(requests_raw):
        fld = f"$.requests[{i}]"
        req = v.obj(item, fld)
        if req is None:
            continue
        rid = v.str_(req, "id", fld)
        if rid is not None:
            v.unique(request_ids, rid, f"{fld}.id")
        ident = v.str_(req, "identity", fld)
        svc = v.str_(req, "service", fld)
        tenant = v.str_(req, "tenant", fld)
        entry = v.str_(req, "entry", fld)
        v.int_(req, "deadline", fld, lo=0, hi=256)
        v.int_(req, "start", fld, lo=0, hi=256, required=False)
        v.int_list(req, "retries", fld, required=False, lo=0, hi=256)
        v.int_(req, "generation", fld, lo=0, hi=64, required=False)
        v.str_(req, "label", fld)
        v.str_(req, "branch", fld, required=False)
        v.bool_(req, "fanout", fld)
        v.bool_(req, "require_all_branches", fld)
        if ident is not None and ident not in identity_ids:
            v.add("E_REFERENCE", f"{fld}.identity", f"unknown identity {ident!r}")
        if svc is not None and svc not in service_ids:
            v.add("E_REFERENCE", f"{fld}.service", f"unknown service {svc!r}")
        if tenant is not None and tenant not in tenant_set:
            v.add("E_REFERENCE", f"{fld}.tenant", f"unknown tenant {tenant!r}")
        if ident is not None and tenant is not None and ident in identity_tenant:
            if identity_tenant[ident] != tenant:
                v.add("E_ACTION", f"{fld}.tenant", "request tenant disagrees with the calling identity")
        if entry is not None:
            if entry not in node_kind:
                v.add("E_REFERENCE", f"{fld}.entry", f"unknown node {entry!r}")
            elif node_kind[entry] != "proxy":
                v.add("E_GRAPH", f"{fld}.entry", "requests must enter at a proxy node")
        eligible = v.str_list(req, "eligible", fld)
        for j, e in enumerate(eligible or []):
            if e not in service_ids:
                v.add("E_REFERENCE", f"{fld}.eligible[{j}]", f"unknown service {e!r}")

    # -- actions ----------------------------------------------------------
    actions_raw = v.arr(case.get("actions", []), "$.actions") or []
    action_ids: set[str] = set()
    actions_by_id: dict[str, dict[str, Any]] = {}
    for i, item in enumerate(actions_raw):
        fld = f"$.actions[{i}]"
        act = v.obj(item, fld)
        if act is None:
            continue
        aid = v.str_(act, "id", fld)
        if aid is not None:
            v.unique(action_ids, aid, f"{fld}.id")
            actions_by_id[aid] = act
        kind = v.enum(act, "kind", fld, _ACTION_KINDS)
        proxy = v.str_(act, "proxy", fld)
        if proxy is not None:
            if proxy not in node_kind:
                v.add("E_REFERENCE", f"{fld}.proxy", f"unknown node {proxy!r}")
            elif node_kind[proxy] != "proxy":
                v.add("E_GRAPH", f"{fld}.proxy", "actions must target a proxy node")
        after = v.str_list(act, "after", fld, required=False) or []
        for j, dep in enumerate(after):
            if dep == aid:
                v.add("E_ACTION", f"{fld}.after[{j}]", "action may not depend on itself")
        slots = v.int_list(act, "slots", fld, required=False, lo=0, hi=63)
        if "slots" in act and slots is not None and not slots:
            v.add("E_RANGE", f"{fld}.slots", "at least one permitted slot is required")
        if "step" in act:
            v.int_(act, "step", fld, lo=0, hi=4096)
        if kind == "barrier":
            v.int_(act, "generation", fld, lo=0, hi=64)
            v.int_list(act, "wait_slots", fld, required=False, lo=0, hi=63)
            if "wait" in act:
                v.int_(act, "wait", fld, lo=0, hi=63)
        elif kind in {"retire_inbound", "retire_egress"}:
            v.str_(act, "version", fld)
            v.int_(act, "generation", fld, lo=0, hi=64, required=False)
        elif kind in {"install_inbound", "switch_egress"}:
            v.str_(act, "version", fld)

    for aid, act in actions_by_id.items():
        fld = f"$.actions[{actions_raw.index(act)}]"
        for j, dep in enumerate(act.get("after", []) or []):
            if dep not in action_ids:
                v.add("E_REFERENCE", f"{fld}.after[{j}]", f"unknown action {dep!r}")

    # -- timeline (replay) ------------------------------------------------
    if "timeline" in case:
        timeline = v.arr(case["timeline"], "$.timeline") or []
        seen_events: set[str] = set()
        for i, item in enumerate(timeline):
            fld = f"$.timeline[{i}]"
            event = v.obj(item, fld)
            if event is None:
                continue
            eid = v.str_(event, "id", fld)
            if eid is not None:
                v.unique(seen_events, eid, f"{fld}.id")
                if eid not in action_ids:
                    v.add("E_REFERENCE", f"{fld}.id", f"unknown action {eid!r}")
            v.int_(event, "step", fld, lo=0, hi=4096)

    # -- scenarios --------------------------------------------------------
    scenarios = v.obj(case.get("scenarios", {}), "$.scenarios") or {}
    latency = v.arr(scenarios.get("latency", []), "$.scenarios.latency") or []
    latency_ids: set[str] = set()
    for i, item in enumerate(latency):
        fld = f"$.scenarios.latency[{i}]"
        scen = v.obj(item, fld)
        if scen is None:
            continue
        sid = v.str_(scen, "id", fld)
        if sid is not None:
            v.unique(latency_ids, sid, f"{fld}.id")
        overrides = v.obj(scen.get("links", {}), f"{fld}.links") or {}
        for lid, patch in overrides.items():
            lfld = f"{fld}.links.{lid}"
            if lid not in link_ids:
                v.add("E_REFERENCE", lfld, f"unknown link {lid!r}")
                continue
            if not isinstance(patch, dict):
                v.add("E_TYPE", lfld, "expected an object")
                continue
            for key in ("delivery", "forward"):
                if key in patch:
                    v.int_(patch, key, lfld, lo=0, hi=64)

    if "fault" in scenarios:
        fault = v.obj(scenarios["fault"], "$.scenarios.fault")
        if fault is not None:
            dom = v.str_(fault, "domain", "$.scenarios.fault")
            if dom is not None and dom not in domains:
                v.add("E_REFERENCE", "$.scenarios.fault.domain", f"unknown fault domain {dom!r}")
            fail_at = v.int_(fault, "fail_at", "$.scenarios.fault", lo=0, hi=256)
            recover_at = v.int_(fault, "recover_at", "$.scenarios.fault", lo=0, hi=256)
            v.int_(fault, "recover_stretch", "$.scenarios.fault", lo=0, hi=64, required=False)
            if fail_at is not None and recover_at is not None and recover_at < fail_at:
                v.add("E_SCENARIO", "$.scenarios.fault.recover_at", "recover_at precedes fail_at")

    # -- baseline order ---------------------------------------------------
    baseline = case.get("baseline_order")
    if baseline is not None:
        items = v.arr(baseline, "$.baseline_order") or []
        seen: set[str] = set()
        for i, aid in enumerate(items):
            if not isinstance(aid, str):
                v.add("E_TYPE", f"$.baseline_order[{i}]", "expected an action id string")
                continue
            if aid not in action_ids:
                v.add("E_REFERENCE", f"$.baseline_order[{i}]", f"unknown action {aid!r}")
            elif aid in seen:
                v.add("E_DUPLICATE", f"$.baseline_order[{i}]", f"duplicate action {aid!r}")
            seen.add(aid)

    # -- declared limits --------------------------------------------------
    limit_map = {
        "proxies": (14, len(proxy_ids), "$.nodes"),
        "actions": (16, len(actions_raw), "$.actions"),
        "slots": (10, int((case.get("plan") or {}).get("slots", 10)), "$.plan.slots"),
        "ttl": (6, int(case.get("ttl", 0)) if isinstance(case.get("ttl"), int) else 0, "$.ttl"),
    }
    for name, (cap, actual, field) in limit_map.items():
        if actual > cap:
            v.add("E_LIMIT", field, f"{name} must be at most {cap}, got {actual}")
    if v.errors:
        raise CaseError(v.errors)
    return case
