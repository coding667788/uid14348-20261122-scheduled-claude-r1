"""Planner tests, including an exhaustive cross-check on a small topology."""

from __future__ import annotations

import itertools
import json
import os
import unittest

from helpers import BaseCase, clone, load, run_cli


class PlanShapeTests(BaseCase):
    def test_plan_is_safe_for_every_scenario(self):
        code, payload = run_cli("plan", load("leak-window.json"))
        self.assertEqual(code, 0)
        self.assertOk(payload)
        for scenario in payload["scenarios"]:
            self.assertTrue(scenario["isolated"], msg=scenario["scenario"])
            self.assertTrue(scenario["reachable"], msg=scenario["scenario"])
            self.assertEqual(scenario["failures"], [])

    def test_plan_respects_ordering_constraints(self):
        code, payload = run_cli("plan", load("entry-switch.json"))
        self.assertEqual(code, 0)
        self.assertOk(payload)
        slot = {entry["id"]: entry["slot"] for entry in payload["table"]}
        for row in payload["constraints"]:
            self.assertLess(
                slot[row["before"]],
                slot[row["after"]],
                msg=f"{row['before']} must precede {row['after']}",
            )

    def test_barrier_may_wait_inside_its_slot(self):
        code, payload = run_cli("plan", load("entry-switch.json"))
        self.assertOk(payload)
        barriers = [e for e in payload["table"] if e["kind"] == "barrier"]
        self.assertTrue(barriers)
        for barrier in barriers:
            self.assertEqual(barrier["fire_step"], barrier["slot"] + barrier["wait"])
            self.assertGreaterEqual(barrier["wait"], 0)

    def test_plan_is_deterministic(self):
        first = run_cli("plan", load("golden-path.json"))[1]
        second = run_cli("plan", load("golden-path.json"))[1]
        self.assertEqual(first, second)

    def test_objective_is_lexicographic(self):
        code, payload = run_cli("plan", load("golden-path.json"))
        objective = payload["objective"]
        self.assertIsInstance(objective["paused_proxy_steps"], int)
        self.assertIsInstance(objective["release_steps"], int)
        self.assertEqual(
            objective["lexicographic"],
            [
                objective["paused_proxy_steps"],
                objective["release_steps"],
                objective["action_id_sequence"],
            ],
        )

    def test_plan_without_actions_reports_static_safety(self):
        code, payload = run_cli("plan", load("loop-ttl.json"))
        self.assertEqual(code, 0)
        # The only actions are a no-op pause/resume pair, so the objective is
        # dominated by the pause cost.
        self.assertOk(payload)


class UnsatisfiableTests(BaseCase):
    def test_joint_infeasibility_uses_failure_prefix(self):
        code, payload = run_cli("plan", load("joint-infeasible.json"))
        self.assertEqual(code, 0)
        self.assertFalse(payload["ok"])
        error = payload["error"]
        self.assertEqual(error["code"], "E_UNSATISFIABLE")
        self.assertEqual(error["reason"], "baseline_order is schedulable but not safe")
        prefix = error["failure_prefix"]
        self.assertIn("prefix", prefix)
        self.assertIn("violation", prefix)
        self.assertLessEqual(len(prefix["prefix"]), 2)

    def test_scheduling_conflict_is_reported_when_baseline_breaks_order(self):
        # joint-infeasible has no table at all, so the diagnostic path runs; a
        # reversed baseline then has to be reported as a scheduling conflict.
        case = load("joint-infeasible.json")
        case["actions"].append(
            {
                "id": "zz-after",
                "kind": "pause",
                "proxy": "px-a",
                "after": ["sw-b-v2"],
            }
        )
        case["baseline_order"] = ["zz-after", "sw-b-v2"]
        code, payload = run_cli("plan", case)
        self.assertFalse(payload["ok"])
        error = payload["error"]
        self.assertEqual(error["reason"], "baseline_order violates the scheduling constraints")
        self.assertIn("baseline_conflict", error)
        conflict = error["baseline_conflict"]
        self.assertIn("code", conflict)
        self.assertIn("prefix", conflict)

    def test_explicit_after_conflict_names_both_actions(self):
        case = load("joint-infeasible.json")
        case["actions"].append(
            {
                "id": "zz-after",
                "kind": "pause",
                "proxy": "px-a",
                "after": ["sw-b-v2"],
            }
        )
        case["baseline_order"] = ["zz-after", "sw-b-v2"]
        code, payload = run_cli("plan", case)
        self.assertFalse(payload["ok"])
        conflict = payload["error"].get("baseline_conflict")
        self.assertIsNotNone(conflict)
        self.assertEqual(conflict["code"], "explicit_after")
        self.assertEqual(conflict["before"], "sw-b-v2")
        self.assertEqual(conflict["after"], "zz-after")

    def test_dependency_cycle_is_reported(self):
        case = load("golden-path.json")
        first, second = case["actions"][0]["id"], case["actions"][1]["id"]
        case["actions"][0]["after"] = [second]
        case["actions"][1]["after"] = [first]
        code, payload = run_cli("plan", case)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "E_SCHEDULE")
        self.assertIn("cycle", payload["error"])

    def test_unreachable_request_makes_plan_unsatisfiable(self):
        case = load("golden-path.json")
        case["requests"][0]["deadline"] = 0
        code, payload = run_cli("plan", case)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "E_UNSATISFIABLE")


class ScenarioMatrixTests(BaseCase):
    def test_fault_matrix_is_included(self):
        code, payload = run_cli("plan", load("entry-switch.json"))
        self.assertOk(payload)
        names = [entry["scenario"] for entry in payload["scenarios"]]
        self.assertIn("baseline", names)
        self.assertIn("slow-old", names)
        self.assertIn("baseline+fault", names)
        self.assertIn("slow-old+fault", names)

    def test_every_scenario_has_a_receipt(self):
        code, payload = run_cli("plan", load("entry-switch.json"))
        self.assertOk(payload)
        for scenario in payload["scenarios"]:
            self.assertTrue(scenario["receipts"], msg=scenario["scenario"])


class BruteForceCrossCheck(unittest.TestCase):
    """Cross-check the planner against an exhaustive enumeration.

    The topology is tiny: one proxy, one barrier wait choice, one switch and
    two shipping requests.  All slot assignments are enumerated by hand and
    each is simulated directly, so the planner's answer can be compared with
    the true optimum rather than with another instance of the same search.
    """

    def test_planner_matches_exhaustive_optimum(self):
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from solution.model import compile_case, make_latency, make_fault, simulate, horizon_for
        from solution.plan import (
            RELEASE_KINDS,
            build_graph,
            cost_horizon,
            paused_steps,
            scenarios,
        )
        from solution.schema import validate_case

        case = load("golden-path.json")
        compiled = compile_case(validate_case(case))
        graph = build_graph(compiled)
        beds = scenarios(compiled)
        window = cost_horizon(compiled, graph)

        ids = graph.ids
        slots = compiled.policy_slots
        barrier_id = "a-barrier-b-gen2"
        waits = graph.waits.get(barrier_id, (0,))

        best: tuple | None = None
        checked = 0
        for assignment in itertools.product(range(slots), repeat=len(ids)):
            mapping = dict(zip(ids, assignment))
            # Ordering: explicit and implicit dependencies.
            ok = True
            for aid in ids:
                for dep in graph.edges.get(aid, ()):
                    if mapping[dep] >= mapping[aid]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            for wait in waits:
                wait_map = {barrier_id: wait}
                fire = {
                    aid: mapping[aid] + (wait_map.get(aid, 0) if compiled.actions[aid].kind == "barrier" else 0)
                    for aid in ids
                }
                if any(fire[dep] >= fire[aid] for aid in ids for dep in graph.edges.get(aid, ())):
                    continue
                checked += 1
                safe = True
                for bed in beds:
                    trace = simulate(
                        compiled,
                        mapping,
                        make_latency(compiled, bed.latency),
                        make_fault(compiled, bed.fault),
                        window,
                        waits=wait_map,
                    )
                    if trace.violations:
                        safe = False
                        break
                if not safe:
                    continue
                paused = paused_steps(compiled, mapping, wait_map, window)
                releases = len(
                    {mapping[a] for a in mapping if compiled.actions[a].kind in RELEASE_KINDS}
                )
                sequence = tuple(aid for _, aid in sorted((mapping[a], a) for a in ids))
                key = (paused, releases, sequence)
                if best is None or key < best[0]:
                    best = (key, mapping, wait_map)

        self.assertIsNotNone(best, "the exhaustive search must find at least one safe table")
        self.assertGreater(checked, 10)

        code, payload = run_cli("plan", case)
        self.assertTrue(payload["ok"])
        planner_key = (
            payload["objective"]["paused_proxy_steps"],
            payload["objective"]["release_steps"],
            tuple(payload["objective"]["action_id_sequence"]),
        )
        self.assertEqual(planner_key, best[0])


class LimitTests(unittest.TestCase):
    def test_fourteen_proxies_and_sixteen_actions_are_accepted(self):
        case = {
            "case_version": 1,
            "name": "max-shape",
            "ttl": 6,
            "horizon": 20,
            "tenants": ["t"],
            "services": [{"id": "svc", "accepts": [{"tenant": "t", "label": "L1"}]}],
            "nodes": [
                {"id": f"p{i:02d}", "kind": "proxy", "region": "r"} for i in range(14)
            ]
            + [{"id": "n", "kind": "service", "region": "r", "service": "svc"}],
            "links": [
                {"id": f"l{i:02d}", "from": f"p{i:02d}", "to": f"p{i + 1:02d}", "delivery": 1, "forward": 0, "domain": "d"}
                for i in range(13)
            ]
            + [{"id": "l13", "from": "p13", "to": "n", "delivery": 1, "forward": 0, "domain": "d"}],
            "identities": [{"id": "i", "tenant": "t", "service": "svc"}],
            "policy": {
                "inbound": [
                    {
                        "id": f"in{i:02d}",
                        "version": "v1",
                        "proxy": f"p{i:02d}",
                        "generation": 1,
                        "tenant": "t",
                        "service": "svc",
                        "identity": "i",
                        "label": "L0" if i == 0 else "L1",
                    }
                    for i in range(14)
                ],
                "egress": [
                    {
                        "id": f"eg{i:02d}",
                        "version": "v1",
                        "proxy": f"p{i:02d}",
                        "generation": 1,
                        "tenant": "t",
                        "service": "svc",
                        "identity": "i",
                        "label_in": "L0" if i == 0 else "L1",
                        "label_out": "L1",
                        "link": f"l{i:02d}",
                    }
                    for i in range(14)
                ],
                "initial": {
                    f"p{i:02d}": {"inbound": ["v1"], "egress": ["v1"]} for i in range(14)
                },
            },
            "requests": [
                {
                    "id": "r",
                    "identity": "i",
                    "tenant": "t",
                    "service": "svc",
                    "entry": "p00",
                    "label": "L0",
                    "generation": 1,
                    "start": 0,
                    "deadline": 18,
                    "eligible": ["svc"],
                }
            ],
            "actions": [
                {"id": f"a{i:02d}", "kind": "pause", "proxy": f"p{i:02d}"} for i in range(8)
            ]
            + [
                {"id": f"b{i:02d}", "kind": "resume", "proxy": f"p{i:02d}"} for i in range(8)
            ],
            "timeline": [],
            "plan": {"slots": 10},
            "scenarios": {},
            "replay": {"latency": "baseline"},
            "baseline_order": [f"a{i:02d}" for i in range(8)] + [f"b{i:02d}" for i in range(8)],
        }
        code, payload = run_cli("check", case)
        self.assertEqual(code, 0, msg=json.dumps(payload, ensure_ascii=False)[:400])
        self.assertEqual(payload["counts"]["nodes"], 15)
        self.assertEqual(payload["counts"]["actions"], 16)


if __name__ == "__main__":
    unittest.main()
