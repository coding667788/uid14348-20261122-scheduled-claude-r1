"""Replay semantics: identity, TTL, generations, barriers, pauses, faults."""

from __future__ import annotations

import json
import unittest

from helpers import BaseCase, clone, load, run_cli


def minimal(path_links=None, **overrides):
    """A one-tenant, one-proxy chain used by the small unit-style tests.

    entry px-a --(rewrites L0->L1)--> svc-orders, which accepts (t-alpha, L1).
    """

    case = {
        "case_version": 1,
        "name": "minimal",
        "ttl": 4,
        "horizon": 16,
        "tenants": ["t-alpha", "t-beta"],
        "services": [{"id": "svc", "accepts": [{"tenant": "t-alpha", "label": "L1"}]}],
        "nodes": [
            {"id": "px-a", "kind": "proxy", "region": "r1"},
            {"id": "n-svc", "kind": "service", "region": "r1", "service": "svc"},
        ],
        "links": path_links
        or [{"id": "a-s", "from": "px-a", "to": "n-svc", "delivery": 1, "forward": 0, "domain": "d"}],
        "identities": [{"id": "id-a", "tenant": "t-alpha", "service": "svc"}],
        "policy": {
            "inbound": [
                {
                    "id": "in-a",
                    "version": "v1",
                    "proxy": "px-a",
                    "generation": 1,
                    "tenant": "t-alpha",
                    "service": "svc",
                    "identity": "id-a",
                    "label": "L0",
                }
            ],
            "egress": [
                {
                    "id": "eg-a",
                    "version": "v1",
                    "proxy": "px-a",
                    "generation": 1,
                    "tenant": "t-alpha",
                    "service": "svc",
                    "identity": "id-a",
                    "label_in": "L0",
                    "label_out": "L1",
                    "link": "a-s",
                }
            ],
            "initial": {"px-a": {"inbound": ["v1"], "egress": ["v1"]}},
        },
        "requests": [
            {
                "id": "req",
                "identity": "id-a",
                "tenant": "t-alpha",
                "service": "svc",
                "entry": "px-a",
                "label": "L0",
                "generation": 1,
                "start": 0,
                "deadline": 10,
                "eligible": ["svc"],
            }
        ],
        "actions": [],
        "timeline": [],
        "plan": {"slots": 8},
        "scenarios": {},
        "replay": {"latency": "baseline"},
        "baseline_order": [],
    }
    case.update(overrides)
    return case


class HappyPathTests(BaseCase):
    def test_delivery_rewrites_label_but_keeps_tenant(self):
        payload = run_cli("replay", minimal())[1]
        self.assertOk(payload)
        self.assertTrue(payload["isolated"])
        self.assertTrue(payload["reachable"])
        receipt = payload["receipts"]["req"]["main"]
        self.assertEqual(receipt["tenant"], "t-alpha")
        self.assertEqual(receipt["label"], "L1")
        self.assertEqual(receipt["step"], 1)
        self.assertEqual(payload["violations"], [])

    def test_replay_is_deterministic(self):
        first = run_cli("replay", minimal())[1]
        second = run_cli("replay", minimal())[1]
        self.assertEqual(first, second)

    def test_tenant_isolation_violation_is_classified(self):
        case = minimal()
        case["services"][0]["accepts"] = [{"tenant": "t-beta", "label": "L1"}]
        payload = run_cli("replay", case)[1]
        self.assertFalse(payload["isolated"])
        violation = payload["earliest_violation"]
        self.assertEqual(violation["code"], "tenant_isolation")
        self.assertEqual(violation["tenant"], "t-alpha")
        self.assertEqual(violation["label"], "L1")
        self.assertEqual(violation["step"], 1)

    def test_prefix_ends_at_the_violation_step(self):
        case = minimal()
        case["services"][0]["accepts"] = []
        payload = run_cli("replay", case)[1]
        prefix = payload["violation_prefix"]
        self.assertTrue(prefix)
        self.assertLessEqual(max(e["step"] for e in prefix), payload["earliest_violation"]["step"])

    def test_rejected_inbound_is_reported_with_reason(self):
        case = minimal()
        case["policy"]["inbound"][0]["label"] = "L9"
        payload = run_cli("replay", case)[1]
        self.assertIn("inbound_rejected", {d["code"] for d in payload["drops"]})
        self.assertTrue(payload["rejections"])

    def test_missing_egress_mapping_is_reported(self):
        case = minimal()
        case["policy"]["egress"] = []
        case["policy"]["initial"]["px-a"]["egress"] = []
        payload = run_cli("replay", case)[1]
        self.assertIn("no_egress_mapping", {d["code"] for d in payload["drops"]})


class TtlTests(BaseCase):
    def test_delivery_exactly_at_ttl_edge_is_accepted(self):
        case = minimal(
            path_links=[
                {"id": "a-s", "from": "px-a", "to": "n-svc", "delivery": 3, "forward": 0, "domain": "d"}
            ]
        )
        payload = run_cli("replay", case)[1]
        self.assertTrue(payload["reachable"])
        # ttl 4 => alive on steps 0..3; a 3-step link arrives at step 3.
        self.assertEqual(payload["receipts"]["req"]["main"]["step"], 3)

    def test_delivery_one_step_past_ttl_is_dropped(self):
        case = minimal(
            path_links=[
                {"id": "a-s", "from": "px-a", "to": "n-svc", "delivery": 4, "forward": 0, "domain": "d"}
            ]
        )
        payload = run_cli("replay", case)[1]
        self.assertFalse(payload["reachable"])
        self.assertEqual(payload["drops"][0]["code"], "ttl_expired")
        self.assertEqual(payload["drops"][0]["step"], 4)
        self.assertEqual(payload["drops"][0]["ttl_left"], 0)

    def test_loop_loses_ttl_and_is_never_delivered(self):
        case = minimal()
        case["nodes"].insert(1, {"id": "px-b", "kind": "proxy", "region": "r2"})
        case["links"] = [
            {"id": "a-b", "from": "px-a", "to": "px-b", "delivery": 1, "forward": 0, "domain": "d"},
            {"id": "b-a", "from": "px-b", "to": "px-a", "delivery": 1, "forward": 0, "domain": "d"},
        ]
        case["policy"]["inbound"].append(
            {
                "id": "in-a-ret",
                "version": "v1",
                "proxy": "px-a",
                "generation": 1,
                "tenant": "t-alpha",
                "service": "svc",
                "identity": "id-a",
                "label": "L9",
            }
        )
        case["policy"]["inbound"].append(
            {
                "id": "in-b",
                "version": "v1",
                "proxy": "px-b",
                "generation": 1,
                "tenant": "t-alpha",
                "service": "svc",
                "identity": "id-a",
                "label": "L1",
            }
        )
        case["policy"]["egress"][0]["link"] = "a-b"
        case["policy"]["egress"].append(
            {
                "id": "eg-b",
                "version": "v1",
                "proxy": "px-b",
                "generation": 1,
                "tenant": "t-alpha",
                "service": "svc",
                "identity": "id-a",
                "label_in": "L1",
                "label_out": "L9",
                "link": "b-a",
            }
        )
        case["policy"]["egress"].append(
            {
                "id": "eg-a-ret",
                "version": "v1",
                "proxy": "px-a",
                "generation": 1,
                "tenant": "t-alpha",
                "service": "svc",
                "identity": "id-a",
                "label_in": "L9",
                "label_out": "L1",
                "link": "a-b",
            }
        )
        case["policy"]["initial"]["px-b"] = {"inbound": ["v1"], "egress": ["v1"]}
        payload = run_cli("replay", case)[1]
        self.assertFalse(payload["reachable"])
        self.assertTrue(all(d["code"] == "ttl_expired" for d in payload["drops"]))
        self.assertEqual(payload["deliveries"], [])


class BarrierTests(BaseCase):
    def setUp(self):
        self.case = load("entry-switch.json")

    def test_early_barrier_is_unproven_and_retire_is_rejected(self):
        payload = run_cli("replay", self.case)[1]
        self.assertOk(payload)
        codes = {v["code"] for v in payload["violations"]}
        self.assertIn("barrier_unproven", codes)
        self.assertIn("retire_without_drain", codes)
        failed = [b for b in payload["barriers"] if not b["ok"]]
        self.assertTrue(failed)
        self.assertTrue(failed[0]["blocking"])

    def test_barrier_witness_lists_every_blocking_marker(self):
        payload = run_cli("replay", self.case)[1]
        witness = next(b for b in payload["barriers"] if not b["ok"])
        for marker in witness["blocking"]:
            self.assertEqual(marker["generation"], 1)
            self.assertLess(marker["generation"], witness["generation"])
            self.assertIn("ttl_left", marker)
            self.assertGreaterEqual(marker["ttl_left"], 0)

    def test_waiting_barrier_proves_the_drain(self):
        case = clone(self.case)
        # The barrier is scheduled at step 1 but waits until step 6, by which
        # time the slow generation-1 marker has landed.
        case["timeline"] = [
            {"id": "inst-b-v2", "step": 0},
            {"id": "bar-b-gen2", "step": 1},
            {"id": "ret-in-b-v1", "step": 8},
            {"id": "ret-eg-b-v1", "step": 8},
        ]
        for action in case["actions"]:
            if action["kind"] == "barrier":
                action["wait"] = 6
        payload = run_cli("replay", case)[1]
        ok = [b for b in payload["barriers"] if b["ok"]]
        self.assertTrue(ok)
        retire_records = [
            r for r in payload["actions_applied"] if r["kind"].startswith("retire")
        ]
        self.assertTrue(all(r["retired"] for r in retire_records))

    def test_closed_generation_rejects_new_injection(self):
        case = clone(self.case)
        case["actions"] = [a for a in case["actions"] if a["kind"] == "barrier"]
        for action in case["actions"]:
            action["wait_slots"] = [0]
        case["baseline_order"] = [a["id"] for a in case["actions"]]
        case["timeline"] = [{"id": "bar-b-gen2", "step": 5}]
        case["requests"] = [case["requests"][0]]
        case["requests"][0].update(
            {"start": 6, "entry": "px-b", "label": "L1", "deadline": 10}
        )
        case["scenarios"] = {}
        payload = run_cli("replay", case)
        self.assertEqual(payload[0], 0, msg=json.dumps(payload[1])[:400])
        payload = payload[1]
        self.assertIn("stale_injection", {r["code"] for r in payload["rejections"]})

    def test_generation_mismatch_is_explained(self):
        case = load("golden-path.json")
        case["requests"][0]["generation"] = 2
        payload = run_cli("replay", case)[1]
        # A generation-2 request needs generation-2 entry rules; without them
        # it is rejected, and the rejection says why.
        self.assertTrue(payload["rejections"])
        self.assertEqual(payload["rejections"][0]["code"], "inbound_rejected")


class PauseTests(BaseCase):
    def test_queued_marker_expires_while_paused(self):
        payload = run_cli("replay", load("pause-resume.json"))[1]
        codes = {d["code"] for d in payload["drops"]}
        self.assertIn("expired_while_paused", codes)
        # The queued marker is never forwarded after the pause.
        queued = [d for d in payload["drops"] if d["code"] == "expired_while_paused"]
        for drop in queued:
            self.assertGreaterEqual(drop["ttl_left"], 0)

    def test_marker_arriving_on_resume_is_forwarded(self):
        payload = run_cli("replay", load("pause-resume.json"))[1]
        self.assertIn("req-arrives-on-resume", payload["receipts"])

    def test_paused_proxy_rejects_new_injection(self):
        case = load("pause-resume.json")
        case["timeline"] = [{"id": "pause-b", "step": 0}, {"id": "resume-b", "step": 5}]
        case["requests"][0]["start"] = 1
        case["requests"][0]["entry"] = "px-b"
        case["requests"][0]["label"] = "L1"
        payload = run_cli("replay", case)[1]
        # A request entering at a paused proxy is refused explicitly.
        self.assertIn("entry_paused", {r["code"] for r in payload["rejections"]})


class FaultTests(BaseCase):
    def test_fault_drops_in_flight_marker_and_never_revives_it(self):
        payload = run_cli("replay", load("fault-ttl-edge.json"))[1]
        self.assertTrue(payload["scenario"]["fault"])
        dropped = [d for d in payload["drops"] if d["code"] == "fault_dropped"]
        self.assertTrue(dropped)
        for drop in dropped:
            self.assertEqual(payload["marker_fates"][drop["marker"]]["outcome"], "dropped")

    def test_recovery_stretch_applies_after_the_window(self):
        payload = run_cli("replay", load("fault-ttl-edge.json"))[1]
        self.assertEqual(payload["scenario"]["fault_window"]["recover_stretch"], 1)
        ttl_drops = [d for d in payload["drops"] if d["code"] == "ttl_expired"]
        self.assertTrue(ttl_drops)
        for drop in ttl_drops:
            self.assertEqual(drop["ttl_left"], 0)

    def test_expired_marker_is_never_reported_as_recovered(self):
        payload = run_cli("replay", load("fault-ttl-edge.json"))[1]
        for marker, fate in payload["marker_fates"].items():
            if fate.get("code") in {"fault_dropped", "ttl_expired"}:
                self.assertNotEqual(fate["outcome"], "delivered")


class ReceiptTests(BaseCase):
    def test_fanout_branches_have_independent_receipts(self):
        payload = run_cli("replay", load("fanout-receipts.json"))[1]
        receipts = payload["receipts"]["req-fan"]
        self.assertEqual(sorted(receipts), ["req-fan#a-alt", "req-fan#a-b"])
        for branch in receipts.values():
            self.assertEqual(branch["tenant"], "t-alpha")

    def test_duplicate_delivery_keeps_the_first_receipt(self):
        payload = run_cli("replay", load("loop-ttl.json"))[1]
        self.assertTrue(payload["duplicate_deliveries"])
        dup = payload["duplicate_deliveries"][0]
        kept = payload["receipts"][dup["request"]][dup["branch"]]
        self.assertEqual(kept["marker"], dup["kept_marker"])
        self.assertEqual(kept["step"], dup["kept_step"])
        self.assertIn("reason", dup)

    def test_late_delivery_is_recorded_with_reason(self):
        case = load("golden-path.json")
        case["requests"][0]["deadline"] = 0
        payload = run_cli("replay", case)[1]
        self.assertTrue(payload["late_deliveries"])
        self.assertIn("reason", payload["late_deliveries"][0])
        self.assertFalse(payload["reachable"])

    def test_off_list_delivery_is_a_violation(self):
        payload = run_cli("replay", load("leak-window.json"))[1]
        codes = {v["code"] for v in payload["violations"]}
        self.assertIn("tenant_isolation", codes)
        self.assertIn("unreachable", codes)


class GenerationFutureTests(BaseCase):
    def test_same_marker_state_has_different_futures_per_barrier(self):
        payload = run_cli("replay", load("barrier-futures.json"))[1]
        self.assertOk(payload)
        barriers = sorted(payload["barriers"], key=lambda b: b["step"])
        self.assertGreaterEqual(len(barriers), 2)
        self.assertFalse(barriers[0]["ok"])
        self.assertTrue(barriers[-1]["ok"])
        self.assertEqual(barriers[-1]["generation"], 3)


class CompressionTests(BaseCase):
    def test_timeline_is_compressed_when_nothing_is_in_flight(self):
        payload = run_cli("replay", load("golden-path.json"))[1]
        compression = payload["compression"]
        self.assertGreater(compression["steps_skipped"], 0)
        self.assertLess(compression["steps_visited"], compression["reachability_end"])
        self.assertIn(0, compression["visited"])

    def test_visited_steps_are_sorted_and_unique(self):
        payload = run_cli("replay", load("loop-ttl.json"))[1]
        visited = payload["compression"]["visited"]
        self.assertEqual(visited, sorted(set(visited)))


if __name__ == "__main__":
    unittest.main()
