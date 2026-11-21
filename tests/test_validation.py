"""Structural validation and CLI contract tests."""

from __future__ import annotations

import json
import os
import unittest

from helpers import EXAMPLES, clone, load, run_cli


class ValidationTests(unittest.TestCase):
    def test_all_examples_are_structurally_valid(self):
        for name in sorted(os.listdir(EXAMPLES)):
            if not name.endswith(".json"):
                continue
            with self.subTest(example=name):
                code, payload = run_cli("check", os.path.join(EXAMPLES, name))
                self.assertEqual(code, 0, msg=json.dumps(payload, ensure_ascii=False)[:500])
                self.assertTrue(payload["ok"])

    def test_missing_member_reports_field_and_code(self):
        code, payload = run_cli("check", {"case_version": 1})
        self.assertEqual(code, 2)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "E_MISSING")
        fields = {e["field"] for e in payload["errors"]}
        self.assertIn("$.ttl", fields)
        self.assertIn("$.policy", fields)

    def test_unknown_reference_is_located(self):
        case = load("golden-path.json")
        case["links"][0]["to"] = "does-not-exist"
        code, payload = run_cli("check", case)
        self.assertEqual(code, 2)
        self.assertEqual(payload["error"]["code"], "E_REFERENCE")
        self.assertEqual(payload["error"]["field"], "$.links[0].to")

    def test_duplicate_identifier_is_located(self):
        case = load("golden-path.json")
        case["nodes"].append(dict(case["nodes"][0]))
        code, payload = run_cli("check", case)
        self.assertEqual(code, 2)
        self.assertEqual(payload["error"]["code"], "E_DUPLICATE")
        self.assertIn("$.nodes[", payload["error"]["field"])

    def test_bad_enum_is_located(self):
        case = load("golden-path.json")
        case["actions"][0]["kind"] = "explode"
        code, payload = run_cli("check", case)
        self.assertEqual(code, 2)
        self.assertEqual(payload["error"]["code"], "E_ENUM")
        self.assertEqual(payload["error"]["field"], "$.actions[0].kind")

    def test_declared_limits_are_enforced(self):
        case = load("golden-path.json")
        case["ttl"] = 7
        code, payload = run_cli("check", case)
        self.assertEqual(code, 2)
        self.assertIn(payload["error"]["code"], {"E_RANGE", "E_LIMIT"})
        self.assertIn(
            "$.ttl", {e["field"] for e in payload.get("errors", [payload["error"]])}
        )

    def test_action_limit_is_enforced(self):
        case = load("golden-path.json")
        for i in range(20):
            case["actions"].append(
                {"id": f"a-extra-{i}", "kind": "pause", "proxy": "px-a"}
            )
        code, payload = run_cli("check", case)
        self.assertEqual(code, 2)
        self.assertEqual(payload["error"]["code"], "E_LIMIT")
        self.assertEqual(payload["error"]["field"], "$.actions")

    def test_request_tenant_must_match_identity(self):
        case = load("golden-path.json")
        case["requests"][0]["tenant"] = "t-beta"
        code, payload = run_cli("check", case)
        self.assertEqual(code, 2)
        self.assertEqual(payload["error"]["code"], "E_ACTION")
        self.assertEqual(payload["error"]["field"], "$.requests[0].tenant")

    def test_unknown_timeline_action_is_a_reference_error(self):
        case = load("golden-path.json")
        case["timeline"].append({"id": "no-such-action", "step": 3})
        code, payload = run_cli("check", case)
        self.assertEqual(code, 2)
        self.assertEqual(payload["error"]["code"], "E_REFERENCE")

    def test_usage_error(self):
        code, payload = run_cli("nonsense", load("golden-path.json"))
        self.assertEqual(code, 2)
        self.assertEqual(payload["error"]["code"], "E_USAGE")

    def test_missing_file(self):
        code, payload = run_cli("check", os.path.join(EXAMPLES, "nope.json"))
        self.assertEqual(code, 2)
        self.assertEqual(payload["error"]["code"], "E_IO")


class CheckSummaryTests(unittest.TestCase):
    def test_summary_counts(self):
        code, payload = run_cli("check", os.path.join(EXAMPLES, "golden-path.json"))
        self.assertEqual(code, 0)
        self.assertEqual(payload["counts"]["nodes"], 3)
        self.assertEqual(payload["counts"]["links"], 2)
        self.assertEqual(payload["limits"], {"proxies": 14, "actions": 16, "slots": 10, "ttl": 6})


if __name__ == "__main__":
    unittest.main()
