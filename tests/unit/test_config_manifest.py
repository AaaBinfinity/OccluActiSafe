"""N02 配置和运行清单的行为测试。"""

from copy import deepcopy
import contextlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from occluactisafe.common.config import DEFAULTS, PROJECT_ROOT, load_config, parse_config_cli
from occluactisafe.common.run_manifest import create_run_manifest


class ConfigTests(unittest.TestCase):
    def test_precedence_selected_config_and_false_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "configs/base.yaml").write_text("runtime: {seed: 1}\nlogging: {keep_truth_separate: true}\n", encoding="utf8")
            (root / "default.yaml").write_text("runtime: {seed: 7}\nlogging: {keep_truth_separate: true}\n", encoding="utf8")
            (root / "selected.yaml").write_text("runtime: {seed: 9}\nlogging: {save_failed_episodes: false}\n", encoding="utf8")
            source, cli = parse_config_cli([])
            self.assertIsNone(source)
            self.assertEqual(cli, {})
            config = load_config(project_root=root, default_config_file="default.yaml", cli_overrides=cli)
            self.assertEqual(config["runtime"]["seed"], 7)
            selected, cli = parse_config_cli(["--config", "selected.yaml", "--seed", "0", "--no-keep-truth-separate"])
            config = load_config(project_root=root, default_config_file="default.yaml", config_file=selected,
                                 overrides={"runtime": {"seed": 8}, "logging": {"keep_truth_separate": True}}, cli_overrides=cli)
            self.assertEqual(config["runtime"]["seed"], 0)
            self.assertIs(config["logging"]["keep_truth_separate"], False)
            self.assertIs(config["logging"]["save_failed_episodes"], False)
            config = load_config(project_root=root, default_config_file="default.yaml", config_file=selected)
            self.assertEqual(config["runtime"]["seed"], 9)
            self.assertIs(config["logging"]["keep_truth_separate"], True)
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit):
                parse_config_cli(["--unexpected-option"])
            self.assertIn("unrecognized arguments", errors.getvalue())

    def test_paths_are_independent_of_cwd(self):
        original = Path.cwd()
        try:
            config = load_config()
            with tempfile.TemporaryDirectory() as tmp:
                os.chdir(tmp)
                self.assertEqual(load_config(), config)
            self.assertEqual(config["paths"]["data_root"], str((PROJECT_ROOT / "data").resolve()))
            self.assertEqual(config["paths"]["output_root"], str((PROJECT_ROOT / "artifacts").resolve()))
        finally:
            os.chdir(original)

    def test_strict_values_and_period_consistency(self):
        cases = [
            {"unknown": 1}, {"runtime": {"seed": True}}, {"runtime": {"seed": -1}},
            {"logging": {"save_failed_episodes": 0}}, {"clock": {"sensor_dt": math.nan}},
            {"clock": {"sensor_dt": math.inf}}, {"clock": {"sensor_dt": 0.11}},
            {"clock": {"actuator_dt": 0}}, {"prediction": {"future_steps": 0}},
            {"prediction": {"future_steps": 1}}, {"planning": {"horizon_s": 3.05}},
            {"paths": {"data_root": "https://example.invalid/secret"}},
        ]
        for bad in cases:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                load_config(overrides=bad)
        self.assertEqual(DEFAULTS["runtime"]["seed"], 0)

    def test_yaml_rejects_duplicate_and_unknown_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "configs/base.yaml").write_text("{}\n", encoding="utf8")
            file = root / "task.yaml"
            for text in ("runtime:\n  seed: 1\n  seed: 2\n", "runtime:\n  wrong: 5\n", "[1, 2]\n"):
                file.write_text(text, encoding="utf8")
                with self.subTest(text=text), self.assertRaises(ValueError):
                    load_config(project_root=root, config_file="task.yaml")


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        self.root.mkdir()
        (self.root / "src").mkdir()
        (self.root / "src/module.py").write_text("value = 1\n", encoding="utf8")
        (self.root / "tests/fixtures").mkdir(parents=True)
        (self.root / "tests/fixtures/golden.json").write_text('{"field": 1}\n', encoding="utf8")
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(["git", "-C", str(self.root), "add", "src/module.py", "tests/fixtures/golden.json"], check=True)
        subprocess.run(["git", "-C", str(self.root), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
        self.output = Path(self.temp.name) / "runs"
        self.config = load_config()
        self.data = {"name": "fixture", "version": "1", "sha256": "a" * 64}

    def create(self, **changes):
        args = {"config": self.config, "input_mode": "analytic_observation", "data": self.data,
                "output_root": self.output, "project_root": self.root}
        args.update(changes)
        return create_run_manifest(**args)

    def test_unique_run_id_hash_and_persisted_content(self):
        first = self.create()
        second = self.create()
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertEqual(first["config"]["sha256"], second["config"]["sha256"])
        self.assertEqual(first["data"], self.data)
        self.assertEqual(first["seed"], 0)
        self.assertEqual(first["input_mode"], "analytic_observation")
        self.assertIs(first["source"]["working_tree_dirty"], False)
        self.assertEqual(len(first["source"]["git_commit"]), 40)
        self.assertEqual(json.loads((self.output / first["run_id"] / "manifest.json").read_text()), first)
        resolved_path = self.output / first["run_id"] / "resolved_config.yaml"
        self.assertEqual(load_config(config_file=resolved_path), self.config)
        self.assertIn("python", first["environment"])
        (self.root / "src/module.py").write_text("value = 2\n", encoding="utf8")
        third = self.create()
        self.assertIs(third["source"]["working_tree_dirty"], True)
        self.assertNotEqual(first["source"]["source_sha256"], third["source"]["source_sha256"])
        (self.root / "tests/fixtures/golden.json").write_text('{"field": 2}\n', encoding="utf8")
        fourth = self.create()
        self.assertNotEqual(third["source"]["source_sha256"], fourth["source"]["source_sha256"])

    def test_relative_output_root_does_not_depend_on_cwd(self):
        original = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as other:
                os.chdir(other)
                run = self.create(output_root="artifacts")
            self.assertTrue((self.root / "artifacts" / run["run_id"] / "manifest.json").is_file())
        finally:
            os.chdir(original)

    def test_reject_credentials_urls_invalid_data_and_nan(self):
        cases = []
        for key, value in (("api_key", "private"), ("nested", {"token": "private"}),
                           ("url", "https://user:private@example.invalid/"), ("nan", float("nan"))):
            cfg = deepcopy(self.config)
            cfg["extra"] = {key: value}
            cases.append({"config": cfg})
        cases.extend([{"input_mode": "Bearer abc"}, {"data": {"name": "x", "version": "1", "sha256": "bad"}},
                      {"data": {"name": "x", "version": "1", "sha256": "a" * 64, "password": "private"}}])
        for bad in cases:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.create(**bad)
        self.assertFalse(self.output.exists())

    def test_missing_data_is_explicit(self):
        self.assertIsNone(self.create(data=None)["data"])


if __name__ == "__main__":
    unittest.main()
