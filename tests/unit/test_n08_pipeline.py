"""N08默认入口、跨cwd、可复核图表及失败留存，不访问真实数据。"""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from occluactisafe.communication.diagnostics import load_diagnostic_config, run_diagnostics, render_diagnostic_plots

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "scripts/check_communication.py"


class CommunicationPipelineTests(unittest.TestCase):
    def test_config_precedence_invalid_fields_and_duplicate_yaml(self):
        config = load_diagnostic_config(overrides={"seed": 29}, cli_overrides={"seed": 0})
        self.assertEqual(config["seed"], 0)
        self.assertTrue(Path(config["output_root"]).is_absolute())
        with self.assertRaises(ValueError): load_diagnostic_config(overrides={"future_truth": True})
        with self.assertRaises(ValueError): load_diagnostic_config(overrides={"seed": True})
        with self.assertRaises(ValueError): load_diagnostic_config(overrides={"output_root": "https://example.com"})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "duplicate.yaml"
            path.write_text("seed: 1\nseed: 2\n", encoding="utf-8")
            with self.assertRaises(ValueError): load_diagnostic_config(path)

    def test_import_no_work_and_ide_no_cli_in_three_cwds_with_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)
            for index, cwd in enumerate((ROOT, ROOT / "scripts", target)):
                output = target / f"out-{index}"
                code = (
                    "import importlib.util, pathlib, sys; "
                    f"spec=importlib.util.spec_from_file_location('n08_entry', {str(ENTRY)!r}); "
                    "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
                    f"assert not pathlib.Path({str(output)!r}).exists(); "
                    "assert 'torch' not in sys.modules; "
                    f"module.OVERRIDES={{'output_root': {str(output)!r}}}; "
                    "raise SystemExit(module.main([]))"
                )
                proc = subprocess.run([sys.executable, "-B", "-c", code], cwd=cwd,
                                      capture_output=True, text=True, encoding="utf-8", timeout=60)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                runs = list(output.iterdir()); self.assertEqual(len(runs), 1)
                run = runs[0]
                report = json.loads((run / "result.json").read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "passed")
                self.assertEqual(report["checks_failed"], 0)
                with (run / "capacity_queue.csv").open(encoding="utf-8") as stream:
                    records = list(csv.DictReader(stream))
                self.assertEqual(len(records), 20)
                self.assertTrue(any(r["reason"] == "queue_timeout" for r in records))
                self.assertTrue((run / "message_lifecycle.jsonl").stat().st_size > 0)
                self.assertTrue(all(c["passed"] for c in json.loads((run / "checks.json").read_text(encoding="utf-8"))))
                manifest = json.loads((run / "plot_manifest.json").read_text(encoding="utf-8"))
                self.assertFalse(manifest["smoothing"]); self.assertFalse(manifest["downsampling"])
                for name, sha in {**manifest["sources"], **manifest["figures"]}.items():
                    self.assertEqual(hashlib.sha256((run / name).read_bytes()).hexdigest(), sha)
                if index == 0:
                    regenerated = render_diagnostic_plots(run, output_dir=target / "regenerated")
                    for name in ("capacity_queue.png", "duplex_timing.png"):
                        self.assertEqual(regenerated["figures"][name], manifest["figures"][name])
                env = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
                self.assertIsNone(env["data"]); self.assertIsNone(env["weights"])
                self.assertEqual(env["config"]["seed"], 17)

    def test_plot_failure_retains_raw_records_and_entry_missing_config_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "out"
            config = load_diagnostic_config(overrides={"output_root": str(output)})
            with patch("occluactisafe.communication.diagnostics.render_diagnostic_plots", side_effect=RuntimeError("injected plot failure")):
                with self.assertRaisesRegex(RuntimeError, "injected plot failure"):
                    run_diagnostics(config)
            run = next(output.iterdir())
            status = json.loads((run / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["reason"], "injected plot failure")
            self.assertTrue((run / "plot_failure.json").is_file())
            self.assertTrue((run / "capacity_queue.csv").is_file())
            self.assertTrue((run / "message_lifecycle.jsonl").stat().st_size > 0)
            self.assertTrue((run / "evidence_index.json").is_file())
            proc = subprocess.run([sys.executable, "-B", str(ENTRY), "--config", str(Path(temp) / "missing.yaml")],
                                  capture_output=True, text=True, encoding="utf-8", timeout=20)
            self.assertEqual(proc.returncode, 1)
            self.assertIn("N08失败", proc.stderr)
            # 缺失输出盘/父目录不自动换路径。
            config = load_diagnostic_config(overrides={"output_root": str(Path(temp) / "not-mounted" / "runs")})
            with self.assertRaises(FileNotFoundError): run_diagnostics(config)


if __name__ == "__main__":
    unittest.main()
