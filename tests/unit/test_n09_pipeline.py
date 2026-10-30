"""N09默认入口、图表来源与故障记录；独立于真实数据/GPU。"""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from occluactisafe.fusion.diagnostics import load_diagnostic_config, run_diagnostics, render_diagnostic_plots

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "scripts/check_fusion.py"


class FusionPipelineTests(unittest.TestCase):
    def test_strict_config_precedence_and_duplicate_yaml(self):
        c = load_diagnostic_config(overrides={"seed": 29}, cli_overrides={"seed": 0})
        self.assertEqual(c["seed"], 0)
        self.assertTrue(Path(c["output_root"]).is_absolute())
        self.assertEqual(c["filter"]["default_position_variance"], 1)
        for override in ({"future_truth": True}, {"seed": True}, {"output_root": "https://example.com"}):
            with self.assertRaises(ValueError): load_diagnostic_config(overrides=override)
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "duplicate.yaml"; p.write_text("seed: 1\nseed: 2\n", encoding="utf-8")
            with self.assertRaises(ValueError): load_diagnostic_config(p)

    def test_ide_no_cli_three_cwds_import_no_work_and_plot_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index, cwd in enumerate((ROOT, ROOT / "scripts", root)):
                output = root / f"out-{index}"
                code = ("import importlib.util,pathlib,sys; "
                        f"spec=importlib.util.spec_from_file_location('n09_entry',{str(ENTRY)!r}); "
                        "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module); "
                        f"assert not pathlib.Path({str(output)!r}).exists();assert 'torch' not in sys.modules; "
                        f"module.OVERRIDES={{'output_root':{str(output)!r}}};raise SystemExit(module.main([]))")
                proc = subprocess.run([sys.executable, "-B", "-c", code], cwd=cwd, capture_output=True,
                                      text=True, encoding="utf-8", timeout=60)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                runs = list(output.iterdir()); self.assertEqual(len(runs), 1)
                run = runs[0]
                result = json.loads((run / "result.json").read_text(encoding="utf-8"))
                self.assertEqual(result["status"], "passed")
                self.assertEqual(result["source_state_rows"], 100)
                self.assertEqual(result["source_measurements"], 94)
                self.assertEqual(result["final_unified_targets"], 3)
                with (run / "source_states.csv").open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 100)
                self.assertEqual({int(r["step"]) for r in rows}, set(range(1, 21)))
                manifest = json.loads((run / "plot_manifest.json").read_text(encoding="utf-8"))
                self.assertFalse(manifest["smoothing"]); self.assertFalse(manifest["downsampling"])
                for name, sha in {**manifest["sources"], **manifest["figures"]}.items():
                    self.assertEqual(hashlib.sha256((run / name).read_bytes()).hexdigest(), sha)
                if index == 0:
                    rebuilt = render_diagnostic_plots(run, output_dir=root / "rebuilt")
                    for name in ("source_filter_history.png", "association_gates.png"):
                        self.assertEqual(rebuilt["figures"][name], manifest["figures"][name])
                original = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
                self.assertIsNone(original["data"]); self.assertIsNone(original["weights"])

    def test_injected_plot_failure_retains_raw_state_and_missing_config_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = load_diagnostic_config(overrides={"output_root": str(root / "failed")})
            with patch("occluactisafe.fusion.diagnostics.render_diagnostic_plots", side_effect=RuntimeError("injected plot failure")):
                with self.assertRaisesRegex(RuntimeError, "injected plot failure"): run_diagnostics(config)
            run = next((root / "failed").iterdir())
            status = json.loads((run / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "failed"); self.assertEqual(status["reason"], "injected plot failure")
            for name in ("source_states.csv", "input_packets.json", "lifecycle_history.jsonl", "plot_failure.json", "evidence_index.json"):
                self.assertTrue((run / name).is_file(), name)
            proc = subprocess.run([sys.executable, "-B", str(ENTRY), "--config", str(root / "missing.yaml")],
                                  capture_output=True, text=True, encoding="utf-8", timeout=20)
            self.assertEqual(proc.returncode, 1); self.assertIn("N09失败", proc.stderr)
            config = load_diagnostic_config(overrides={"output_root": str(root / "not-mounted" / "out")})
            with self.assertRaises(FileNotFoundError): run_diagnostics(config)


if __name__ == "__main__": unittest.main()
