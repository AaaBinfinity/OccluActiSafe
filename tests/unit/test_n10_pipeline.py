"""N10默认入口、原始诊断/图表哈希及失败留存。"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch
from occluactisafe.risk.diagnostics import load_diagnostic_config, run_diagnostics, render_diagnostic_plots

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "scripts/check_risk.py"


class RiskPipelineTests(unittest.TestCase):
    def test_strict_config_and_duplicate_yaml(self):
        c = load_diagnostic_config(overrides={"output_root": "artifacts/a"}, cli_overrides={"output_root": "artifacts/b"})
        self.assertEqual(Path(c["output_root"]), ROOT / "artifacts/b")
        self.assertEqual(c["interval"]["min_interval_s"], .01)
        for override in ({"future_truth": True}, {"output_root": "https://example.com"}, {"output_root": ""}):
            with self.assertRaises(ValueError): load_diagnostic_config(overrides=override)
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "duplicate.yaml"; p.write_text("schema_version: x\nschema_version: y\n", encoding="utf-8")
            with self.assertRaises(ValueError): load_diagnostic_config(p)

    def test_ide_no_cli_three_cwds_import_no_work_and_rebuild_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index, cwd in enumerate((ROOT, ROOT / "scripts", root)):
                output = root / f"out-{index}"
                code = ("import importlib.util,pathlib,sys; "
                        f"spec=importlib.util.spec_from_file_location('n10_entry',{str(ENTRY)!r}); "
                        "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module); "
                        f"assert not pathlib.Path({str(output)!r}).exists();assert 'torch' not in sys.modules; "
                        f"module.OVERRIDES={{'output_root':{str(output)!r}}};raise SystemExit(module.main([]))")
                proc = subprocess.run([sys.executable, "-B", "-W", "error", "-c", code], cwd=cwd, capture_output=True,
                                      text=True, encoding="utf-8", timeout=60)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                runs = list(output.iterdir()); self.assertEqual(len(runs), 1); run = runs[0]
                result = json.loads((run / "result.json").read_text(encoding="utf-8"))
                self.assertEqual(result["status"], "passed")
                self.assertEqual(result["checks_passed"], 22)
                self.assertEqual(result["risk_scenarios"], 13)
                self.assertEqual(result["candidate_rows"], 14)
                manifest = json.loads((run / "plot_manifest.json").read_text(encoding="utf-8"))
                self.assertFalse(manifest["smoothing"]); self.assertFalse(manifest["downsampling"])
                self.assertEqual(len(manifest["figures"]), 6)
                for name, sha in {**manifest["sources"], **manifest["figures"]}.items():
                    self.assertEqual(hashlib.sha256((run / name).read_bytes()).hexdigest(), sha)
                original = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
                self.assertIsNone(original["data"]); self.assertIsNone(original["weights"])
                self.assertFalse(original["cuda_used"])
                if index == 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        rebuilt = render_diagnostic_plots(run, output_dir=root / "rebuilt")
                    for name in ("geometry_intervals.png", "raw_scores_coverage.png", "visibility_constraints.png"):
                        self.assertEqual(rebuilt["figures"][name], manifest["figures"][name])

    def test_injected_plot_failure_retains_raw_and_missing_mount_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); output = root / "failed"
            config = load_diagnostic_config(overrides={"output_root": str(output)})
            with patch("occluactisafe.risk.diagnostics.render_diagnostic_plots", side_effect=RuntimeError("injected plot failure")):
                with self.assertRaisesRegex(RuntimeError, "injected plot failure"): run_diagnostics(config)
            run = next(output.iterdir())
            status = json.loads((run / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["reason"], "injected plot failure")
            for name in ("evaluations.json", "risk_rows.csv", "interval_subdivision.json", "grids.json", "plot_failure.json", "evidence_index.json"):
                self.assertTrue((run / name).is_file(), name)
            proc = subprocess.run([sys.executable, "-B", str(ENTRY), "--config", str(root / "missing.yaml")],
                                  capture_output=True, text=True, encoding="utf-8", timeout=20)
            self.assertEqual(proc.returncode, 1); self.assertIn("N10失败", proc.stderr)
            config = load_diagnostic_config(overrides={"output_root": str(root / "not-mounted" / "out")})
            with self.assertRaises(FileNotFoundError): run_diagnostics(config)


if __name__ == "__main__": unittest.main()
