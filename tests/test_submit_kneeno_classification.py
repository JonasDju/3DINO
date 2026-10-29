"""The standalone eval's SLURM script (``dinov2/eval/submit_kneeno_classification.sh``): runs it with ``srun`` and the
venv's python stubbed, and checks which config and checkpoint it hands the evaluation, the ``CONFIG_FILE_NAME`` it
exports and the name it gives the job's log.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "dinov2" / "eval" / "submit_kneeno_classification.sh"
DEFAULT_CONFIG = "dinov2/configs/eval/kneeno-vitl-official.yaml"


class SubmitKneeNoClassificationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.repo = self.root / "repo"
        self.calls = self.root / "calls.txt"
        bin_dir, venv_bin, kneeno = self.root / "bin", self.repo / ".venv" / "bin", self.root / "KneeNo" / "data"
        for directory in (bin_dir, venv_bin, kneeno, self.repo / "configs"):
            directory.mkdir(parents=True)
        (kneeno / "prepare_data.py").touch()
        (bin_dir / "srun").write_text('#!/bin/bash\nexec "$@"\n')
        # Every call is recorded with the CONFIG_FILE_NAME it sees
        (venv_bin / "python").write_text(
            f'#!/bin/bash\necho "$* CONFIG_FILE_NAME=$CONFIG_FILE_NAME" >> {self.calls}\n'
        )
        for script in (bin_dir / "srun", venv_bin / "python"):
            script.chmod(0o755)
        (self.repo / DEFAULT_CONFIG).parent.mkdir(parents=True)
        (self.repo / DEFAULT_CONFIG).touch()
        for name in ("query2label.yaml", "query2label.yml"):
            (self.repo / "configs" / name).touch()
        self.checkpoint = self.root / "teacher_checkpoint.pth"
        self.checkpoint.touch()
        # The log SLURM opened before the script ran (#SBATCH --output, relative to the submit directory)
        (self.repo / "slurm_logs").mkdir()
        (self.repo / "slurm_logs" / "123.log").touch()
        self.env = {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "SLURM_SUBMIT_DIR": str(self.repo),
            "SLURM_JOB_ID": "123",
            "KNEENO_DIR": str(self.root / "KneeNo"),
            "TMP": str(self.root / "scratch"),
            "CHECKPOINT": str(self.checkpoint),
        }

    def _run(self, **env):
        result = subprocess.run(["bash", str(SCRIPT)], env={**self.env, **env}, capture_output=True, text=True)
        lines = self.calls.read_text().splitlines() if self.calls.exists() else []
        return result, [line for line in lines if "kneeno_classification" in line]

    def _logs(self):
        return sorted(p.name for p in (self.repo / "slurm_logs").iterdir())

    def test_config(self):
        cases = {
            "default": ({}, DEFAULT_CONFIG),
            "config": ({"CONFIG": "configs/query2label.yaml"}, "configs/query2label.yaml"),
            "base_dir_relative": (
                {"CONFIG_BASE_DIR": "configs", "CONFIG_FILE_NAME": "query2label"},
                "configs/query2label.yaml",
            ),
            "base_dir_absolute": (
                {"CONFIG_BASE_DIR": str(self.repo / "configs"), "CONFIG_FILE_NAME": "query2label"},
                "configs/query2label.yaml",
            ),
        }
        for case, (env, config) in cases.items():
            with self.subTest(case):
                self.calls.unlink(missing_ok=True)
                for log in (self.repo / "slurm_logs").iterdir():
                    log.unlink()
                (self.repo / "slurm_logs" / "123.log").touch()
                result, launches = self._run(**env)
                self.assertEqual(result.returncode, 0, result.stderr)
                # The evaluation gets the config's path and, either way, its file name without the .yaml for the
                # output folders
                path = self.repo / config
                self.assertEqual(
                    launches,
                    [
                        f"-m dinov2.eval.kneeno_classification --config {path} --checkpoint {self.checkpoint} "
                        f"--device cuda:0 CONFIG_FILE_NAME={path.stem}"
                    ],
                )
                # The job's log is renamed after the config and the split
                self.assertEqual(self._logs(), [f"type-{path.stem}.level-1.0.rep-0.123.log"])

    def test_default_checkpoint(self):
        # The official ViT-L weights, which the default config describes
        default = self.root / "public_checkpoints" / "3dino" / "3dino_vit_weights.pth"
        default.parent.mkdir(parents=True)
        default.touch()
        result, launches = self._run(CHECKPOINT="", HPCWORK=str(self.root))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"--checkpoint {default} ", launches[0])

    def test_log_name_has_the_split(self):
        result, _ = self._run(
            CONFIG_BASE_DIR="configs", CONFIG_FILE_NAME="query2label", SPLIT_LEVEL="0.05", SPLIT_REP="3"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._logs(), ["type-query2label.level-0.05.rep-3.123.log"])

    def test_config_errors(self):
        cases = {
            "config_and_file_name": (
                {"CONFIG": "configs/query2label.yaml", "CONFIG_FILE_NAME": "query2label"},
                "not both",
            ),
            "config_and_base_dir": ({"CONFIG": "configs/query2label.yaml", "CONFIG_BASE_DIR": "configs"}, "not both"),
            "base_dir_only": ({"CONFIG_BASE_DIR": "configs"}, "must be set together"),
            "file_name_only": ({"CONFIG_FILE_NAME": "query2label"}, "must be set together"),
            "file_name_path": (
                {"CONFIG_BASE_DIR": ".", "CONFIG_FILE_NAME": "configs/query2label"},
                "without the .yaml",
            ),
            "file_name_with_yaml": (
                {"CONFIG_BASE_DIR": "configs", "CONFIG_FILE_NAME": "query2label.yaml"},
                "without the .yaml",
            ),
            "config_not_yaml": ({"CONFIG": "configs/query2label.yml"}, "must be a .yaml file"),
        }
        for case, (env, match) in cases.items():
            with self.subTest(case):
                result, launches = self._run(**env)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(match, result.stderr)
                self.assertEqual(launches, [])
                # Invalid CONFIG* vars stop the script before the log is renamed
                self.assertEqual(self._logs(), ["123.log"])

    def test_missing_inputs_fail_before_the_data_extraction(self):
        cases = {
            "config": ({"CONFIG_BASE_DIR": "configs", "CONFIG_FILE_NAME": "missing"}, "Eval config not found"),
            "checkpoint": ({"CHECKPOINT": str(self.root / "missing.pth")}, "Checkpoint not found"),
        }
        for case, (env, match) in cases.items():
            with self.subTest(case):
                self.calls.unlink(missing_ok=True)
                result, _ = self._run(**env)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(match, result.stderr)
                # Not even prepare_data.py ran
                self.assertFalse(self.calls.exists())


if __name__ == "__main__":
    unittest.main()
