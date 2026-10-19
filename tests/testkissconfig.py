"""Check KISS configuration routing without running analysis or accelerator tests."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class KissConfigTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="blckbaux-kiss-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.config = self.root / ".config/kiss.toml"
        self.config.parent.mkdir()
        self.config.write_text("[global]\n", encoding="utf-8")
        self.ignore = self.root / ".config/kiss.ignore"
        self.ignore.write_text(
            "# Repository-relative paths\nskipped/\n", encoding="utf-8"
        )
        (self.root / "skipped").mkdir()
        self.upstream = self.root / "upstream"
        self.upstream.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
        self.upstream.chmod(0o755)

    def invoke(self, *args):
        return subprocess.run(
            [str(ROOT / "tools/kiss"), *args],
            cwd=self.root,
            env=os.environ
            | {
                "PIXI_PROJECT_ROOT": str(self.root),
                "KISS_UPSTREAM_BIN": str(self.upstream),
            },
            capture_output=True,
            text=True,
            check=False,
        )

    def test_default(self):
        result = self.invoke("check", ".")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            ["--config", str(self.config), "check", ".", "--ignore", "skipped"],
        )
        self.assertFalse((self.root / ".kissconfig").exists())

    def test_custom(self):
        custom = self.root / "custom.toml"
        custom.write_text("[global]\n", encoding="utf-8")
        self.config.unlink()
        for option in (("--config", "custom.toml"), ("--config=custom.toml",)):
            with self.subTest(option=option):
                result = self.invoke(*option, "rules")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), [*option, "rules"])

    def test_python(self):
        result = self.invoke("test", "--lang", "python")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            ["--config", str(self.config), "test", "--lang", "python"],
        )

    def test_missing(self):
        for path in (self.config, self.ignore):
            with self.subTest(path=path.name):
                saved = path.read_text(encoding="utf-8")
                path.unlink()
                result = self.invoke("check", ".")
                self.assertEqual(result.returncode, 2)
                self.assertIn(str(path), result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertFalse((self.root / ".kissconfig").exists())
                path.write_text(saved, encoding="utf-8")

    def test_rust(self):
        target = "//build/buck2/tests:all"
        (self.root / "tools").mkdir()
        (self.root / "tools/buck2w").symlink_to(self.upstream)
        for option in ((), ("--config", str(self.config))):
            with self.subTest(option=option):
                result = self.invoke(*option, "test", "--dry-run")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), ["targets", target])
