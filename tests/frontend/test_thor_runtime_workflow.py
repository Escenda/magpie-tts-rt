from __future__ import annotations

import ast
from pathlib import Path
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "thor-runtime.yml"
BENCHMARK_STEP = "- name: Run installed real-synthesis smoke and Thor benchmark"


def python_heredoc_after(source: str, marker: str) -> str:
    marker_offset = source.index(marker)
    heredoc_offset = source.index("<<'PY'\n", marker_offset) + len("<<'PY'\n")
    lines = source[heredoc_offset:].splitlines(keepends=True)
    heredoc_lines: list[str] = []
    for line in lines:
        if line.strip() == "PY":
            return textwrap.dedent("".join(heredoc_lines))
        heredoc_lines.append(line)
    raise ValueError("benchmark Python heredoc has no terminator")


class ThorRuntimeWorkflowTests(unittest.TestCase):
    def test_benchmark_receipt_script_defines_its_validation_dependencies(
        self,
    ) -> None:
        script = python_heredoc_after(
            WORKFLOW.read_text(encoding="utf-8"),
            BENCHMARK_STEP,
        )
        tree = ast.parse(script)
        compile(tree, str(WORKFLOW), "exec")

        imported_modules = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_names = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assigned_names = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }

        self.assertIn("copy", imported_modules)
        self.assertIn("ValidationError", imported_names)
        self.assertIn("validator", assigned_names)


if __name__ == "__main__":
    unittest.main()
